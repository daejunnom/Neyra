import type { BagCandidates, Candidate, CandidateView, EngineCommand, EngineEvent, RunStatus, SearchConfig, SearchSnapshot, TrainingField } from './types';
import { base } from '$app/paths';
import { version } from '$app/environment';

type WorkerSlot = {
  id: number;
  worker: Worker;
  status: RunStatus;
  snapshot: SearchSnapshot | null;
  incumbent: { candidate: Candidate; data: unknown } | null;
  watching: boolean;
};
type OwnedCandidate = { slot: WorkerSlot; candidate: Candidate };
type ArchivedCandidate = OwnedCandidate & { data: unknown };

export const MAX_WORKERS = 128;

/** The browser can expose fewer logical processors than the operating system. */
export function workerLimit(hardware = typeof navigator === 'undefined' ? 1 : navigator.hardwareConcurrency): number {
  return Math.max(1, Math.min(MAX_WORKERS, Number.isFinite(hardware) && hardware > 0 ? Math.floor(hardware) : 1));
}

/** Automatic mode follows the browser; explicit counts can override its hint. */
export function workerCount(requested: number, hardware = typeof navigator === 'undefined' ? 1 : navigator.hardwareConcurrency): number {
  const cores = workerLimit(hardware);
  const desired = Number.isFinite(requested) && requested > 0 ? Math.floor(requested) : cores;
  return Math.max(1, Math.min(MAX_WORKERS, desired));
}

export function workerSeed(seed: number, index: number): number {
  return ((seed - 1 + index * 2654435761) % 4294967295) + 1;
}

/** A fresh, nonzero u32 for this page entry; a manual seed stays valid within the tab. */
export function randomSeed(): number {
  const value = new Uint32Array(1);
  do { crypto.getRandomValues(value); } while (value[0] === 0);
  return value[0];
}

function atBudget(candidate: Candidate): number {
  return candidate.cdf.find((point) => point.pieces === candidate.budgetPieces)?.probability ?? candidate.probability;
}

/** Complete results use the same lock-cost and probability order as Rust. */
export function betterComplete(a: Candidate, b: Candidate): boolean {
  const aCost = a.budgetPieces ?? Number.MAX_SAFE_INTEGER;
  const bCost = b.budgetPieces ?? Number.MAX_SAFE_INTEGER;
  return aCost < bCost || (aCost === bCost &&
    (atBudget(a) > atBudget(b) || (atBudget(a) === atBudget(b) && a.probability > b.probability)));
}

/** Displayed efficiency includes the opening bag; count at least one bag. */
export function countedPieces(candidate: Candidate): number {
  return Math.max(7, candidate.placed);
}

export function spinsPerBag(candidate: Candidate): number {
  return candidate.spins * 7 / countedPieces(candidate);
}

function probabilityFactor(candidate: Candidate): number {
  return Number.isFinite(candidate.probability) ? Math.max(0, Math.min(1, candidate.probability)) : 0;
}

/** Use the already verified counted probability, including the chosen opening scope. */
export function probabilityAdjustedSpinsPerBag(candidate: Candidate): number {
  return spinsPerBag(candidate) * probabilityFactor(candidate);
}

function fieldQuality(candidate: Candidate): number {
  return Number.isFinite(candidate.fieldQuality) ? candidate.fieldQuality : -Infinity;
}

function sameForecast(a: Candidate, b: Candidate): boolean {
  const aBags = a.forecastSpinsByBag ?? [];
  const bBags = b.forecastSpinsByBag ?? [];
  return a.forecastSpins === b.forecastSpins && a.forecastPlacements === b.forecastPlacements &&
    aBags.length === bBags.length && aBags.every((spins, bag) => spins === bBags[bag]);
}

/** Compare foundations by future field quality, then productive routes by risk-adjusted efficiency. */
export function betterPartial(a: Candidate, b: Candidate): boolean {
  const aProbability = probabilityFactor(a);
  const bProbability = probabilityFactor(b);
  if (a.bagCount === 1 && b.bagCount > 1) return false;
  if (b.bagCount === 1 && a.bagCount > 1) return true;
  if (a.bagCount === 1 && b.bagCount === 1) {
    if (fieldQuality(a) !== fieldQuality(b)) return fieldQuality(a) > fieldQuality(b);
    const aStart = Number.isFinite(a.firstBagProbability) ? Math.max(0, Math.min(1, a.firstBagProbability!)) : 0;
    const bStart = Number.isFinite(b.firstBagProbability) ? Math.max(0, Math.min(1, b.firstBagProbability!)) : 0;
    if (aStart !== bStart) return aStart > bStart;
    if (aProbability !== bProbability) return aProbability > bProbability;
    if (a.spins !== b.spins) return a.spins > b.spins;
    return a.placed < b.placed;
  }
  const aRate = a.spins * aProbability * countedPieces(b);
  const bRate = b.spins * bProbability * countedPieces(a);
  if (aRate !== bRate) return aRate > bRate;
  if (aProbability !== bProbability) return aProbability > bProbability;
  if (a.bagCount !== b.bagCount) return a.bagCount > b.bagCount;
  if (a.spins !== b.spins) return a.spins > b.spins;
  if (fieldQuality(a) !== fieldQuality(b)) return fieldQuality(a) > fieldQuality(b);
  return a.placed < b.placed || (a.placed === b.placed && a.probability > b.probability);
}

function sameBagProbabilities(a: Candidate, b: Candidate): boolean {
  const aBags = a.bagProbabilities ?? [];
  const bBags = b.bagProbabilities ?? [];
  return aBags.length === bBags.length && aBags.every((probability, bag) => probability === bBags[bag]);
}

/** A plan ID alone cannot distinguish an improved record from its old export. */
function samePartialRecord(a: Candidate, b: Candidate): boolean {
  return a.id === b.id && a.spins === b.spins && a.placed === b.placed && a.bagCount === b.bagCount &&
    fieldQuality(a) === fieldQuality(b) && sameForecast(a, b) && a.probability === b.probability &&
    sameBagProbabilities(a, b) &&
    a.firstBagProbability === b.firstBagProbability && a.firstBagSymmetry === b.firstBagSymmetry &&
    a.firstBagSymmetryAssumed === b.firstBagSymmetryAssumed && a.frameCount === b.frameCount &&
    a.rows.length === b.rows.length && a.rows.every((row, y) => row === b.rows[y]);
}

function complete(candidate: Candidate | null | undefined): candidate is Candidate {
  return !!candidate?.complete && candidate.qualified && candidate.verified && candidate.budgetPieces !== null;
}

export class SearchClient {
  private slots: WorkerSlot[] = [];
  private runId = '';
  private runConfig: SearchConfig | null = null;
  private intent: 'running' | 'paused' | 'cancelled' = 'cancelled';
  private status: RunStatus = 'idle';
  private currentOwner: number | null = null;
  private observingCurrent = false;
  private nextFocusAt = 0;
  private lastProgressAt = 0;
  private progressTimer: ReturnType<typeof setTimeout> | undefined;
  private pendingExports = new Map<string, { owner: number; view: CandidateView; candidateId: string }>();
  private bagRecords = new Map<number, ArchivedCandidate[]>();
  private selectedArchive: ArchivedCandidate | null = null;
  private bestSpinsByBag = Array<number>(101).fill(0);
  private shareSpinBenchmarks = true;

  constructor(private receive: (event: EngineEvent) => void,
    private contribute: (field: TrainingField) => void = () => {}) {}

  start(config: SearchConfig) {
    this.dispose();
    this.runId = crypto.randomUUID();
    this.runConfig = { ...config, cnn_evaluation: false };
    const runId = this.runId;
    this.intent = 'running';
    this.status = 'loading';
    this.shareSpinBenchmarks = config.min_success_percent === 100;
    this.currentOwner = null;
    this.nextFocusAt = 0;
    this.lastProgressAt = 0;
    const count = workerCount(config.worker_count);
    const { worker_count: _workerCount, ...engineConfig } = config;
    let creationFailure = '';
    for (let index = 0; index < count; index++) {
      let worker: Worker | undefined;
      try {
        worker = new Worker(new URL('./search.worker.ts', import.meta.url), { type: 'module' });
        const id = this.slots.length;
        const slot: WorkerSlot = { id, worker, status: 'loading', snapshot: null, incumbent: null,
          watching: this.observingCurrent && id === 0 };
        worker.onmessage = (event: MessageEvent<EngineEvent>) => {
          if (this.runId !== runId || event.data.runId !== runId) return;
          this.handle(slot, event.data);
        };
        worker.onerror = (event) => {
          if (this.runId === runId) this.failed(slot, event.message || '탐색기를 불러오지 못했어요.');
        };
        this.slots.push(slot);
      } catch (error) {
        worker?.terminate();
        creationFailure = error instanceof Error ? error.message : String(error);
      }
    }
    // Divide the root work among the workers that were actually constructed.
    // A synchronous creation failure must not leave a hole in the lane range.
    for (const slot of this.slots) {
      try {
        slot.worker.postMessage({
          type: 'start', runId,
          config: { ...engineConfig, cnn_evaluation: false, seed: workerSeed(config.seed, slot.id), search_profile: slot.id % 4,
            search_lane: slot.id, search_lanes: this.slots.length },
          assetBase: `${location.origin}${base}/wasm/`, assetVersion: version, watchCurrent: slot.watching
        } satisfies EngineCommand);
      } catch (error) {
        this.failed(slot, error instanceof Error ? error.message : String(error));
      }
    }
    if (this.slots.length === 0) {
      this.status = 'failed';
      this.receive({ type: 'error', runId, message: creationFailure || '이 환경에서 탐색기를 시작할 수 없어요.' });
    } else if (this.slots.some((slot) => slot.status !== 'failed')) {
      this.receive({ type: 'status', runId, status: 'loading' });
      if (creationFailure) this.receive({ type: 'warning', runId, message: `일부 탐색기를 시작하지 못해 ${this.slots.length}개로 진행합니다.` });
    }
  }

  pause() {
    if (this.intent !== 'running') return;
    this.intent = 'paused';
    this.broadcast('pause');
    this.publishStatus();
  }

  /** Stream changing full current histories only while that tab is visible.
   * Verified record archives and compact progress remain independent. */
  observeCurrent(enabled: boolean) {
    if (this.observingCurrent === enabled) return;
    this.observingCurrent = enabled;
    this.watchCurrentOwner(enabled ? this.currentCandidateOwner()?.slot.id : undefined);
  }

  resume() {
    if (this.intent !== 'paused') return;
    this.intent = 'running';
    this.nextFocusAt = performance.now() + 2000;
    this.broadcast('resume');
    this.publishStatus();
  }

  cancel() {
    this.intent = 'cancelled';
    this.broadcast('cancel');
    this.publishStatus();
  }

  /** Pin the actual selected certificate; later rankings do not switch its replay. */
  selectCandidate(id: string | null): boolean {
    if (id === null) {
      this.selectedArchive = null;
      this.publishProgress(true);
      return true;
    }
    const record = [...this.bagRecords.values()].flat().find((item) => this.ownedId(item) === id);
    if (!record) return false;
    this.selectedArchive = record;
    this.publishProgress(true);
    return true;
  }

  export(view: 'current' | 'best' | 'partial' = 'best', requestId = crypto.randomUUID()): boolean {
    const owner = view === 'current' ? this.currentCandidateOwner() :
      view === 'partial' ? this.partialOwner() : this.selectedArchive ?? this.bestOwner() ?? this.partialOwner();
    if (!owner) return false;
    const selection: CandidateView = view === 'current' ? 'current' : owner.candidate.complete ? 'best' : 'partial';
    if (selection === 'best' && owner.slot.incumbent?.candidate.id === owner.candidate.id) {
      // A completed certificate remains usable even if its worker later fails.
      this.receive({ type: 'export', runId: this.runId, requestId, data: owner.slot.incumbent.data });
      return true;
    }
    const archived = this.findArchive(owner);
    if (selection === 'partial' && archived) {
      this.receive({ type: 'export', runId: this.runId, requestId, data: archived.data });
      return true;
    }
    if (owner.slot.status === 'failed') {
      this.receive({ type: 'export-error', runId: this.runId, requestId, message: '이 경로의 기록을 받기 전에 탐색기가 중단됐어요. 이미 보관한 결과는 계속 저장할 수 있습니다.' });
      return false;
    }
    const requestView: CandidateView = selection === 'partial' ? `candidate:${owner.candidate.id}` : selection;
    this.pendingExports.set(requestId, { owner: owner.slot.id, view: requestView, candidateId: owner.candidate.id });
    try {
      owner.slot.worker.postMessage({ type: 'export', runId: this.runId, requestId, view: requestView } satisfies EngineCommand);
      return true;
    } catch {
      this.pendingExports.delete(requestId);
      this.receive({ type: 'export-error', runId: this.runId, requestId,
        message: '이 후보의 문서를 요청하지 못했어요. 다른 후보를 선택해 주세요.' });
      return false;
    }
  }

  /** Export the retained certified boundaries, including a selection displaced
   * by later records. A run is one training family to keep its related boards
   * together when splitting training and validation data. */
  trainingFields(): TrainingField[] {
    if (!this.runConfig || !this.runId) return [];
    const records: OwnedCandidate[] = [...this.bagRecords.values()].flat();
    if (this.selectedArchive) records.push(this.selectedArchive);
    for (const slot of this.slots) {
      if (slot.incumbent) records.push({ slot, candidate: slot.incumbent.candidate });
    }
    const seen = new Set<string>();
    const fields: TrainingField[] = [];
    for (const record of records) {
      const id = this.ownedId(record);
      const field = this.trainingField(record);
      if (seen.has(id) || !field) continue;
      seen.add(id);
      fields.push(field);
    }
    return fields.sort((a, b) => a.source.bagBoundary - b.source.bagBoundary || a.id.localeCompare(b.id));
  }

  private trainingField(record: OwnedCandidate): TrainingField | null {
    if (!this.runConfig || !this.runId) return null;
    const config = this.runConfig;
    const candidate = record.candidate;
    const id = this.ownedId(record);
    // A terminal result may stop inside a bag. Only completed post-clear
    // boundaries with a captured representative are useful collection origins.
    if (!candidate.verified || !candidate.qualified ||
        !Number.isInteger(candidate.bagCount) || candidate.bagCount < 1 ||
        candidate.placed !== candidate.bagCount * 7 ||
        candidate.frames.length !== candidate.frameCount || candidate.rows.length !== 24 ||
        !candidate.rows.some((row) => row !== 0) ||
        !candidate.rows.every((row) => Number.isInteger(row) && row >= 0 && row < 1023)) return null;
    return { schemaVersion: 1, id: `${this.runId}:${id}`, rows: [...candidate.rows], source: {
      kind: 'search', family: `search:${this.runId}`,
      reference: `neyra:run:${this.runId}/candidate:${id}`,
      runId: this.runId, candidateId: id, bagBoundary: candidate.bagCount,
      probability: candidate.probability, firstBagProbability: candidate.firstBagProbability ?? null,
      ignoreFirstBagProbability: config.ignore_first_bag_probability, verified: true, qualified: true,
      seed: workerSeed(config.seed, record.slot.id), runSeed: config.seed,
      modelMode: 'heuristic', buildVersion: version,
      rules: 'neyra-srsplus-sonicdrop-clearonlyhold-v1',
      searchConfig: {
        minSuccessPercent: config.min_success_percent,
        bagMinSuccessPercent: config.bag_min_success_percent,
        firstBagMinSuccessPercent: config.first_bag_min_success_percent,
        riskMinSpinsPerBag: config.risk_min_spins_per_bag,
        pieceGoal: config.piece_goal, maxPieces: config.max_pieces, beamWidth: config.beam_width
      }
    } };
  }

  private contributeRecord(record: OwnedCandidate) {
    const field = this.trainingField(record);
    if (!field) return;
    // Data transport is optional and must never interrupt search or alter a record.
    try { this.contribute(field); } catch { /* The transport reports its own state. */ }
  }

  dispose() {
    clearTimeout(this.progressTimer);
    this.progressTimer = undefined;
    for (const slot of this.slots) slot.worker.terminate();
    this.slots = [];
    this.pendingExports.clear();
    this.bagRecords.clear();
    this.selectedArchive = null;
    this.bestSpinsByBag.fill(0);
    // Invalidate queued messages before a new generation starts.
    this.runId = '';
    this.runConfig = null;
    this.intent = 'cancelled';
  }

  private broadcast(type: 'pause' | 'resume' | 'cancel') {
    for (const slot of this.slots) {
      if (slot.status !== 'failed' && slot.status !== 'completed' && slot.status !== 'cancelled') {
        slot.worker.postMessage({ type, runId: this.runId } satisfies EngineCommand);
      }
    }
  }

  private handle(slot: WorkerSlot, message: EngineEvent) {
    if (message.type === 'progress') {
      this.updateSnapshot(slot, message.snapshot);
      this.publishProgress();
    } else if (message.type === 'status') {
      // A start acknowledgement can arrive after pause/cancel was requested.
      // Keep waiting for the matching control acknowledgement from every worker.
      slot.status = message.status;
      this.publishStatus();
      this.publishProgress(true);
    } else if (message.type === 'incumbent') {
      if (complete(message.candidate) &&
          (!slot.incumbent || betterComplete(message.candidate, slot.incumbent.candidate))) {
        slot.incumbent = { candidate: message.candidate, data: message.data };
        this.contributeRecord({ slot, candidate: message.candidate });
      }
      this.publishProgress();
    } else if (message.type === 'bag-record') {
      this.retainBagRecord(slot, message.candidate, message.data);
      this.publishProgress();
    } else if (message.type === 'error') {
      this.failed(slot, message.message);
    } else if (message.type === 'export' || message.type === 'export-error') {
      const request = this.pendingExports.get(message.requestId);
      if (!request || request.owner !== slot.id) return;
      this.pendingExports.delete(message.requestId);
      if (message.type === 'export-error') {
        this.receive({ type: 'export-error', runId: this.runId, requestId: message.requestId, message: message.message });
      } else {
        const candidate = (message.data as { candidate?: Candidate } | null)?.candidate;
        if (request.view !== 'current' && (!candidate || candidate.id !== request.candidateId)) {
          this.receive({ type: 'export-error', runId: this.runId, requestId: message.requestId,
            message: '선택한 후보의 기록이 바뀌었어요. 후보를 다시 선택해 주세요.' });
          return;
        }
        if (message.snapshot) this.updateSnapshot(slot, message.snapshot);
        if (request.view === 'current') {
          this.currentOwner = slot.id;
          this.nextFocusAt = performance.now() + 2000;
        }
        this.publishProgress(true);
        this.receive({ type: 'export', runId: this.runId, requestId: message.requestId, data: message.data });
      }
    }
  }

  private failed(slot: WorkerSlot, message: string) {
    if (slot.status === 'failed') return;
    slot.status = 'failed';
    slot.worker.terminate();
    for (const [requestId, request] of this.pendingExports) {
      if (request.owner === slot.id) {
        this.pendingExports.delete(requestId);
        this.receive({ type: 'export-error', runId: this.runId, requestId,
          message: '문서를 받기 전에 이 후보의 탐색기가 중단됐어요. 보관된 최선 경로를 선택해 주세요.' });
      }
    }
    if (this.slots.every((candidate) => candidate.status === 'failed')) {
      this.status = 'failed';
      this.receive({ type: 'error', runId: this.runId, message });
    } else {
      this.receive({ type: 'warning', runId: this.runId, message: '한 탐색기가 중단되어 남은 탐색기로 계속 진행합니다.' });
      this.publishStatus();
    }
    this.publishProgress(true);
  }

  private publishStatus() {
    const available = this.slots.filter((slot) => slot.status !== 'failed');
    if (available.length === 0) return;
    let next: RunStatus;
    if (this.intent === 'cancelled') {
      next = available.every((slot) => slot.status === 'cancelled' || slot.status === 'completed') ? 'cancelled' : this.status;
    } else if (this.intent === 'paused') {
      next = available.every((slot) => slot.status === 'paused' || slot.status === 'completed') ? 'paused' : this.status;
    } else if (available.every((slot) => slot.status === 'completed')) {
      next = 'completed';
    } else if (available.some((slot) => slot.status === 'running')) {
      next = 'running';
    } else {
      next = 'loading';
    }
    if (next !== this.status) {
      this.status = next;
      this.receive({ type: 'status', runId: this.runId, status: next });
    }
  }

  private bestOwner(): OwnedCandidate | null {
    let best: OwnedCandidate | null = null;
    for (const slot of this.slots) {
      const candidates = [slot.status === 'failed' ? null : slot.snapshot?.best, slot.incumbent?.candidate];
      for (const candidate of candidates) {
        if (complete(candidate) && (!best || betterComplete(candidate, best.candidate))) best = { slot, candidate };
      }
    }
    return best;
  }

  private partialOwner(): OwnedCandidate | null {
    let best: OwnedCandidate | null = null;
    for (const records of this.bagRecords.values()) {
      for (const record of records) {
        if (!best || betterPartial(record.candidate, best.candidate)) best = record;
      }
    }
    for (const slot of this.slots) {
      const candidate = slot.snapshot?.bestPartial;
      if (slot.status !== 'failed' && candidate?.verified && candidate.qualified && !candidate.complete &&
          (!best || betterPartial(candidate, best.candidate))) best = { slot, candidate };
    }
    return best;
  }

  private ownedId(item: OwnedCandidate): string {
    return `w${item.slot.id}:${item.candidate.id}`;
  }

  private findArchive(item: OwnedCandidate): ArchivedCandidate | null {
    const matches = (record: ArchivedCandidate) => record.slot.id === item.slot.id &&
      samePartialRecord(record.candidate, item.candidate);
    if (this.selectedArchive && matches(this.selectedArchive)) return this.selectedArchive;
    return this.bagRecords.get(item.candidate.bagCount)?.find(matches) ?? null;
  }

  private retainBagRecord(slot: WorkerSlot, candidate: Candidate, data: unknown) {
    const exported = (data as { candidate?: Candidate } | null)?.candidate;
    if (!candidate.verified || !candidate.qualified || candidate.complete ||
        !Number.isInteger(candidate.bagCount) || candidate.bagCount < 1 || candidate.bagCount > 100 ||
        candidate.placed !== candidate.bagCount * 7 || candidate.frames.length !== candidate.frameCount ||
        !exported || !samePartialRecord(candidate, exported) || exported.frames.length !== exported.frameCount) {
      this.failed(slot, '가방별 후보의 검증 기록이 일치하지 않아요.');
      return;
    }
    const record: ArchivedCandidate = { slot, candidate, data };
    const sameId = (item: ArchivedCandidate) => item.slot.id === slot.id && item.candidate.id === candidate.id;
    const previous = this.selectedArchive && sameId(this.selectedArchive) ? this.selectedArchive :
      this.bagRecords.get(candidate.bagCount)?.find(sameId);
    if (previous && betterPartial(previous.candidate, candidate)) return;
    if (this.selectedArchive && sameId(this.selectedArchive)) this.selectedArchive = record;
    const records = (this.bagRecords.get(candidate.bagCount) ?? []).filter((item) => !sameId(item));
    records.push(record);
    records.sort((a, b) => betterPartial(a.candidate, b.candidate) ? -1 :
      betterPartial(b.candidate, a.candidate) ? 1 : 0);
    // Full certificate ownership remains separate even when two workers find
    // similar boards. Keep just the two displayed records, plus one user pin.
    this.bagRecords.set(candidate.bagCount, records.slice(0, 2));
    if (records.slice(0, 2).includes(record)) this.contributeRecord(record);
  }

  private publishedBagRecords(): BagCandidates[] {
    return [...this.bagRecords.entries()].sort(([a], [b]) => a - b).map(([bagCount, records]) => ({
      bagCount,
      best: { ...records[0].candidate, id: this.ownedId(records[0]) },
      runnerUp: records[1] ? { ...records[1].candidate, id: this.ownedId(records[1]) } : null
    }));
  }

  private updateSnapshot(slot: WorkerSlot, snapshot: SearchSnapshot) {
    const merge = (next: Candidate | null, previous: Candidate | null | undefined, exactRecord = false) => {
      const sameRows = !!next && !!previous && next.rows.length === previous.rows.length &&
        next.rows.every((row, y) => row === previous.rows[y]);
      if (next && next.frames.length === 0 && next.frameCount > 0 && previous?.id === next.id &&
          (!exactRecord || samePartialRecord(next, previous)) &&
          previous.verified === next.verified && previous.placed === next.placed && previous.spins === next.spins &&
          sameRows &&
          previous.frames.length === next.frameCount) return { ...next, frames: previous.frames };
      return next;
    };
    slot.snapshot = { ...snapshot,
      current: merge(snapshot.current, slot.snapshot?.current),
      best: merge(snapshot.best, slot.snapshot?.best),
      bestPartial: merge(snapshot.bestPartial ?? null, slot.snapshot?.bestPartial, true) };
    this.mergeBenchmarks(snapshot.bestSpinsByBag);
  }

  private mergeBenchmarks(incoming: number[] | undefined) {
    if (!Array.isArray(incoming)) return;
    let changed = false;
    for (let bag = 1; bag < Math.min(incoming.length, this.bestSpinsByBag.length); bag++) {
      const spins = incoming[bag];
      if (Number.isInteger(spins) && spins <= 40 && spins > this.bestSpinsByBag[bag]) {
        this.bestSpinsByBag[bag] = spins;
        changed = true;
      }
    }
    // Raw spin maxima remain useful progress, but cannot compare the probability
    // spent by different routes when a run permits construction failures.
    if (!changed || !this.shareSpinBenchmarks || this.intent === 'cancelled') return;
    // Share only a small monotonic scheduling benchmark. Each worker keeps
    // its own boards, plans, supply probabilities and replay certificates.
    for (const slot of this.slots) {
      if (!['failed', 'completed', 'cancelled'].includes(slot.status)) {
        slot.worker.postMessage({ type: 'benchmarks', runId: this.runId,
          bestSpinsByBag: [...this.bestSpinsByBag] } satisfies EngineCommand);
      }
    }
  }

  private watchCurrentOwner(owner: number | undefined) {
    for (const slot of this.slots) {
      const enabled = slot.id === owner;
      if (slot.watching === enabled || slot.status === 'failed') continue;
      slot.watching = enabled;
      slot.worker.postMessage({ type: 'watch', runId: this.runId, enabled } satisfies EngineCommand);
    }
  }

  private currentCandidateOwner(): OwnedCandidate | null {
    const ready = this.slots.filter((slot) => slot.snapshot?.current && slot.status !== 'failed');
    if (ready.length === 0) return null;
    const now = performance.now();
    let slot = ready.find((candidate) => candidate.id === this.currentOwner);
    if (!slot || (this.intent === 'running' && now >= this.nextFocusAt)) {
      const previous = ready.findIndex((candidate) => candidate.id === this.currentOwner);
      slot = ready[(previous + 1) % ready.length];
      this.currentOwner = slot.id;
      this.nextFocusAt = now + 2000;
    }
    return { slot, candidate: slot.snapshot!.current! };
  }

  private publishProgress(force = false) {
    const now = performance.now();
    if (!force && now - this.lastProgressAt < 100) {
      if (!this.progressTimer) {
        const runId = this.runId;
        this.progressTimer = setTimeout(() => {
          this.progressTimer = undefined;
          if (this.runId === runId) this.publishProgress(true);
        }, 100 - (now - this.lastProgressAt));
      }
      return;
    }
    clearTimeout(this.progressTimer);
    this.progressTimer = undefined;
    this.lastProgressAt = now;
    const values = this.slots.flatMap((slot) => slot.snapshot ? [slot.snapshot] : []);
    if (values.length === 0) return;
    const current = this.currentCandidateOwner();
    const best = this.bestOwner();
    const partial = this.partialOwner();
    this.watchCurrentOwner(this.observingCurrent ? current?.slot.id : undefined);
    const sum = (read: (value: SearchSnapshot) => number) => values.reduce((value, next) => value + read(next), 0);
    const max = (read: (value: SearchSnapshot) => number) => Math.max(0, ...values.map(read));
    const selected = current?.slot.snapshot;
    const owned = (item: OwnedCandidate | null) => item ? { ...item.candidate, id: this.ownedId(item) } : null;
    this.receive({ type: 'progress', runId: this.runId, snapshot: {
      phase: selected?.phase ?? 'searching',
      expanded: sum((value) => value.expanded),
      candidates: sum((value) => value.candidates),
      verifiedCandidates: sum((value) => value.verifiedCandidates),
      deepestSpins: max((value) => value.deepestSpins),
      qualifiedBags: max((value) => value.qualifiedBags ?? 0),
      qualifiedSpins: max((value) => value.qualifiedSpins ?? 0),
      bestSpinsByBag: [...this.bestSpinsByBag],
      pruned: sum((value) => value.pruned ?? 0),
      selectivePruned: sum((value) => value.selectivePruned ?? 0),
      duplicates: sum((value) => value.duplicates ?? 0),
      bags: selected?.bags ?? 0,
      checked: selected?.checked ?? 0,
      success: selected?.success ?? 0,
      failed: selected?.failed ?? 0,
      frontier: sum((value) => value.frontier),
      finished: this.status === 'completed',
      current: owned(current), best: owned(best), bestPartial: owned(partial),
      bestByBag: this.publishedBagRecords(), selectedCandidate: owned(this.selectedArchive),
      workers: this.slots.filter((slot) => !['failed', 'completed', 'cancelled'].includes(slot.status)).length
    } });
  }
}
