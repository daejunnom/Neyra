import type { Candidate, CandidateView, EngineCommand, EngineEvent, RunStatus, SearchConfig, SearchSnapshot } from './types';
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

function fieldQuality(candidate: Candidate): number {
  return Number.isFinite(candidate.fieldQuality) ? candidate.fieldQuality : -Infinity;
}

/** Compare the displayed efficiency record without floating-point division. */
export function betterPartial(a: Candidate, b: Candidate): boolean {
  const aRate = a.spins * countedPieces(b);
  const bRate = b.spins * countedPieces(a);
  return aRate > bRate || (aRate === bRate &&
    (a.spins > b.spins || (a.spins === b.spins &&
      (fieldQuality(a) > fieldQuality(b) || (fieldQuality(a) === fieldQuality(b) &&
        (a.placed < b.placed || (a.placed === b.placed && a.probability > b.probability)))))));
}

/** A plan ID alone cannot distinguish an improved record from its old export. */
function samePartialRecord(a: Candidate, b: Candidate): boolean {
  return a.id === b.id && a.spins === b.spins && a.placed === b.placed &&
    fieldQuality(a) === fieldQuality(b) && a.probability === b.probability &&
    a.firstBagProbability === b.firstBagProbability && a.firstBagSymmetry === b.firstBagSymmetry &&
    a.firstBagSymmetryAssumed === b.firstBagSymmetryAssumed && a.frameCount === b.frameCount &&
    a.rows.length === b.rows.length && a.rows.every((row, y) => row === b.rows[y]);
}

function partialArchiveKey(item: OwnedCandidate): string {
  const value = item.candidate;
  return JSON.stringify([item.slot.id, value.id, value.spins, value.placed, fieldQuality(value),
    value.probability, value.firstBagProbability, value.firstBagSymmetry, value.firstBagSymmetryAssumed,
    value.frameCount, value.rows]);
}

function complete(candidate: Candidate | null | undefined): candidate is Candidate {
  return !!candidate?.complete && candidate.qualified && candidate.verified && candidate.budgetPieces !== null;
}

export class SearchClient {
  private slots: WorkerSlot[] = [];
  private runId = '';
  private intent: 'running' | 'paused' | 'cancelled' = 'cancelled';
  private status: RunStatus = 'idle';
  private currentOwner: number | null = null;
  private nextFocusAt = 0;
  private lastProgressAt = 0;
  private progressTimer: ReturnType<typeof setTimeout> | undefined;
  private pendingExports = new Map<string, { owner: number; view: CandidateView }>();
  private pendingDetails = new Map<number, { requestId: string }>();
  private partialPeak: OwnedCandidate | null = null;
  private partialArchive: ArchivedCandidate | null = null;
  private pendingPartialArchive: { requestId: string; owner: number; key: string } | null = null;
  private rejectedPartialArchives = new Set<string>();
  private bestSpinsByBag = Array<number>(101).fill(0);

  constructor(private receive: (event: EngineEvent) => void) {}

  start(config: SearchConfig) {
    this.dispose();
    this.runId = crypto.randomUUID();
    const runId = this.runId;
    this.intent = 'running';
    this.status = 'loading';
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
        const slot: WorkerSlot = { id, worker, status: 'loading', snapshot: null, incumbent: null, watching: id === 0 };
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
          config: { ...engineConfig, seed: workerSeed(config.seed, slot.id), search_profile: slot.id % 4,
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

  export(view: CandidateView = 'best', requestId = crypto.randomUUID()): boolean {
    const owner = view === 'current' ? this.currentCandidateOwner() :
      view === 'partial' ? this.partialOwner() : this.bestOwner() ?? this.partialOwner();
    if (!owner) return false;
    const selection: CandidateView = view === 'current' ? 'current' : owner.candidate.complete ? 'best' : 'partial';
    if (selection === 'best' && owner.slot.incumbent?.candidate.id === owner.candidate.id) {
      // A completed certificate remains usable even if its worker later fails.
      this.receive({ type: 'export', runId: this.runId, requestId, data: owner.slot.incumbent.data });
      return true;
    }
    if (selection === 'partial' && this.partialArchive?.slot.id === owner.slot.id &&
        samePartialRecord(this.partialArchive.candidate, owner.candidate)) {
      this.receive({ type: 'export', runId: this.runId, requestId, data: this.partialArchive.data });
      return true;
    }
    if (owner.slot.status === 'failed') {
      this.receive({ type: 'export-error', runId: this.runId, requestId, message: '이 경로의 기록을 받기 전에 탐색기가 중단됐어요. 이미 보관한 결과는 계속 저장할 수 있습니다.' });
      return false;
    }
    this.pendingExports.set(requestId, { owner: owner.slot.id, view: selection });
    try {
      owner.slot.worker.postMessage({ type: 'export', runId: this.runId, requestId, view: selection } satisfies EngineCommand);
      return true;
    } catch {
      this.pendingExports.delete(requestId);
      this.receive({ type: 'export-error', runId: this.runId, requestId,
        message: '이 후보의 문서를 요청하지 못했어요. 다른 후보를 선택해 주세요.' });
      return false;
    }
  }

  dispose() {
    clearTimeout(this.progressTimer);
    this.progressTimer = undefined;
    for (const slot of this.slots) slot.worker.terminate();
    this.slots = [];
    this.pendingExports.clear();
    this.pendingDetails.clear();
    this.partialPeak = null;
    this.partialArchive = null;
    this.pendingPartialArchive = null;
    this.rejectedPartialArchives.clear();
    this.bestSpinsByBag.fill(0);
    // Invalidate queued messages before a new generation starts.
    this.runId = '';
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
    } else if (message.type === 'detail') {
      const pending = this.pendingDetails.get(slot.id);
      if (!pending || pending.requestId !== message.requestId) return;
      this.pendingDetails.delete(slot.id);
      this.updateSnapshot(slot, message.snapshot);
      this.publishProgress(true);
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
      }
      this.publishProgress();
    } else if (message.type === 'error') {
      this.failed(slot, message.message);
    } else if (message.type === 'export' || message.type === 'export-error') {
      const archive = this.pendingPartialArchive;
      if (archive?.requestId === message.requestId) {
        if (archive.owner !== slot.id) return;
        this.pendingPartialArchive = null;
        if (message.type === 'export') {
          if (message.snapshot) this.updateSnapshot(slot, message.snapshot);
          const candidate = (message.data as { candidate?: Candidate } | null)?.candidate;
          if (candidate?.verified && candidate.qualified && !candidate.complete &&
              candidate.frames.length === candidate.frameCount) {
            // The engine can improve between request and reply. Keep that newer
            // certificate, but a late reply cannot overwrite a higher record.
            if (!this.partialArchive || betterPartial(candidate, this.partialArchive.candidate)) {
              this.partialArchive = { slot, candidate, data: message.data };
            }
          } else {
            this.rejectedPartialArchives.add(archive.key);
          }
        } else {
          this.rejectedPartialArchives.add(archive.key);
        }
        this.publishProgress(true);
        return;
      }
      const request = this.pendingExports.get(message.requestId);
      if (!request || request.owner !== slot.id) return;
      this.pendingExports.delete(message.requestId);
      if (message.type === 'export-error') {
        this.receive({ type: 'export-error', runId: this.runId, requestId: message.requestId, message: message.message });
      } else {
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
    this.pendingDetails.delete(slot.id);
    if (this.pendingPartialArchive?.owner === slot.id) this.pendingPartialArchive = null;
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
    let best = this.partialPeak;
    for (const slot of this.slots) {
      const candidate = slot.snapshot?.bestPartial;
      if (candidate?.verified && candidate.qualified && !candidate.complete &&
          (!best || betterPartial(candidate, best.candidate) ||
            (best.slot.id === slot.id && samePartialRecord(best.candidate, candidate) &&
              candidate.frames.length >= best.candidate.frames.length))) best = { slot, candidate };
    }
    const archived = this.partialArchive;
    if (archived && (!best || betterPartial(archived.candidate, best.candidate) ||
        (best.slot.id === archived.slot.id && samePartialRecord(best.candidate, archived.candidate)))) best = archived;
    this.partialPeak = best;
    return best;
  }

  private archivePartial(item: OwnedCandidate | null) {
    if (!item || item.slot.status === 'failed' || this.pendingPartialArchive ||
        (this.partialArchive && !betterPartial(item.candidate, this.partialArchive.candidate)) ||
        this.rejectedPartialArchives.has(partialArchiveKey(item))) return;
    const requestId = crypto.randomUUID();
    this.pendingPartialArchive = { requestId, owner: item.slot.id, key: partialArchiveKey(item) };
    // Archive only the new global record. Ordinary progress stays compact and
    // this internal request never triggers a download in the user's browser.
    item.slot.worker.postMessage({ type: 'export', runId: this.runId, requestId, view: 'partial' } satisfies EngineCommand);
  }

  private updateSnapshot(slot: WorkerSlot, snapshot: SearchSnapshot) {
    const merge = (next: Candidate | null, previous: Candidate | null | undefined) => {
      if (next && next.frames.length === 0 && next.frameCount > 0 && previous?.id === next.id &&
          previous.verified === next.verified && previous.placed === next.placed && previous.spins === next.spins &&
          previous.rows.length === next.rows.length && previous.rows.every((row, y) => row === next.rows[y]) &&
          previous.frames.length === next.frameCount) return { ...next, frames: previous.frames };
      return next;
    };
    slot.snapshot = { ...snapshot,
      current: merge(snapshot.current, slot.snapshot?.current),
      best: merge(snapshot.best, slot.snapshot?.best),
      bestPartial: merge(snapshot.bestPartial ?? null, slot.snapshot?.bestPartial) };
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
    if (!changed || this.intent === 'cancelled') return;
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

  private requestPartialDetail(item: OwnedCandidate | null) {
    if (!item || item.slot.status === 'failed' || item.candidate.frames.length >= item.candidate.frameCount || this.pendingDetails.has(item.slot.id)) return;
    const requestId = crypto.randomUUID();
    this.pendingDetails.set(item.slot.id, { requestId });
    item.slot.worker.postMessage({ type: 'detail', runId: this.runId, requestId } satisfies EngineCommand);
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
    this.watchCurrentOwner(current?.slot.id);
    if (!best) {
      this.archivePartial(partial);
      if (!this.pendingPartialArchive) this.requestPartialDetail(partial);
    }
    const sum = (read: (value: SearchSnapshot) => number) => values.reduce((value, next) => value + read(next), 0);
    const max = (read: (value: SearchSnapshot) => number) => Math.max(0, ...values.map(read));
    const selected = current?.slot.snapshot;
    const owned = (item: OwnedCandidate | null) => item ? { ...item.candidate, id: `w${item.slot.id}:${item.candidate.id}` } : null;
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
      workers: this.slots.filter((slot) => !['failed', 'completed', 'cancelled'].includes(slot.status)).length
    } });
  }
}
