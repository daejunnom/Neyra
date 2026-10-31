import type { Candidate, CandidateView, EngineCommand, EngineEvent, RunStatus, SearchConfig, SearchSnapshot } from './types';
import { base } from '$app/paths';
import { version } from '$app/environment';

type WorkerSlot = {
  id: number;
  worker: Worker;
  status: RunStatus;
  snapshot: SearchSnapshot | null;
  incumbent: { candidate: Candidate; data: unknown } | null;
};
type OwnedCandidate = { slot: WorkerSlot; candidate: Candidate };

/** Automatic mode leaves CPU headroom; an explicit count still respects the device. */
export function workerCount(requested: number, hardware = navigator.hardwareConcurrency): number {
  const cores = Number.isFinite(hardware) && hardware > 0 ? Math.floor(hardware) : 1;
  const automatic = Math.max(1, Math.floor(cores / 2));
  return Math.max(1, Math.min(4, cores, requested === 0 ? automatic : requested));
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
        const slot: WorkerSlot = { id: index, worker, status: 'loading', snapshot: null, incumbent: null };
        worker.onmessage = (event: MessageEvent<EngineEvent>) => {
          if (this.runId !== runId || event.data.runId !== runId) return;
          this.handle(slot, event.data);
        };
        worker.onerror = (event) => {
          if (this.runId === runId) this.failed(slot, event.message || '탐색기를 불러오지 못했어요.');
        };
        worker.postMessage({
          type: 'start', runId,
          config: { ...engineConfig, seed: workerSeed(config.seed, index), search_profile: index },
          assetBase: `${location.origin}${base}/wasm/`, assetVersion: version
        } satisfies EngineCommand);
        this.slots.push(slot);
      } catch (error) {
        worker?.terminate();
        creationFailure = error instanceof Error ? error.message : String(error);
      }
    }
    if (this.slots.length === 0) {
      this.status = 'failed';
      this.receive({ type: 'error', runId, message: creationFailure || '이 환경에서 탐색기를 시작할 수 없어요.' });
    } else {
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

  export(view: CandidateView = 'best') {
    const owner = view === 'best' ? this.bestOwner() : this.currentCandidateOwner();
    if (!owner) return;
    const requestId = crypto.randomUUID();
    if (view === 'best' && owner.slot.incumbent?.candidate.id === owner.candidate.id) {
      // A completed certificate remains usable even if its worker later fails.
      this.receive({ type: 'export', runId: this.runId, requestId, data: owner.slot.incumbent.data });
      return;
    }
    if (owner.slot.status === 'failed') {
      this.receive({ type: 'warning', runId: this.runId, message: '이 작업의 현재 후보를 저장할 수 없어요. 보관된 완주 결과는 계속 저장할 수 있습니다.' });
      return;
    }
    this.pendingExports.set(requestId, { owner: owner.slot.id, view });
    owner.slot.worker.postMessage({ type: 'export', runId: this.runId, requestId, view } satisfies EngineCommand);
  }

  dispose() {
    clearTimeout(this.progressTimer);
    this.progressTimer = undefined;
    for (const slot of this.slots) slot.worker.terminate();
    this.slots = [];
    this.pendingExports.clear();
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
      slot.snapshot = message.snapshot;
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
      }
      this.publishProgress();
    } else if (message.type === 'error') {
      this.failed(slot, message.message);
    } else if (message.type === 'export' || message.type === 'export-error') {
      const request = this.pendingExports.get(message.requestId);
      if (!request || request.owner !== slot.id) return;
      this.pendingExports.delete(message.requestId);
      if (message.type === 'export-error') {
        this.receive({ type: 'warning', runId: this.runId, message: message.message });
      } else {
        if (message.snapshot) slot.snapshot = message.snapshot;
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
      if (request.owner === slot.id) this.pendingExports.delete(requestId);
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
      pruned: sum((value) => value.pruned ?? 0),
      bags: selected?.bags ?? 0,
      checked: selected?.checked ?? 0,
      success: selected?.success ?? 0,
      failed: selected?.failed ?? 0,
      frontier: sum((value) => value.frontier),
      finished: this.status === 'completed',
      current: owned(current), best: owned(best),
      workers: this.slots.filter((slot) => !['failed', 'completed', 'cancelled'].includes(slot.status)).length
    } });
  }
}
