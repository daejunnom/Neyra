import type { Candidate, EngineCommand, EngineEvent, FocusedInput, FocusedSpinLabel, SearchSnapshot } from './types';

type WasmEngine = {
  advance(work: number): void;
  snapshot(): string;
  snapshot_compact(): string;
  observe_benchmarks(values: string): void;
  export_selected_json(selection: string): string;
  free(): void;
};
type WasmModule = {
  default(options: { module_or_path: string }): Promise<unknown>;
  NeyraEngine: new (config: string) => WasmEngine;
  label_board(input: string, config: string): string;
  spin_model_json(): string;
};

const scope = globalThis as unknown as {
  postMessage(message: EngineEvent): void;
  onmessage: ((event: MessageEvent<EngineCommand>) => void) | null;
};
let engine: WasmEngine | null = null;
let runId = '';
let requested: 'running' | 'paused' | 'cancelled' = 'cancelled';
let running = false;
const tasks = new MessageChannel();
let taskEpoch = 0;
let taskQueued = false;
let lastSnapshot = 0;
let incumbentId = '';
let watchCurrent = false;
let detailedCurrentKey = '';
let modulePromise: Promise<WasmModule> | undefined;
let bestSpinsByBag = Array<number>(101).fill(0);
let bagRecordVersions = new Map<string, string>();
let loadedModule: WasmModule | null = null;
let fixedLabelConfig = '';
let learningEnabled = false;
let pendingObservation: { id: string; input: FocusedInput } | null = null;
let waitingObservationDecision = false;

function send(message: EngineEvent) { scope.postMessage(message); }

function observeBenchmarks(incoming: number[]) {
  if (!Array.isArray(incoming)) return;
  let changed = false;
  for (let bag = 1; bag < Math.min(incoming.length, bestSpinsByBag.length); bag++) {
    const spins = incoming[bag];
    if (Number.isInteger(spins) && spins <= 40 && spins > bestSpinsByBag[bag]) {
      bestSpinsByBag[bag] = spins;
      changed = true;
    }
  }
  if (changed && engine) engine.observe_benchmarks(JSON.stringify(bestSpinsByBag));
}

async function wasm(assetBase: string, build: string): Promise<WasmModule> {
  modulePromise ??= (async () => {
    // Keep the loader and Wasm on the same app build, including deployments
    // where only Rust changed and the JavaScript worker hash stayed identical.
    const loader = new URL('neyra_core.js', assetBase);
    const binary = new URL('neyra_core_bg.wasm', assetBase);
    loader.searchParams.set('build', build);
    binary.searchParams.set('build', build);
    const module = await import(/* @vite-ignore */ loader.href) as WasmModule;
    await module.default({ module_or_path: binary.href });
    return module;
  })();
  return modulePromise;
}

function recordCounters(value: SearchSnapshot) {
  return { expanded: value.expanded, candidates: value.candidates,
    verifiedCandidates: value.verifiedCandidates, pruned: value.pruned };
}

function readSnapshot(full = false): SearchSnapshot | null {
  if (!engine) return null;
  const compact = JSON.parse(engine.snapshot_compact()) as SearchSnapshot;
  // Capture new certified records before advertising their IDs. A later step
  // can replace Rust's second-place archive before an on-demand request arrives.
  // Routine snapshots remain compact; the coordinator retains just two per bag.
  const captured = new Map<string, Candidate>();
  const retained = new Map<string, string>();
  for (const group of compact.bestByBag ?? []) {
    for (const candidate of [group.best, group.runnerUp]) {
      if (!candidate) continue;
      const revision = JSON.stringify(candidate);
      retained.set(candidate.id, revision);
      if (bagRecordVersions.get(candidate.id) === revision) continue;
      const data = JSON.parse(engine.export_selected_json(`candidate:${candidate.id}`));
      const record = data.candidate as Candidate | undefined;
      if (!record || record.id !== candidate.id || record.bagCount !== group.bagCount ||
          !record.verified || !record.qualified || record.complete ||
          record.frames.length !== record.frameCount) {
        throw new Error('가방별 후보의 검증 기록을 가져오지 못했어요.');
      }
      captured.set(record.id, record);
      // This is a selected certificate archive. Its aggregate search snapshot
      // is unrelated to that selection and would repeat other routes' replays.
      delete data.snapshot;
      send({ type: 'bag-record', runId, candidate: record, data, counters: recordCounters(compact) });
    }
  }
  bagRecordVersions = retained;
  const current = compact.current;
  const key = current ? JSON.stringify([current.id, current.verified, current.frameCount,
    current.placed, current.spins, current.rows]) : '';
  // A verified witness can use another order than the same plan's preview.
  // Only changed replay identities need another full history transfer.
  if (full || (watchCurrent && key !== detailedCurrentKey)) {
    const capturedCurrent = current ? captured.get(current.id) : undefined;
    if (!full && capturedCurrent) {
      compact.current = capturedCurrent;
    } else {
      // Full snapshots contain only current/best/partial histories; every
      // per-depth entry stays compact in Rust. Avoid serializing P7 proofs
      // merely to display a changing current field.
      const detail = JSON.parse(engine.snapshot()) as SearchSnapshot;
      compact.current = detail.current;
      if (full) {
        compact.best = detail.best;
        compact.bestPartial = detail.bestPartial;
      }
    }
    detailedCurrentKey = key;
  }
  return compact;
}

function retainIncumbent(value: SearchSnapshot) {
  if (engine && value.best && value.best.id !== incumbentId) {
    const data = JSON.parse(engine.export_selected_json('best'));
    incumbentId = value.best.id;
    // The archive includes full frames even if this worker only sends compact
    // routine updates. Retain complete results across a worker failure.
    send({ type: 'incumbent', runId, candidate: data.candidate, data, counters: recordCounters(value) });
  }
}

function snapshot(full = false): SearchSnapshot | null {
  const value = readSnapshot(full);
  if (!value || !engine) return null;
  lastSnapshot = performance.now();
  send({ type: 'progress', runId, snapshot: value });
  retainIncumbent(value);
  return value;
}

function fail(error: unknown) {
  running = false;
  invalidateTask();
  const message = error instanceof Error ? error.message : String(error);
  send({ type: 'error', runId, message });
}

function schedule() {
  if (taskQueued || !running) return;
  taskQueued = true;
  // A message task yields to controls without the nested timer minimum delay.
  // Queue one task at a time instead of a self-perpetuating microtask chain.
  tasks.port2.postMessage(taskEpoch);
}

function invalidateTask() {
  taskEpoch++;
  taskQueued = false;
}

tasks.port1.onmessage = ({ data }: MessageEvent<number>) => {
  if (data !== taskEpoch) return;
  taskQueued = false;
  tick();
};

function tick() {
  if (!running || !engine) return;
  try {
    if (learningEnabled && pendingObservation && loadedModule) {
      const job = pendingObservation;
      pendingObservation = null;
      // The collector is unchanged. Only the supplied origin is chosen deeply;
      // a completed bounded job yields before another one is assigned.
      waitingObservationDecision = true;
      try {
        fixedLabelConfig ||= JSON.stringify(JSON.parse(loadedModule.spin_model_json()).labelConfig);
        const label = JSON.parse(loadedModule.label_board(JSON.stringify(job.input), fixedLabelConfig)) as FocusedSpinLabel;
        send({ type: 'observation', runId, observationId: job.id, label });
      } catch (error) {
        send({ type: 'observation-error', runId, observationId: job.id,
          message: error instanceof Error ? error.message : String(error) });
      }
      return;
    }
    if (learningEnabled && waitingObservationDecision) return;
    const started = performance.now();
    do { engine.advance(1); } while (performance.now() - started < 12);
    if (performance.now() - lastSnapshot >= 200) {
      const state = snapshot();
      if (state?.finished) {
        running = false;
        send({ type: 'status', runId, status: 'completed' });
      }
    }
    if (running) schedule();
  } catch (error) { fail(error); }
}

scope.onmessage = async ({ data }) => {
  if (data.type === 'start') {
    running = false;
    requested = 'running';
    invalidateTask();
    engine?.free();
    engine = null;
    incumbentId = '';
    watchCurrent = data.watchCurrent;
    detailedCurrentKey = '';
    bestSpinsByBag.fill(0);
    bagRecordVersions.clear();
    pendingObservation = null;
    waitingObservationDecision = false;
    learningEnabled = data.learningEnabled === true;
    runId = data.runId;
    const requestedId = runId;
    try {
      const module = await wasm(data.assetBase, data.assetVersion);
      if (runId !== requestedId) return;
      loadedModule = module;
      // Pause/cancel can arrive while the async module is loading.
      // Public Pages always runs the established search. A stale saved config
      // or an old caller must not re-enable learned ordering or training here.
      engine = new module.NeyraEngine(JSON.stringify({ ...data.config, cnn_evaluation: false }));
      if (bestSpinsByBag.some((spins) => spins > 0)) {
        engine.observe_benchmarks(JSON.stringify(bestSpinsByBag));
      }
      running = requested === 'running';
      snapshot();
      send({ type: 'status', runId, status: requested });
      if (running) schedule();
    } catch (error) { if (runId === requestedId) fail(error); }
    return;
  }
  if (data.runId !== runId) return;
  if (data.type === 'learning-mode') {
    learningEnabled = data.enabled;
    if (!learningEnabled) { pendingObservation = null; waitingObservationDecision = false; }
    if (running) schedule();
    return;
  }
  if (data.type === 'observe') {
    if (learningEnabled && requested !== 'cancelled') {
      pendingObservation = { id: data.observationId, input: data.input };
      waitingObservationDecision = false;
      if (running) schedule();
    }
    return;
  }
  if (data.type === 'observation-idle') {
    pendingObservation = null;
    waitingObservationDecision = false;
    if (running) schedule();
    return;
  }
  if (data.type === 'observation-wait') {
    if (learningEnabled) waitingObservationDecision = true;
    return;
  }
  if (data.type === 'benchmarks') {
    try { if (requested !== 'cancelled') observeBenchmarks(data.bestSpinsByBag); }
    catch (error) { fail(error); }
    return;
  }
  if (data.type === 'watch') {
    watchCurrent = data.enabled;
    try { if (watchCurrent) snapshot(true); } catch (error) { fail(error); }
    return;
  }
  if (data.type === 'detail') {
    try {
      const value = readSnapshot(true);
      if (value) {
        send({ type: 'detail', runId, requestId: data.requestId, snapshot: value });
        retainIncumbent(value);
      }
    } catch (error) { fail(error); }
    return;
  }
  if (data.type === 'export') {
    try {
      if (!engine) throw new Error('탐색기가 준비된 뒤 저장해 주세요.');
      const result = JSON.parse(engine.export_selected_json(data.view));
      // No engine.advance can interleave these calls. The updated preview and
      // exported certificate refer to the same worker state.
      const state = readSnapshot(true);
      if (!result.candidate) throw new Error('아직 저장할 후보가 없어요.');
      send({ type: 'export', runId, requestId: data.requestId, data: result, snapshot: state ?? undefined });
    } catch (error) {
      send({ type: 'export-error', runId, requestId: data.requestId, message: error instanceof Error ? error.message : String(error) });
    }
    return;
  }
  try {
    if (data.type === 'pause') {
      requested = 'paused';
      running = false;
      invalidateTask();
      snapshot();
      send({ type: 'status', runId, status: 'paused' });
    } else if (data.type === 'resume' && requested !== 'cancelled') {
      requested = 'running';
      running = engine !== null;
      snapshot();
      send({ type: 'status', runId, status: running ? 'running' : 'loading' });
      if (running) schedule();
    } else if (data.type === 'cancel') {
      requested = 'cancelled';
      running = false;
      invalidateTask();
      pendingObservation = null;
      waitingObservationDecision = false;
      snapshot();
      send({ type: 'status', runId, status: 'cancelled' });
    }
  } catch (error) { fail(error); }
};
