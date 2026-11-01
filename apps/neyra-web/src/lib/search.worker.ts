import type { Candidate, EngineCommand, EngineEvent, SearchSnapshot } from './types';
import { QUIET_TT_BATCH_ENTRIES, quietTranspositionBatch } from './transpositions';

type WasmEngine = {
  advance(work: number): void;
  snapshot(): string;
  snapshot_compact(): string;
  observe_benchmarks(values: string): void;
  drain_transpositions(limit: number): string;
  observe_transpositions(batch: string): void;
  export_selected_json(selection: string): string;
  free(): void;
};
type WasmModule = {
  default(options: { module_or_path: string }): Promise<unknown>;
  NeyraEngine: { new (config: string): WasmEngine; with_ga_model?(config: string, model: string): WasmEngine };
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
let transpositionSharing = false;
let guideFallback: string | null = null;

function send(message: EngineEvent) { scope.postMessage(message); }

function stopTranspositionSharing() {
  if (!transpositionSharing) return;
  transpositionSharing = false;
  send({ type: 'warning', runId, message: '중복 검증 공유를 사용할 수 없어 각 탐색기의 자체 검증으로 계속합니다.' });
}

function shareTranspositions() {
  if (!engine || !transpositionSharing) return;
  try {
    const batch = quietTranspositionBatch(JSON.parse(engine.drain_transpositions(QUIET_TT_BATCH_ENTRIES)));
    if (!batch) { stopTranspositionSharing(); return; }
    if (batch.entries.length) send({ type: 'transpositions', runId, batch });
  } catch { stopTranspositionSharing(); }
}

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

function readSnapshot(full = false): SearchSnapshot | null {
  if (!engine) return null;
  const compact = JSON.parse(engine.snapshot_compact()) as SearchSnapshot;
  if (guideFallback) compact.evaluator = { enabled: false, requested: 'ga-policy', active: 'heuristic', modelId: null, weightsSha256: null,
    fallbackReason: guideFallback, scope: 'CPU heuristic fallback after local GA guide initialization failure', learningComplete: false };
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
      send({ type: 'bag-record', runId, candidate: record, data });
    }
  }
  bagRecordVersions = retained;
  // Frontier metadata remains compact. Full P7 evidence is requested only
  // when the user selects a plan; hundreds of cost points may share one plan.
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
    send({ type: 'incumbent', runId, candidate: data.candidate, data });
  }
}

function snapshot(full = false): SearchSnapshot | null {
  const value = readSnapshot(full);
  if (!value || !engine) return null;
  lastSnapshot = performance.now();
  shareTranspositions();
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
    transpositionSharing = data.transpositionSharing === true;
    guideFallback = null;
    runId = data.runId;
    const requestedId = runId;
    const mayInitialize = () => runId === requestedId && requested !== 'cancelled';
    try {
      const module = await wasm(data.assetBase, data.assetVersion);
      // An explicit cancel while the module is loading must not allocate a
      // new engine afterwards. Already initialized archives remain available.
      if (!mayInitialize()) return;
      // Pause/cancel can arrive while the async module is loading.
      // The explicit local GA guide is inference only. It never re-enables
      // retired CNN collection, training, upload or model service routes.
      const configJson = JSON.stringify({ ...data.config, cnn_evaluation: false, board_evaluator: 'heuristic' });
      if (data.gaModelJson) {
        try {
          if (new TextEncoder().encode(data.gaModelJson).byteLength > 32 * 1024 * 1024) throw new Error('로컬 GA 파일은 최대 32 MiB입니다.');
          if (!module.NeyraEngine.with_ga_model) throw new Error('현재 엔진에는 로컬 GA 가이드 기능이 없습니다.');
          engine = module.NeyraEngine.with_ga_model(configJson, data.gaModelJson);
        } catch (error) {
          guideFallback = error instanceof Error ? error.message : String(error);
          engine = new module.NeyraEngine(configJson);
          send({ type: 'warning', runId, message: `GA 가이드를 적용하지 못해 CPU 기본 평가기로 계속합니다: ${guideFallback}` });
        }
      } else engine = new module.NeyraEngine(configJson);
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
  if (data.type === 'benchmarks') {
    try { if (requested !== 'cancelled') observeBenchmarks(data.bestSpinsByBag); }
    catch (error) { fail(error); }
    return;
  }
  if (data.type === 'transpositions') {
    if (!engine || !transpositionSharing || requested === 'cancelled') return;
    const batch = quietTranspositionBatch(data.batch);
    if (!batch) return;
    try { engine.observe_transpositions(JSON.stringify(batch)); }
    catch { stopTranspositionSharing(); }
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
      snapshot();
      send({ type: 'status', runId, status: 'cancelled' });
    }
  } catch (error) { fail(error); }
};
