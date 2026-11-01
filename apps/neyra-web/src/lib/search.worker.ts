import type { BoardEvaluation, Candidate, EngineCommand, EngineEvent, SearchSnapshot } from './types';

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
  evaluate_board(rows: string): string;
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
let evaluateBoard: WasmModule['evaluate_board'] | null = null;
const boardEvaluations = new Map<string, BoardEvaluation | null>();
const EVALUATION_CACHE_LIMIT = 64;

function send(message: EngineEvent) { scope.postMessage(message); }

function withBoardEvaluation(candidate: Candidate | null, compute = false): Candidate | null {
  if (!candidate || !evaluateBoard) return candidate;
  const key = JSON.stringify(candidate.rows);
  let evaluation = boardEvaluations.get(key);
  if (!boardEvaluations.has(key) && compute) {
    evaluation = JSON.parse(evaluateBoard(key)) as BoardEvaluation | null;
    // Empty/out-of-domain boards deliberately have no estimate. Cache that
    // answer too, rather than repeatedly evaluating the initial empty field.
    if (evaluation !== null && (typeof evaluation?.modelId !== 'string' || typeof evaluation.scope !== 'string' ||
        evaluation.experimental !== true || ![2, 3].includes(evaluation.horizonBags) ||
        !Number.isInteger(evaluation.workBudget) || evaluation.workBudget < 1 || evaluation.workBudget > 4194304 ||
        ![evaluation.shapePreference, evaluation.observedSpins, evaluation.observedChoices,
          evaluation.adjustment].every(Number.isFinite) || evaluation.observedSpins < 0 ||
        evaluation.observedChoices < 0 || Math.abs(evaluation.adjustment) > 12)) {
      throw new Error('CNN 필드 예측의 형식을 확인하지 못했어요.');
    }
    boardEvaluations.set(key, evaluation);
    if (boardEvaluations.size > EVALUATION_CACHE_LIMIT) {
      boardEvaluations.delete(boardEvaluations.keys().next().value!);
    }
  }
  return evaluation ? { ...candidate, boardEvaluation: evaluation } : candidate;
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
      const evaluated = withBoardEvaluation(record, true)!;
      data.candidate = evaluated;
      captured.set(record.id, evaluated);
      // This is a selected certificate archive. Its aggregate search snapshot
      // is unrelated to that selection and would repeat other routes' replays.
      delete data.snapshot;
      send({ type: 'bag-record', runId, candidate: evaluated, data });
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
    compact.current = withBoardEvaluation(compact.current, true);
    detailedCurrentKey = key;
  }
  // Unchanged compact updates only reuse cached values. New certified records
  // and the changed visible current board are the inference entry points.
  compact.current = withBoardEvaluation(compact.current);
  compact.best = withBoardEvaluation(compact.best, !!compact.best && compact.best.id !== incumbentId);
  compact.bestPartial = withBoardEvaluation(compact.bestPartial, full);
  return compact;
}

function retainIncumbent(value: SearchSnapshot) {
  if (engine && value.best && value.best.id !== incumbentId) {
    const data = JSON.parse(engine.export_selected_json('best'));
    data.candidate = withBoardEvaluation(data.candidate, true);
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
    boardEvaluations.clear();
    evaluateBoard = null;
    runId = data.runId;
    const requestedId = runId;
    try {
      const module = await wasm(data.assetBase, data.assetVersion);
      if (runId !== requestedId) return;
      if (data.config.cnn_evaluation) {
        if (typeof module.evaluate_board !== 'function') throw new Error('CNN 평가기가 포함된 탐색기를 불러오지 못했어요.');
        evaluateBoard = module.evaluate_board;
      }
      // Pause/cancel can arrive while the async module is loading.
      engine = new module.NeyraEngine(JSON.stringify(data.config));
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
      result.candidate = withBoardEvaluation(result.candidate, true);
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
