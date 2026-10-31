import type { EngineCommand, EngineEvent, SearchSnapshot } from './types';

type WasmEngine = {
  advance(work: number): void;
  snapshot(): string;
  snapshot_compact(): string;
  export_selected_json(selection: string): string;
  free(): void;
};
type WasmModule = {
  default(options: { module_or_path: string }): Promise<unknown>;
  NeyraEngine: new (config: string) => WasmEngine;
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

function send(message: EngineEvent) { scope.postMessage(message); }

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
  const current = compact.current;
  const key = current ? JSON.stringify([current.id, current.verified, current.frameCount,
    current.placed, current.spins, current.rows]) : '';
  // A verified witness can use another order than the same plan's preview.
  // Only changed replay identities need another full history transfer.
  if (full || (watchCurrent && key !== detailedCurrentKey)) {
    const value = JSON.parse(engine.snapshot()) as SearchSnapshot;
    detailedCurrentKey = key;
    return value;
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
    runId = data.runId;
    const requestedId = runId;
    try {
      const module = await wasm(data.assetBase, data.assetVersion);
      if (runId !== requestedId) return;
      // Pause/cancel can arrive while the async module is loading.
      engine = new module.NeyraEngine(JSON.stringify(data.config));
      running = requested === 'running';
      snapshot();
      send({ type: 'status', runId, status: requested });
      if (running) schedule();
    } catch (error) { if (runId === requestedId) fail(error); }
    return;
  }
  if (data.runId !== runId) return;
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
