import type { EngineCommand, EngineEvent, SearchSnapshot } from './types';

type WasmEngine = {
  advance(work: number): void;
  snapshot(): string;
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
let timer: ReturnType<typeof setTimeout> | undefined;
let lastSnapshot = 0;
let incumbentId = '';
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

function readSnapshot(): SearchSnapshot | null {
  return engine ? JSON.parse(engine.snapshot()) as SearchSnapshot : null;
}

function snapshot(): SearchSnapshot | null {
  const value = readSnapshot();
  if (!value || !engine) return null;
  lastSnapshot = performance.now();
  send({ type: 'progress', runId, snapshot: value });
  if (value.best && value.best.id !== incumbentId) {
    const data = JSON.parse(engine.export_selected_json('best'));
    incumbentId = value.best.id;
    // Send a complete certificate only when an incumbent changes. Main-thread
    // retention protects it if this independent worker subsequently fails.
    send({ type: 'incumbent', runId, candidate: value.best, data });
  }
  return value;
}

function fail(error: unknown) {
  running = false;
  clearTimeout(timer);
  const message = error instanceof Error ? error.message : String(error);
  send({ type: 'error', runId, message });
}

function schedule() {
  clearTimeout(timer);
  // Workers share no engine state. Each bounded task yields to host controls.
  timer = setTimeout(tick, 0);
}

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
    clearTimeout(timer);
    engine?.free();
    engine = null;
    incumbentId = '';
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
  if (data.type === 'export') {
    try {
      if (!engine) throw new Error('탐색기가 준비된 뒤 저장해 주세요.');
      const result = JSON.parse(engine.export_selected_json(data.view));
      // No engine.advance can interleave these calls. The updated preview and
      // exported certificate refer to the same worker state.
      const state = readSnapshot();
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
      clearTimeout(timer);
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
      clearTimeout(timer);
      snapshot();
      send({ type: 'status', runId, status: 'cancelled' });
    }
  } catch (error) { fail(error); }
};
