import type { EngineCommand, EngineEvent, SearchSnapshot } from './types';

type WasmEngine = {
  advance(work: number): void;
  snapshot(): string;
  export_json(): string;
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
let running = false;
let cancelled = false;
let timer: ReturnType<typeof setTimeout> | undefined;
let lastSnapshot = 0;
let modulePromise: Promise<WasmModule> | undefined;

function send(message: EngineEvent) { scope.postMessage(message); }

async function wasm(assetBase: string): Promise<WasmModule> {
  modulePromise ??= (async () => {
    const url = new URL('neyra_core.js', assetBase).href;
    const module = await import(/* @vite-ignore */ url) as WasmModule;
    await module.default({ module_or_path: new URL('neyra_core_bg.wasm', assetBase).href });
    return module;
  })();
  return modulePromise;
}

function snapshot(): SearchSnapshot | null {
  if (!engine) return null;
  const value = JSON.parse(engine.snapshot()) as SearchSnapshot;
  lastSnapshot = performance.now();
  send({ type: 'progress', runId, snapshot: value });
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
  // A host task yields to pause/cancel messages between bounded Rust batches.
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
    cancelled = false;
    clearTimeout(timer);
    runId = data.runId;
    const requestedId = runId;
    try {
      const module = await wasm(data.assetBase);
      if (runId !== requestedId) return;
      engine?.free();
      engine = new module.NeyraEngine(JSON.stringify(data.config));
      running = !cancelled;
      snapshot();
      send({ type: 'status', runId, status: running ? 'running' : 'cancelled' });
      if (running) schedule();
    } catch (error) { if (runId === requestedId) fail(error); }
    return;
  }
  if (data.runId !== runId) return;
  try {
    if (data.type === 'pause') {
      running = false;
      clearTimeout(timer);
      snapshot();
      send({ type: 'status', runId, status: 'paused' });
    } else if (data.type === 'resume' && engine && !cancelled) {
      running = true;
      send({ type: 'status', runId, status: 'running' });
      schedule();
    } else if (data.type === 'cancel') {
      running = false;
      cancelled = true;
      clearTimeout(timer);
      snapshot();
      send({ type: 'status', runId, status: 'cancelled' });
    } else if (data.type === 'export' && engine) {
      send({ type: 'export', runId, data: JSON.parse(engine.export_json()) });
    }
  } catch (error) { fail(error); }
};
