import type { EngineCommand, EngineEvent, SearchConfig } from './types';
import { base } from '$app/paths';

export class SearchClient {
  private worker: Worker | null = null;
  private runId = '';

  constructor(private receive: (event: EngineEvent) => void) {}

  start(config: SearchConfig) {
    this.worker?.terminate();
    this.runId = crypto.randomUUID();
    const runId = this.runId;
    this.worker = new Worker(new URL('./search.worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (event: MessageEvent<EngineEvent>) => {
      const message = event.data;
      if (message.runId && message.runId !== this.runId) return;
      this.receive(message);
    };
    this.worker.onerror = (event) => {
      this.receive({ type: 'error', runId, message: event.message || '탐색기를 불러오지 못했어요.' });
    };
    this.post({ type: 'start', runId, config, assetBase: `${location.origin}${base}/wasm/` });
  }

  pause() { this.post({ type: 'pause', runId: this.runId }); }
  resume() { this.post({ type: 'resume', runId: this.runId }); }
  cancel() { this.post({ type: 'cancel', runId: this.runId }); }
  export() { this.post({ type: 'export', runId: this.runId }); }
  dispose() { this.worker?.terminate(); this.worker = null; }
  private post(message: EngineCommand) { this.worker?.postMessage(message); }
}
