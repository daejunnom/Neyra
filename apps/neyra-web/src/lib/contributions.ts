import type { TrainingField } from './types';

export type ContributionPhase = 'ready' | 'sending' | 'retrying' | 'paused' | 'disabled' | 'unavailable';
export interface ContributionState {
  enabled: boolean;
  phase: ContributionPhase;
  queued: number;
  acknowledged: number;
  dropped: number;
}

type Pending = { field: TrainingField; key: string; order: number };
type PublicIngestion = { url: string; anonKey: string };
const MAX_QUEUE = 128;
const MAX_BATCH = 8;
const MAX_BODY_BYTES = 32 * 1024;
const MAX_SEEN = 4096;
const MAX_ATTEMPTS = 6;
const FLUSH_DELAY = 3000;

/** This guard catches accidental privileged-key configuration. The server is
 * still responsible for verifying the JWT, payload and storage permissions. */
function publicIngestion(config: PublicIngestion): boolean {
  try {
    if (new URL(config.url).protocol !== 'https:') return false;
    const part = config.anonKey.split('.')[1];
    if (!part) return false;
    return JSON.parse(atob(part.replace(/-/g, '+').replace(/_/g, '/'))).role === 'anon';
  } catch { return false; }
}

/** Bounded, anonymous field contribution. It has no search or model-training API. */
export class ContributionClient {
  private queue: Pending[] = [];
  private seen = new Map<string, true>();
  private acknowledged = 0;
  private dropped = 0;
  private order = 0;
  private attempts = 0;
  private enabled = true;
  private disposed = false;
  private generation = 0;
  private inFlight: AbortController | null = null;
  private inFlightKeys = new Set<string>();
  private timer: ReturnType<typeof setTimeout> | undefined;
  private phase: ContributionPhase = 'ready';
  private readonly configured: boolean;

  constructor(private config: PublicIngestion, private receive: (state: ContributionState) => void) {
    this.configured = publicIngestion(config);
    this.publish();
  }

  get state(): ContributionState {
    return { enabled: this.enabled,
      phase: !this.enabled ? 'disabled' : !this.configured ? 'unavailable' : this.phase,
      queued: this.queue.length, acknowledged: this.acknowledged, dropped: this.dropped };
  }

  setEnabled(enabled: boolean) {
    if (this.disposed || enabled === this.enabled) return;
    this.enabled = enabled;
    if (!enabled) {
      this.generation++;
      clearTimeout(this.timer);
      this.timer = undefined;
      this.inFlight?.abort();
      for (const entry of this.queue) this.seen.delete(entry.key);
      this.queue = [];
      this.attempts = 0;
      this.phase = 'ready';
    } else this.schedule();
    this.publish();
  }

  enqueue(field: TrainingField) {
    if (this.disposed || !this.enabled || !this.configured) return;
    // These are inputs only. Remote training must derive new labels rather
    // than trust any browser's certificate or previous board evaluation.
    const key = `${field.source.runId}:${field.source.bagBoundary}:${field.rows.join(',')}`;
    if (this.seen.has(key)) return;
    const copy = JSON.parse(JSON.stringify(field)) as TrainingField;
    const entry = { field: copy, key, order: this.order++ };
    this.seen.set(key, true);
    if (this.seen.size > MAX_SEEN) this.seen.delete(this.seen.keys().next().value!);
    this.queue.push(entry);
    this.queue.sort((a, b) => b.field.source.bagBoundary - a.field.source.bagBoundary || a.order - b.order);
    if (this.queue.length > MAX_QUEUE) {
      const index = this.queue.findLastIndex((item) => !this.inFlightKeys.has(item.key));
      const [evicted] = this.queue.splice(index, 1);
      this.dropped++;
      // Keep the seen marker: a repeated shallow record must not refill the queue.
      if (evicted === entry) { this.publish(); return; }
    }
    this.schedule();
    this.publish();
  }

  /** A deliberate retry or an online transition gives a stopped batch a new bounded lease. */
  retry() {
    if (this.disposed || !this.enabled || !this.configured || this.inFlight) return;
    this.attempts = 0;
    this.phase = 'ready';
    this.schedule(0);
    this.publish();
  }

  dispose() {
    this.disposed = true;
    this.generation++;
    clearTimeout(this.timer);
    this.timer = undefined;
    this.inFlight?.abort();
    this.queue = [];
    this.seen.clear();
  }

  private publish() { this.receive(this.state); }

  private schedule(delay = FLUSH_DELAY) {
    if (this.disposed || !this.enabled || !this.configured || !this.queue.length ||
        this.inFlight || this.timer || this.phase === 'paused') return;
    this.timer = setTimeout(() => { this.timer = undefined; void this.flush(); }, delay);
  }

  private batch(): { entries: Pending[]; body: string } {
    const entries: Pending[] = [];
    const encoder = new TextEncoder();
    let body = '';
    for (const entry of this.queue.slice(0, MAX_BATCH)) {
      const trial = JSON.stringify({ schemaVersion: 2, fields: [...entries, entry].map((item) => item.field) });
      if (encoder.encode(trial).byteLength > MAX_BODY_BYTES) break;
      entries.push(entry);
      body = trial;
    }
    return { entries, body };
  }

  private remove(entries: Pending[]) {
    const removed = new Set(entries.map((entry) => entry.key));
    this.queue = this.queue.filter((entry) => !removed.has(entry.key));
  }

  private async flush() {
    if (this.disposed || !this.enabled || !this.configured || this.inFlight || !this.queue.length) return;
    const { entries, body } = this.batch();
    if (!entries.length) {
      this.queue.shift();
      this.dropped++;
      this.schedule();
      this.publish();
      return;
    }
    const controller = new AbortController();
    const generation = this.generation;
    this.inFlight = controller;
    this.inFlightKeys = new Set(entries.map((entry) => entry.key));
    this.phase = 'sending';
    this.publish();
    const timeout = setTimeout(() => controller.abort(), 15000);
    let retryDelay: number | null = null;
    try {
      const response = await fetch(this.config.url, {
        method: 'POST', credentials: 'omit', referrerPolicy: 'no-referrer',
        headers: { 'Content-Type': 'application/json', apikey: this.config.anonKey,
          Authorization: `Bearer ${this.config.anonKey}` },
        body, signal: controller.signal
      });
      if (this.disposed || !this.enabled || generation !== this.generation) return;
      if (!response.ok) {
        // Invalid records cannot become valid by retrying unchanged bytes.
        if ([400, 413, 422].includes(response.status)) {
          this.remove(entries);
          this.dropped += entries.length;
          this.attempts = 0;
          this.phase = 'ready';
          return;
        }
        if ([401, 403, 404].includes(response.status)) {
          this.phase = 'paused';
          return;
        }
        const retryAfter = Number(response.headers.get('Retry-After'));
        if (response.status === 429 && Number.isFinite(retryAfter) && retryAfter > 0) {
          retryDelay = Math.min(60000, Math.max(1000, retryAfter * 1000));
        }
        throw new Error('Contribution request was not acknowledged');
      }
      const result = await response.json() as { accepted?: number; duplicates?: number };
      if (this.disposed || !this.enabled || generation !== this.generation) return;
      if (!Number.isInteger(result.accepted) || !Number.isInteger(result.duplicates) ||
          result.accepted! < 0 || result.duplicates! < 0 || result.accepted! + result.duplicates! !== entries.length) {
        throw new Error('Contribution acknowledgement does not match the batch');
      }
      this.remove(entries);
      this.acknowledged += entries.length;
      this.attempts = 0;
      this.phase = 'ready';
    } catch {
      if (this.disposed || !this.enabled || generation !== this.generation) return;
      this.attempts++;
      this.phase = this.attempts >= MAX_ATTEMPTS ? 'paused' : 'retrying';
      retryDelay ??= Math.min(60000, 2000 * 2 ** (this.attempts - 1));
    } finally {
      clearTimeout(timeout);
      this.inFlight = null;
      this.inFlightKeys.clear();
      if (!this.disposed) {
        this.schedule(retryDelay ?? FLUSH_DELAY);
        this.publish();
      }
    }
  }
}
