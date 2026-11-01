import type { FocusedObservation, TrainingContribution } from './types';

export type ContributionPhase = 'ready' | 'sending' | 'retrying' | 'paused' | 'disabled' | 'unavailable';
export interface ContributionState {
  enabled: boolean;
  phase: ContributionPhase;
  queued: number;
  observationQueued: number;
  acknowledged: number;
  dropped: number;
}

type Pending = { serialized: string; bytes: number; bagBoundary: number; key: string; order: number };
type PendingObservation = { id: string; body: string; key: string };
export type ObservationDelivery = { id: string; status: 'acknowledged' | 'rejected' };
type Batch = { entries: Pending[]; observation?: PendingObservation; body: string };
type PublicIngestion = { url: string; anonKey: string };
export const CONTRIBUTION_CONSENT_KEY = 'neyra-contribution-consent-v2';

/** Earlier default-on preferences are not explicit consent to this collection. */
export function hasContributionConsent(stored: string | null): boolean {
  try {
    const value = stored === null ? null : JSON.parse(stored);
    return value?.version === 2 && value?.scope === 'verified-search-process' && value?.decision === 'allow';
  } catch { return false; }
}

export function contributionConsentValue(enabled: boolean): string {
  return JSON.stringify({ version: 2, scope: 'verified-search-process', decision: enabled ? 'allow' : 'deny' });
}

const MAX_QUEUE = 128;
const MAX_BATCH = 8;
// Leave room for Postgres JSONB whitespace and the service RPC wrapper.
const MAX_BODY_BYTES = 24 * 1024;
const MAX_FIELD_BYTES = 6144;
const MAX_SEEN = 4096;
// The coordinator uses the same cap for active jobs and unacknowledged labels.
// Observations never evict another observation: descendants need its receipt.
export const MAX_PENDING_OBSERVATIONS = 8;
const MAX_OBSERVATION_BYTES = 512 * 1024;
const MAX_ATTEMPTS = 6;
const FLUSH_DELAY = 3000;
const encoder = new TextEncoder();
const envelopePrefix = (schemaVersion: number, key: string) =>
  `${JSON.stringify({ schemaVersion, consent: { version: 2,
    scope: 'verified-search-process', decision: 'allow' } }).slice(0, -1)},"${key}":[`;
const TRACE_PREFIX = envelopePrefix(3, 'fields');
const OBSERVATION_PREFIX = envelopePrefix(4, 'observations');
const ENVELOPE_SUFFIX = ']}';
const TRACE_ENVELOPE_BYTES = encoder.encode(TRACE_PREFIX + ENVELOPE_SUFFIX).byteLength;

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

/** Bounded anonymous transport for retained traces and measured continuations. */
export class ContributionClient {
  private queue: Pending[] = [];
  private observations: PendingObservation[] = [];
  private seen = new Map<string, true>();
  private acknowledged = 0;
  private dropped = 0;
  private order = 0;
  private attempts = 0;
  private enabled = false;
  private disposed = false;
  private generation = 0;
  private inFlight: AbortController | null = null;
  private inFlightKeys = new Set<string>();
  private timer: ReturnType<typeof setTimeout> | undefined;
  private phase: ContributionPhase = 'ready';
  private readonly configured: boolean;

  constructor(private config: PublicIngestion, private receive: (state: ContributionState) => void,
    private delivered: (delivery: ObservationDelivery) => void = () => {}) {
    this.configured = publicIngestion(config);
    this.publish();
  }

  get state(): ContributionState {
    return { enabled: this.enabled,
      phase: !this.enabled ? 'disabled' : !this.configured ? 'unavailable' : this.phase,
      queued: this.queue.length + this.observations.length, observationQueued: this.observations.length,
      acknowledged: this.acknowledged, dropped: this.dropped };
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
      for (const entry of this.observations) this.seen.delete(entry.key);
      this.queue = [];
      this.observations = [];
      this.attempts = 0;
      this.phase = 'ready';
    } else this.schedule();
    this.publish();
  }

  enqueue(field: TrainingContribution) {
    if (this.disposed || !this.enabled || !this.configured) return;
    // Traces are untrusted observations, separate from labels and inference inputs.
    // Distinct retained paths can share a board without sharing a search history.
    const key = `${field.id}:${field.source.bagBoundary}:${field.rows.join(',')}`;
    if (this.seen.has(key)) return;
    const serialized = JSON.stringify(field);
    const bytes = encoder.encode(serialized).byteLength;
    if (!field.trace || bytes > MAX_FIELD_BYTES) {
      this.dropped++;
      this.publish();
      return;
    }
    // The immutable wire representation owns the queued snapshot. Avoid a
    // second full object graph and repeat serialization on every HTTP retry.
    const entry = { serialized, bytes, bagBoundary: field.source.bagBoundary, key, order: this.order++ };
    this.seen.set(key, true);
    if (this.seen.size > MAX_SEEN) this.seen.delete(this.seen.keys().next().value!);
    this.queue.push(entry);
    this.queue.sort((a, b) => b.bagBoundary - a.bagBoundary || a.order - b.order);
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

  /** The caller retains descendants until an acknowledgement for this ID.
   * Saturation rejects new work instead of silently dropping its parent. */
  enqueueObservation(observation: FocusedObservation): boolean {
    if (this.disposed || !this.enabled || !this.configured ||
        this.observations.length >= MAX_PENDING_OBSERVATIONS) return false;
    const key = `observation:${observation.id}`;
    if (this.seen.has(key)) return false;
    const serialized = JSON.stringify(observation);
    if (encoder.encode(serialized).byteLength > MAX_OBSERVATION_BYTES) {
      this.dropped++;
      this.publish();
      return false;
    }
    this.observations.push({ id: observation.id, body: OBSERVATION_PREFIX + serialized + ENVELOPE_SUFFIX, key });
    this.seen.set(key, true);
    if (this.seen.size > MAX_SEEN) this.seen.delete(this.seen.keys().next().value!);
    this.schedule(0);
    this.publish();
    return true;
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
    this.observations = [];
    this.seen.clear();
  }

  private publish() { this.receive(this.state); }

  private schedule(delay = FLUSH_DELAY) {
    if (this.disposed || !this.enabled || !this.configured || (!this.queue.length && !this.observations.length) ||
        this.inFlight || this.phase === 'paused') return;
    if (this.timer) {
      // New observations can overtake a trace batching delay, but never an
      // automatic failure backoff or the server's Retry-After interval.
      if (delay !== 0 || this.phase !== 'ready') return;
      clearTimeout(this.timer);
    }
    this.timer = setTimeout(() => { this.timer = undefined; void this.flush(); }, delay);
  }

  private batch(): Batch {
    // A single observation leaves enough room for all bounded proof frames.
    // Parents are sent before the coordinator admits their descendants.
    const observation = this.observations[0];
    if (observation) return { entries: [], observation, body: observation.body };
    const entries: Pending[] = [];
    let bytes = TRACE_ENVELOPE_BYTES;
    for (const entry of this.queue.slice(0, MAX_BATCH)) {
      const nextBytes = bytes + entry.bytes + (entries.length ? 1 : 0);
      if (nextBytes > MAX_BODY_BYTES) break;
      entries.push(entry);
      bytes = nextBytes;
    }
    return { entries, body: entries.length ? TRACE_PREFIX + entries.map(entry => entry.serialized).join(',') + ENVELOPE_SUFFIX : '' };
  }

  private remove(entries: Pending[]) {
    const removed = new Set(entries.map((entry) => entry.key));
    this.queue = this.queue.filter((entry) => !removed.has(entry.key));
  }

  private settle(batch: Batch, status: ObservationDelivery['status']) {
    if (batch.observation) {
      const entry = batch.observation;
      this.observations = this.observations.filter((pending) => pending.key !== entry.key);
      // Delivery callbacks cannot turn a successful HTTP receipt into a retry.
      try { this.delivered({ id: entry.id, status }); } catch { /* Consumer is optional. */ }
    } else this.remove(batch.entries);
  }

  private async flush() {
    if (this.disposed || !this.enabled || !this.configured || this.inFlight ||
        (!this.queue.length && !this.observations.length)) return;
    const batch = this.batch();
    const { entries, observation, body } = batch;
    const count = observation ? 1 : entries.length;
    if (!count) {
      this.queue.shift();
      this.dropped++;
      this.schedule();
      this.publish();
      return;
    }
    const controller = new AbortController();
    const generation = this.generation;
    this.inFlight = controller;
    this.inFlightKeys = new Set(observation ? [observation.key] : entries.map((entry) => entry.key));
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
          this.settle(batch, 'rejected');
          this.dropped += count;
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
          result.accepted! < 0 || result.duplicates! < 0 || result.accepted! + result.duplicates! !== count) {
        throw new Error('Contribution acknowledgement does not match the batch');
      }
      this.settle(batch, 'acknowledged');
      this.acknowledged += count;
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
        this.schedule(retryDelay ?? (this.observations.length ? 0 : FLUSH_DELAY));
        this.publish();
      }
    }
  }
}
