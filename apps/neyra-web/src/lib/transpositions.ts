import type { QuietTranspositionBatch } from './types';

export const QUIET_TT_POLICY = 'neyra-exact-quiet-tt-v1' as const;
export const QUIET_TT_BATCH_ENTRIES = 32;
export const QUIET_TT_BATCH_BYTES = 16 * 1024;
const QUIET_TT_CAPACITY = 2048;
const MAX_KEY_LENGTH = 1024;
const MAX_WORKERS = 128;
const EMPTY_BATCH_BYTES = JSON.stringify({ schemaVersion: 1, policy: QUIET_TT_POLICY, entries: [] }).length;

/** These are complete oriented execution keys, not occupancy hashes.
 * Only Rust may declare an exact quiet-policy rejection. */
export function quietTranspositionBatch(value: unknown): QuietTranspositionBatch | null {
  if (!value || typeof value !== 'object') return null;
  const batch = value as Partial<QuietTranspositionBatch>;
  if (batch.schemaVersion !== 1 || batch.policy !== QUIET_TT_POLICY ||
      !Array.isArray(batch.entries) || batch.entries.length > QUIET_TT_BATCH_ENTRIES) return null;
  let bytes = EMPTY_BATCH_BYTES;
  for (const key of batch.entries) {
    if (typeof key !== 'string' || key.length < 2 || key.length > MAX_KEY_LENGTH ||
        key.length % 2 !== 0 || !/^[0-9a-f]+$/.test(key)) return null;
    bytes += key.length + 3;
    if (bytes > QUIET_TT_BATCH_BYTES) return null;
  }
  return { schemaVersion: 1, policy: QUIET_TT_POLICY, entries: [...batch.entries] };
}

type Entry = { sequence: number; key: string; seen: Uint32Array };

/** Single-writer, bounded, immutable-key relay for this browser run.
 * Peers pull only at their existing progress cadence. Missing or evicted
 * entries cost another local verification; they never block a worker.
 * No shared memory, partial proofs, probability unions or reflected keys. */
export class QuietTranspositionBroker {
  private ring: (Entry | undefined)[] = new Array(QUIET_TT_CAPACITY);
  private byKey = new Map<string, Entry>();
  private cursors = new Map<number, number>();
  private next = 0;

  clear() {
    this.ring.fill(undefined);
    this.byKey.clear();
    this.cursors.clear();
    this.next = 0;
  }

  accept(worker: number, value: unknown): number {
    if (!Number.isInteger(worker) || worker < 0 || worker >= MAX_WORKERS) return 0;
    const batch = quietTranspositionBatch(value);
    if (!batch) return 0;
    let added = 0;
    for (const key of batch.entries) {
      const existing = this.byKey.get(key);
      if (existing) {
        this.markSeen(existing, worker);
        continue;
      }
      const sequence = this.next++;
      const index = sequence % QUIET_TT_CAPACITY;
      const replaced = this.ring[index];
      if (replaced) this.byKey.delete(replaced.key);
      const entry: Entry = { sequence, key, seen: new Uint32Array(4) };
      this.markSeen(entry, worker);
      this.ring[index] = entry;
      this.byKey.set(key, entry);
      added++;
    }
    return added;
  }

  take(worker: number): QuietTranspositionBatch | null {
    if (!Number.isInteger(worker) || worker < 0 || worker >= MAX_WORKERS) return null;
    let cursor = Math.max(this.cursors.get(worker) ?? 0, this.next - QUIET_TT_CAPACITY);
    let bytes = EMPTY_BATCH_BYTES;
    const entries: string[] = [];
    while (cursor < this.next && entries.length < QUIET_TT_BATCH_ENTRIES) {
      const entry = this.ring[cursor % QUIET_TT_CAPACITY];
      if (!entry || entry.sequence !== cursor || this.hasSeen(entry, worker)) {
        cursor++;
        continue;
      }
      if (bytes + entry.key.length + 3 > QUIET_TT_BATCH_BYTES) break;
      bytes += entry.key.length + 3;
      entries.push(entry.key);
      this.markSeen(entry, worker);
      cursor++;
    }
    this.cursors.set(worker, cursor);
    return entries.length ? { schemaVersion: 1, policy: QUIET_TT_POLICY, entries } : null;
  }

  private hasSeen(entry: Entry, worker: number): boolean {
    return (entry.seen[worker >>> 5] & (1 << (worker & 31))) !== 0;
  }

  private markSeen(entry: Entry, worker: number) {
    entry.seen[worker >>> 5] |= 1 << (worker & 31);
  }
}
