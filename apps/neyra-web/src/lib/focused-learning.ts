import type { FocusedInput, FocusedObservation, FocusedSpinLabel, FocusedWitness, TrainingField } from './types';

export type FocusedOrigin = {
  input: FocusedInput;
  rootCandidateId: string;
  parentObservationId: string | null;
  parentWitnessId: number | null;
  gain: number;
};

const sameRows = (a: number[], b: number[]) => a.length === b.length && a.every((row, y) => row === b[y]);
const rows = (value: unknown): value is number[] => Array.isArray(value) && value.length === 24 &&
  value.every(row => Number.isInteger(row) && row >= 0 && row < 1023) && value.some(Boolean);

/** Identity only. A physical certificate is never reflected or rewritten. */
export function focusedBoardKey(value: number[]): string {
  const reflection = value.map(row => {
    let result = 0;
    for (let x = 0; x < 10; x++) result |= ((row >>> x) & 1) << (9 - x);
    return result;
  });
  for (let y = 0; y < value.length; y++) {
    if (reflection[y] !== value[y]) return (reflection[y] < value[y] ? reflection : value).join(',');
  }
  return value.join(',');
}

export function focusedRoot(field: TrainingField): FocusedOrigin | null {
  if (field.source.bagBoundary < 2 || field.source.bagBoundary > 510 || !rows(field.rows)) return null;
  const source = field.source;
  return { input: { id: field.id, rows: [...field.rows], source: {
    kind: 'search', family: `search:${source.runId}`, reference: source.reference,
    bagBoundary: source.bagBoundary, runId: source.runId, rootCandidateId: source.candidateId,
    rootBagBoundary: source.bagBoundary, rootSeed: source.seed, buildVersion: source.buildVersion,
    rules: source.rules, trust: 'client-reported'
  } }, rootCandidateId: source.candidateId, parentObservationId: null, parentWitnessId: null, gain: 0 };
}

/** Read exact native evidence before using a result field as another origin. */
function certifiedChain(label: FocusedSpinLabel, last: FocusedWitness): boolean {
  const byId = new Map(label.certifiedWitnesses.map(witness => [witness.id, witness]));
  const chain: FocusedWitness[] = [];
  const seen = new Set<number>();
  let current: FocusedWitness | undefined = last;
  while (current) {
    if (seen.has(current.id) || chain.length >= label.config.horizonBags) return false;
    seen.add(current.id);
    chain.unshift(current);
    if (current.parentId === null) break;
    current = byId.get(current.parentId);
    if (!current) return false;
  }
  if (!chain.length || chain[0].parentId !== null) return false;
  let previous = label.input.rows, spins = 0;
  for (let index = 0; index < chain.length; index++) {
    const witness = chain[index];
    if (witness.bags !== index + 1 || witness.checked !== 5040 || witness.success !== 5040 ||
        witness.quietSafeSupplies !== 5040 || witness.firstBagSymmetryAssumed !== false ||
        !sameRows(witness.startRows, previous) || witness.startSpins !== spins ||
        !rows(witness.endRows) || witness.frames.length !== 7) return false;
    let counted = spins;
    for (let lock = 0; lock < witness.frames.length; lock++) {
      const frame = witness.frames[lock];
      if (!Number.isInteger(frame.lines) || frame.lines < 0 || frame.lines > 1 ||
          frame.placed !== index * 7 + lock + 1 || frame.spins !== counted + frame.lines ||
          (frame.lines === 1 && (!frame.spin || frame.piece === 'O'))) return false;
      counted += frame.lines;
    }
    if (witness.endSpins !== counted || !sameRows(witness.frames[6].rows, witness.endRows)) return false;
    previous = witness.endRows;
    spins = counted;
  }
  return true;
}

export function focusedDescendants(observation: FocusedObservation): FocusedOrigin[] {
  const label = observation.label;
  if (label.policy !== 'neyra-fixed-bag-continuation-v2' || label.schemaVersion !== 2 ||
      label.config.horizonBags < 2 || label.config.horizonBags > 3) return [];
  const result: FocusedOrigin[] = [];
  for (const witness of label.certifiedWitnesses) {
    const depth = label.input.source.bagBoundary + witness.bags;
    if (depth > 510 || !certifiedChain(label, witness)) continue;
    result.push({ input: {
      id: `${observation.id}:certificate-${witness.id}`, rows: [...witness.endRows],
      source: { ...label.input.source, bagBoundary: depth,
        reference: `neyra:observation:${observation.id}/certificate:${witness.id}` }
    }, rootCandidateId: observation.rootCandidateId, parentObservationId: observation.id,
    parentWitnessId: witness.id, gain: witness.endSpins });
  }
  return result.sort((a, b) => b.input.source.bagBoundary - a.input.source.bagBoundary ||
    b.gain - a.gain || a.input.id.localeCompare(b.input.id));
}

/** Shared, bounded frontier gives descendants priority over more shallow roots. */
export class FocusedFrontier {
  private pending: FocusedOrigin[] = [];
  private seen = new Set<string>();
  private shallowJobs = new Map<number, number>();
  private shallowRoots = new Map<number, number>();

  get length() { return this.pending.length; }

  add(origin: FocusedOrigin) {
    if (this.seen.size >= 4096 || !rows(origin.input.rows) || !this.eligible(origin)) return;
    const key = focusedBoardKey(origin.input.rows);
    if (this.seen.has(key)) return;
    this.seen.add(key);
    this.pending.push(origin);
    this.pending.sort((a, b) => b.input.source.bagBoundary - a.input.source.bagBoundary ||
      Number(b.parentObservationId !== null) - Number(a.parentObservationId !== null) ||
      b.gain - a.gain || a.input.id.localeCompare(b.input.id));
    if (this.pending.length > 128) this.pending.length = 128;
  }

  private eligible(origin: FocusedOrigin): boolean {
    const depth = origin.input.source.bagBoundary;
    if (depth !== 2 && depth !== 3) return true;
    return (this.shallowJobs.get(depth) ?? 0) < 4 &&
      (origin.parentObservationId !== null || (this.shallowRoots.get(depth) ?? 0) < 2);
  }

  take() {
    const origin = this.pending.shift();
    if (!origin) return null;
    const depth = origin.input.source.bagBoundary;
    if (depth === 2 || depth === 3) {
      this.shallowJobs.set(depth, (this.shallowJobs.get(depth) ?? 0) + 1);
      if (origin.parentObservationId === null) this.shallowRoots.set(depth, (this.shallowRoots.get(depth) ?? 0) + 1);
      // Wide shallow sibling sets must not consume the whole observation quota.
      // Exhausting this small allowance lets the preserved search seek deeper roots.
      this.pending = this.pending.filter((entry) => this.eligible(entry));
    }
    return origin;
  }

  /** A final broad-search snapshot can precede its completed acknowledgement.
   * Return a job assigned in that gap; it has consumed no collector work. */
  release(origin: FocusedOrigin) {
    const depth = origin.input.source.bagBoundary;
    if (depth === 2 || depth === 3) {
      this.shallowJobs.set(depth, Math.max(0, (this.shallowJobs.get(depth) ?? 0) - 1));
      if (origin.parentObservationId === null) {
        this.shallowRoots.set(depth, Math.max(0, (this.shallowRoots.get(depth) ?? 0) - 1));
      }
    }
    this.pending.push(origin);
    this.pending.sort((a, b) => b.input.source.bagBoundary - a.input.source.bagBoundary ||
      Number(b.parentObservationId !== null) - Number(a.parentObservationId !== null) ||
      b.gain - a.gain || a.input.id.localeCompare(b.input.id));
    if (this.pending.length > 128) this.pending.length = 128;
  }

  clear(resetBudget = true) {
    this.pending = [];
    this.seen.clear();
    if (resetBudget) { this.shallowJobs.clear(); this.shallowRoots.clear(); }
  }
}
