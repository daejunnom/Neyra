import type { FocusedInput, FocusedObservation, FocusedReachableWitness, FocusedSpinLabel, FocusedWitness, TrainingField } from './types';

export type FocusedOrigin = {
  input: FocusedInput;
  rootCandidateId: string;
  parentObservationId: string | null;
  parentWitnessId: number | null;
  gain: number;
};

const sameRows = (a: number[], b: number[]) => a.length === b.length && a.every((row, y) => row === b[y]);
const physicalRows = (value: unknown): value is number[] => Array.isArray(value) && value.length === 24 &&
  value.every(row => Number.isInteger(row) && row >= 0 && row < 1023);
const rows = (value: unknown): value is number[] => physicalRows(value) && value.some(Boolean);
const record = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === 'object' && !Array.isArray(value);
const integer = (value: unknown, min: number, max: number): value is number =>
  typeof value === 'number' && Number.isSafeInteger(value) && value >= min && value <= max;

/** Check the recorded placement/clear arithmetic; SRS input reachability is
 * independently replayed by Rust, never inferred by this browser guard. */
function reachableFrames(start: number[], startSpins: number, parentBags: number,
  frames: unknown, targets: unknown): { rows: number[]; spins: number; deleted: number } | null {
  if (!Array.isArray(frames) || !frames.length || frames.length > 7 ||
    !Array.isArray(targets) || targets.length !== frames.length) return null;
  let board = [...start], spins = startSpins, deleted = 0;
  const used = new Set<string>();
  for (let index = 0; index < frames.length; index++) {
    const frame = frames[index], target = targets[index];
    if (!record(frame) || !record(target) || typeof frame.piece !== 'string' ||
      !['I', 'J', 'L', 'O', 'S', 'T', 'Z'].includes(frame.piece) || used.has(frame.piece) ||
      target.piece !== frame.piece || !Array.isArray(frame.placedCells) || frame.placedCells.length !== 4 ||
      !Array.isArray(target.cells) || target.cells.length !== 4 || !physicalRows(frame.rows) ||
      !integer(frame.lines, 0, 1) || typeof frame.spin !== 'boolean' ||
      (frame.lines === 1 && !frame.spin) || (frame.piece === 'O' && frame.spin) ||
      frame.placed !== parentBags * 7 + index + 1) return null;
    const physicalToLogical = Array.from({ length: 32 }, (_, row) => row)
      .filter(row => (deleted & (1 << row)) === 0);
    const logical: [number, number][] = [];
    for (const cell of frame.placedCells) {
      if (!Array.isArray(cell) || cell.length !== 2 || !integer(cell[0], 0, 9) ||
        !integer(cell[1], 0, 23) || (board[cell[1]] & (1 << cell[0]))) return null;
      const y = physicalToLogical[cell[1]];
      if (y === undefined) return null;
      board[cell[1]] |= 1 << cell[0];
      logical.push([cell[0], y]);
    }
    logical.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (JSON.stringify(logical) !== JSON.stringify(target.cells)) return null;
    const cleared = board.flatMap((row, y) => row === 1023 ? [y] : []);
    if (cleared.length !== frame.lines) return null;
    for (const y of cleared) deleted = (deleted | (1 << physicalToLogical[y])) >>> 0;
    board = board.filter(row => row !== 1023);
    while (board.length < 24) board.push(0);
    spins += frame.lines;
    if (frame.spins !== spins || !sameRows(board, frame.rows)) return null;
    used.add(frame.piece);
  }
  return { rows: board, spins, deleted };
}

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
function certifiedChain(label: FocusedSpinLabel, last: FocusedWitness,
  byId: ReadonlyMap<number, FocusedWitness>): boolean {
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
  if (label.schemaVersion === 3) return reachableDescendants(observation);
  if (label.policy !== 'neyra-fixed-bag-continuation-v2' || label.schemaVersion !== 2 ||
      label.config.horizonBags < 2 || label.config.horizonBags > 3 || !label.certifiedWitnesses.length) return [];
  const byId = new Map(label.certifiedWitnesses.map(witness => [witness.id, witness]));
  if (byId.size !== label.certifiedWitnesses.length) return [];
  const result: FocusedOrigin[] = [];
  for (const witness of label.certifiedWitnesses) {
    const depth = label.input.source.bagBoundary + witness.bags;
    if (depth > 510 || !certifiedChain(label, witness, byId)) continue;
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

/** V3 continuation is a replayed physical queue, not an all-supply certificate.
 * Keep its orientation and parent receipt even when training identity is mirrored. */
function reachableChain(label: FocusedSpinLabel, last: FocusedReachableWitness,
  byId: ReadonlyMap<number, FocusedReachableWitness>): boolean {
  const chain: FocusedReachableWitness[] = [], seen = new Set<number>();
  let current: FocusedReachableWitness | undefined = last;
  while (current) {
    if (seen.has(current.id) || chain.length >= label.config.horizonBags) return false;
    seen.add(current.id); chain.unshift(current);
    if (current.parentId === null) break;
    current = byId.get(current.parentId);
    if (!current) return false;
  }
  if (!chain.length || chain[0].parentId !== null) return false;
  const pieces = ['I', 'J', 'L', 'O', 'S', 'T', 'Z'];
  const inputs = new Set(['left', 'right', 'cw', 'ccw', 'half', 'sonic', 'hard_drop']);
  let previous = label.input.rows, spins = 0;
  for (let index = 0; index < chain.length; index++) {
    const witness = chain[index];
    if (witness.trust !== 'input-reachable' || witness.wholeBag !== true || witness.bags !== index + 1 ||
      !sameRows(witness.startRows, previous) || witness.startSpins !== spins || !physicalRows(witness.endRows) ||
      witness.frames.length !== 7 || witness.targets.length !== 7 || witness.representativeQueue.length !== 7 ||
      witness.representativeInputs.length !== 7 || new Set(witness.representativeQueue).size !== 7 ||
      !Number.isInteger(witness.deletedRows) || witness.deletedRows < 0 || witness.deletedRows > 0xffffffff) return false;
    const replay = reachableFrames(previous, spins, index, witness.frames, witness.targets);
    if (!replay || replay.deleted !== witness.deletedRows || replay.spins !== witness.endSpins ||
      !sameRows(replay.rows, witness.endRows)) return false;
    for (let lock = 0; lock < 7; lock++) {
      const frame = witness.frames[lock], piece = witness.representativeQueue[lock];
      const path = witness.representativeInputs[lock];
      if (!Number.isInteger(piece) || piece < 0 || piece > 6 || frame.piece !== pieces[piece] ||
        witness.targets[lock].piece !== frame.piece || !Array.isArray(path) || !path.length ||
        path[path.length - 1] !== 'hard_drop' || path.slice(0, -1).includes('hard_drop') ||
        path.some(input => !inputs.has(input)) || !physicalRows(frame.rows) ||
        !Number.isInteger(frame.lines) || frame.lines < 0 || frame.lines > 1 ||
        frame.placed !== index * 7 + lock + 1 || frame.spins !== spins + frame.lines ||
        (frame.lines === 1 && (!frame.spin || frame.piece === 'O'))) return false;
      spins += frame.lines;
    }
    if (witness.endSpins !== spins || !sameRows(witness.frames[6].rows, witness.endRows)) return false;
    previous = witness.endRows;
  }
  return true;
}

/** Versioned observation shape, physical frame continuity and honest horizons.
 * This does not turn a client-reported input path into a P7 certificate. */
export function validNextBagLabel(label: FocusedSpinLabel): boolean {
  try {
    const witnesses = label.reachableWholeBagWitnesses, measured = label.measured;
    const config = label.config, lookahead = measured.lookahead;
    if (label.schemaVersion !== 3 || label.task !== 'naga-next-bag-spin-v3' ||
      label.policy !== 'bounded-next-bag-spin-v3' || label.targetHorizonBags !== 1 ||
      config.observationVersion !== 3 || !integer(config.horizonBags, 2, 3) ||
      !integer(config.workBudget, 1, 4194304) || !integer(config.beamWidth, 4, 128) ||
      !integer(config.profile, 0, 3) || !integer(config.seed, 0, Number.MAX_SAFE_INTEGER) ||
      !integer(config.maxNodes, config.horizonBags, 256) || !integer(config.attemptsPerNode, 1, 16) ||
      typeof config.sourceRevision !== 'string' || !config.sourceRevision ||
      !rows(label.input.rows) || !integer(label.input.source.bagBoundary, 2, 510) ||
      !Array.isArray(witnesses) || witnesses.length > Math.min(config.maxNodes, 32) ||
      !Array.isArray(label.certifiedWitnesses) || label.certifiedWitnesses.length ||
      !integer(label.targets.observedSpins, 0, 6) || label.targets.foundSpin !== (label.targets.observedSpins > 0) ||
      measured.censored !== true || !integer(measured.workUsed, 0, config.workBudget) ||
      !integer(measured.generationWork, 0, measured.workUsed) ||
      measured.verificationWork !== witnesses.length * 7 ||
      measured.generationWork + measured.verificationWork !== measured.workUsed ||
      ['certifiedWholeBagChains', 'productiveCertifiedWholeBagChains', 'canonicalSuppliesChecked',
        'quietPolicyRejections', 'supplyRejections', 'graphOverflows'].some(key => measured[key] !== 0) ||
      measured.maxCertifiedObservedSpins !== null || !record(lookahead) ||
      lookahead.targetHorizonBags !== 1 || lookahead.requestedBags !== config.horizonBags ||
      lookahead.witnessCapacity !== Math.min(config.maxNodes, 32) ||
      !integer(lookahead.deepestAttemptedBag, 1, config.horizonBags) ||
      !integer(lookahead.deepestReachedPlacements, 0, lookahead.deepestAttemptedBag * 7) ||
      lookahead.reachableWholeBagChains !== witnesses.length ||
      measured.geometricWholeBagChains !== witnesses.length ||
      !integer(measured.maxReachableObservedSpins, label.targets.observedSpins, config.horizonBags * 6)) return false;
    const byId = new Map(witnesses.map(witness => [witness.id, witness]));
    if (byId.size !== witnesses.length || witnesses.some(witness => !integer(witness.id, 1, Number.MAX_SAFE_INTEGER) ||
      !reachableChain(label, witness, byId))) return false;
    const depths = measured.depths;
    if (!Array.isArray(depths) || depths.length !== config.horizonBags || depths.some((depth, index) => {
      const at = witnesses.filter(witness => witness.bags === index + 1);
      return !record(depth) || depth.bags !== index + 1 || depth.geometricWholeBagChains !== at.length ||
        depth.productiveGeometricWholeBagChains !== at.filter(witness => witness.endSpins > 0).length ||
        depth.certifiedWholeBagChains !== 0 || depth.productiveCertifiedWholeBagChains !== 0 ||
        depth.maxCertifiedSpins !== null || depth.maxGeometricWholeBagSpins !==
          (at.length ? Math.max(...at.map(witness => witness.endSpins)) : null);
    }) || lookahead.deepestCompletedBag !== Math.max(0, ...witnesses.map(witness => witness.bags))) return false;
    const validatePrefix = (value: unknown, target: boolean): boolean => {
      if (value === null || value === undefined) return target ? label.targets.observedSpins === 0 : lookahead.deepestReachedPlacements === 0;
      if (!record(value) || value.certifiedParentId !== null ||
        !integer(value.completedParentBags, 0, config.horizonBags - 1) ||
        !integer(value.additionalPlacements, 1, 7)) return false;
      const parent = value.reachableParentId === undefined || value.reachableParentId === null
        ? null : byId.get(value.reachableParentId as number);
      if ((value.completedParentBags === 0) !== (parent === null) ||
        (value.completedParentBags > 0 && (!parent || parent.bags !== value.completedParentBags)) ||
        (target && value.completedParentBags !== 0)) return false;
      const result = reachableFrames(parent ? parent.endRows : label.input.rows,
        parent ? parent.endSpins : 0, value.completedParentBags, value.frames, value.targets);
      return result !== null && Array.isArray(value.frames) && value.frames.length === value.additionalPlacements &&
        result.spins === value.observedSpins &&
        (target ? value.observedSpins === label.targets.observedSpins :
          lookahead.deepestReachedPlacements === value.completedParentBags * 7 + value.additionalPlacements);
    };
    if (!validatePrefix(label.targetWitness, true) || !validatePrefix(label.reachableWitness, false)) return false;
    const compensation = lookahead.bestTwoBagPath;
    if (compensation === null) return !witnesses.some(witness => witness.bags === 2);
    if (!record(compensation)) return false;
    const witness = byId.get(compensation.witnessId as number);
    return witness?.bags === 2 && compensation.firstBagSpins === witness.startSpins &&
      compensation.secondBagSpins === witness.endSpins - witness.startSpins && compensation.totalSpins === witness.endSpins;
  } catch { return false; }
}

function reachableDescendants(observation: FocusedObservation): FocusedOrigin[] {
  const label = observation.label, witnesses = label.reachableWholeBagWitnesses;
  if (!validNextBagLabel(label) || !witnesses) return [];
  const byId = new Map(witnesses.map(witness => [witness.id, witness]));
  if (byId.size !== witnesses.length) return [];
  const result: FocusedOrigin[] = [];
  for (const witness of witnesses) {
    const depth = label.input.source.bagBoundary + witness.bags;
    if (depth > 510 || !rows(witness.endRows) || !reachableChain(label, witness, byId)) continue;
    result.push({ input: {
      id: `${observation.id}:reachable-${witness.id}`, rows: [...witness.endRows],
      source: { ...label.input.source, bagBoundary: depth,
        reference: `neyra:observation:${observation.id}/reachable:${witness.id}` }
    }, rootCandidateId: observation.rootCandidateId, parentObservationId: observation.id,
    parentWitnessId: witness.id, gain: witness.endSpins });
  }
  return result.sort(compareOrigins);
}

function compareOrigins(a: FocusedOrigin, b: FocusedOrigin): number {
  return b.input.source.bagBoundary - a.input.source.bagBoundary ||
    Number(b.parentObservationId !== null) - Number(a.parentObservationId !== null) ||
    b.gain - a.gain || a.input.id.localeCompare(b.input.id);
}

/** Shared, bounded frontier gives descendants priority over more shallow roots. */
export class FocusedFrontier {
  private pending: FocusedOrigin[] = [];
  private pendingByKey = new Map<string, FocusedOrigin>();
  // Only dispatched origins consume history. An offered board trimmed from the
  // bounded frontier has not been observed and must remain eligible later.
  private seen = new Set<string>();
  private shallowJobs = new Map<number, number>();
  private shallowRoots = new Map<number, number>();

  get length() { return this.pending.length; }

  add(origin: FocusedOrigin) {
    if (this.seen.size >= 4096 || !rows(origin.input.rows) || !this.eligible(origin)) return;
    const key = focusedBoardKey(origin.input.rows);
    if (this.seen.has(key)) return;
    const previous = this.pendingByKey.get(key);
    if (previous) {
      if (compareOrigins(origin, previous) >= 0) return;
      this.pending.splice(this.pending.indexOf(previous), 1);
    }
    this.pendingByKey.set(key, origin);
    this.pending.push(origin);
    this.pending.sort(compareOrigins);
    if (this.pending.length > 128) {
      const removed = this.pending.pop()!;
      this.pendingByKey.delete(focusedBoardKey(removed.input.rows));
    }
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
    const key = focusedBoardKey(origin.input.rows);
    this.pendingByKey.delete(key);
    this.seen.add(key);
    const depth = origin.input.source.bagBoundary;
    if (depth === 2 || depth === 3) {
      this.shallowJobs.set(depth, (this.shallowJobs.get(depth) ?? 0) + 1);
      if (origin.parentObservationId === null) this.shallowRoots.set(depth, (this.shallowRoots.get(depth) ?? 0) + 1);
      // Wide shallow sibling sets must not consume the whole observation quota.
      // Exhausting this small allowance lets the preserved search seek deeper roots.
      this.pending = this.pending.filter((entry) => {
        if (this.eligible(entry)) return true;
        this.pendingByKey.delete(focusedBoardKey(entry.input.rows));
        return false;
      });
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
    this.seen.delete(focusedBoardKey(origin.input.rows));
    this.add(origin);
  }

  clear(resetBudget = true) {
    this.pending = [];
    this.pendingByKey.clear();
    this.seen.clear();
    if (resetBudget) { this.shallowJobs.clear(); this.shallowRoots.clear(); }
  }
}
