/** Bounded client-reported observations. Shape checks do not re-prove SRS or P7. */
export const MAX_OBSERVATION_BYTES = 512 * 1024;
const REVISION = 'sha256:05af5a80cc12bc8316d3f98d8a7c51a92c1883fb831f5a476c9880c5d516cd9f';
const RULES = 'neyra-srsplus-sonicdrop-clearonlyhold-v1';
type Obj = Record<string, unknown>;
export class ObservationError extends Error {
  constructor(public status: number, message: string) { super(message); }
}
function requireValue(condition: unknown, message = 'Invalid focused observation'): asserts condition {
  if (!condition) throw new ObservationError(400, message);
}
function obj(value: unknown, keys: string[]): Obj {
  requireValue(value && typeof value === 'object' && !Array.isArray(value));
  const result = value as Obj;
  requireValue(Object.keys(result).length === keys.length && keys.every(key => Object.hasOwn(result, key)), 'Unexpected observation fields');
  return result;
}
function int(value: unknown, low = 0, high = Number.MAX_SAFE_INTEGER): number {
  requireValue(typeof value === 'number' && Number.isSafeInteger(value) && value >= low && value <= high, 'Invalid observation count');
  return value;
}
function str(value: unknown, max: number): string {
  requireValue(typeof value === 'string' && value.length > 0 && value.length <= max && !/[\x00-\x1f]/.test(value));
  return value;
}
function list(value: unknown, min: number, max: number): unknown[] {
  requireValue(Array.isArray(value) && value.length >= min && value.length <= max, 'Invalid observation array');
  return value;
}
function rows(value: unknown): number[] { return list(value, 24, 24).map(row => int(row, 0, 1022)); }
function same(a: number[], b: number[]): boolean { return a.every((v, i) => v === b[i]); }
function cells(value: unknown, logical: boolean): number[][] {
  const result = list(value, 4, 4).map(cell => {
    const pair = list(cell, 2, 2); return [int(pair[0], 0, 9), int(pair[1], 0, logical ? 47 : 23)];
  });
  requireValue(new Set(result.map(cell => cell.join(','))).size === 4, 'Repeated placement cell');
  return result;
}
function piece(value: unknown): string { const p = str(value, 1); requireValue('IOTSZJL'.includes(p)); return p; }
function targets(value: unknown, length: number): void {
  const pieces = list(value, length, length).map(raw => {
    const t = obj(raw, ['piece', 'cells']); cells(t.cells, true); return piece(t.piece);
  });
  requireValue(new Set(pieces).size === length, 'Repeated bag piece');
}
function frames(value: unknown, start: number[], startSpins: number, placed: number, length: number) {
  let current = start.slice(), spins = startSpins;
  const pieces = new Set<string>();
  for (const raw of list(value, length, length)) {
    const f = obj(raw, ['rows', 'piece', 'placedCells', 'spins', 'placed', 'lines', 'spin']);
    const p = piece(f.piece); requireValue(!pieces.has(p)); pieces.add(p);
    const next = rows(f.rows), lines = int(f.lines, 0, 1), locked = cells(f.placedCells, false);
    requireValue(typeof f.spin === 'boolean' && (lines === 0 || f.spin && p !== 'O'));
    for (const [x, y] of locked) {
      requireValue((current[y] & (1 << x)) === 0, 'Overlapping placement'); current[y] |= 1 << x;
    }
    const full = current.filter(row => row === 1023).length;
    requireValue(full === lines, 'Line-clear mismatch');
    current = current.filter(row => row !== 1023);
    while (current.length < 24) current.push(0);
    spins += lines; placed++;
    requireValue(f.spins === spins && f.placed === placed && same(next, current), 'Frame transition mismatch');
  }
  return { rows: current, spins };
}
function countOrNull(value: unknown, high: number): number | null { return value === null ? null : int(value, 0, high); }

export function cleanObservation(value: unknown) {
  if (new TextEncoder().encode(JSON.stringify(value)).length > MAX_OBSERVATION_BYTES) {
    throw new ObservationError(413, 'Focused observation exceeds limit');
  }
  const o = obj(value, ['schemaVersion', 'kind', 'id', 'runId', 'rootCandidateId', 'sequence', 'parentObservationId', 'parentWitnessId', 'label']);
  requireValue(o.schemaVersion === 1 && o.kind === 'focused-spin-observation');
  const runId = str(o.runId, 80), rootCandidateId = str(o.rootCandidateId, 100), sequence = int(o.sequence, 1, 1000000);
  requireValue(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(runId) && /^w\d+:plan-\d+$/.test(rootCandidateId));
  requireValue(o.id === `${runId}:observation-${sequence}`, 'Observation identity mismatch');
  const parentObservationId = o.parentObservationId === null ? null : str(o.parentObservationId, 120);
  const parentWitnessId = o.parentWitnessId === null ? null : int(o.parentWitnessId, 1);
  requireValue((parentObservationId === null) === (parentWitnessId === null), 'Incomplete parent link');
  if (parentObservationId !== null) {
    requireValue(parentObservationId.startsWith(`${runId}:observation-`));
    const parentSequence = parentObservationId.slice(`${runId}:observation-`.length);
    requireValue(/^[1-9]\d{0,6}$/.test(parentSequence) && Number(parentSequence) < sequence, 'Parent must precede child');
  }
  const label = obj(o.label, ['schemaVersion', 'task', 'policy', 'input', 'config', 'targets', 'measured', 'certifiedWitnesses', 'reachableWitness']);
  requireValue(label.schemaVersion === 2 && label.task === 'naga-spin-observations-v2' && label.policy === 'neyra-fixed-bag-continuation-v2');
  const input = obj(label.input, ['id', 'rows', 'source']), board = rows(input.rows);
  requireValue(board.some(Boolean), 'Empty first-bag fields are excluded');
  const source = obj(input.source, ['kind', 'family', 'reference', 'bagBoundary', 'runId', 'rootCandidateId', 'rootBagBoundary', 'rootSeed', 'buildVersion', 'rules', 'trust']);
  const bagBoundary = int(source.bagBoundary, 2, 512), rootBagBoundary = int(source.rootBagBoundary, 2, bagBoundary);
  int(source.rootSeed, 1, 0xffffffff); str(source.buildVersion, 96);
  requireValue(source.kind === 'search' && source.family === `search:${runId}` && source.runId === runId && source.rootCandidateId === rootCandidateId && source.rules === RULES && source.trust === 'client-reported');
  requireValue(input.id === (parentObservationId === null ? `${runId}:${rootCandidateId}` : `${parentObservationId}:certificate-${parentWitnessId}`));
  requireValue(source.reference === (parentObservationId === null ? `neyra:run:${runId}/candidate:${rootCandidateId}` : `neyra:observation:${parentObservationId}/certificate:${parentWitnessId}`));
  requireValue(parentObservationId !== null || bagBoundary === rootBagBoundary);
  const config = obj(label.config, ['horizonBags', 'workBudget', 'beamWidth', 'profile', 'seed', 'maxNodes', 'attemptsPerNode', 'sourceRevision']);
  const fixed: Obj = {horizonBags: 2, workBudget: 131072, beamWidth: 48, profile: 0, seed: 1, maxNodes: 96, attemptsPerNode: 6, sourceRevision: REVISION};
  requireValue(Object.keys(fixed).every(key => config[key] === fixed[key]), 'Collector policy differs from fixed training policy');
  const target = obj(label.targets, ['foundSpin', 'observedSpins']), observedSpins = int(target.observedSpins, 0, 12);
  requireValue(target.foundSpin === (observedSpins > 0));
  const m = obj(label.measured, ['maxReachableObservedSpins', 'geometricWholeBagChains', 'productiveGeometricWholeBagChains', 'certifiedWholeBagChains', 'productiveCertifiedWholeBagChains', 'maxCertifiedObservedSpins', 'depths', 'workUsed', 'generationWork', 'verificationWork', 'geometricPlacementsConsidered', 'canonicalSuppliesChecked', 'quietPolicyRejections', 'supplyRejections', 'graphOverflows', 'nodeCapacityDrops', 'censored', 'stopReason', 'evidenceKind', 'distinctness', 'workUnit']);
  requireValue(m.maxReachableObservedSpins === observedSpins && m.censored === true);
  const work = int(m.workUsed, 0, 131072);
  requireValue(int(m.generationWork, 0, work) + int(m.verificationWork, 0, work) === work, 'Inconsistent work accounting');
  for (const key of ['geometricPlacementsConsidered', 'canonicalSuppliesChecked', 'quietPolicyRejections', 'supplyRejections', 'graphOverflows', 'nodeCapacityDrops']) int(m[key]);
  const geometric = int(m.geometricWholeBagChains), certified = int(m.certifiedWholeBagChains, 0, geometric);
  int(m.productiveGeometricWholeBagChains, 0, geometric); int(m.productiveCertifiedWholeBagChains, 0, certified);
  requireValue(['work_budget', 'bounded_frontier_exhausted'].includes(String(m.stopReason)));
  requireValue(m.evidenceKind === 'input-reachable prefixes; exact quiet-safe P7 certificates for every expanded whole bag');
  requireValue(m.distinctness === 'fixed piece-to-logical-target assignment chains including deleted rows and boundary boards; productive means at least one actual spin since the supplied origin; not input sequences, supply orders or success probability');
  requireValue(m.workUnit === 'one bounded GenerationJob.advance or VerificationJob.advance; latter expands one execution node or checks at most 64 canonical supplies; forecast refinement disabled');
  const witnesses = list(label.certifiedWitnesses, certified, certified);
  // maxNodes limits expandable parents, not all certified terminal witnesses.
  requireValue(certified <= 4096, 'Witness capacity exceeded');
  const byId = new Map<number, { bags: number; rows: number[]; spins: number }>();
  for (const raw of witnesses) {
    const w = obj(raw, ['id', 'parentId', 'bags', 'startRows', 'endRows', 'startSpins', 'endSpins', 'targets', 'checked', 'success', 'quietSafeSupplies', 'firstBagSymmetryAssumed', 'representativeQueue', 'representativeInputs', 'frames']);
    const id = int(w.id, 1), bags = int(w.bags, 1, 2);
    const parent = w.parentId === null ? undefined : byId.get(int(w.parentId, 1));
    requireValue(!byId.has(id) && (w.parentId === null || parent) && bags === (parent?.bags ?? 0) + 1);
    const startRows = parent?.rows ?? board, startSpins = parent?.spins ?? 0;
    requireValue(same(rows(w.startRows), startRows) && w.startSpins === startSpins);
    const endRows = rows(w.endRows), endSpins = int(w.endSpins, startSpins, Math.min(observedSpins, startSpins + 6));
    targets(w.targets, 7);
    requireValue(w.checked === 5040 && w.success === 5040 && w.quietSafeSupplies === 5040 && w.firstBagSymmetryAssumed === false);
    requireValue(new Set(list(w.representativeQueue, 7, 7).map(v => int(v, 0, 6))).size === 7);
    for (const path of list(w.representativeInputs, 7, 7)) {
      const actions = list(path, 1, 256);
      requireValue(actions.every(a => ['left', 'right', 'cw', 'ccw', 'half', 'sonic', 'hard_drop'].includes(String(a))) && actions[actions.length - 1] === 'hard_drop');
    }
    const end = frames(w.frames, startRows, startSpins, (bags - 1) * 7, 7);
    requireValue(same(end.rows, endRows) && end.spins === endSpins);
    byId.set(id, {bags, rows: endRows, spins: endSpins});
  }
  const records = [...byId.values()], maxCertified = records.length ? Math.max(...records.map(w => w.spins)) : null;
  requireValue(m.maxCertifiedObservedSpins === maxCertified && m.productiveCertifiedWholeBagChains === records.filter(w => w.spins > 0).length);
  let geometricSum = 0, productiveSum = 0;
  for (const [i, raw] of list(m.depths, 2, 2).entries()) {
    const d = obj(raw, ['bags', 'geometricWholeBagChains', 'productiveGeometricWholeBagChains', 'certifiedWholeBagChains', 'productiveCertifiedWholeBagChains', 'maxGeometricWholeBagSpins', 'maxCertifiedSpins']);
    requireValue(d.bags === i + 1);
    const count = int(d.geometricWholeBagChains), productive = int(d.productiveGeometricWholeBagChains, 0, count), group = records.filter(w => w.bags === i + 1);
    requireValue(d.certifiedWholeBagChains === group.length && d.productiveCertifiedWholeBagChains === group.filter(w => w.spins > 0).length);
    requireValue(d.maxCertifiedSpins === (group.length ? Math.max(...group.map(w => w.spins)) : null));
    countOrNull(d.maxGeometricWholeBagSpins, (i + 1) * 6);
    geometricSum += count; productiveSum += productive;
  }
  requireValue(geometricSum === geometric && productiveSum === m.productiveGeometricWholeBagChains);
  if (label.reachableWitness !== null) {
    const r = obj(label.reachableWitness, ['certifiedParentId', 'completedParentBags', 'additionalPlacements', 'observedSpins', 'targets', 'frames']);
    const parent = r.certifiedParentId === null ? undefined : byId.get(int(r.certifiedParentId, 1));
    requireValue(r.certifiedParentId === null || parent);
    const bags = int(r.completedParentBags, 0, 1), placements = int(r.additionalPlacements, 1, 7);
    requireValue(bags === (parent?.bags ?? 0) && r.observedSpins === observedSpins);
    targets(r.targets, placements);
    const end = frames(r.frames, parent?.rows ?? board, parent?.spins ?? 0, bags * 7, placements);
    requireValue(end.spins === observedSpins);
  } else requireValue(observedSpins === 0 || maxCertified === observedSpins, 'Missing observed-spin witness');
  return {schemaVersion: 1, kind: 'focused-spin-observation', id: o.id as string, runId, rootCandidateId, sequence,
    parentObservationId, parentWitnessId, label: {...label, input: {...input, source: {...source, bagBoundary, rootBagBoundary}}}};
}
