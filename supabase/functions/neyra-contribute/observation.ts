/** Bounded client-reported observations. Shape checks do not re-prove SRS or P7. */
export const MAX_OBSERVATION_BYTES = 512 * 1024;
const REVISION = 'sha256:05af5a80cc12bc8316d3f98d8a7c51a92c1883fb831f5a476c9880c5d516cd9f';
const NEXT_BAG_REVISION = 'sha256:fc94c1a31763439465ab676db88ed5ba746696885451a1848b39f8adf70788d4';
const REFINED_REVISION = 'sha256:78bd36da629a393e140b433eee6eaa9f08ddcda80fd322a857d099a91b75acff';
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
function frames(value: unknown, start: number[], startSpins: number, placed: number, length: number, logicalTargets?: unknown) {
  let current = start.slice(), spins = startSpins, deleted = 0;
  const expected = logicalTargets === undefined ? null : list(logicalTargets, length, length);
  const pieces = new Set<string>();
  for (const [index, raw] of list(value, length, length).entries()) {
    const f = obj(raw, ['rows', 'piece', 'placedCells', 'spins', 'placed', 'lines', 'spin']);
    const p = piece(f.piece); requireValue(!pieces.has(p)); pieces.add(p);
    const next = rows(f.rows), lines = int(f.lines, 0, 1), locked = cells(f.placedCells, false);
    requireValue(typeof f.spin === 'boolean' && (lines === 0 || f.spin && p !== 'O'));
    if (expected) requireValue(p !== 'O' || !f.spin);
    const logicalRows = Array.from({length: 32}, (_, y) => y).filter(y => !(deleted & (1 << y)));
    const logical: number[][] = [];
    for (const [x, y] of locked) {
      requireValue((current[y] & (1 << x)) === 0, 'Overlapping placement'); current[y] |= 1 << x;
      requireValue(logicalRows[y] !== undefined); logical.push([x, logicalRows[y]]);
    }
    if (expected) {
      const target = obj(expected[index], ['piece', 'cells']); logical.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      requireValue(target.piece === p && JSON.stringify(target.cells) === JSON.stringify(logical), 'Logical target and actual placed cells differ');
    }
    const full = current.filter(row => row === 1023).length;
    requireValue(full === lines, 'Line-clear mismatch');
    current.forEach((row, y) => { if (row === 1023) deleted = (deleted | (1 << logicalRows[y])) >>> 0; });
    current = current.filter(row => row !== 1023);
    while (current.length < 24) current.push(0);
    spins += lines; placed++;
    requireValue(f.spins === spins && f.placed === placed && same(next, current), 'Frame transition mismatch');
  }
  return { rows: current, spins, deleted };
}
function countOrNull(value: unknown, high: number): number | null { return value === null ? null : int(value, 0, high); }

/** V3 has a one-bag target and a separate, replayed two-bag observation. */
function validateNextBag(label: Obj, board: number[], refined = false): void {
  const target = obj(label.targets, ['foundSpin', 'observedSpins']), observed = int(target.observedSpins, 0, 6);
  requireValue(target.foundSpin === (observed > 0));
  const m = obj(label.measured, ['maxReachableObservedSpins', 'geometricWholeBagChains', 'productiveGeometricWholeBagChains', 'certifiedWholeBagChains', 'productiveCertifiedWholeBagChains', 'maxCertifiedObservedSpins', 'depths', 'workUsed', 'generationWork', 'verificationWork', 'geometricPlacementsConsidered', 'canonicalSuppliesChecked', 'quietPolicyRejections', 'supplyRejections', 'graphOverflows', 'nodeCapacityDrops', 'censored', 'stopReason', 'evidenceKind', 'distinctness', 'workUnit', 'lookahead', ...(refined ? ['dependencyWork'] : [])]);
  const maximum = int(m.maxReachableObservedSpins, observed, 12), work = int(m.workUsed, 0, 131072);
  requireValue(m.censored === true && int(m.generationWork, 0, work) + int(m.verificationWork, 0, work) + (refined ? int(m.dependencyWork, 0, work) : 0) === work);
  for (const key of ['geometricPlacementsConsidered', 'quietPolicyRejections', 'supplyRejections', 'graphOverflows', 'nodeCapacityDrops']) int(m[key]);
  requireValue(m.certifiedWholeBagChains === 0 && m.productiveCertifiedWholeBagChains === 0 && m.maxCertifiedObservedSpins === null
    && m.canonicalSuppliesChecked === 0 && m.quietPolicyRejections === 0 && m.supplyRejections === 0 && m.graphOverflows === 0
    && list(label.certifiedWitnesses, 0, 0).length === 0, 'Input reachability is not a P7 certificate');
  requireValue((refined ? ['phase_quotas', 'bounded_refinement_complete'] : ['work_budget', 'bounded_frontier_exhausted', 'witness_capacity']).includes(String(m.stopReason)));
  requireValue(m.evidenceKind === (refined ? 'refined input-reachable one-bag targets per supplied board and selected actual first-bag endpoint; no P7 or quiet-safe coverage claimed' : 'input-reachable prefixes and replayed whole-bag chains; no P7 or quiet-safe coverage claimed'));
  requireValue(m.distinctness === (refined ? 'physical continuation states by board, actual spins and inherited debt within each origin; per-arm local maxima remain separate; not supply orders or success probability' : 'whole-bag continuation states include physical board, depth, actual spin count and inherited debt; one retained physical path per state; not supply orders or success probability'));
  requireValue(m.workUnit === (refined ? 'one bounded GenerationJob.advance, one retained lock replay, or one origin-only component target query; replay budgets reserved; P7 verification disabled' : 'one bounded GenerationJob.advance, or one retained lock replay; seven replay units per whole-bag witness; forecast refinement and P7 verification disabled'));
  const lookahead = obj(m.lookahead, ['targetHorizonBags', 'requestedBags', 'deepestAttemptedBag', 'deepestCompletedBag', 'deepestReachedPlacements', 'reachableWholeBagChains', 'witnessCapacity', 'capacityDrops', 'duplicateContinuations', 'bestTwoBagPath']);
  requireValue(lookahead.targetHorizonBags === 1 && lookahead.requestedBags === 2 && lookahead.witnessCapacity === 32);
  int(lookahead.capacityDrops); int(lookahead.duplicateContinuations);
  const rawWitnesses = list(label.reachableWholeBagWitnesses, 0, 32);
  requireValue(lookahead.reachableWholeBagChains === rawWitnesses.length && m.geometricWholeBagChains === rawWitnesses.length
    && (refined || m.verificationWork === rawWitnesses.length * 7));
  const byId = new Map<number, {bags: number; rows: number[]; spins: number; parentId: number | null}>();
  for (const raw of rawWitnesses) {
    const w = obj(raw, ['id', 'parentId', 'bags', 'startRows', 'endRows', 'startSpins', 'endSpins', 'deletedRows', 'targets', 'representativeQueue', 'representativeInputs', 'frames', 'trust', 'wholeBag']);
    const id = int(w.id, 1), bags = int(w.bags, 1, 2), parentId = w.parentId === null ? null : int(w.parentId, 1);
    const parent = parentId === null ? undefined : byId.get(parentId);
    requireValue(!byId.has(id) && (parentId === null || parent) && bags === (parent?.bags ?? 0) + 1
      && w.trust === 'input-reachable' && w.wholeBag === true);
    const startRows = parent?.rows ?? board, startSpins = parent?.spins ?? 0;
    requireValue(same(rows(w.startRows), startRows) && w.startSpins === startSpins);
    const endRows = rows(w.endRows), endSpins = int(w.endSpins, startSpins, Math.min(maximum, startSpins + 6));
    int(w.deletedRows, 0, 0xffffffff); targets(w.targets, 7);
    const queue = list(w.representativeQueue, 7, 7).map(v => int(v, 0, 6));
    requireValue(new Set(queue).size === 7);
    const display = list(w.frames, 7, 7);
    for (const [index, path] of list(w.representativeInputs, 7, 7).entries()) {
      const actions = list(path, 1, 256);
      requireValue(actions.every(a => ['left', 'right', 'cw', 'ccw', 'half', 'sonic', 'hard_drop'].includes(String(a)))
        && actions[actions.length - 1] === 'hard_drop' && !actions.slice(0, -1).includes('hard_drop'));
      requireValue((display[index] as Obj).piece === ['I', 'J', 'L', 'O', 'S', 'T', 'Z'][queue[index]], 'Supply and actual placement pieces differ');
    }
    const end = frames(w.frames, startRows, startSpins, (bags - 1) * 7, 7, w.targets);
    requireValue(same(end.rows, endRows) && end.spins === endSpins && end.deleted === w.deletedRows && (bags !== 1 || end.spins <= observed));
    byId.set(id, {bags, rows: endRows, spins: endSpins, parentId});
  }
  const records = [...byId.values()], deepest = records.length ? Math.max(...records.map(w => w.bags)) : 0;
  requireValue(m.productiveGeometricWholeBagChains === records.filter(w => w.spins > 0).length && lookahead.deepestCompletedBag === deepest);
  int(lookahead.deepestAttemptedBag, Math.max(1, deepest), 2);
  for (const [i, raw] of list(m.depths, 2, 2).entries()) {
    const d = obj(raw, ['bags', 'geometricWholeBagChains', 'productiveGeometricWholeBagChains', 'certifiedWholeBagChains', 'productiveCertifiedWholeBagChains', 'maxGeometricWholeBagSpins', 'maxCertifiedSpins']);
    const group = records.filter(w => w.bags === i + 1);
    requireValue(d.bags === i + 1 && d.geometricWholeBagChains === group.length
      && d.productiveGeometricWholeBagChains === group.filter(w => w.spins > 0).length
      && d.certifiedWholeBagChains === 0 && d.productiveCertifiedWholeBagChains === 0 && d.maxCertifiedSpins === null
      && d.maxGeometricWholeBagSpins === (group.length ? Math.max(...group.map(w => w.spins)) : null));
  }
  const prefix = (raw: unknown, firstBag: boolean): number => {
    if (raw === null || raw === undefined) { requireValue(lookahead.deepestReachedPlacements === 0 && (!firstBag || observed === 0), 'Missing actual prefix witness'); return 0; }
    const hasParent = raw && typeof raw === 'object' && Object.hasOwn(raw, 'reachableParentId');
    const r = obj(raw, ['certifiedParentId', 'completedParentBags', 'additionalPlacements', 'observedSpins', 'targets', 'frames', ...(hasParent ? ['reachableParentId'] : [])]);
    const parentId = r.reachableParentId ?? null, parent = parentId === null ? undefined : byId.get(int(parentId, 1));
    requireValue(r.certifiedParentId === null && (parentId === null || parent));
    const bags = int(r.completedParentBags, 0, firstBag ? 0 : 1), placements = int(r.additionalPlacements, 1, 7);
    requireValue(bags === (parent?.bags ?? 0)); targets(r.targets, placements);
    const end = frames(r.frames, parent?.rows ?? board, parent?.spins ?? 0, bags * 7, placements, r.targets);
    requireValue(end.spins === r.observedSpins && end.spins <= maximum && (!firstBag || end.spins === observed));
    return bags * 7 + placements;
  };
  const targetDepth = prefix(label.targetWitness, true), prefixDepth = prefix(label.reachableWitness, false);
  requireValue(lookahead.deepestReachedPlacements === Math.max(deepest * 7, prefixDepth) && targetDepth <= Number(lookahead.deepestReachedPlacements));
  if (lookahead.bestTwoBagPath !== null) {
    const best = obj(lookahead.bestTwoBagPath, ['witnessId', 'firstBagSpins', 'secondBagSpins', 'totalSpins']);
    const last = byId.get(int(best.witnessId, 1)), first = last?.parentId ? byId.get(last.parentId) : null;
    requireValue(last?.bags === 2 && first?.bags === 1 && best.firstBagSpins === first.spins
      && best.secondBagSpins === last.spins - first.spins && best.totalSpins === last.spins, 'Two-bag gains must share one actual chain');
  } else requireValue(!records.some(w => w.bags === 2));
}

const pieceNames = ['I', 'O', 'T', 'S', 'Z', 'J', 'L'];
function refObject(value: unknown): Obj { requireValue(value && typeof value === 'object' && !Array.isArray(value)); return value as Obj; }
function refCanonical(value: unknown): string {
  if (Array.isArray(value)) return '[' + value.map(refCanonical).join(',') + ']';
  if (value && typeof value === 'object') return '{' + Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([k, v]) => JSON.stringify(k) + ':' + refCanonical(v)).join(',') + '}';
  return JSON.stringify(value);
}
const refSame = (a: unknown, b: unknown) => refCanonical(a) === refCanonical(b);

function exact(value: Obj, keys: string[]) {
  obj(value, keys);
}
function localPrefix(raw: unknown, start: number[]) {
  if (raw === null) return { placements: 0, spins: 0 };
  const w = refObject(raw); exact(w, ["placements", "observedSpins", "deletedRows", "targets", "representativeQueue", "representativeInputs", "frames", "trust"]);
  const count = int(w.placements, 1, 7);
  if (w.trust !== "input-reachable" || !Array.isArray(w.representativeQueue) || w.representativeQueue.length !== count
    || new Set(w.representativeQueue.map(x => int(x, 0, 6))).size !== count
    || !Array.isArray(w.representativeInputs) || w.representativeInputs.length !== count) throw new ObservationError(400, "후보의 실제 미노·입력 경로가 필요합니다.");
  for (const [index, path] of w.representativeInputs.entries()) {
    if (!Array.isArray(path) || !path.length || path.length > 256 || path.at(-1) !== "hard_drop" || path.slice(0, -1).includes("hard_drop")
      || path.some(action => !["left", "right", "cw", "ccw", "half", "sonic", "hard_drop"].includes(String(action)))) throw new ObservationError(400, "후보 입력 경로가 잘못되었습니다.");
    if (!Array.isArray(w.frames) || refObject(w.frames[index]).piece !== ["I", "J", "L", "O", "S", "T", "Z"][Number(w.representativeQueue[index])]) throw new ObservationError(400, "후보 공급 순서와 배치가 다릅니다.");
  }
  const result = frames(w.frames, start, 0, 0, count, w.targets);
  if (w.deletedRows !== result.deleted || w.observedSpins !== result.spins) throw new ObservationError(400, "후보의 국소 한 가방 결과가 다릅니다.");
  return { placements: count, spins: result.spins };
}
function searchObservation(raw: unknown, start: number[], quota: number, attemptsLimit: number) {
  const o = refObject(raw);
  exact(o, ["allocatedWork", "workUsed", "generationWork", "verificationWork", "attempts", "targets", "targetWitness", "deepestWitness", "deepestReachedPlacements", "completedWholeBagCandidates", "distinctWholeBagCandidates", "censored", "stopReason"]);
  const used = int(o.workUsed, 0, quota), generation = int(o.generationWork, 0, used), verification = int(o.verificationWork, 0, used);
  const attempts = int(o.attempts, 0, attemptsLimit), targets = refObject(o.targets); exact(targets, ["foundSpin", "observedSpins"]);
  const spins = int(targets.observedSpins, 0, 6), reached = int(o.deepestReachedPlacements, 0, 7);
  const completed = int(o.completedWholeBagCandidates, 0, Number.MAX_SAFE_INTEGER);
  int(o.distinctWholeBagCandidates, 0, completed);
  if (o.allocatedWork !== quota || used !== generation + verification || targets.foundSpin !== (spins > 0)
    || o.censored !== true || typeof o.stopReason !== "string" || !o.stopReason || o.stopReason.length > 100
    || (completed > 0 && reached !== 7) || (used > 0 && attempts === 0)) throw new ObservationError(400, "집중 후보의 작업 예산·표지가 다릅니다.");
  const target = localPrefix(o.targetWitness, start), deepest = localPrefix(o.deepestWitness, start);
  if (target.spins !== spins || target.placements > reached || deepest.placements !== reached || deepest.spins > spins
    || (reached > 0 && target.placements === 0)) throw new ObservationError(400, "집중 후보의 최선·최심 경로가 표지와 다릅니다.");
  return { used, generation, verification, attempts, spins, reached, completed, prefixVerification: target.placements + deepest.placements };
}

function validateRefinement(label: Obj): Obj {
  const r = refObject(label.refinement), m = refObject(label.measured), config = refObject(label.config);
  exact(r, ["schemaVersion", "policy", "selectionSeed", "selectionScoreMin", "selectionScoreMax", "selectionBandCount", "candidatePoolCapacity", "candidatePoolSize", "candidatePoolDrops", "candidateDuplicates", "armLimit", "armQuota", "dependencyWorkBudget", "dependencyWorkUsed", "proofCapacity", "proofCapacityDrops", "root", "candidates", "arms"]);
  const workBudget = int(config.workBudget, 1024, 4194304), maxNodes = int(config.maxNodes, 4, 4096), attemptsLimit = int(config.attemptsPerNode, 1, 64);
  const dependencyBudget = Math.min(4096, Math.floor(workBudget / 32));
  const quota = Math.floor((workBudget - dependencyBudget) / 4);
  if (r.schemaVersion !== 1 || r.policy !== "uniform-score-band-three-arm-equal-quota-v1" || r.armLimit !== 3 || r.armQuota !== quota
    || r.dependencyWorkBudget !== dependencyBudget || r.dependencyWorkUsed !== m.dependencyWork
    || r.proofCapacity !== Math.min(maxNodes, 32)) throw new ObservationError(400, "집중 관측의 버전·고정 할당량이 다릅니다.");
  int(r.selectionSeed, 0, 4294967295); int(r.dependencyWorkUsed, 0, dependencyBudget);
  const poolCapacity = int(r.candidatePoolCapacity, 1, 4096), poolSize = int(r.candidatePoolSize, 0, poolCapacity);
  if (typeof r.selectionScoreMin !== "number" || !Number.isFinite(r.selectionScoreMin) || typeof r.selectionScoreMax !== "number"
    || !Number.isFinite(r.selectionScoreMax) || r.selectionScoreMin > r.selectionScoreMax
    || r.selectionBandCount !== (r.selectionScoreMin === r.selectionScoreMax ? 1 : 4)
    || (!poolSize && (r.selectionScoreMin !== 0 || r.selectionScoreMax !== 0))) throw new ObservationError(400, "후보 점수 구간의 범위가 다릅니다.");
  for (const key of ["candidatePoolDrops", "candidateDuplicates", "proofCapacityDrops"]) int(r[key], 0, Number.MAX_SAFE_INTEGER);
  const witnesses = label.reachableWholeBagWitnesses as Obj[], byId = new Map(witnesses.map(w => [Number(w.id), w]));
  if (!Array.isArray(r.candidates) || r.candidates.length !== witnesses.length || !Array.isArray(r.arms) || r.arms.length > Math.min(3, poolSize)) throw new ObservationError(400, "집중 후보·하위 관측 개수가 다릅니다.");
  const root = searchObservation(r.root, rows(refObject(label.input).rows), quota, attemptsLimit);
  if (!refSame(refObject(r.root).targets, label.targets)
    || root.verification !== witnesses.filter(w => w.parentId === null).length * 7 + root.prefixVerification) throw new ObservationError(400, "루트의 한 가방 표지·입력 검증량이 다릅니다.");
  const phaseBest = (parentId: number | null, startSpins: number, best: number) => {
    if (witnesses.some(w => w.parentId === parentId && Number(w.endSpins) - startSpins > best)) throw new ObservationError(400, "관측된 한 가방 최선보다 큰 실제 후속 경로가 남아 있습니다.");
    if (label.reachableWitness != null) {
      const prefix = refObject(label.reachableWitness);
      if ((prefix.reachableParentId ?? null) === parentId && Number(prefix.observedSpins) - startSpins > best) throw new ObservationError(400, "최심 경로의 실제 스핀이 해당 한 가방 최선을 넘습니다.");
    }
  };
  phaseBest(null, 0, root.spins);
  let used = root.used, generation = root.generation, verification = root.verification;
  const selected = new Set<number>();
  for (const raw of r.arms) {
    const arm = refObject(raw); exact(arm, ["candidateWitnessId", "selectionStratum", "selectionStrata", "selectionPoolSize", "selectionIndex", "observation"]);
    const id = int(arm.candidateWitnessId, 1, Number.MAX_SAFE_INTEGER), candidate = byId.get(id);
    if (!candidate || candidate.bags !== 1 || selected.has(id) || !(candidate.endRows as number[]).some(Boolean)) throw new ObservationError(400, "하위 관측은 서로 다른 실제 첫 가방 후보에서 시작해야 합니다.");
    const strata = int(arm.selectionStrata, 1, 4); int(arm.selectionStratum, 0, strata - 1);
    if (strata !== r.selectionBandCount) throw new ObservationError(400, "후보 선택 구간 개수가 다릅니다.");
    const size = int(arm.selectionPoolSize, 1, poolSize); int(arm.selectionIndex, 0, size - 1);
    const observed = searchObservation(arm.observation, rows(candidate.endRows), quota, attemptsLimit);
    if (observed.used === 0 || observed.verification !== witnesses.filter(w => w.parentId === id).length * 7 + observed.prefixVerification) throw new ObservationError(400, "후보의 실제 진행·입력 검증량이 다릅니다.");
    phaseBest(id, Number(candidate.endSpins), observed.spins);
    selected.add(id); used += observed.used; generation += observed.generation; verification += observed.verification;
  }
  if (witnesses.some(w => w.bags === 2 && !selected.has(Number(w.parentId)))) throw new ObservationError(400, "두 번째 가방 증거의 하위 관측이 없습니다.");
  let dependencyUsed = 0; const seen = new Set<number>();
  for (const raw of r.candidates) {
    const candidate = refObject(raw); exact(candidate, ["witnessId", "structuralQuality", "dependency"]);
    const id = int(candidate.witnessId, 1, Number.MAX_SAFE_INTEGER), witness = byId.get(id);
    if (!witness || seen.has(id) || typeof candidate.structuralQuality !== "number" || !Number.isFinite(candidate.structuralQuality)) throw new ObservationError(400, "후보의 현재 필드 진단이 다릅니다.");
    seen.add(id);
    if (candidate.dependency !== null) {
      if (!selected.has(id) || witness.bags !== 1) throw new ObservationError(400, "의존성 진단은 선택한 첫 가방 후보에만 붙습니다.");
      dependencyUsed += componentDependency(candidate.dependency, witness);
    }
  }
  if (dependencyUsed !== r.dependencyWorkUsed || used + dependencyUsed !== m.workUsed || generation !== m.generationWork || verification !== m.verificationWork) throw new ObservationError(400, "집중 관측의 총 작업량이 다릅니다.");
  return label;
}

function componentDependency(raw: unknown, witness: Obj) {
  const d = refObject(raw); exact(d, ["schemaVersion", "policy", "connectivity", "scope", "maxOrdersPerComponent", "workUsed", "totalOrders", "testedOrders", "successfulOrders", "components"]);
  if (d.schemaVersion !== 1 || d.policy !== "touching-components-origin-only-v1" || d.connectivity !== "orthogonal-logical-target-cells"
    || d.maxOrdersPerComponent !== 64 || typeof d.scope !== "string" || !d.scope || d.scope.length > 1000 || !Array.isArray(d.components) || d.components.length < 1 || d.components.length > 7) throw new ObservationError(400, "연결 성분 의존성 근사의 정의가 다릅니다.");
  const targets = (witness.targets as unknown[]).map(refObject), adjacent = targets.map(() => new Set<number>());
  targets.forEach((a, i) => targets.forEach((b, j) => { if (i < j && (a.cells as number[][]).some(x => (b.cells as number[][]).some(y => Math.abs(x[0] - y[0]) + Math.abs(x[1] - y[1]) === 1))) { adjacent[i].add(j); adjacent[j].add(i); } }));
  const remaining = new Set(targets.map((_, i) => i)), expected: string[] = [];
  while (remaining.size) { const stack = [remaining.values().next().value!], group: string[] = []; remaining.delete(stack[0]);
    while (stack.length) { const at = stack.pop()!; group.push(String(targets[at].piece)); for (const next of adjacent[at]) if (remaining.delete(next)) stack.push(next); }
    expected.push(group.sort().join("")); }
  const actual: string[] = []; let work = 0, total = 0, tested = 0, success = 0;
  for (const raw of d.components) {
    const c = refObject(raw); exact(c, ["pieces", "totalOrders", "testedOrders", "successfulOrders", "interruptedOrders", "workUsed", "exhaustive"]);
    if (!Array.isArray(c.pieces) || !c.pieces.length || c.pieces.some(p => !pieceNames.includes(String(p))) || new Set(c.pieces).size !== c.pieces.length) throw new ObservationError(400, "연결 성분의 미노가 다릅니다.");
    const factorial = c.pieces.reduce<number>((n, _, index) => n * (index + 1), 1);
    const n = int(c.testedOrders, 0, Math.min(64, factorial)), s = int(c.successfulOrders, 0, n);
    int(c.interruptedOrders, 0, 1); const w = int(c.workUsed, 0, 4096);
    if (c.totalOrders !== factorial || c.exhaustive !== (n === factorial)) throw new ObservationError(400, "성분별 순열 관측 범위가 다릅니다.");
    actual.push(c.pieces.map(String).sort().join("")); work += w; total += factorial; tested += n; success += s;
  }
  if (!refSame(expected.sort(), actual.sort()) || d.workUsed !== work || d.totalOrders !== total || d.testedOrders !== tested || d.successfulOrders !== success) throw new ObservationError(400, "성분별 합계는 독립 진단의 합이어야 합니다.");
  return work;
}

function validateOriginSelection(raw: unknown) {
  const s = refObject(raw); exact(s, ["policy", "seed", "draw", "stratum", "strata", "poolSize", "evaluator", "scoreMin", "scoreMax"]);
  const size = int(s.poolSize, 1, 128), strata = int(s.strata, 1, Math.min(4, size));
  int(s.stratum, 0, strata - 1); int(s.seed, 0, 4294967295); int(s.draw, 0, 4294967295);
  if (s.policy !== "uniform-score-stratum-v1" || !["heuristic", "cnn", "unknown"].includes(String(s.evaluator))
    || (s.scoreMin === null) !== (s.scoreMax === null) || (s.evaluator === "unknown") !== (s.scoreMin === null) || (s.scoreMin !== null && (typeof s.scoreMin !== "number" || !Number.isFinite(s.scoreMin)
      || typeof s.scoreMax !== "number" || !Number.isFinite(s.scoreMax) || s.scoreMin > s.scoreMax))) throw new ObservationError(400, "시작 필드의 무작위 구간 선택 출처가 다릅니다.");
}

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
  const refined = (o.label as Obj)?.schemaVersion === 4;
  const nextBag = refined || (o.label as Obj)?.schemaVersion === 3;
  const hasTargetWitness = nextBag && Object.hasOwn(o.label as Obj, 'targetWitness');
  const label = obj(o.label, ['schemaVersion', 'task', 'policy', 'input', 'config', 'targets', 'measured', 'certifiedWitnesses', 'reachableWitness',
    ...(nextBag ? ['targetHorizonBags', 'reachableWholeBagWitnesses', ...(hasTargetWitness ? ['targetWitness'] : [])] : []), ...(refined ? ['refinement'] : [])]);
  requireValue(nextBag ? label.task === (refined ? 'naga-next-bag-spin-v4' : 'naga-next-bag-spin-v3') && label.policy === (refined ? 'concentrated-next-bag-spin-v4' : 'bounded-next-bag-spin-v3') && label.targetHorizonBags === 1
    : label.schemaVersion === 2 && label.task === 'naga-spin-observations-v2' && label.policy === 'neyra-fixed-bag-continuation-v2');
  const input = obj(label.input, ['id', 'rows', 'source']), board = rows(input.rows);
  requireValue(board.some(Boolean), 'Empty first-bag fields are excluded');
  const source = obj(input.source, ['kind', 'family', 'reference', 'bagBoundary', 'runId', 'rootCandidateId', 'rootBagBoundary', 'rootSeed', 'buildVersion', 'rules', 'trust', ...(refined && Object.hasOwn(input.source as Obj, 'selection') ? ['selection'] : [])]);
  if (source.selection !== undefined) validateOriginSelection(source.selection);
  const bagBoundary = int(source.bagBoundary, 2, 512), rootBagBoundary = int(source.rootBagBoundary, 2, bagBoundary);
  int(source.rootSeed, 1, 0xffffffff); str(source.buildVersion, 96);
  requireValue(source.kind === 'search' && source.family === `search:${runId}` && source.runId === runId && source.rootCandidateId === rootCandidateId && source.rules === RULES && source.trust === 'client-reported');
  const link = nextBag ? 'reachable' : 'certificate';
  requireValue(input.id === (parentObservationId === null ? `${runId}:${rootCandidateId}` : `${parentObservationId}:${link}-${parentWitnessId}`));
  requireValue(source.reference === (parentObservationId === null ? `neyra:run:${runId}/candidate:${rootCandidateId}` : `neyra:observation:${parentObservationId}/${link}:${parentWitnessId}`));
  requireValue(parentObservationId !== null || bagBoundary === rootBagBoundary);
  const config = obj(label.config, ['horizonBags', 'workBudget', 'beamWidth', 'profile', 'seed', 'maxNodes', 'attemptsPerNode', 'sourceRevision', ...(nextBag ? ['observationVersion'] : [])]);
  const fixed: Obj = {horizonBags: 2, workBudget: 131072, beamWidth: 48, profile: 0, seed: 1, maxNodes: 96, attemptsPerNode: refined ? 16 : 6,
    sourceRevision: refined ? REFINED_REVISION : nextBag ? NEXT_BAG_REVISION : REVISION, ...(nextBag ? {observationVersion: refined ? 4 : 3} : {})};
  requireValue(Object.keys(fixed).every(key => config[key] === fixed[key]), 'Collector policy differs from fixed training policy');
  if (nextBag) {
    validateNextBag(label, board, refined);
    if (refined) validateRefinement(label);
    return {schemaVersion: 1, kind: 'focused-spin-observation', id: o.id as string, runId, rootCandidateId, sequence,
      parentObservationId, parentWitnessId, label: {...label, input: {...input, source: {...source, bagBoundary, rootBagBoundary}}}};
  }
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
