/** Explicit-consent traces and fixed-policy observations, all client-reported. */
import { cleanObservation, ObservationError } from './observation.ts';
const ALLOWED_ORIGIN = 'https://daejunnom.github.io';
const RULES = 'neyra-srsplus-sonicdrop-clearonlyhold-v1';
const MAX_BODY = 576 * 1024;
const MAX_TRACE_BODY = 32 * 1024;
type ObjectValue = Record<string, unknown>;
class Rejection extends Error {
  constructor(public status: number, message: string) { super(message); }
}

function object(value: unknown): ObjectValue {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Rejection(400, 'Invalid object');
  return value as ObjectValue;
}
function text(value: unknown, limit = 240): string {
  if (typeof value !== 'string' || !value.length || value.length > limit || /[\x00-\x1f]/.test(value)) {
    throw new Rejection(400, 'Invalid provenance');
  }
  return value;
}
function number(value: unknown, min: number, max: number, integer = false): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) {
    throw new Rejection(400, 'Invalid numeric field');
  }
  return value;
}
function bool(value: unknown): boolean {
  if (typeof value !== 'boolean') throw new Rejection(400, 'Invalid flag');
  return value;
}
function onlyKeys(value: ObjectValue, allowed: string[]) {
  if (Object.keys(value).some(key => !allowed.includes(key))) throw new Rejection(400, 'Unexpected input field');
}
function board(value: unknown): number[] {
  if (!Array.isArray(value) || value.length !== 24) throw new Rejection(400, 'Invalid trace board');
  return value.map(row => number(row, 0, 1022, true));
}
function nullableNumber(value: unknown, min: number, max: number, integer = false): number | null {
  return value === null ? null : number(value, min, max, integer);
}
function mass(rows: number[]): number {
  return rows.reduce((total, row) => { while (row) { total++; row &= row - 1; } return total; }, 0);
}
function equalRows(a: number[], b: number[]): boolean { return a.every((row, i) => row === b[i]); }
function cleanTrace(value: unknown, endRows: number[], bagDepth: number, probability: number,
  modelMode: 'heuristic' | 'cnn', modelId?: string, weightsSha256?: string) {
  const t = object(value);
  onlyKeys(t, ['schemaVersion', 'kind', 'prefixOmittedBags', 'start', 'steps', 'evaluation', 'search', 'outcome']);
  if (t.schemaVersion !== 1 || t.kind !== 'retained-path') throw new Rejection(400, 'Invalid trace version');
  const omitted = number(t.prefixOmittedBags, 0, 99, true);
  const start = object(t.start); onlyKeys(start, ['rows', 'spins', 'placed']);
  const first = {rows: board(start.rows), spins: number(start.spins, 0, 40, true), placed: number(start.placed, 0, 700, true)};
  if (first.placed !== omitted * 7 || first.spins > first.placed ||
      (omitted === 0 && (first.spins !== 0 || first.rows.some(Boolean)))) throw new Rejection(400, 'Invalid trace origin');
  if (!Array.isArray(t.steps) || ![7, 14].includes(t.steps.length)) throw new Rejection(400, 'Invalid trace window');
  let previous = first;
  const steps = t.steps.map(raw => {
    const step = object(raw); onlyKeys(step, ['piece', 'rows', 'spins', 'placed', 'lines', 'spin']);
    const piece = text(step.piece, 1); if (!'IOTSZJL'.includes(piece)) throw new Rejection(400, 'Invalid trace piece');
    const current = {piece, rows: board(step.rows), spins: number(step.spins, 0, 40, true),
      placed: number(step.placed, 1, 700, true), lines: number(step.lines, 0, 1, true), spin: bool(step.spin)};
    if (current.placed !== previous.placed + 1 || current.spins !== previous.spins + current.lines ||
        (current.lines === 1 && (!current.spin || piece === 'O')) ||
        mass(current.rows) !== mass(previous.rows) + 4 - 10 * current.lines) throw new Rejection(400, 'Inconsistent trace transition');
    previous = current; return current;
  });
  if (previous.placed !== bagDepth * 7 || !equalRows(previous.rows, endRows) ||
      omitted !== Math.max(0, bagDepth - 2)) throw new Rejection(400, 'Trace endpoint mismatch');
  const o = object(t.outcome); onlyKeys(o, ['spins', 'placed', 'bagCount', 'countedProbability']);
  if (o.spins !== previous.spins || o.placed !== previous.placed || o.bagCount !== bagDepth ||
      o.countedProbability !== probability) throw new Rejection(400, 'Trace outcome mismatch');
  const e = object(t.evaluation);
  onlyKeys(e, ['kind', 'rawQuality', 'rankingScore', 'forecastSpins', 'forecastSpinsByBag', 'forecastPlacements', 'modelId', 'weightsSha256']);
  if (e.kind !== modelMode || !Array.isArray(e.forecastSpinsByBag) || e.forecastSpinsByBag.length > 2 ||
    (modelMode === 'cnn' ? e.modelId !== modelId || e.weightsSha256 !== weightsSha256
      : e.modelId !== undefined || e.weightsSha256 !== undefined)) throw new Rejection(400, 'Invalid trace evaluator');
  const evaluation = {kind: modelMode, ...(modelMode === 'cnn' ? {modelId, weightsSha256} : {}),
    rawQuality: nullableNumber(e.rawQuality, -1e12, 1e12),
    rankingScore: nullableNumber(e.rankingScore, -1e12, 1e12), forecastSpins: nullableNumber(e.forecastSpins, 0, 12, true),
    forecastSpinsByBag: e.forecastSpinsByBag.map(v => number(v, 0, 6, true)),
    forecastPlacements: nullableNumber(e.forecastPlacements, 0, 14, true)};
  const s = object(t.search); onlyKeys(s, ['scope', 'budgetStart', 'expanded', 'candidates', 'verifiedCandidates', 'pruned']);
  if (s.scope !== 'worker-total-at-record') throw new Rejection(400, 'Invalid trace work scope');
  const search = {scope: s.scope, budgetStart: number(s.budgetStart, 0, 700, true),
    expanded: nullableNumber(s.expanded, 0, Number.MAX_SAFE_INTEGER, true),
    candidates: nullableNumber(s.candidates, 0, Number.MAX_SAFE_INTEGER, true),
    verifiedCandidates: nullableNumber(s.verifiedCandidates, 0, Number.MAX_SAFE_INTEGER, true),
    pruned: nullableNumber(s.pruned, 0, Number.MAX_SAFE_INTEGER, true)};
  return {schemaVersion: 1, kind: 'retained-path', prefixOmittedBags: omitted, start: first, steps, evaluation, search,
    outcome: {spins: previous.spins, placed: previous.placed, bagCount: bagDepth, countedProbability: probability}};
}
// jsonb text has separator spaces. Check that size before submitting the RPC,
// so a valid HTTP batch cannot fail later merely because storage adds spaces.
function storageJson(value: unknown): string {
  if (Array.isArray(value)) return '[' + value.map(storageJson).join(', ') + ']';
  if (value && typeof value === 'object') return '{' + Object.entries(value).map(([key, v]) => JSON.stringify(key) + ': ' + storageJson(v)).join(', ') + '}';
  return JSON.stringify(value) ?? 'null';
}
function cleanInput(value: unknown) {
  const field = object(value);
  onlyKeys(field, ['schemaVersion', 'id', 'rows', 'source', 'trace']);
  if (field.schemaVersion !== 1) throw new Rejection(400, 'Invalid input version');
  if (!Array.isArray(field.rows) || field.rows.length !== 24 || !field.rows.some(row => row !== 0)) {
    throw new Rejection(400, 'A nonempty completed board is required');
  }
  const rows = field.rows.map(row => number(row, 0, 1022, true));
  const source = object(field.source);
  onlyKeys(source, ['kind', 'family', 'reference', 'runId', 'candidateId', 'bagBoundary', 'probability',
    'firstBagProbability', 'ignoreFirstBagProbability', 'verified', 'qualified', 'seed', 'runSeed',
    'modelMode', 'evaluatorModelId', 'evaluatorWeightsSha256', 'buildVersion', 'rules', 'searchConfig']);
  const runId = text(source.runId, 80);
  if (!/^[a-zA-Z0-9-]{16,80}$/.test(runId) || source.family !== `search:${runId}` ||
      source.kind !== 'search' || !['heuristic', 'cnn'].includes(String(source.modelMode)) || source.rules !== RULES ||
      source.verified !== true || source.qualified !== true) throw new Rejection(400, 'Invalid search provenance');
  const modelMode = source.modelMode as 'heuristic' | 'cnn';
  const modelId = modelMode === 'cnn' ? text(source.evaluatorModelId, 200) : undefined;
  const weightsSha256 = modelMode === 'cnn' ? text(source.evaluatorWeightsSha256, 64) : undefined;
  if (modelMode === 'cnn' ? !/^[0-9a-f]{64}$/.test(weightsSha256!) || number(source.bagBoundary, 1, 128, true) < 2
    : source.evaluatorModelId !== undefined || source.evaluatorWeightsSha256 !== undefined) {
    throw new Rejection(400, 'Invalid evaluator provenance');
  }
  const candidateId = text(source.candidateId, 100);
  if (!/^w\d+:plan-\d+$/.test(candidateId) || field.id !== `${runId}:${candidateId}` ||
      source.reference !== `neyra:run:${runId}/candidate:${candidateId}`) throw new Rejection(400, 'Invalid record identity');
  const c = object(source.searchConfig);
  onlyKeys(c, ['minSuccessPercent', 'bagMinSuccessPercent', 'firstBagMinSuccessPercent',
    'riskMinSpinsPerBag', 'pieceGoal', 'maxPieces', 'beamWidth']);
  if (![16, 32, 64, 128].includes(c.beamWidth as number)) throw new Rejection(400, 'Invalid beam width');
  const result = {schemaVersion: 1, id: text(field.id), rows, source: {
    kind: 'search', family: source.family, reference: source.reference, runId, candidateId,
    bagBoundary: number(source.bagBoundary, 1, 128, true),
    probability: number(source.probability, 0, 1),
    firstBagProbability: source.firstBagProbability === null ? null : number(source.firstBagProbability, 0, 1),
    ignoreFirstBagProbability: bool(source.ignoreFirstBagProbability), verified: true, qualified: true,
    seed: number(source.seed, 0, 0xffffffff, true), runSeed: number(source.runSeed, 0, 0xffffffff, true),
    modelMode, ...(modelMode === 'cnn' ? {evaluatorModelId: modelId, evaluatorWeightsSha256: weightsSha256} : {}),
    buildVersion: text(source.buildVersion, 96), rules: RULES,
    trust: 'unverified-contribution',
    searchConfig: {
      minSuccessPercent: number(c.minSuccessPercent, 0, 100),
      bagMinSuccessPercent: number(c.bagMinSuccessPercent, 0, 100),
      firstBagMinSuccessPercent: number(c.firstBagMinSuccessPercent, 0, 100),
      riskMinSpinsPerBag: number(c.riskMinSpinsPerBag, 0, 6, true),
      pieceGoal: number(c.pieceGoal, 1, 1000, true), maxPieces: number(c.maxPieces, 1, 700, true),
      beamWidth: number(c.beamWidth, 16, 128, true)
    }
  }, trace: cleanTrace(field.trace, rows, number(source.bagBoundary, 1, 100, true), number(source.probability, 0, 1),
    modelMode, modelId, weightsSha256)};
  if (new TextEncoder().encode(JSON.stringify(result)).length > 6144 ||
      new TextEncoder().encode(storageJson(result)).length > 8192) throw new Rejection(413, 'Contribution exceeds storage limit');
  return result;
}
async function digest(value: string): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value))),
    byte => byte.toString(16).padStart(2, '0')).join('');
}
function canonicalJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value !== null && typeof value === 'object') return `{${Object.keys(value).sort()
    .map(key => `${JSON.stringify(key)}:${canonicalJson((value as ObjectValue)[key])}`).join(',')}}`;
  return JSON.stringify(value) ?? 'null';
}
async function readBody(request: Request): Promise<ObjectValue> {
  const declared = Number(request.headers.get('content-length'));
  if (declared > MAX_BODY) throw new Rejection(413, 'Batch exceeds limit');
  if (!request.headers.get('content-type')?.toLowerCase().startsWith('application/json')) {
    throw new Rejection(415, 'JSON required');
  }
  const reader = request.body?.getReader();
  if (!reader) throw new Rejection(400, 'Missing body');
  const chunks: Uint8Array[] = []; let bytes = 0;
  while (true) {
    const next = await reader.read();
    if (next.done) break;
    bytes += next.value.length;
    if (bytes > MAX_BODY) { await reader.cancel(); throw new Rejection(413, 'Batch exceeds limit'); }
    chunks.push(next.value);
  }
  const body = new Uint8Array(bytes); let offset = 0;
  for (const chunk of chunks) { body.set(chunk, offset); offset += chunk.length; }
  try { return object(JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(body))); }
  catch (error) { if (error instanceof Rejection) throw error; throw new Rejection(400, 'Invalid JSON'); }
}
function database() {
  const url = Deno.env.get('SUPABASE_URL');
  const secret = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY') ||
    JSON.parse(Deno.env.get('SUPABASE_SECRET_KEYS') || '{}').default;
  if (!url || !secret) throw new Rejection(503, 'Storage unavailable');
  return {url, headers: {apikey: secret, ...(secret.startsWith('ey') ? {Authorization: `Bearer ${secret}`} : {}),
    'Content-Type': 'application/json'}};
}
async function rest(path: string, init: RequestInit = {}): Promise<Response> {
  const db = database();
  const response = await fetch(`${db.url}/rest/v1/${path}`, {...init,
    headers: {...db.headers, ...(init.headers as Record<string, string> || {})}, signal: AbortSignal.timeout(12000)});
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    if (error.message === 'contribution limit' || error.message === 'contribution queue full') {
      throw new Rejection(429, 'Contribution capacity reached');
    }
    if (error.message === 'observation identity conflict' || error.message === 'unknown observation receipt') {
      throw new Rejection(409, error.message);
    }
    if (['invalid observation', 'invalid batch', 'invalid run', 'invalid observation receipt'].includes(error.message)) {
      throw new Rejection(400, 'Invalid contribution record');
    }
    throw new Rejection(503, 'Storage unavailable');
  }
  return response;
}
async function requireConsumer(request: Request) {
  const token = request.headers.get('x-neyra-admin');
  if (!token || token.length < 32 || token.length > 200) throw new Rejection(403, 'Consumer authorization required');
  const settings = await (await rest('neyra_contribution_settings?name=eq.studio-consumer&select=token_sha256&limit=1')).json();
  const actual = await digest(token); const expected = settings[0]?.token_sha256;
  if (typeof expected !== 'string' || expected.length !== actual.length) throw new Rejection(403, 'Invalid consumer');
  let difference = 0;
  for (let i = 0; i < actual.length; i++) difference |= actual.charCodeAt(i) ^ expected.charCodeAt(i);
  if (difference) throw new Rejection(403, 'Invalid consumer');
}

Deno.serve(async (request: Request) => {
  const origin = request.headers.get('origin');
  const cors: Record<string, string> = {'Vary': 'Origin', 'Cache-Control': 'no-store',
    'Access-Control-Allow-Methods': 'POST, OPTIONS',
    'Access-Control-Allow-Headers': 'authorization, apikey, content-type',
    'Access-Control-Expose-Headers': 'Retry-After'};
  if (origin === ALLOWED_ORIGIN) cors['Access-Control-Allow-Origin'] = ALLOWED_ORIGIN;
  const respond = (body: unknown, status = 200) => new Response(JSON.stringify(body), {
    status, headers: {...cors, 'Content-Type': 'application/json', ...(status === 429 ? {'Retry-After': '60'} : {})}
  });
  try {
    if (origin && origin !== ALLOWED_ORIGIN) throw new Rejection(403, 'Origin not allowed');
    if (request.method === 'OPTIONS') return new Response(null, {status: 204, headers: cors});
    if (request.method !== 'POST') throw new Rejection(405, 'POST required');
    const body = await readBody(request);
    if (body.action !== undefined) {
      await requireConsumer(request);
      if (body.action === 'pull') {
        const limit = number(body.limit ?? 16, 1, 32, true);
        const rows = await (await rest(`neyra_contribution_queue?acknowledged_at=is.null&select=board_key,bag_depth,input&order=bag_depth.desc,created_at.asc&limit=${limit}`)).json();
        return respond({fields: rows.map((row: {board_key: string; bag_depth: number; input: unknown}) =>
          ({queueId: `${row.board_key}:${row.bag_depth}`, input: row.input}))});
      }
      if (body.action === 'ack') {
        if (!Array.isArray(body.ids) || !body.ids.length || body.ids.length > 32 ||
            body.ids.some(id => typeof id !== 'string' || !/^[0-9a-f]{64}:[1-9]\d{0,2}$/.test(id) ||
              Number(id.split(':')[1]) > 128)) throw new Rejection(400, 'Invalid acknowledgement');
        const ids = [...new Set(body.ids as string[])];
        const receipts = ids.map(id => { const [key, depth] = id.split(':');
          return `and(board_key.eq.${key},bag_depth.eq.${depth})`; });
        const response = await rest(`neyra_contribution_queue?or=(${receipts.join(',')})&acknowledged_at=is.null`,
          {method: 'PATCH', headers: {'Prefer': 'return=representation'}, body: JSON.stringify({acknowledged_at: new Date().toISOString()})});
        // A repeat ACK is successful even when the original PATCH response was
        // lost. Exact depth predicates cannot retire a newer queued revision.
        return respond({acknowledged: (await response.json()).length, acknowledgedIds: ids});
      }
      if (body.action === 'stats') {
        const [response, observations] = await Promise.all([
          rest('neyra_contribution_queue?acknowledged_at=is.null&select=board_key', {method: 'HEAD', headers: {'Prefer': 'count=exact'}}),
          rest('neyra_observation_queue?select=queue_key', {method: 'HEAD', headers: {'Prefer': 'count=exact'}})
        ]);
        return respond({pending: Number(response.headers.get('content-range')?.split('/')[1] ?? 0),
          pendingObservations: Number(observations.headers.get('content-range')?.split('/')[1] ?? 0)});
      }
      if (body.action === 'pull-observations') {
        const limit = number(body.limit ?? 1, 1, 4, true);
        const records = await (await rest(`neyra_observation_queue?select=queue_key,observation&order=created_at.asc,sequence.asc,queue_key.asc&limit=${limit}`)).json();
        return respond({observations: await Promise.all(records.map(async (row: {queue_key: string; observation: unknown}) =>
          ({queueId: row.queue_key, observation: row.observation,
            payloadSha256: await digest(canonicalJson(row.observation))})))});
      }
      if (body.action === 'ack-observations') {
        if (!Array.isArray(body.ids) || !body.ids.length || body.ids.length > 32 ||
          body.ids.some(id => typeof id !== 'string' || !/^[0-9a-f]{64}$/.test(id))) throw new Rejection(400, 'Invalid observation acknowledgement');
        const ids = [...new Set(body.ids as string[])];
        const filter = `queue_key=in.(${ids.join(',')})`;
        const [queued, recorded] = await Promise.all([
          rest(`neyra_observation_queue?${filter}&select=queue_key,observation`).then(r => r.json()),
          rest(`neyra_observation_receipts?${filter}&select=queue_key,payload_sha256`).then(r => r.json())
        ]);
        const hashes = new Map<string, string>(recorded.map((row: {queue_key: string; payload_sha256: string}) =>
          [row.queue_key, row.payload_sha256]));
        // Legacy queued observations acquire the same immutable receipt as new
        // admissions. Raw data is only removed in the receipt transaction.
        for (const row of queued) hashes.set(row.queue_key, await digest(canonicalJson(row.observation)));
        if (ids.some(id => !hashes.has(id))) throw new Rejection(409, 'Unknown observation receipt');
        const response = await rest('rpc/neyra_ack_observations', {method: 'POST', body: JSON.stringify({
          p_receipts: ids.map(queue_key => ({queue_key, payload_sha256: hashes.get(queue_key)}))
        })});
        const result = await response.json();
        return respond({...result, receipts: ids.map(queueId => ({queueId, state: 'imported'}))});
      }
      throw new Rejection(400, 'Unknown consumer action');
    }
    const consent = object(body.consent); onlyKeys(consent, ['version', 'scope', 'decision']);
    if (consent.version !== 2 || consent.scope !== 'verified-search-process' || consent.decision !== 'allow') {
      throw new Rejection(400, 'Explicit current contribution consent required');
    }
    if (body.schemaVersion === 4) {
      onlyKeys(body, ['schemaVersion', 'consent', 'observations']);
      if (!Array.isArray(body.observations) || body.observations.length !== 1) throw new Rejection(400, 'One focused observation is required');
      const observation = cleanObservation(body.observations[0]);
      const queueId = await digest(observation.id);
      const payloadSha256 = await digest(canonicalJson(observation));
      const response = await rest('rpc/neyra_admit_observation_v2', {method: 'POST', body: JSON.stringify({
        p_key: queueId, p_observation: observation, p_payload_sha256: payloadSha256
      })});
      const result = await response.json();
      return respond({...result, receipt: {kind: 'focused-observation', id: observation.id,
        queueId, payloadSha256, state: result.state}});
    }
    onlyKeys(body, ['schemaVersion', 'consent', 'fields']);
    if (body.schemaVersion !== 3 || !Array.isArray(body.fields) || !body.fields.length || body.fields.length > 8) {
      throw new Rejection(400, 'Invalid contribution batch');
    }
    const records = await Promise.all(body.fields.map(async value => {
      const input = cleanInput(value);
      // Retain distinct run/path observations, including known geometry. D1
      // separately deduplicates occupancy; public traces keep their identity.
      return {board_key: await digest(input.id + ':' + input.rows.join(',')), bag_depth: input.source.bagBoundary, input};
    }));
    if (new TextEncoder().encode(storageJson(records)).length > MAX_TRACE_BODY) throw new Rejection(413, 'Batch exceeds storage limit');
    const response = await rest('rpc/neyra_admit_contributions', {method: 'POST', body: JSON.stringify({p_records: records})});
    return respond({...await response.json(), receipts: records.map(record =>
      ({id: record.input.id, bagBoundary: record.bag_depth}))});
  } catch (error) {
    const rejected = error instanceof Rejection || error instanceof ObservationError;
    return respond({error: rejected ? error.message : 'Service temporarily unavailable'}, rejected ? error.status : 503);
  }
});
