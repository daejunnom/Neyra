/** Bounded public board-input ingress. Training data is remeasured privately. */
const ALLOWED_ORIGIN = 'https://daejunnom.github.io';
const RULES = 'neyra-srsplus-sonicdrop-clearonlyhold-v1';
const MAX_BODY = 32 * 1024;
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
function cleanInput(value: unknown) {
  const field = object(value);
  onlyKeys(field, ['schemaVersion', 'id', 'rows', 'source']);
  if (field.schemaVersion !== 1) throw new Rejection(400, 'Invalid input version');
  if (!Array.isArray(field.rows) || field.rows.length !== 24 || !field.rows.some(row => row !== 0)) {
    throw new Rejection(400, 'A nonempty completed board is required');
  }
  const rows = field.rows.map(row => number(row, 0, 1022, true));
  const source = object(field.source);
  onlyKeys(source, ['kind', 'family', 'reference', 'runId', 'candidateId', 'bagBoundary', 'probability',
    'firstBagProbability', 'ignoreFirstBagProbability', 'verified', 'qualified', 'seed', 'runSeed',
    'modelMode', 'buildVersion', 'rules', 'searchConfig']);
  const runId = text(source.runId, 80);
  if (!/^[a-zA-Z0-9-]{16,80}$/.test(runId) || source.family !== `search:${runId}` ||
      source.kind !== 'search' || source.modelMode !== 'heuristic' || source.rules !== RULES ||
      source.verified !== true || source.qualified !== true) throw new Rejection(400, 'Invalid search provenance');
  const candidateId = text(source.candidateId, 100);
  if (!/^w\d+:plan-\d+$/.test(candidateId) || field.id !== `${runId}:${candidateId}` ||
      source.reference !== `neyra:run:${runId}/candidate:${candidateId}`) throw new Rejection(400, 'Invalid record identity');
  const c = object(source.searchConfig);
  onlyKeys(c, ['minSuccessPercent', 'bagMinSuccessPercent', 'firstBagMinSuccessPercent',
    'riskMinSpinsPerBag', 'pieceGoal', 'maxPieces', 'beamWidth']);
  if (![16, 32, 64, 128].includes(c.beamWidth as number)) throw new Rejection(400, 'Invalid beam width');
  return {schemaVersion: 1, id: text(field.id), rows, source: {
    kind: 'search', family: source.family, reference: source.reference, runId, candidateId,
    bagBoundary: number(source.bagBoundary, 1, 128, true),
    probability: number(source.probability, 0, 1),
    firstBagProbability: source.firstBagProbability === null ? null : number(source.firstBagProbability, 0, 1),
    ignoreFirstBagProbability: bool(source.ignoreFirstBagProbability), verified: true, qualified: true,
    seed: number(source.seed, 0, 0xffffffff, true), runSeed: number(source.runSeed, 0, 0xffffffff, true),
    modelMode: 'heuristic', buildVersion: text(source.buildVersion, 96), rules: RULES,
    trust: 'unverified-contribution',
    searchConfig: {
      minSuccessPercent: number(c.minSuccessPercent, 0, 100),
      bagMinSuccessPercent: number(c.bagMinSuccessPercent, 0, 100),
      firstBagMinSuccessPercent: number(c.firstBagMinSuccessPercent, 0, 100),
      riskMinSpinsPerBag: number(c.riskMinSpinsPerBag, 0, 6, true),
      pieceGoal: number(c.pieceGoal, 1, 1000, true), maxPieces: number(c.maxPieces, 1, 700, true),
      beamWidth: number(c.beamWidth, 16, 128, true)
    }
  }};
}
async function digest(value: string): Promise<string> {
  return Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(value))),
    byte => byte.toString(16).padStart(2, '0')).join('');
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
        return respond({acknowledged: (await response.json()).length});
      }
      if (body.action === 'stats') {
        const response = await rest('neyra_contribution_queue?acknowledged_at=is.null&select=board_key',
          {method: 'HEAD', headers: {'Prefer': 'count=exact'}});
        return respond({pending: Number(response.headers.get('content-range')?.split('/')[1] ?? 0)});
      }
      throw new Rejection(400, 'Unknown consumer action');
    }
    onlyKeys(body, ['schemaVersion', 'fields']);
    if (body.schemaVersion !== 2 || !Array.isArray(body.fields) || !body.fields.length || body.fields.length > 8) {
      throw new Rejection(400, 'Invalid contribution batch');
    }
    const records = await Promise.all(body.fields.map(async value => {
      const input = cleanInput(value);
      return {board_key: await digest(JSON.stringify(input.rows)), bag_depth: input.source.bagBoundary, input};
    }));
    const response = await rest('rpc/neyra_admit_contributions', {method: 'POST', body: JSON.stringify({p_records: records})});
    return respond(await response.json());
  } catch (error) {
    return respond({error: error instanceof Rejection ? error.message : 'Service temporarily unavailable'},
      error instanceof Rejection ? error.status : 503);
  }
});
