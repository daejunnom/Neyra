-- Additive service-only relay for explicit-consent focused observations (v4).
-- The private Site durably stores each receipt before acknowledging/removing it.
create table public.neyra_observation_queue (
  queue_key text primary key check (queue_key ~ '^[0-9a-f]{64}$'),
  observation jsonb not null check (jsonb_typeof(observation) = 'object'),
  payload_bytes integer not null check (payload_bytes between 1 and 786432),
  run_id text not null check (run_id ~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
  sequence integer not null check (sequence between 1 and 1000000),
  bag_depth smallint not null check (bag_depth between 2 and 512),
  created_at timestamptz not null default now(),
  check (payload_bytes = octet_length(observation::text)),
  unique (run_id, sequence)
);
create index neyra_observation_fifo on public.neyra_observation_queue (created_at, sequence, queue_key);
alter table public.neyra_observation_queue enable row level security;
revoke all on public.neyra_observation_queue from public, anon, authenticated;
grant select, insert, delete on public.neyra_observation_queue to service_role;
create policy neyra_observation_service on public.neyra_observation_queue
  for all to service_role using (true) with check (true);

create function public.neyra_admit_observation(p_key text, p_observation jsonb)
returns jsonb language plpgsql security invoker set search_path = '' as $$
declare
  bytes integer;
  usage integer;
  run text;
  today date := (now() at time zone 'UTC')::date;
begin
  bytes := octet_length(p_observation::text);
  run := p_observation->>'runId';
  if p_key is null or p_key !~ '^[0-9a-f]{64}$' or jsonb_typeof(p_observation) <> 'object'
    or bytes > 786432 or run is null
    or run !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    or p_observation->>'kind' is distinct from 'focused-spin-observation'
    or p_observation #>> '{label,input,source,trust}' is distinct from 'client-reported'
  then raise exception 'invalid observation'; end if;
  -- A bounded queue needs one admission lock, including count/byte checks.
  perform pg_catalog.pg_advisory_xact_lock(71022, 4);
  if exists (select 1 from public.neyra_observation_queue where queue_key = p_key) then
    return jsonb_build_object('accepted', 0, 'duplicates', 1);
  end if;
  insert into public.neyra_contribution_limits as limits (bucket, day, fields)
    values ('observations:global', today, 1)
    on conflict (bucket, day) do update set fields = limits.fields + 1
      where limits.fields < 1024 returning fields into usage;
  if usage is null or usage > 1024 then raise exception 'contribution limit'; end if;
  insert into public.neyra_contribution_limits as limits (bucket, day, fields)
    values ('observations:run:' || run, today, 1)
    on conflict (bucket, day) do update set fields = limits.fields + 1
      where limits.fields < 64 returning fields into usage;
  if usage is null or usage > 64 then raise exception 'contribution limit'; end if;
  if (select count(*) from public.neyra_observation_queue) >= 512
    or (select coalesce(sum(payload_bytes), 0) from public.neyra_observation_queue) + bytes > 134217728
  then raise exception 'contribution queue full'; end if;
  insert into public.neyra_observation_queue (queue_key, observation, payload_bytes, run_id, sequence, bag_depth)
    values (p_key, p_observation, bytes, run, (p_observation->>'sequence')::integer,
      (p_observation #>> '{label,input,source,bagBoundary}')::smallint);
  delete from public.neyra_contribution_limits where day < today - 2;
  return jsonb_build_object('accepted', 1, 'duplicates', 0);
end;
$$;
revoke all on function public.neyra_admit_observation(text, jsonb) from public, anon, authenticated;
grant execute on function public.neyra_admit_observation(text, jsonb) to service_role;
comment on table public.neyra_observation_queue is
  'Fixed-policy bounded spin observations submitted by opt-in clients. Client-reported evidence, not independently verified supply certificates. Private consumer only; delete after durable import.';

-- Retain only compact receipts after the private consumer has stored the raw
-- observation. A lost HTTP response must not re-admit data or consume quota.
create table public.neyra_observation_receipts (
  queue_key text primary key check (queue_key ~ '^[0-9a-f]{64}$'),
  payload_sha256 text not null check (payload_sha256 ~ '^[0-9a-f]{64}$'),
  created_at timestamptz not null default now(),
  acknowledged_at timestamptz
);
alter table public.neyra_observation_receipts enable row level security;
revoke all on public.neyra_observation_receipts from public, anon, authenticated;
grant select, insert, update on public.neyra_observation_receipts to service_role;
create policy neyra_observation_receipts_service on public.neyra_observation_receipts
  for all to service_role using (true) with check (true);

-- The Edge Function calculates the hash from recursively key-sorted JSON. This
-- RPC is service-only; it is not an anonymous hashing or admission endpoint.
create function public.neyra_admit_observation_v2(p_key text, p_observation jsonb, p_payload_sha256 text)
returns jsonb language plpgsql security invoker set search_path = '' as $$
declare
  bytes integer;
  usage integer;
  run text;
  receipt public.neyra_observation_receipts%rowtype;
  queued public.neyra_observation_queue%rowtype;
  today date := (now() at time zone 'UTC')::date;
begin
  bytes := octet_length(p_observation::text);
  run := p_observation->>'runId';
  if p_key is null or p_key !~ '^[0-9a-f]{64}$' or p_payload_sha256 is null
    or p_payload_sha256 !~ '^[0-9a-f]{64}$' or jsonb_typeof(p_observation) is distinct from 'object'
    or bytes > 786432 or run is null
    or run !~ '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    or p_observation->>'kind' is distinct from 'focused-spin-observation'
    or p_observation #>> '{label,input,source,trust}' is distinct from 'client-reported'
  then raise exception 'invalid observation'; end if;
  perform pg_catalog.pg_advisory_xact_lock(71022, 4);
  select * into receipt from public.neyra_observation_receipts where queue_key = p_key;
  if found then
    if receipt.payload_sha256 <> p_payload_sha256 then raise exception 'observation identity conflict'; end if;
    return jsonb_build_object('accepted', 0, 'duplicates', 1,
      'state', case when receipt.acknowledged_at is null then 'queued' else 'imported' end);
  end if;
  -- Preserve pre-upgrade queued data. A duplicate cannot silently overwrite an
  -- existing observation with the same id and a different measured payload.
  select * into queued from public.neyra_observation_queue where queue_key = p_key;
  if found then
    if queued.observation <> p_observation then raise exception 'observation identity conflict'; end if;
    insert into public.neyra_observation_receipts(queue_key, payload_sha256, created_at)
      values (p_key, p_payload_sha256, queued.created_at);
    return jsonb_build_object('accepted', 0, 'duplicates', 1, 'state', 'queued');
  end if;
  insert into public.neyra_contribution_limits as limits (bucket, day, fields)
    values ('observations:global', today, 1)
    on conflict (bucket, day) do update set fields = limits.fields + 1
      where limits.fields < 1024 returning fields into usage;
  if usage is null or usage > 1024 then raise exception 'contribution limit'; end if;
  insert into public.neyra_contribution_limits as limits (bucket, day, fields)
    values ('observations:run:' || run, today, 1)
    on conflict (bucket, day) do update set fields = limits.fields + 1
      where limits.fields < 64 returning fields into usage;
  if usage is null or usage > 64 then raise exception 'contribution limit'; end if;
  if (select count(*) from public.neyra_observation_queue) >= 512
    or (select coalesce(sum(payload_bytes), 0) from public.neyra_observation_queue) + bytes > 134217728
  then raise exception 'contribution queue full'; end if;
  insert into public.neyra_observation_queue (queue_key, observation, payload_bytes, run_id, sequence, bag_depth)
    values (p_key, p_observation, bytes, run, (p_observation->>'sequence')::integer,
      (p_observation #>> '{label,input,source,bagBoundary}')::smallint);
  insert into public.neyra_observation_receipts(queue_key, payload_sha256) values (p_key, p_payload_sha256);
  delete from public.neyra_contribution_limits where day < today - 2;
  return jsonb_build_object('accepted', 1, 'duplicates', 0, 'state', 'queued');
end;
$$;
revoke all on function public.neyra_admit_observation_v2(text, jsonb, text) from public, anon, authenticated;
grant execute on function public.neyra_admit_observation_v2(text, jsonb, text) to service_role;

create function public.neyra_ack_observations(p_receipts jsonb)
returns jsonb language plpgsql security invoker set search_path = '' as $$
declare
  entry jsonb;
  existing_hash text;
  exists_in_queue boolean;
  removed integer;
begin
  if jsonb_typeof(p_receipts) is distinct from 'array' or jsonb_array_length(p_receipts) not between 1 and 32 then
    raise exception 'invalid observation receipt';
  end if;
  perform pg_catalog.pg_advisory_xact_lock(71022, 4);
  for entry in select value from jsonb_array_elements(p_receipts) loop
    if jsonb_typeof(entry) is distinct from 'object' or entry->>'queue_key' is null
      or entry->>'queue_key' !~ '^[0-9a-f]{64}$' or entry->>'payload_sha256' is null
      or entry->>'payload_sha256' !~ '^[0-9a-f]{64}$' then raise exception 'invalid observation receipt'; end if;
    select payload_sha256 into existing_hash from public.neyra_observation_receipts where queue_key = entry->>'queue_key';
    if found and existing_hash <> entry->>'payload_sha256' then raise exception 'observation identity conflict'; end if;
    select exists(select 1 from public.neyra_observation_queue where queue_key = entry->>'queue_key') into exists_in_queue;
    if existing_hash is null and not exists_in_queue then raise exception 'unknown observation receipt'; end if;
    insert into public.neyra_observation_receipts as receipts (queue_key, payload_sha256, acknowledged_at)
      values (entry->>'queue_key', entry->>'payload_sha256', now())
      on conflict (queue_key) do update set acknowledged_at = coalesce(receipts.acknowledged_at, excluded.acknowledged_at);
  end loop;
  delete from public.neyra_observation_queue where queue_key in
    (select value->>'queue_key' from jsonb_array_elements(p_receipts));
  get diagnostics removed = row_count;
  return jsonb_build_object('acknowledged', removed);
end;
$$;
revoke all on function public.neyra_ack_observations(jsonb) from public, anon, authenticated;
grant execute on function public.neyra_ack_observations(jsonb) to service_role;
comment on table public.neyra_observation_receipts is
  'Service-only immutable content identities. Raw observations stay in the bounded relay only until durable private import; receipts preserve retry idempotency afterward.';
