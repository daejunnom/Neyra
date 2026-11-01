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
