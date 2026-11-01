-- Public Pages submit untrusted inputs and bounded machine traces after opt-in.
-- The owner-private Site imports this queue into D1 and generates fresh labels.
-- No browser role can read or write these tables or call the service RPC.
create table public.neyra_contribution_queue (
  -- Legacy column name: schema-3 keys hash the contribution ID and occupancy,
  -- preserving distinct run/path traces. D1 deduplicates training geometry.
  board_key text primary key check (board_key ~ '^[0-9a-f]{64}$'),
  input jsonb not null check (jsonb_typeof(input) = 'object' and octet_length(input::text) <= 8192),
  bag_depth smallint not null check (bag_depth between 1 and 128),
  created_at timestamptz not null default now(),
  acknowledged_at timestamptz
);
create index neyra_contribution_pending on public.neyra_contribution_queue (bag_depth desc, created_at, board_key)
  where acknowledged_at is null;
create index neyra_contribution_acknowledged on public.neyra_contribution_queue (acknowledged_at)
  where acknowledged_at is not null;

create table public.neyra_contribution_settings (
  name text primary key check (name = 'studio-consumer'),
  token_sha256 text not null check (token_sha256 ~ '^[0-9a-f]{64}$')
);
create table public.neyra_contribution_limits (
  bucket text not null,
  day date not null,
  fields integer not null check (fields between 0 and 20000),
  primary key (bucket, day)
);

alter table public.neyra_contribution_queue enable row level security;
alter table public.neyra_contribution_settings enable row level security;
alter table public.neyra_contribution_limits enable row level security;
revoke all on public.neyra_contribution_queue, public.neyra_contribution_settings,
  public.neyra_contribution_limits from public, anon, authenticated;
grant select, insert, update, delete on public.neyra_contribution_queue, public.neyra_contribution_limits to service_role;
grant select on public.neyra_contribution_settings to service_role;
create policy neyra_queue_service on public.neyra_contribution_queue for all to service_role using (true) with check (true);
create policy neyra_limits_service on public.neyra_contribution_limits for all to service_role using (true) with check (true);
create policy neyra_settings_service on public.neyra_contribution_settings for select to service_role using (true);

-- Invoker rights, explicit service-role execution, atomic bounded admission.
create or replace function public.neyra_admit_contributions(p_records jsonb)
returns jsonb language plpgsql security invoker set search_path = '' as $$
declare
  amount integer;
  additions integer;
  pending_additions integer;
  admitted jsonb;
  usage integer;
  inserted integer;
  entry record;
  today date := (now() at time zone 'UTC')::date;
begin
  if jsonb_typeof(p_records) is distinct from 'array' then raise exception 'invalid batch'; end if;
  amount := jsonb_array_length(p_records);
  if amount < 1 or amount > 8 or octet_length(p_records::text) > 32768 then
    raise exception 'invalid batch';
  end if;
  -- Serialize admission count and quota accounting. Retries and duplicates do
  -- not consume daily capacity, including when the original receipt was lost.
  perform pg_catalog.pg_advisory_xact_lock(71022, 3);
  select coalesce(jsonb_agg(item), '[]'::jsonb), count(*)::integer,
    coalesce(sum(case when queue.board_key is null or queue.acknowledged_at is not null then 1 else 0 end), 0)::integer
    into admitted, additions, pending_additions
    from (select distinct on (item->>'board_key') item
      from jsonb_array_elements(p_records) as records(item)
      order by item->>'board_key', (item->>'bag_depth')::smallint desc, item #>> '{input,id}') selected
    left join public.neyra_contribution_queue queue on queue.board_key = item->>'board_key'
    where queue.board_key is null or (item->>'bag_depth')::smallint > queue.bag_depth;
  if additions = 0 then return jsonb_build_object('accepted', 0, 'duplicates', amount); end if;
  insert into public.neyra_contribution_limits as limits (bucket, day, fields)
    values ('global', today, additions)
    on conflict (bucket, day) do update set fields = limits.fields + excluded.fields
      where limits.fields + excluded.fields <= 20000
    returning fields into usage;
  if usage is null or usage > 20000 then raise exception 'contribution limit'; end if;
  for entry in select item #>> '{input,source,runId}' as run_id, count(*)::integer as count
    from jsonb_array_elements(admitted) as records(item) group by 1
  loop
    if entry.run_id is null or entry.run_id !~ '^[a-zA-Z0-9-]{16,80}$' then
      raise exception 'invalid run';
    end if;
    insert into public.neyra_contribution_limits as limits (bucket, day, fields)
      values ('run:' || entry.run_id, today, entry.count)
      on conflict (bucket, day) do update set fields = limits.fields + excluded.fields
      returning fields into usage;
    if usage > 512 then raise exception 'contribution limit'; end if;
  end loop;
  if (select count(*) from public.neyra_contribution_queue where acknowledged_at is null) + pending_additions > 50000 then
    raise exception 'contribution queue full';
  end if;
  insert into public.neyra_contribution_queue as queue (board_key, input, bag_depth)
    select distinct on (item->>'board_key') item->>'board_key', item->'input', (item->>'bag_depth')::smallint
    from jsonb_array_elements(admitted) as records(item)
    order by item->>'board_key', (item->>'bag_depth')::smallint desc, item #>> '{input,id}'
    on conflict (board_key) do update set input = excluded.input, bag_depth = excluded.bag_depth,
      acknowledged_at = null where excluded.bag_depth > queue.bag_depth;
  get diagnostics inserted = row_count;
  delete from public.neyra_contribution_limits where day < today - 2;
  delete from public.neyra_contribution_queue where acknowledged_at < now() - interval '7 days';
  return jsonb_build_object('accepted', inserted, 'duplicates', amount - inserted);
end;
$$;
revoke all on function public.neyra_admit_contributions(jsonb) from public, anon, authenticated;
grant execute on function public.neyra_admit_contributions(jsonb) to service_role;

comment on table public.neyra_contribution_queue is 'Untrusted board inputs and machine traces after explicit consent. Relabel privately; trace scores are not supervised CNN targets or trusted supply proofs.';
