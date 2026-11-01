-- Apply only after choosing an authorized project and inspecting its schema.
-- This is a schema definition, not a fabricated Supabase migration history.
-- Agent access uses direct SQL; do not add this schema to exposed Data API schemas.
begin;
create schema if not exists neyra_learning;
revoke all on schema neyra_learning from public, anon, authenticated;
grant usage on schema neyra_learning to service_role;

create table if not exists neyra_learning.boards (
    board_hash text primary key check (board_hash ~ '^[0-9a-f]{64}$'),
    mirror_group_hash text not null check (mirror_group_hash ~ '^[0-9a-f]{64}$'),
    rows smallint[] not null check (
        array_ndims(rows) = 1 and array_lower(rows, 1) = 1
        and cardinality(rows) = 24 and array_position(rows, null) is null
        and 0 <= all(rows) and 1023 >= all(rows) and 0 <> any(rows)
    ),
    created_at timestamptz not null default now()
);
create index if not exists boards_mirror_group on neyra_learning.boards (mirror_group_hash);

create table if not exists neyra_learning.inputs (
    input_hash text primary key check (input_hash ~ '^[0-9a-f]{64}$'),
    board_hash text not null references neyra_learning.boards (board_hash),
    source_kind text not null check (source_kind in ('opener', 'search', 'perturbation')),
    source_family text not null check (length(source_family) between 1 and 256),
    bag_boundary integer not null check (bag_boundary between 1 and 10000),
    payload jsonb not null check (jsonb_typeof(payload) = 'object'),
    created_at timestamptz not null default now()
);
create index if not exists inputs_board on neyra_learning.inputs (board_hash);
create index if not exists inputs_family on neyra_learning.inputs (source_family);

create table if not exists neyra_learning.label_tasks (
    task_hash text primary key check (task_hash ~ '^[0-9a-f]{64}$'),
    horizon_bags smallint not null check (horizon_bags in (2, 3)),
    work_budget bigint not null check (work_budget between 1 and 1000000000),
    source_revision text not null check (length(source_revision) between 1 and 256),
    payload jsonb not null check (jsonb_typeof(payload) = 'object'),
    created_at timestamptz not null default now()
);

create table if not exists neyra_learning.labels (
    label_hash text primary key check (label_hash ~ '^[0-9a-f]{64}$'),
    input_hash text not null references neyra_learning.inputs (input_hash),
    task_hash text not null references neyra_learning.label_tasks (task_hash),
    reachable_spins smallint not null check (reachable_spins between 0 and 18),
    certified_spins smallint check (certified_spins between 0 and reachable_spins),
    geometric_chains bigint not null check (geometric_chains >= 0),
    certified_chains bigint not null check (certified_chains between 0 and geometric_chains),
    work_used bigint not null check (work_used >= 0),
    censored boolean not null,
    evidence_kind text not null check (length(evidence_kind) between 1 and 256),
    stop_reason text not null check (length(stop_reason) between 1 and 256),
    payload jsonb not null check (jsonb_typeof(payload) = 'object'),
    created_at timestamptz not null default now(),
    check ((certified_chains = 0 and certified_spins is null)
        or (certified_chains > 0 and certified_spins is not null))
);
create index if not exists labels_input_task on neyra_learning.labels (input_hash, task_hash);
create index if not exists labels_task on neyra_learning.labels (task_hash);

-- Reference quality is a separate prior, never an observed spin/count label.
create table if not exists neyra_learning.reference_priors (
    prior_hash text primary key check (prior_hash ~ '^[0-9a-f]{64}$'),
    input_hash text not null references neyra_learning.inputs (input_hash),
    prior_score double precision not null check (prior_score between -1000000000 and 1000000000),
    payload jsonb not null check (jsonb_typeof(payload) = 'object'),
    created_at timestamptz not null default now()
);
create index if not exists priors_input on neyra_learning.reference_priors (input_hash);

create table if not exists neyra_learning.models (
    model_hash text primary key check (model_hash ~ '^[0-9a-f]{64}$'),
    model_id text not null check (length(model_id) between 1 and 256),
    dataset_sha256 text not null check (dataset_sha256 ~ '^[0-9a-f]{64}$'),
    weights_sha256 text not null check (weights_sha256 ~ '^[0-9a-f]{64}$'),
    payload jsonb not null check (jsonb_typeof(payload) = 'object'),
    created_at timestamptz not null default now()
);
create index if not exists models_dataset on neyra_learning.models (dataset_sha256);

alter table neyra_learning.boards enable row level security;
alter table neyra_learning.inputs enable row level security;
alter table neyra_learning.label_tasks enable row level security;
alter table neyra_learning.labels enable row level security;
alter table neyra_learning.reference_priors enable row level security;
alter table neyra_learning.models enable row level security;

-- No anonymous/user policy is intentional. Privileged ingestion is append-only.
revoke all on all tables in schema neyra_learning from public, anon, authenticated, service_role;
grant select, insert on all tables in schema neyra_learning to service_role;
alter default privileges in schema neyra_learning
    revoke all on tables from public, anon, authenticated;
commit;
