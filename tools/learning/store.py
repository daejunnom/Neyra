#!/usr/bin/env python3
"""Validate and retain Neyra learning observations in a private Postgres schema.

No remote connection occurs for validate/sql. push/pull use a locally installed
psql and NEYRA_DATABASE_URL; credentials are never placed in arguments or output.
Connected agents can submit generated SQL through their authorized SQL tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import parse_qs, unquote, urlsplit

MAX_RECORD_BYTES = 8 * 1024 * 1024
MAX_INPUT_BYTES = 128 * 1024 * 1024
HASH = re.compile(r"^[0-9a-f]{64}$")
WEIGHTS_HASH_ALGORITHM = "sha256-canonical-json-numeric-v1"
TABLES = {
    "boards": ("inputs", "input_hash"),
    "labels": ("labels", "label_hash"),
    "priors": ("reference_priors", "prior_hash"),
    "models": ("models", "model_hash"),
}


def canonical(value):
    # JSONB preserves numeric values, not exponent spelling or the sign of zero.
    # Normalize integral floats so hashes survive its JSON serialization.
    def normalized(item):
        if isinstance(item, dict):
            return {key: normalized(child) for key, child in item.items()}
        if isinstance(item, list):
            return [normalized(child) for child in item]
        if isinstance(item, float) and math.isfinite(item) and item.is_integer():
            return int(item)
        return item
    return json.dumps(normalized(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def object_value(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def integer(value, name, lower=0, upper=1_000_000_000):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def string(value, name, maximum=256):
    if not isinstance(value, str) or not value or len(value) > maximum or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters")
    return value


def hash_value(value, name):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def validate_input(value):
    value = object_value(value, "input")
    string(value.get("id"), "input.id")
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError("input.rows must have exactly 24 bottom-up row masks")
    for row in rows:
        integer(row, "row mask", 0, 1023)
    if not any(rows) or 1023 in rows:
        raise ValueError("inputs must be nonempty post-clear boards, with no complete row")
    source = object_value(value.get("source"), "input.source")
    if source.get("kind") not in {"opener", "search", "perturbation"}:
        raise ValueError("source.kind must be opener, search, or perturbation")
    string(source.get("family"), "source.family")
    string(source.get("reference"), "source.reference", 4096)
    integer(source.get("bagBoundary"), "source.bagBoundary", 1, 10000)
    return value


def validate(value, kind):
    value = object_value(value, "record")
    encoded = canonical(value).encode("utf-8")
    if len(encoded) > MAX_RECORD_BYTES or b"\\u0000" in encoded:
        raise ValueError("record exceeds the size limit or contains a null character")
    if kind == "boards":
        return validate_input(value)
    if type(value.get("schemaVersion")) is not int or value.get("schemaVersion") != 1:
        raise ValueError("schemaVersion must be 1")
    if kind == "models":
        string(value.get("modelId"), "modelId")
        hash_value(value.get("datasetSha256"), "datasetSha256")
        hash_value(value.get("weightsSha256"), "weightsSha256")
        if value.get("weightsSha256Algorithm") != WEIGHTS_HASH_ALGORITHM:
            raise ValueError("weightsSha256Algorithm must declare the canonical numeric JSON hash")
        if "weights" in value:
            weights = object_value(value["weights"], "weights")
            if not weights or any(not isinstance(values, list) or not values for values in weights.values()):
                raise ValueError("weights must contain named, nonempty numeric arrays")
            for values in weights.values():
                if any(type(number) not in {int, float} or not math.isfinite(number) for number in values):
                    raise ValueError("weights must contain finite numbers")
            if digest(weights) != value["weightsSha256"]:
                raise ValueError("weightsSha256 does not match the supplied weights")
        for key in ("architecture", "training", "metrics"):
            object_value(value.get(key), key)
        return value
    validate_input(value.get("input"))
    if kind == "priors":
        prior = object_value(value.get("prior"), "prior")
        score = prior.get("score")
        if type(score) not in {int, float} or not math.isfinite(score) or abs(score) > 1e9:
            raise ValueError("prior.score must be finite and bounded")
        string(prior.get("reference"), "prior.reference", 4096)
        string(prior.get("reason"), "prior.reason", 4096)
        return value
    config = object_value(value.get("config"), "config")
    horizon = integer(config.get("horizonBags"), "config.horizonBags", 2, 3)
    budget = integer(config.get("workBudget"), "config.workBudget", 1, 4_194_304)
    integer(config.get("beamWidth"), "config.beamWidth", 4, 128)
    integer(config.get("profile"), "config.profile", 0, 3)
    integer(config.get("seed"), "config.seed", 0, 2**64 - 1)
    integer(config.get("maxNodes"), "config.maxNodes", 1, 256)
    integer(config.get("attemptsPerNode"), "config.attemptsPerNode", 1, 16)
    string(config.get("sourceRevision"), "config.sourceRevision")
    object_value(value.get("currentBoardTeacher"), "currentBoardTeacher")
    measured = object_value(value.get("measured"), "measured")
    reachable = integer(measured.get("maxReachableObservedSpins"), "maxReachableObservedSpins", 0, horizon * 6)
    chains = integer(measured.get("geometricWholeBagChains"), "geometricWholeBagChains", 0, 2**63 - 1)
    certified = integer(measured.get("certifiedWholeBagChains"), "certifiedWholeBagChains", 0, chains)
    productive = integer(measured.get("productiveGeometricWholeBagChains"), "productiveGeometricWholeBagChains", 0, chains)
    integer(measured.get("productiveCertifiedWholeBagChains"), "productiveCertifiedWholeBagChains", 0, min(certified, productive))
    if certified:
        integer(measured.get("maxCertifiedObservedSpins"), "maxCertifiedObservedSpins", 0, reachable)
    elif measured.get("maxCertifiedObservedSpins") is not None:
        raise ValueError("no certified chain means an unknown (null) certified maximum")
    integer(measured.get("workUsed"), "workUsed", 0, budget)
    if type(measured.get("censored")) is not bool:
        raise ValueError("censored must explicitly be true or false")
    string(measured.get("evidenceKind"), "evidenceKind")
    string(measured.get("stopReason"), "stopReason")
    if not isinstance(measured.get("depths"), list):
        raise ValueError("measured.depths must be an array")
    seen_depths = set()
    for depth in measured["depths"]:
        object_value(depth, "depth observation")
        bags = integer(depth.get("bags"), "depth.bags", 1, horizon)
        if bags in seen_depths:
            raise ValueError("depth observations must have distinct bag depths")
        seen_depths.add(bags)
        total = integer(depth.get("geometricWholeBagChains"), "depth.geometricWholeBagChains", 0, chains)
        proven = integer(depth.get("certifiedWholeBagChains"), "depth.certifiedWholeBagChains", 0, min(total, certified))
        if "productiveGeometricWholeBagChains" in depth:
            live = integer(depth["productiveGeometricWholeBagChains"], "depth.productiveGeometricWholeBagChains", 0, total)
            integer(depth.get("productiveCertifiedWholeBagChains"), "depth.productiveCertifiedWholeBagChains", 0, min(proven, live))
    if not isinstance(value.get("certifiedWitnesses"), list):
        raise ValueError("certifiedWitnesses must be an array")
    return value


def reject_constant(value):
    raise ValueError(f"nonfinite JSON number {value} is not allowed")


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_records(path, kind):
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError("input file exceeds 128 MiB; split it into bounded batches")
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > MAX_RECORD_BYTES:
            raise ValueError(f"line {number}: record exceeds 8 MiB")
        try:
            value = json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_keys)
            records.append(validate(value, kind))
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"line {number}: {error}") from error
    return records


def sql_text(value):
    # Only literal data is encoded; no user-controlled SQL identifier is accepted.
    data = value.encode("utf-8").hex()
    return f"convert_from(decode('{data}', 'hex'), 'UTF8')"


def sql_json(value):
    return sql_text(canonical(value)) + "::jsonb"


def insert(table, fields):
    return (f"insert into neyra_learning.{table} ({', '.join(fields)}) values "
            f"({', '.join(fields.values())}) on conflict do nothing;")


def input_statements(value):
    rows = value["rows"]
    mirrored = [int(f"{row:010b}"[::-1], 2) for row in rows]
    board_hash = digest(rows)
    group_hash = digest(min(rows, mirrored))
    source = value["source"]
    return [
        insert("boards", {"board_hash": sql_text(board_hash), "mirror_group_hash": sql_text(group_hash),
                          "rows": "ARRAY[" + ",".join(map(str, rows)) + "]::smallint[]"}),
        insert("inputs", {"input_hash": sql_text(digest(value)), "board_hash": sql_text(board_hash),
                          "source_kind": sql_text(source["kind"]), "source_family": sql_text(source["family"]),
                          "bag_boundary": str(source["bagBoundary"]), "payload": sql_json(value)}),
    ]


def statements(value, kind):
    validate(value, kind)
    if kind == "models":
        return [insert("models", {"model_hash": sql_text(digest(value)), "model_id": sql_text(value["modelId"]),
                                  "dataset_sha256": sql_text(value["datasetSha256"]),
                                  "weights_sha256": sql_text(value["weightsSha256"]), "payload": sql_json(value)})]
    source_input = value if kind == "boards" else value["input"]
    result = input_statements(source_input)
    if kind == "boards":
        return result
    if kind == "priors":
        result.append(insert("reference_priors", {"prior_hash": sql_text(digest(value)),
                      "input_hash": sql_text(digest(source_input)), "prior_score": repr(value["prior"]["score"]),
                      "payload": sql_json(value)}))
        return result
    config, measured = value["config"], value["measured"]
    result.append(insert("label_tasks", {"task_hash": sql_text(digest(config)),
                  "horizon_bags": str(config["horizonBags"]), "work_budget": str(config["workBudget"]),
                  "source_revision": sql_text(config["sourceRevision"]), "payload": sql_json(config)}))
    result.append(insert("labels", {"label_hash": sql_text(digest(value)),
                  "input_hash": sql_text(digest(source_input)), "task_hash": sql_text(digest(config)),
                  "reachable_spins": str(measured["maxReachableObservedSpins"]),
                  "certified_spins": "null" if measured["maxCertifiedObservedSpins"] is None else str(measured["maxCertifiedObservedSpins"]),
                  "geometric_chains": str(measured["geometricWholeBagChains"]),
                  "certified_chains": str(measured["certifiedWholeBagChains"]),
                  "work_used": str(measured["workUsed"]), "censored": str(measured["censored"]).lower(),
                  "evidence_kind": sql_text(measured["evidenceKind"]), "stop_reason": sql_text(measured["stopReason"]),
                  "payload": sql_json(value)}))
    return result


def transaction(records, kind):
    body = ["begin;", "set local statement_timeout = '60s';"]
    for value in records:
        body.extend(statements(value, kind))
    return "\n".join(body + ["commit;", ""])


def verify_records(records, kind):
    table, key = TABLES[kind]
    expected = {digest(value) for value in records}
    if not expected:
        return
    query = (f"select payload::text from neyra_learning.{table} where {key} in ("
             + ",".join(sql_text(value) for value in sorted(expected)) + ");")
    returned = [validate(json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_keys), kind)
                for line in run_sql(query).splitlines() if line.strip()]
    if {digest(value) for value in returned} != expected:
        raise RuntimeError("database round-trip mismatch; do not treat this batch as verified")


def connection_environment():
    url = os.environ.get("NEYRA_DATABASE_URL")
    if not url:
        raise ValueError("set NEYRA_DATABASE_URL for the selected private database")
    parsed = urlsplit(url)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname or not parsed.username:
        raise ValueError("NEYRA_DATABASE_URL must be a Postgres connection URL")
    options = parse_qs(parsed.query, strict_parsing=True)
    if set(options) - {"sslmode", "connect_timeout"}:
        raise ValueError("connection URL supports only sslmode and connect_timeout options")
    if options.get("sslmode", ["verify-full"]) != ["verify-full"]:
        raise ValueError("remote storage requires sslmode=verify-full")
    timeout = int(options.get("connect_timeout", ["15"])[0])
    integer(timeout, "connect_timeout", 1, 120)
    env = dict(os.environ)
    env.update(PGHOST=parsed.hostname, PGPORT=str(parsed.port or 5432), PGUSER=unquote(parsed.username),
               PGDATABASE=unquote(parsed.path.lstrip("/")) or "postgres", PGSSLMODE="verify-full",
               PGCONNECT_TIMEOUT=str(timeout), PGAPPNAME="neyra-learning-store")
    if parsed.password is not None:
        env["PGPASSWORD"] = unquote(parsed.password)
    # Keep libpq's configured root certificate / password file selectors intact.
    env.pop("NEYRA_DATABASE_URL", None)
    return env


def run_sql(query):
    try:
        result = subprocess.run(["psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1"],
                                input=query, text=True, capture_output=True,
                                env=connection_environment(), timeout=90, check=False)
    except FileNotFoundError as error:
        raise ValueError("install psql or submit the generated SQL with an authorized database tool") from error
    if result.returncode:
        # Server error detail may contain record data. Keep it out of default logs.
        raise RuntimeError(f"database statement failed (psql exit {result.returncode}); no success was recorded")
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate", "sql", "push", "pull"])
    parser.add_argument("--kind", choices=TABLES, default="labels")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--after-key", default="")
    args = parser.parse_args()
    if args.action == "pull":
        integer(args.limit, "limit", 1, 10000)
        if args.after_key:
            hash_value(args.after_key, "after-key")
        if not args.output:
            raise ValueError("pull requires --output")
        table, key = TABLES[args.kind]
        query = (f"select payload::text from neyra_learning.{table} where {key} > {sql_text(args.after_key)} "
                 f"order by {key} limit {args.limit};")
        output = run_sql(query)
        records = [validate(json.loads(line, parse_constant=reject_constant, object_pairs_hook=unique_keys), args.kind)
                   for line in output.splitlines() if line.strip()]
        args.output.write_text("".join(canonical(record) + "\n" for record in records), encoding="utf-8")
        print(json.dumps({"records": len(records), "nextAfterKey": digest(records[-1]) if records else None}))
        return
    if not args.input:
        raise ValueError("validate/sql/push require --input")
    records = read_records(args.input, args.kind)
    if args.action == "sql":
        if not args.output:
            raise ValueError("sql requires --output")
        args.output.write_text(transaction(records, args.kind), encoding="utf-8")
    elif args.action == "push":
        integer(args.batch_size, "batch-size", 1, 256)
        committed = 0
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            try:
                run_sql(transaction(batch, args.kind))
                verify_records(batch, args.kind)
            except Exception:
                print(json.dumps({"verifiedDocuments": committed, "complete": False,
                                  "note": "The current batch may have committed; retrying identical documents is idempotent."}), file=sys.stderr)
                raise
            committed += len(batch)
    print(json.dumps({"kind": args.kind, "records": len(records), "uniqueDocuments": len({digest(x) for x in records}),
                      "action": args.action, "remoteWrite": args.action == "push"}))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(f"neyra store: {error}", file=sys.stderr)
        raise SystemExit(1)
