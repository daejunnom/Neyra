"""Strict v2 contracts for occupancy-only, measured-spin learning.

Legacy Terrain teachers, opener preferences and path-count objectives are rejected.
The weights hash is stable across Python, Rust/JS clients and JSON spellings.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct

TASK = "naga-spin-observations-v2"
POLICY = "neyra-fixed-bag-continuation-v2"
ARCHITECTURE = {"name": "neyra-spin-cnn-v2", "input": [1, 24, 10],
                "channels": [24, 32, 48], "kernel": 3, "padding": 1,
                "poolOutput": [4, 2], "poolKernel": [6, 5],
                "hidden": 32, "heads": 2, "parameters": 33442}
WEIGHT_LAYOUT = [
    ("conv1.weight", 216), ("conv1.bias", 24),
    ("conv2.weight", 6912), ("conv2.bias", 32),
    ("conv3.weight", 13824), ("conv3.bias", 48),
    ("hidden.weight", 12288), ("hidden.bias", 32),
    ("output.weight", 64), ("output.bias", 2),
]
WEIGHTS_HASH_ALGORITHM = "sha256-f32le-neyraspin-v2"
MAX_RECORD_BYTES = 8 * 1024 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024
FORBIDDEN = {"currentBoardTeacher", "referencePrior", "priorIsMeasuredOutcome",
             "terrainScore", "fieldQuality", "rankingScore", "shapePreference",
             "observedChoices", "boardEvaluation", "prior"}
HASH = re.compile(r"^[0-9a-f]{64}$")


def integer(value, name, lower=0, upper=1_000_000_000):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def text_value(value, name, limit=4096):
    if not isinstance(value, str) or not value.strip() or len(value) > limit or "\0" in value:
        raise ValueError(f"{name} must be a bounded nonempty string")
    return value


def reject_legacy(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN:
                raise ValueError(f"Legacy evaluation data is excluded from spin v2: {key}")
            reject_legacy(child)
    elif isinstance(value, list):
        for child in value:
            reject_legacy(child)


def validate_rows(rows, *, nonempty=True):
    if not isinstance(rows, list) or len(rows) != 24:
        raise ValueError("A board needs 24 bottom-up row masks")
    for row in rows:
        integer(row, "row mask", 0, 1022)
    if nonempty and not any(rows):
        raise ValueError("An empty first-bag board is outside the learning task")
    return rows


def canonical_rows(rows):
    validate_rows(rows)
    reflected = [int(f"{row:010b}"[::-1], 2) for row in rows]
    return min(tuple(rows), tuple(reflected))


def validate_sample(value, minimum_bag=2):
    if not isinstance(value, dict) or set(value) != {"id", "rows", "source"}:
        raise ValueError("A v2 input has exactly id, rows and source")
    reject_legacy(value)
    text_value(value["id"], "id", 256)
    validate_rows(value["rows"])
    source = value["source"]
    if not isinstance(source, dict) or source.get("kind") not in {"opener", "search", "perturbation"}:
        raise ValueError("Invalid source kind")
    text_value(source.get("family"), "source.family", 256)
    text_value(source.get("reference"), "source.reference")
    integer(source.get("bagBoundary"), "source.bagBoundary", minimum_bag, 10000)
    return value


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Missing labeling config")
    required = {"horizonBags", "workBudget", "beamWidth", "profile", "seed",
                "maxNodes", "attemptsPerNode", "sourceRevision"}
    if set(config) != required:
        raise ValueError("Labeling policy must declare the exact v2 budget fields")
    integer(config["horizonBags"], "horizonBags", 2, 3)
    integer(config["workBudget"], "workBudget", 1, 4194304)
    integer(config["beamWidth"], "beamWidth", 4, 128)
    integer(config["profile"], "profile", 0, 3)
    integer(config["seed"], "seed", 0, 2**64 - 1)
    integer(config["maxNodes"], "maxNodes", 1, 256)
    integer(config["attemptsPerNode"], "attemptsPerNode", 1, 16)
    text_value(config["sourceRevision"], "sourceRevision", 256)
    return config


def validate_label(value):
    if not isinstance(value, dict) or value.get("schemaVersion") != 2 or value.get("task") != TASK:
        raise ValueError("Only newly collected spin-only schemaVersion 2 labels are accepted")
    reject_legacy(value)
    if value.get("policy") != POLICY:
        raise ValueError("Wrong measured-spin policy")
    validate_sample(value.get("input"))
    config = validate_config(value.get("config"))
    targets, measured = value.get("targets"), value.get("measured")
    if not isinstance(targets, dict) or set(targets) != {"foundSpin", "observedSpins"}:
        raise ValueError("V2 targets are foundSpin and observedSpins only")
    if not isinstance(measured, dict) or measured.get("censored") is not True:
        raise ValueError("Bounded observations must retain censoring")
    spins = integer(targets["observedSpins"], "observedSpins", 0, 6 * config["horizonBags"])
    if type(targets["foundSpin"]) is not bool or targets["foundSpin"] != (spins > 0):
        raise ValueError("foundSpin disagrees with the observed spin count")
    if measured.get("maxReachableObservedSpins") != spins:
        raise ValueError("Targets disagree with measured reachable spins")
    work = integer(measured.get("workUsed"), "workUsed", 0, config["workBudget"])
    generation = integer(measured.get("generationWork"), "generationWork", 0, work)
    verification = integer(measured.get("verificationWork"), "verificationWork", 0, work)
    if generation + verification != work:
        raise ValueError("Work accounting does not sum to the declared total")
    certified = measured.get("maxCertifiedObservedSpins")
    if certified is not None:
        integer(certified, "maxCertifiedObservedSpins", 0, spins)
    witnesses = value.get("certifiedWitnesses")
    if not isinstance(witnesses, list) or measured.get("certifiedWholeBagChains") != len(witnesses):
        raise ValueError("Certified chain count disagrees with witness records")
    by_id = {}
    for witness in witnesses:
        if not isinstance(witness, dict):
            raise ValueError("Invalid certificate witness")
        identity = integer(witness.get("id"), "certificate id", 1, 2**64-1)
        if identity in by_id:
            raise ValueError("Duplicate certificate id")
        parent_id = witness.get("parentId")
        parent = by_id.get(parent_id) if parent_id is not None else None
        if parent_id is not None and parent is None:
            raise ValueError("Certificate parent must precede its child")
        bags = integer(witness.get("bags"), "certificate bags", 1, config["horizonBags"])
        if bags != (parent["bags"] + 1 if parent else 1):
            raise ValueError("Certificate horizon is incoherent")
        start = parent["endRows"] if parent else value["input"]["rows"]
        start_spins = parent["endSpins"] if parent else 0
        if witness.get("startRows") != start or witness.get("startSpins") != start_spins:
            raise ValueError("Certificate parent board or count mismatch")
        validate_rows(witness.get("endRows"), nonempty=False)
        integer(witness.get("endSpins"), "certificate spins", start_spins, min(spins, start_spins + 6))
        if any(witness.get(key) != 5040 for key in ("checked", "success", "quietSafeSupplies")):
            raise ValueError("Expanded parents require exact full-supply quiet-safe proof")
        if witness.get("firstBagSymmetryAssumed") is not False:
            raise ValueError("A supplied deep board cannot assume first-bag symmetry")
        by_id[identity] = witness
    observed_max = max((w["endSpins"] for w in witnesses), default=None)
    if certified != observed_max:
        raise ValueError("Certified maximum must remain null when unobserved")
    return value


def weights_hash(weights):
    if not isinstance(weights, dict) or set(weights) != {key for key, _ in WEIGHT_LAYOUT}:
        raise ValueError("Unexpected model tensors")
    h = hashlib.sha256()
    for key, size in WEIGHT_LAYOUT:
        values = weights[key]
        if not isinstance(values, list) or len(values) != size:
            raise ValueError(f"Wrong tensor shape: {key}")
        for number in values:
            if type(number) not in (int, float) or not math.isfinite(number) or abs(number) > 1000:
                raise ValueError(f"Nonfinite or excessive model weight: {key}")
        h.update(struct.pack("<" + "f" * size, *values))
    return h.hexdigest()


def validate_model(value):
    if not isinstance(value, dict) or value.get("schemaVersion") != 2 or value.get("task") != TASK:
        raise ValueError("Only spin-only v2 models are accepted")
    reject_legacy(value)
    if value.get("trained") is not True or value.get("device") != "cpu":
        raise ValueError("A trained CPU model is required")
    if value.get("architecture") != ARCHITECTURE:
        raise ValueError("Unexpected v2 architecture")
    text_value(value.get("modelId"), "modelId", 128)
    for key in ("datasetSha256", "weightsSha256"):
        if not isinstance(value.get(key), str) or not HASH.fullmatch(value[key]):
            raise ValueError(f"Invalid {key}")
    if value.get("weightsSha256Algorithm") != WEIGHTS_HASH_ALGORITHM:
        raise ValueError("Wrong weights hash algorithm")
    if weights_hash(value.get("weights")) != value["weightsSha256"]:
        raise ValueError("Model weights hash mismatch")
    config = validate_config(value.get("labelConfig"))
    if value.get("horizonBags") != config["horizonBags"] or value.get("workBudget") != config["workBudget"]:
        raise ValueError("Model scope disagrees with its labeling policy")
    return value


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant: {value}")


def unique_keys(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def read_jsonl(path, kind="labels"):
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Learning file exceeds the bounded input limit")
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        if len(line.encode()) > MAX_RECORD_BYTES:
            raise ValueError(f"Line {line_number} exceeds the record limit")
        value = json.loads(line, object_pairs_hook=unique_keys, parse_constant=reject_constant)
        validate_label(value) if kind == "labels" else validate_sample(value)
        records.append(value)
    return records
