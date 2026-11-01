"""Second-stage observed completion cost: explicit progress, fixed rule profile.

This contract never treats unfinished-search locks as exact completion costs or
minimum-cost lower bounds. It does not alter the first-stage occupancy task.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
from pathlib import Path

from spin_contract import (HASH, MAX_FILE_BYTES, MAX_RECORD_BYTES, PRECISION_TASK,
                           canonical_rows, reject_constant, unique_keys, validate_model,
                           validate_sample)

TASK = "naga-completion-cost-v1"
POLICY = "input-replayed-completion-cost-v1"
RULE_PROFILE = "neyra-srs-plus-24-spin-single-clear-hold-v1"
HASH_ALGORITHM = "sha256-f32le-neyracompletion-v1"
COST_SCALE = 140
ARCHITECTURE = {"name": "neyra-completion-cnn-v1", "input": [1, 24, 10],
                "channels": [24, 32, 48], "kernel": 3, "padding": 1,
                "poolOutput": [4, 2], "poolKernel": [6, 5],
                "context": ["remainingSpins/40"], "hidden": 32, "heads": 1,
                "parameters": 33441}
LAYOUT = [("conv1.weight", 216), ("conv1.bias", 24),
          ("conv2.weight", 6912), ("conv2.bias", 32),
          ("conv3.weight", 13824), ("conv3.bias", 48),
          ("hidden.weight", 12320), ("hidden.bias", 32),
          ("output.weight", 32), ("output.bias", 1)]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def bounded_json(path):
    path = Path(path)
    if path.stat().st_size > MAX_RECORD_BYTES:
        raise ValueError("JSON artifact exceeds the bounded size")
    return json.loads(path.read_text(), object_pairs_hook=unique_keys, parse_constant=reject_constant)


def jsonl(path):
    path = Path(path)
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Dataset exceeds the bounded size")
    result = []
    with path.open("rb") as stream:
        while line := stream.readline(MAX_RECORD_BYTES + 1):
            if len(line) > MAX_RECORD_BYTES:
                raise ValueError("JSONL record exceeds the bounded size")
            if line.strip():
                result.append(json.loads(line, object_pairs_hook=unique_keys, parse_constant=reject_constant))
    return result


def weights_hash(weights):
    if not isinstance(weights, dict) or set(weights) != {name for name, _ in LAYOUT}:
        raise ValueError("Unexpected completion tensors")
    result = hashlib.sha256()
    for name, size in LAYOUT:
        values = weights[name]
        if not isinstance(values, list) or len(values) != size or any(
                type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 1000 for v in values):
            raise ValueError(f"Invalid completion tensor: {name}")
        result.update(struct.pack("<" + "f" * size, *values))
    return result.hexdigest()


def initialize(parent):
    parent = validate_model(parent)
    if parent["schemaVersion"] != 5 or parent["task"] != PRECISION_TASK:
        raise ValueError("Stage two requires an explicitly selected trained v5 parent")
    weights = {name: copy.deepcopy(parent["weights"][name]) for name, _ in LAYOUT if name.startswith("conv")}
    weights["hidden.weight"] = [value for out in range(32)
                                for value in [*parent["weights"]["hidden.weight"][out * 384:(out + 1) * 384], 0.0]]
    weights["hidden.bias"] = copy.deepcopy(parent["weights"]["hidden.bias"])
    weights["output.weight"], weights["output.bias"] = [0.0] * 32, [0.0]
    weight_hash = weights_hash(weights)
    return {"schemaVersion": 1, "task": TASK, "modelId": "naga-completion-v1-init-" + weight_hash[:12],
            "trained": False, "device": "cpu", "architecture": ARCHITECTURE,
            "rulesProfile": RULE_PROFILE, "observationPolicy": POLICY,
            "outputNames": ["observedRemainingLocksScaled"], "costScale": COST_SCALE,
            "weightsSha256Algorithm": HASH_ALGORITHM, "weightsSha256": weight_hash, "weights": weights,
            "initialization": {"kind": "v5-occupancy-features-completion-head-reset-v1",
                               "sourceModelId": parent["modelId"], "sourceWeightsSha256": parent["weightsSha256"],
                               "sourceTask": parent["task"], "reused": ["conv1", "conv2", "conv3", "hidden.occupancy"],
                               "reset": ["hidden.remainingSpins", "output"]},
            "searchOrderingEnabled": False, "learningComplete": False,
            "scope": "Untrained progress-conditioned completion head; existing one-bag outputs were not reinterpreted."}


def validate_completion_model(value):
    init = value.get("initialization", {})
    if (type(value.get("schemaVersion")) is not int or value["schemaVersion"] != 1
            or value.get("task") != TASK or value.get("architecture") != ARCHITECTURE
            or type(value.get("trained")) is not bool or value.get("device") != "cpu"
            or value.get("rulesProfile") != RULE_PROFILE or value.get("observationPolicy") != POLICY
            or value.get("outputNames") != ["observedRemainingLocksScaled"] or value.get("costScale") != COST_SCALE
            or value.get("searchOrderingEnabled") is not False
            or init.get("kind") != "v5-occupancy-features-completion-head-reset-v1"
            or init.get("sourceTask") != PRECISION_TASK
            or init.get("reused") != ["conv1", "conv2", "conv3", "hidden.occupancy"]
            or init.get("reset") != ["hidden.remainingSpins", "output"]):
        raise ValueError("Invalid completion task, progress, rule or initialization contract")
    for identifier in (value.get("modelId"), init.get("sourceModelId")):
        if not isinstance(identifier, str) or not 1 <= len(identifier) <= 128:
            raise ValueError("Invalid completion model identity")
    for key in (init.get("sourceWeightsSha256"), value.get("weightsSha256")):
        if not isinstance(key, str) or not HASH.fullmatch(key):
            raise ValueError("Invalid completion provenance digest")
    if value.get("weightsSha256Algorithm") != HASH_ALGORITHM or weights_hash(value.get("weights")) != value["weightsSha256"]:
        raise ValueError("Completion model digest mismatch")
    if value["trained"]:
        if not isinstance(value.get("datasetSha256"), str) or not HASH.fullmatch(value["datasetSha256"]):
            raise ValueError("Trained completion model needs its exact dataset digest")
    elif (any(value["weights"]["output.weight"] + value["weights"]["output.bias"])
          or any(value["weights"]["hidden.weight"][out * 385 + 384] for out in range(32))):
        raise ValueError("Untrained completion context and output weights must be reset")
    return value


def validate_target(record):
    if (record.get("schemaVersion") != 1 or record.get("kind") != "completion-cost-target"
            or record.get("task") != TASK or record.get("policy") != POLICY
            or record.get("rulesProfile") != RULE_PROFILE):
        raise ValueError("Invalid completion target task or rules")
    validate_sample(record.get("input"))
    source = record["input"]["source"]
    if source["kind"] != "search" or source.get("split") not in {"train", "validation", "test"} or not source.get("group"):
        raise ValueError("Completion labels need durable search family/group/split provenance")
    completed, remaining = record.get("completedSpins"), record.get("remainingSpins")
    cost = record.get("targets", {}).get("observedRemainingLocks")
    evidence = record.get("evidence", {})
    if (type(completed) is not int or not 0 <= completed < 40 or remaining != 40 - completed
            or type(cost) is not int or not remaining <= cost <= 700
            or evidence.get("physicalReplay") is not True or evidence.get("representativeSupplyVerified") is not True
            or evidence.get("allSuppliesVerified") is not False):
        raise ValueError("Completion labels require an actual complete representative path")
    for key in (evidence.get("proofSha256"), record.get("collectionPolicySha256")):
        if not isinstance(key, str) or not HASH.fullmatch(key):
            raise ValueError("Missing immutable completion proof/policy identity")
    return record


def selection_key(record):
    return (canonical_rows(record["input"]["rows"]), record["remainingSpins"],
            record["rulesProfile"], record["collectionPolicySha256"])
