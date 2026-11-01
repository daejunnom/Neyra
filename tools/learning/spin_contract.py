"""Versioned contracts for occupancy-only, measured-spin learning.

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
EXPLORATION_POLICY = "neyra-seeded-bag-continuation-v2"
NEXT_BAG_TASK = "naga-next-bag-spin-v3"
NEXT_BAG_POLICY = "bounded-next-bag-spin-v3"
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
    if "group" in source:
        text_value(source["group"], "source.group", 256)
    if "split" in source and source["split"] not in {"train", "validation", "test"}:
        raise ValueError("Invalid assigned source split")
    return value


def validate_config(config, observation_version=2):
    if not isinstance(config, dict):
        raise ValueError("Missing labeling config")
    required = {"horizonBags", "workBudget", "beamWidth", "profile", "seed",
                "maxNodes", "attemptsPerNode", "sourceRevision"}
    if observation_version == 3:
        required.add("observationVersion")
    if set(config) != required or config.get("observationVersion", 2) != observation_version:
        raise ValueError("Labeling policy must declare its exact versioned budget fields")
    integer(config["horizonBags"], "horizonBags", 2, 3)
    integer(config["workBudget"], "workBudget", 1, 4194304)
    integer(config["beamWidth"], "beamWidth", 4, 128)
    integer(config["profile"], "profile", 0, 3)
    integer(config["seed"], "seed", 0, 2**64 - 1)
    integer(config["maxNodes"], "maxNodes", 1, 256)
    if observation_version == 3 and config["maxNodes"] < config["horizonBags"]:
        raise ValueError("Continuation node capacity must cover its requested horizon")
    integer(config["attemptsPerNode"], "attemptsPerNode", 1, 16)
    text_value(config["sourceRevision"], "sourceRevision", 256)
    return config


def validate_label(value):
    if isinstance(value, dict) and "measurementGroup" in value:
        text_value(value["measurementGroup"], "measurementGroup", 256)
    if isinstance(value, dict) and value.get("schemaVersion") == 3:
        return validate_next_bag_label(value)
    return validate_v2_label(value)


def validate_v2_label(value):
    if not isinstance(value, dict) or value.get("schemaVersion") != 2 or value.get("task") != TASK:
        raise ValueError("Only newly collected spin-only schemaVersion 2 labels are accepted")
    reject_legacy(value)
    if value.get("policy") == POLICY:
        if "exploration" in value:
            raise ValueError("A fixed-policy label cannot carry an exploration seed")
    elif value.get("policy") == EXPLORATION_POLICY:
        exploration = value.get("exploration")
        if not isinstance(exploration, dict) or set(exploration) != {"schemaVersion", "seed", "ordinal", "scope"}:
            raise ValueError("Seeded observations require the exact exploration metadata")
        integer(exploration["schemaVersion"], "exploration.schemaVersion", 1, 1)
        if exploration["scope"] != "proposal-order-only":
            raise ValueError("Unsupported exploration semantics")
        integer(exploration["seed"], "exploration.seed", 0, 2**32 - 1)
        integer(exploration["ordinal"], "exploration.ordinal", 0, 2**32 - 1)
    else:
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


PIECES = ("I", "J", "L", "O", "S", "T", "Z")
ACTIONS = {"left", "right", "cw", "ccw", "half", "sonic", "hard_drop"}


def validate_frames(frames, targets, start_rows, start_spins, prior_bags, count):
    """Validate supplied geometry/accounting only; this is not SRS input replay."""
    if not isinstance(frames, list) or len(frames) != count or not isinstance(targets, list) or len(targets) != count:
        raise ValueError("Actual locks and typed logical targets must have the declared count")
    board, spins, deleted, inventory = list(start_rows), start_spins, 0, set()
    for index, (frame, target) in enumerate(zip(frames, targets)):
        if not isinstance(frame, dict) or not isinstance(target, dict):
            raise ValueError("Invalid physical frame or logical target")
        piece = frame.get("piece")
        if piece not in PIECES or piece in inventory or target.get("piece") != piece:
            raise ValueError("A physical bag must use distinct matching piece labels")
        inventory.add(piece)
        lines = integer(frame.get("lines"), "frame lines", 0, 1)
        if type(frame.get("spin")) is not bool or (lines and (not frame["spin"] or piece == "O")):
            raise ValueError("Only non-O Spin Singles are scoring placements")
        cells, logical = frame.get("placedCells"), target.get("cells")
        if not isinstance(cells, list) or len(cells) != 4 or not isinstance(logical, list) or len(logical) != 4:
            raise ValueError("A mino and its logical target require four distinct cells")
        remaining_rows = [y for y in range(32) if not (deleted & (1 << y))]
        normalized, occupied = [], set()
        for cell in cells:
            if not isinstance(cell, list) or len(cell) != 2:
                raise ValueError("Invalid physical cell")
            x, y = integer(cell[0], "cell x", 0, 9), integer(cell[1], "cell y", 0, 23)
            if (x, y) in occupied or board[y] & (1 << x) or y >= len(remaining_rows):
                raise ValueError("Physical cells overlap or exceed logical-row scope")
            occupied.add((x, y)); normalized.append((x, remaining_rows[y]))
            board[y] |= 1 << x
        logical_cells = []
        for cell in logical:
            if not isinstance(cell, list) or len(cell) != 2:
                raise ValueError("Invalid logical cell")
            logical_cells.append((integer(cell[0], "logical x", 0, 9), integer(cell[1], "logical y", 0, 31)))
        if sorted(normalized) != sorted(logical_cells):
            raise ValueError("Typed target lost its exact deleted-row context")
        full = [y for y, row in enumerate(board) if row == 1023]
        if len(full) != lines:
            raise ValueError("Reported clears disagree with the supplied physical cells")
        for y in full:
            deleted |= 1 << remaining_rows[y]
        board = [row for row in board if row != 1023] + [0] * len(full)
        spins += lines
        validate_rows(frame.get("rows"), nonempty=False)
        if (frame["rows"] != board or frame.get("spins") != spins
                or frame.get("placed") != prior_bags * 7 + index + 1):
            raise ValueError("Physical board, spin and lock accounting is discontinuous")
    return board, spins, deleted


def validate_next_bag_label(value):
    if (value.get("schemaVersion") != 3 or value.get("task") != NEXT_BAG_TASK
            or value.get("policy") != NEXT_BAG_POLICY or value.get("targetHorizonBags") != 1
            or "exploration" in value):
        raise ValueError("V3 requires its separate fixed one-bag target policy")
    reject_legacy(value)
    validate_sample(value.get("input"))
    config = validate_config(value.get("config"), 3)
    targets, measured = value.get("targets"), value.get("measured")
    if not isinstance(targets, dict) or set(targets) != {"foundSpin", "observedSpins"}:
        raise ValueError("One-bag targets are foundSpin and observedSpins only")
    spins = integer(targets["observedSpins"], "observedSpins", 0, 6)
    if type(targets["foundSpin"]) is not bool or targets["foundSpin"] != (spins > 0):
        raise ValueError("One-bag discovery target disagrees with its count")
    if not isinstance(measured, dict) or measured.get("censored") is not True:
        raise ValueError("Bounded continuation observations must remain censored")
    maximum = integer(measured.get("maxReachableObservedSpins"), "reachable maximum", spins, config["horizonBags"] * 6)
    work = integer(measured.get("workUsed"), "workUsed", 0, config["workBudget"])
    generation = integer(measured.get("generationWork"), "generationWork", 0, work)
    verification = integer(measured.get("verificationWork"), "verificationWork", 0, work)
    if generation + verification != work:
        raise ValueError("Continuation work accounting does not sum to its total")
    if (value.get("certifiedWitnesses") != [] or measured.get("maxCertifiedObservedSpins") is not None
            or any(measured.get(key) != 0 for key in ("certifiedWholeBagChains", "productiveCertifiedWholeBagChains",
                "canonicalSuppliesChecked", "quietPolicyRejections", "supplyRejections", "graphOverflows"))):
        raise ValueError("Input-reachable evidence cannot claim P7 or quiet-safe coverage")
    witnesses, lookahead = value.get("reachableWholeBagWitnesses"), measured.get("lookahead")
    if (not isinstance(witnesses, list) or len(witnesses) > 32 or not isinstance(lookahead, dict)
            or lookahead.get("targetHorizonBags") != 1 or lookahead.get("requestedBags") != config["horizonBags"]
            or lookahead.get("reachableWholeBagChains") != len(witnesses)):
        raise ValueError("Lookahead evidence does not match its separate measurement horizon")
    capacity = integer(lookahead.get("witnessCapacity"), "witness capacity", 1, 32)
    if len(witnesses) > capacity or verification != len(witnesses) * 7:
        raise ValueError("Retained whole-bag witnesses disagree with replay capacity/work")
    by_id, deepest = {}, 0
    for witness in witnesses:
        if not isinstance(witness, dict):
            raise ValueError("Invalid reachable whole-bag witness")
        identity = integer(witness.get("id"), "reachable witness id", 1, 2**53 - 1)
        if (identity in by_id or witness.get("trust") != "input-reachable" or witness.get("wholeBag") is not True
                or any(key in witness for key in ("checked", "success", "quietSafeSupplies", "firstBagSymmetryAssumed"))):
            raise ValueError("Reachable witnesses require distinct identities and existence-only trust")
        parent_id = witness.get("parentId")
        if parent_id is not None:
            integer(parent_id, "reachable parent id", 1, 2**53 - 1)
        parent = by_id.get(parent_id) if parent_id is not None else None
        if parent_id is not None and parent is None:
            raise ValueError("Reachable parent must precede its child")
        prior_bags = parent["bags"] if parent else 0
        start, start_spins = (parent["endRows"], parent["endSpins"]) if parent else (value["input"]["rows"], 0)
        bags = integer(witness.get("bags"), "reachable witness bags", 1, config["horizonBags"])
        if bags != prior_bags + 1 or witness.get("startRows") != start or witness.get("startSpins") != start_spins:
            raise ValueError("Reachable witness parent board, depth or spin count mismatch")
        queue, inputs = witness.get("representativeQueue"), witness.get("representativeInputs")
        if (not isinstance(queue, list) or len(queue) != 7 or sorted(integer(q, "queue piece", 0, 6) for q in queue) != list(range(7))
                or not isinstance(inputs, list) or len(inputs) != 7):
            raise ValueError("A whole-bag witness needs one queue of all seven labels and seven input paths")
        for path in inputs:
            if (not isinstance(path, list) or not 1 <= len(path) <= 256 or path[-1] != "hard_drop"
                    or any(not isinstance(action, str) or action not in ACTIONS for action in path) or "hard_drop" in path[:-1]):
                raise ValueError("Invalid retained physical input path")
        board, end_spins, deleted = validate_frames(witness.get("frames"), witness.get("targets"), start, start_spins, prior_bags, 7)
        if ([PIECES[q] for q in queue] != [frame["piece"] for frame in witness["frames"]]
                or witness.get("endRows") != board or witness.get("endSpins") != end_spins
                or witness.get("deletedRows") != deleted or end_spins > maximum):
            raise ValueError("Reachable endpoint, queue or deletion context disagrees with its frames")
        deepest = max(deepest, bags); by_id[identity] = witness

    def prefix(raw, target=False):
        if raw is None:
            if target and spins:
                raise ValueError("Positive one-bag target needs its physical prefix")
            return 0, 0
        if not isinstance(raw, dict) or raw.get("certifiedParentId") is not None:
            raise ValueError("Reachable prefix cannot claim a certified parent")
        parent_id = raw.get("reachableParentId")
        if parent_id is not None:
            integer(parent_id, "reachable prefix parent id", 1, 2**53 - 1)
        parent = by_id.get(parent_id) if parent_id is not None else None
        if parent_id is not None and parent is None:
            raise ValueError("Reachable prefix parent is absent")
        prior_bags = parent["bags"] if parent else 0
        count = integer(raw.get("additionalPlacements"), "prefix placements", 1, 7)
        if raw.get("completedParentBags") != prior_bags or prior_bags >= config["horizonBags"] or (target and prior_bags != 0):
            raise ValueError("One-bag targets must begin at the supplied root")
        board, end_spins, _ = validate_frames(raw.get("frames"), raw.get("targets"),
            parent["endRows"] if parent else value["input"]["rows"], parent["endSpins"] if parent else 0, prior_bags, count)
        if raw.get("observedSpins") != end_spins or end_spins > maximum or (target and end_spins != spins):
            raise ValueError("Reachable prefix disagrees with its measured target")
        return prior_bags * 7 + count, end_spins

    prefix(value.get("targetWitness"), True)
    reached, _ = prefix(value.get("reachableWitness"))
    if (lookahead.get("deepestCompletedBag") != deepest
            or lookahead.get("deepestReachedPlacements") != max(deepest * 7, reached)):
        raise ValueError("Observed continuation depth disagrees with retained physical evidence")
    integer(lookahead.get("deepestAttemptedBag"), "deepest attempted bag", deepest, config["horizonBags"])
    for key in ("capacityDrops", "duplicateContinuations"):
        integer(lookahead.get(key), key, 0, 2**53 - 1)
    best = lookahead.get("bestTwoBagPath")
    if best is not None:
        if not isinstance(best, dict):
            raise ValueError("Invalid coherent two-bag path")
        end = by_id.get(integer(best.get("witnessId"), "two-bag witness id", 1, 2**53 - 1))
        first = by_id.get(end["parentId"]) if end else None
        if (not end or end["bags"] != 2 or not first or first["bags"] != 1
                or best.get("firstBagSpins") != first["endSpins"]
                or best.get("secondBagSpins") != end["endSpins"] - first["endSpins"]
                or best.get("totalSpins") != end["endSpins"]):
            raise ValueError("Two-bag gains must belong to a single retained parent chain")
    elif any(w["bags"] == 2 for w in witnesses):
        raise ValueError("Retained second-bag paths require coherent lookahead diagnostics")
    if measured.get("geometricWholeBagChains") != len(witnesses):
        raise ValueError("Geometric chain count disagrees with retained endpoints")
    depths = measured.get("depths")
    if not isinstance(depths, list) or len(depths) != config["horizonBags"]:
        raise ValueError("Missing continuation depth accounting")
    for bags, depth in enumerate(depths, 1):
        chosen = [w for w in witnesses if w["bags"] == bags]
        if (not isinstance(depth, dict) or depth.get("bags") != bags
                or depth.get("geometricWholeBagChains") != len(chosen)
                or depth.get("productiveGeometricWholeBagChains") != sum(w["endSpins"] > 0 for w in chosen)
                or depth.get("maxGeometricWholeBagSpins") != max((w["endSpins"] for w in chosen), default=None)
                or depth.get("certifiedWholeBagChains") != 0 or depth.get("productiveCertifiedWholeBagChains") != 0
                or depth.get("maxCertifiedSpins") is not None):
            raise ValueError("Per-depth counts mix reachable observations with certification")
    if measured.get("productiveGeometricWholeBagChains") != sum(w["endSpins"] > 0 for w in witnesses):
        raise ValueError("Productive continuation count is inconsistent")
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
    if not isinstance(value, dict):
        raise ValueError("Missing spin-only model")
    next_bag = value.get("schemaVersion") == 3
    if (next_bag and (value.get("task") != NEXT_BAG_TASK or value.get("targetHorizonBags") != 1)
            or not next_bag and (value.get("schemaVersion") != 2 or value.get("task") != TASK)):
        raise ValueError("Model schema, task and prediction horizon disagree")
    reject_legacy(value)
    if value.get("trained") is not True or value.get("device") != "cpu":
        raise ValueError("A trained CPU model is required")
    if value.get("architecture") != ARCHITECTURE:
        raise ValueError("Unexpected v2 architecture")
    if value.get("outputNames") != ["foundSpinLogit", "observedSpinsScaled"] or value.get("countScale") != 6:
        raise ValueError("Unexpected v2 output semantics")
    if value.get("searchOrderingEnabled") is not False:
        raise ValueError("Private checkpoints cannot enable public search ordering")
    text_value(value.get("modelId"), "modelId", 128)
    for key in ("datasetSha256", "weightsSha256"):
        if not isinstance(value.get(key), str) or not HASH.fullmatch(value[key]):
            raise ValueError(f"Invalid {key}")
    if value.get("datasetHashAlgorithm", "sha256-raw-jsonl-v1") not in {
            "sha256-raw-jsonl-v1", "sha256-neyra-canonical-projected-records-v1"}:
        raise ValueError("Unsupported dataset hash algorithm")
    lineage = value.get("trainingDatasetLineage")
    if lineage is not None:
        if (not isinstance(lineage, list) or not 1 <= len(lineage) <= 256
                or any(not isinstance(item, str) or not HASH.fullmatch(item) for item in lineage)
                or len(set(lineage)) != len(lineage) or value["datasetSha256"] not in lineage):
            raise ValueError("Invalid training dataset lineage")
    if value.get("weightsSha256Algorithm") != WEIGHTS_HASH_ALGORITHM:
        raise ValueError("Wrong weights hash algorithm")
    if weights_hash(value.get("weights")) != value["weightsSha256"]:
        raise ValueError("Model weights hash mismatch")
    config = validate_config(value.get("labelConfig"), 3 if next_bag else 2)
    if value.get("horizonBags") != config["horizonBags"] or value.get("workBudget") != config["workBudget"]:
        raise ValueError("Model scope disagrees with its labeling policy")
    if not next_bag and value.get("targetHorizonBags", config["horizonBags"]) != config["horizonBags"]:
        raise ValueError("Retained v2 outputs cannot be reinterpreted as one-bag predictions")
    return value


def prediction_horizon(model):
    """Keep measurement depth distinct from the model's supervised horizon."""
    return 1 if model["schemaVersion"] == 3 else model["horizonBags"]


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
