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
REFINED_TASK = "naga-next-bag-spin-v4"
CONCENTRATED_POLICY = "concentrated-next-bag-spin-v4"
PRECISION_TASK = "naga-next-bag-spin-v5"
PRECISION_POLICY = "precision-single-bag-spin-v5"
REFINED_TARGET_POLICY = "best-replayed-next-bag-v1"
TARGET_DERIVATION = {"schemaVersion": 1, "policy": REFINED_TARGET_POLICY,
                     "sourceObservationVersion": 4, "sourcePolicy": CONCENTRATED_POLICY,
                     "targetHorizonBags": 1}
PRECISION_TARGET_DERIVATION = {"schemaVersion": 1, "policy": REFINED_TARGET_POLICY,
                               "sourceObservationVersion": 5, "sourcePolicy": PRECISION_POLICY,
                               "targetHorizonBags": 1}
PRECISION_TRAINING_OBJECTIVE = {"schemaVersion": 1, "kind": "asymmetric-smooth-l1-v1",
                                "underestimateWeight": 2, "overestimateWeight": 1, "countScale": 6}
REFINED_DATA_HASH = "sha256-neyra-refined-records-v1"
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
    if observation_version in (3, 4, 5):
        required.add("observationVersion")
    if set(config) != required or config.get("observationVersion", 2) != observation_version:
        raise ValueError("Labeling policy must declare its exact versioned budget fields")
    integer(config["horizonBags"], "horizonBags", 1 if observation_version == 5 else 2,
            1 if observation_version == 5 else 3)
    integer(config["workBudget"], "workBudget", 1, 4194304)
    integer(config["beamWidth"], "beamWidth", 4, 128)
    integer(config["profile"], "profile", 0, 3)
    integer(config["seed"], "seed", 0, 2**64 - 1)
    integer(config["maxNodes"], "maxNodes", 1, 256)
    if observation_version == 3 and config["maxNodes"] < config["horizonBags"]:
        raise ValueError("Continuation node capacity must cover its requested horizon")
    if observation_version == 4 and (config["horizonBags"] != 2 or config["maxNodes"] < 4 or config["workBudget"] < 1024):
        raise ValueError("V4 requires two measured bags, at least four nodes and 1024 work units")
    integer(config["attemptsPerNode"], "attemptsPerNode", 1, 16)
    if observation_version == 5 and (config["workBudget"] < 1024 or config["maxNodes"] < 4 or config["attemptsPerNode"] != 1):
        raise ValueError("V5 requires one inventory, one precision pass, at least four nodes and 1024 work units")
    text_value(config["sourceRevision"], "sourceRevision", 256)
    return config


def validate_label(value):
    if isinstance(value, dict) and "measurementGroup" in value:
        text_value(value["measurementGroup"], "measurementGroup", 256)
    if isinstance(value, dict) and value.get("schemaVersion") == 3:
        return validate_next_bag_label(value)
    if isinstance(value, dict) and value.get("schemaVersion") == 4:
        return validate_next_bag_label(value, concentrated=True)
    if isinstance(value, dict) and value.get("schemaVersion") == 5:
        return validate_next_bag_label(value, concentrated=True, precision=True)
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


def validate_next_bag_label(value, *, concentrated=False, precision=False):
    version = 5 if precision else 4 if concentrated else 3
    task = PRECISION_TASK if precision else REFINED_TASK if concentrated else NEXT_BAG_TASK
    policy = PRECISION_POLICY if precision else CONCENTRATED_POLICY if concentrated else NEXT_BAG_POLICY
    if (value.get("schemaVersion") != version or value.get("task") != task
            or value.get("policy") != policy or value.get("targetHorizonBags") != 1
            or "exploration" in value):
        raise ValueError("V3 requires its separate fixed one-bag target policy")
    reject_legacy(value)
    validate_sample(value.get("input"))
    config = validate_config(value.get("config"), version)
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
    dependency = integer(measured.get("dependencyWork"), "dependencyWork", 0, work) if concentrated else 0
    if generation + verification + dependency != work:
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
    if (len(witnesses) > capacity or (not concentrated and verification != len(witnesses) * 7)
            or concentrated and capacity != min(config["maxNodes"], 32)):
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
                or witness.get("deletedRows") != deleted or end_spins > maximum
                or concentrated and (bags == 1 and end_spins > spins
                    or any(frame["piece"] == "O" and frame["spin"] for frame in witness["frames"]))):
            raise ValueError("Reachable endpoint, queue or deletion context disagrees with its frames")
        deepest = max(deepest, bags); by_id[identity] = witness

    def prefix(raw, target=False):
        if raw is None:
            if target and spins or concentrated and lookahead.get("deepestReachedPlacements", 0) > 0:
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
    integer(lookahead.get("deepestAttemptedBag"), "deepest attempted bag", max(1, deepest) if concentrated else deepest, config["horizonBags"])
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
    if precision:
        validate_precision_refinement(value, by_id)
    elif concentrated:
        validate_refinement(value, by_id)
    return value


def exact_object(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{name} has unexpected or missing fields")
    return value


def validate_local_prefix(value, start):
    if value is None:
        return 0, 0
    exact_object(value, ("placements", "observedSpins", "deletedRows", "targets", "representativeQueue",
                         "representativeInputs", "frames", "trust"), "local prefix")
    count = integer(value["placements"], "local placements", 1, 7)
    queue, inputs = value["representativeQueue"], value["representativeInputs"]
    if (value["trust"] != "input-reachable" or not isinstance(queue, list) or len(queue) != count
            or len({integer(q, "local piece", 0, 6) for q in queue}) != count
            or not isinstance(inputs, list) or len(inputs) != count):
        raise ValueError("Local evidence requires distinct typed pieces and physical input paths")
    for path in inputs:
        if (not isinstance(path, list) or not 1 <= len(path) <= 256 or path[-1] != "hard_drop"
                or "hard_drop" in path[:-1] or any(not isinstance(action, str) or action not in ACTIONS for action in path)):
            raise ValueError("Invalid local physical input path")
    _, spins, deleted = validate_frames(value["frames"], value["targets"], start, 0, 0, count)
    if (value["observedSpins"] != spins or value["deletedRows"] != deleted
            or [PIECES[q] for q in queue] != [frame["piece"] for frame in value["frames"]]
            or any(frame["piece"] == "O" and frame["spin"] for frame in value["frames"])):
        raise ValueError("Local one-bag evidence disagrees with its physical frames")
    return count, spins


def validate_refined_search(value, start, quota, attempts_limit):
    exact_object(value, ("allocatedWork", "workUsed", "generationWork", "verificationWork", "attempts", "targets",
                         "targetWitness", "deepestWitness", "deepestReachedPlacements", "completedWholeBagCandidates",
                         "distinctWholeBagCandidates", "censored", "stopReason"), "local observation")
    used = integer(value["workUsed"], "local work", 0, quota)
    generation = integer(value["generationWork"], "local generation", 0, used)
    verification = integer(value["verificationWork"], "local verification", 0, used)
    attempts = integer(value["attempts"], "local attempts", 0, attempts_limit)
    targets = exact_object(value["targets"], ("foundSpin", "observedSpins"), "local targets")
    spins = integer(targets["observedSpins"], "local spins", 0, 6)
    reached = integer(value["deepestReachedPlacements"], "local reached", 0, 7)
    completed = integer(value["completedWholeBagCandidates"], "local completed", 0, 2**53 - 1)
    integer(value["distinctWholeBagCandidates"], "local distinct", 0, completed)
    if (value["allocatedWork"] != quota or used != generation + verification or targets["foundSpin"] is not (spins > 0)
            or value["censored"] is not True or completed > 0 and reached != 7 or used > 0 and attempts == 0):
        raise ValueError("Local quotas, actual progress or targets disagree")
    text_value(value["stopReason"], "local stop reason", 100)
    target_count, target_spins = validate_local_prefix(value["targetWitness"], start)
    deepest_count, deepest_spins = validate_local_prefix(value["deepestWitness"], start)
    if (target_spins != spins or target_count > reached or deepest_count != reached or deepest_spins > spins
            or reached > 0 and target_count == 0):
        raise ValueError("Local best/deepest physical prefixes disagree with the target")
    return used, generation, verification, target_count + deepest_count


def validate_component_dependency(value, witness):
    exact_object(value, ("schemaVersion", "policy", "connectivity", "scope", "maxOrdersPerComponent", "workUsed",
                         "totalOrders", "testedOrders", "successfulOrders", "components"), "component dependency")
    if (value["schemaVersion"] != 1 or value["policy"] != "touching-components-origin-only-v1"
            or value["connectivity"] != "orthogonal-logical-target-cells" or value["maxOrdersPerComponent"] != 64
            or not isinstance(value["components"], list) or not 1 <= len(value["components"]) <= 7):
        raise ValueError("Unsupported dependency approximation")
    text_value(value["scope"], "dependency scope", 1000)
    targets = witness["targets"]
    adjacent = [set() for _ in targets]
    for i, a in enumerate(targets):
        for j, b in enumerate(targets[i + 1:], i + 1):
            if any(abs(x[0] - y[0]) + abs(x[1] - y[1]) == 1 for x in a["cells"] for y in b["cells"]):
                adjacent[i].add(j); adjacent[j].add(i)
    remaining, expected = set(range(len(targets))), []
    while remaining:
        stack, group = [remaining.pop()], []
        while stack:
            at = stack.pop(); group.append(targets[at]["piece"])
            following = adjacent[at] & remaining
            remaining -= following; stack.extend(following)
        expected.append(tuple(sorted(group)))
    actual, total, tested, success, work = [], 0, 0, 0, 0
    for component in value["components"]:
        exact_object(component, ("pieces", "totalOrders", "testedOrders", "successfulOrders", "interruptedOrders",
                                 "workUsed", "exhaustive"), "dependency component")
        pieces = component["pieces"]
        if (not isinstance(pieces, list) or not pieces or any(p not in PIECES for p in pieces)
                or len(set(pieces)) != len(pieces)):
            raise ValueError("Invalid dependency component pieces")
        factorial = math.factorial(len(pieces))
        n = integer(component["testedOrders"], "component tested orders", 0, min(64, factorial))
        s = integer(component["successfulOrders"], "component successful orders", 0, n)
        integer(component["interruptedOrders"], "component interrupted orders", 0, 64)
        w = integer(component["workUsed"], "component work", 0, 4096)
        if component["totalOrders"] != factorial or component["exhaustive"] is not (n == factorial):
            raise ValueError("Component observation coverage is incoherent")
        actual.append(tuple(sorted(pieces))); total += factorial; tested += n; success += s; work += w
    if (sorted(actual) != sorted(expected) or value["totalOrders"] != total or value["testedOrders"] != tested
            or value["successfulOrders"] != success or value["workUsed"] != work):
        raise ValueError("Dependency components must match actual touching targets and their independent totals")
    return work


def validate_refinement(label, by_id):
    refinement = exact_object(label.get("refinement"), ("schemaVersion", "policy", "selectionSeed", "selectionScoreMin",
        "selectionScoreMax", "selectionBandCount", "candidatePoolCapacity", "candidatePoolSize", "candidatePoolDrops",
        "candidateDuplicates", "armLimit", "armQuota", "dependencyWorkBudget", "dependencyWorkUsed", "proofCapacity",
        "proofCapacityDrops", "root", "candidates", "arms"), "refinement")
    config, measured = label["config"], label["measured"]
    dependency_budget = min(4096, config["workBudget"] // 32)
    quota = (config["workBudget"] - dependency_budget) // 4
    if (type(refinement["schemaVersion"]) is not int or refinement["schemaVersion"] != 1 or refinement["policy"] != "uniform-score-band-three-arm-equal-quota-v1"
            or refinement["armLimit"] != 3 or refinement["armQuota"] != quota
            or refinement["dependencyWorkBudget"] != dependency_budget
            or refinement["dependencyWorkUsed"] != measured["dependencyWork"]
            or refinement["proofCapacity"] != min(config["maxNodes"], 32)):
        raise ValueError("Refinement policy or reserved work differs from its fixed contract")
    integer(refinement["selectionSeed"], "selection seed", 0, 2**32 - 1)
    integer(refinement["dependencyWorkUsed"], "dependency work", 0, dependency_budget)
    capacity = integer(refinement["candidatePoolCapacity"], "candidate pool capacity", 1, 4096)
    pool = integer(refinement["candidatePoolSize"], "candidate pool size", 0, capacity)
    low, high = refinement["selectionScoreMin"], refinement["selectionScoreMax"]
    if (any(type(n) not in (float, int) or not math.isfinite(n) for n in (low, high)) or low > high
            or refinement["selectionBandCount"] != (1 if low == high else 4) or not pool and (low != 0 or high != 0)):
        raise ValueError("Invalid score-band sampling metadata")
    for key in ("candidatePoolDrops", "candidateDuplicates", "proofCapacityDrops"):
        integer(refinement[key], key, 0, 2**53 - 1)
    candidates, arms = refinement["candidates"], refinement["arms"]
    if (not isinstance(candidates, list) or len(candidates) != len(by_id) or not isinstance(arms, list)
            or len(arms) > min(3, pool)):
        raise ValueError("Retained candidates and local observation arms disagree")
    root = refinement["root"]
    used, generation, verification, prefix_work = validate_refined_search(root, label["input"]["rows"], quota, config["attemptsPerNode"])
    if root["targets"] != label["targets"] or verification != 7 * sum(w["parentId"] is None for w in by_id.values()) + prefix_work:
        raise ValueError("Root targets or input verification work disagree")
    selected = set()
    phase_best = {None: root["targets"]["observedSpins"]}
    for arm in arms:
        exact_object(arm, ("candidateWitnessId", "selectionStratum", "selectionStrata", "selectionPoolSize", "selectionIndex", "observation"), "refinement arm")
        identity = integer(arm["candidateWitnessId"], "arm candidate", 1, 2**53 - 1)
        candidate = by_id.get(identity)
        if candidate is None or candidate["bags"] != 1 or not any(candidate["endRows"]) or identity in selected:
            raise ValueError("Each local arm must start from a distinct actual first-bag endpoint")
        strata = integer(arm["selectionStrata"], "arm strata", 1, 4)
        integer(arm["selectionStratum"], "arm stratum", 0, strata - 1)
        size = integer(arm["selectionPoolSize"], "arm selection pool", 1, pool)
        integer(arm["selectionIndex"], "arm selection index", 0, size - 1)
        if strata != refinement["selectionBandCount"]:
            raise ValueError("Arm sampling does not match the recorded score bands")
        u, g, v, prefix_work = validate_refined_search(arm["observation"], candidate["endRows"], quota, config["attemptsPerNode"])
        if u == 0 or v != 7 * sum(w["parentId"] == identity for w in by_id.values()) + prefix_work:
            raise ValueError("An arm must have actual work and matching retained input evidence")
        phase_best[identity] = candidate["endSpins"] + arm["observation"]["targets"]["observedSpins"]
        if any(w["parentId"] == identity and w["endSpins"] > phase_best[identity] for w in by_id.values()):
            raise ValueError("An arm's local best cannot be below its retained whole-bag child gains")
        used += u; generation += g; verification += v; selected.add(identity)
    deepest = label.get("reachableWitness")
    if deepest is not None:
        parent = deepest.get("reachableParentId")
        if parent not in phase_best or deepest["observedSpins"] > phase_best[parent]:
            raise ValueError("The global deepest prefix cannot exceed its corresponding local phase best")
    dependency, seen = 0, set()
    for candidate in candidates:
        exact_object(candidate, ("witnessId", "structuralQuality", "dependency"), "refinement candidate")
        identity = integer(candidate["witnessId"], "candidate id", 1, 2**53 - 1)
        quality = candidate["structuralQuality"]
        if identity not in by_id or identity in seen or type(quality) not in (int, float) or not math.isfinite(quality):
            raise ValueError("Invalid current-field sampling metadata")
        seen.add(identity)
        if candidate["dependency"] is not None:
            if identity not in selected or by_id[identity]["bags"] != 1:
                raise ValueError("Only selected first-bag candidates carry dependency diagnostics")
            dependency += validate_component_dependency(candidate["dependency"], by_id[identity])
    if (dependency != refinement["dependencyWorkUsed"] or used + dependency != measured["workUsed"]
            or generation != measured["generationWork"] or verification != measured["verificationWork"]):
        raise ValueError("Refinement work totals do not sum to the original bounded observation")


def validate_precision_refinement(label, by_id):
    """V5 spends one root quota inside one inventory; diagnostics never add targets."""
    refinement = exact_object(label.get("refinement"), ("schemaVersion", "policy", "selectionSeed", "selectionScoreMin",
        "selectionScoreMax", "selectionBandCount", "candidatePoolCapacity", "candidatePoolSize", "candidatePoolDrops",
        "candidateDuplicates", "armLimit", "armQuota", "dependencyWorkBudget", "dependencyWorkUsed", "proofCapacity",
        "proofCapacityDrops", "root", "candidates", "arms", "intraBag"), "precision refinement")
    config, measured = label["config"], label["measured"]
    dependency_budget = min(4096, config["workBudget"] // 32)
    quota = config["workBudget"] - dependency_budget
    if (type(refinement["schemaVersion"]) is not int or refinement["schemaVersion"] != 2
            or refinement["policy"] != "spin-first-single-bag-transpositions-v1"
            or type(refinement["armLimit"]) is not int or refinement["armLimit"] != 0
            or type(refinement["armQuota"]) is not int or refinement["armQuota"] != 0 or refinement["arms"] != []
            or refinement["dependencyWorkBudget"] != dependency_budget
            or refinement["dependencyWorkUsed"] != measured["dependencyWork"]
            or refinement["proofCapacity"] != min(config["maxNodes"], 32)
            or refinement["candidatePoolCapacity"] != config["maxNodes"]):
        raise ValueError("V5 must spend its declared quota on one root and no successor arms")
    integer(refinement["selectionSeed"], "precision selection seed", 0, 2**32 - 1)
    integer(refinement["dependencyWorkUsed"], "precision dependency work", 0, dependency_budget)
    pool = integer(refinement["candidatePoolSize"], "precision candidate pool", 0, config["maxNodes"])
    low, high = refinement["selectionScoreMin"], refinement["selectionScoreMax"]
    if (any(type(n) not in (int, float) or not math.isfinite(n) for n in (low, high)) or low > high
            or refinement["selectionBandCount"] != (1 if low == high else 4) or not pool and (low != 0 or high != 0)):
        raise ValueError("Invalid precision diagnostic sampling metadata")
    for key in ("candidatePoolDrops", "candidateDuplicates", "proofCapacityDrops"):
        integer(refinement[key], key, 0, 2**53 - 1)
    root = refinement["root"]
    used, generation, verification, prefix_work = validate_refined_search(root, label["input"]["rows"], quota, 1)
    if (root["targets"] != label["targets"] or root["attempts"] != 1
            or verification != 7 * len(by_id) + prefix_work
            or any(w["bags"] != 1 or w["parentId"] is not None or w["startSpins"] != 0 for w in by_id.values())):
        raise ValueError("Precision targets and replay work must all belong to the supplied inventory")
    deepest = label.get("reachableWitness")
    if deepest is not None and (deepest.get("reachableParentId") is not None
            or deepest["observedSpins"] > root["targets"]["observedSpins"]):
        raise ValueError("A precision prefix cannot exceed its retained root maximum")
    candidates = refinement["candidates"]
    if not isinstance(candidates, list) or len(candidates) != len(by_id):
        raise ValueError("Precision candidates must match their retained physical proofs")
    dependency, diagnostics, seen = 0, 0, set()
    for candidate in candidates:
        exact_object(candidate, ("witnessId", "structuralQuality", "dependency"), "precision candidate")
        identity = integer(candidate["witnessId"], "precision candidate id", 1, 2**53 - 1)
        quality = candidate["structuralQuality"]
        if identity not in by_id or identity in seen or type(quality) not in (int, float) or not math.isfinite(quality):
            raise ValueError("Invalid precision diagnostic candidate")
        seen.add(identity)
        if candidate["dependency"] is not None:
            diagnostics += 1
            dependency += validate_component_dependency(candidate["dependency"], by_id[identity])
    if (diagnostics > 3 or dependency != refinement["dependencyWorkUsed"]
            or used + dependency != measured["workUsed"] or generation != measured["generationWork"]
            or verification != measured["verificationWork"]):
        raise ValueError("Precision work must include one root and at most three bounded diagnostics")
    stats = exact_object(refinement["intraBag"], ("schemaVersion", "policy", "expandedStates", "physicalTranspositions",
        "orderTranspositions", "dominatedStates", "spinBoundPruned", "frontierCapacity", "frontierDrops",
        "transpositionCapacity", "transpositionEvictions", "maximumPendingStates"), "precision intra-bag statistics")
    if (type(stats["schemaVersion"]) is not int or stats["schemaVersion"] != 1
            or stats["policy"] != "oriented-occupancy-remaining-labels-spin-dominance-v1"
            or stats["frontierCapacity"] != 126 * config["beamWidth"] * 5 + 1
            or stats["transpositionCapacity"] != config["maxNodes"] * 256):
        raise ValueError("Precision transposition capacity or identity policy mismatch")
    for key in set(stats) - {"schemaVersion", "policy"}:
        integer(stats[key], "intraBag." + key, 0, 2**53 - 1)
    if (stats["orderTranspositions"] > stats["physicalTranspositions"] or stats["expandedStates"] > generation
            or stats["maximumPendingStates"] > stats["frontierCapacity"]
            or measured["lookahead"]["duplicateContinuations"] != stats["physicalTranspositions"] + refinement["candidateDuplicates"]
            or measured["lookahead"]["capacityDrops"] != stats["frontierDrops"] + refinement["candidatePoolDrops"] + refinement["proofCapacityDrops"]):
        raise ValueError("Precision counters disagree with their bounded search accounting")


def validate_refined_record(value):
    """Validate the compact projection; evidence must also be resolved before fitting.

    The ID is an immutable Site identity, not a claim that Python reproduces
    JavaScript locale collation or floating-point JSON spellings. Download page
    hashes and the original proof independently protect the exact data used.
    """
    exact_object(value, ("schemaVersion", "kind", "id", "task", "policy", "targetHorizonBags", "input", "config",
                         "targets", "measurementScope", "measured", "evidence"), "refined target")
    reject_legacy(value)
    precision = value["task"] == PRECISION_TASK
    version = 5 if precision else 4
    if (type(value["schemaVersion"]) is not int or value["schemaVersion"] != 1 or value["kind"] != "refined-next-bag-target" or value["task"] not in {REFINED_TASK, PRECISION_TASK}
            or value["policy"] != REFINED_TARGET_POLICY or value["targetHorizonBags"] != 1
            or not isinstance(value["id"], str) or not HASH.fullmatch(value["id"])):
        raise ValueError("Invalid versioned refined target identity")
    validate_sample(value["input"]); config = validate_config(value["config"], version)
    source = value["input"]["source"]
    text_value(source.get("group"), "refined source group", 256)
    if source.get("split") not in {"train", "validation", "test"}:
        raise ValueError("Refined targets require their actual immutable source split")
    target = exact_object(value["targets"], ("foundSpin", "observedSpins"), "refined targets")
    scope = exact_object(value["measurementScope"], ("kind", "allocatedWork", "workUnit"), "refined scope")
    measured = exact_object(value["measured"], ("workUsed", "generationWork", "verificationWork", "deepestReachedPlacements",
        "completedWholeBagCandidates", "distinctWholeBagCandidates", "attempts", "censored", "stopReason"), "refined measurement")
    evidence = exact_object(value["evidence"], ("observationId", "observationPolicy", "artifact", "candidateWitnessId",
        "armIndex", "targetPath", "measurementGroup", "trust"), "refined evidence")
    spins = integer(target["observedSpins"], "refined spins", 0, 6)
    allocated = integer(scope["allocatedWork"], "refined allocation", 1, config["workBudget"])
    used = integer(measured["workUsed"], "refined actual work", 1, allocated)
    reached = integer(measured["deepestReachedPlacements"], "refined reached", 0, 7)
    if (target["foundSpin"] is not (spins > 0) or spins > reached or measured["censored"] is not True
            or scope["workUnit"] != ("single-bag-expansion-and-input-verification-v5" if precision else "bounded-generation-and-input-verification-v4")
            or scope["kind"] not in ({"root"} if precision else {"root", "candidate-arm"})
            or integer(measured["generationWork"], "refined generation", 0, used)
               + integer(measured["verificationWork"], "refined verification", 0, used) != used):
        raise ValueError("Refined local one-bag target or measured work is incoherent")
    completed = integer(measured["completedWholeBagCandidates"], "refined completed", 0, 2**53 - 1)
    integer(measured["distinctWholeBagCandidates"], "refined distinct", 0, completed)
    integer(measured["attempts"], "refined attempts", 1, 2**53 - 1)
    if precision and (measured["attempts"] != 1 or allocated != config["workBudget"] - min(4096, config["workBudget"] // 32)):
        raise ValueError("V5 refined targets retain exactly one root pass and its full declared quota")
    text_value(measured["stopReason"], "refined stop reason", 100)
    if completed > 0 and reached != 7:
        raise ValueError("A completed candidate requires an actually reached whole inventory")
    identity = evidence["observationId"]
    if (not isinstance(identity, str) or not HASH.fullmatch(identity)
            or evidence["observationPolicy"] != (PRECISION_POLICY if precision else CONCENTRATED_POLICY)
            or evidence["artifact"] != "observations/" + identity + ".json" or evidence["measurementGroup"] != source["group"]
            or evidence["trust"] not in {"client-reported", "owner-cpu-observation"}):
        raise ValueError("Refined target is missing its original observation provenance")
    if scope["kind"] == "root":
        if evidence["candidateWitnessId"] is not None or evidence["armIndex"] is not None or evidence["targetPath"] != "refinement.root.targetWitness":
            raise ValueError("Refined root evidence path is invalid")
    else:
        integer(evidence["candidateWitnessId"], "refined candidate witness", 1, 2**53 - 1)
        index = integer(evidence["armIndex"], "refined arm index", 0, 2)
        if evidence["targetPath"] != f"refinement.arms[{index}].observation.targetWitness":
            raise ValueError("Refined arm evidence path is invalid")
    return value


def validate_refined_evidence(record, label):
    """Resolve a validated target to a validated, physically oriented raw proof."""
    version = 5 if record["task"] == PRECISION_TASK else 4
    if label["schemaVersion"] != version or label["task"] != record["task"] or record["config"] != label["config"]:
        raise ValueError("Refined target and its raw observation policy disagree")
    evidence, source = record["evidence"], record["input"]["source"]
    origin = label["input"]["source"]
    if source["family"] != origin["family"] or any(key in origin and source.get(key) != origin[key] for key in ("group", "split", "runId")):
        raise ValueError("Refined target cannot change its raw measurement lineage or split")
    expected_trust = "client-reported" if label.get("measurementTrust") == "client-reported" else "owner-cpu-observation"
    if evidence["trust"] != expected_trust:
        raise ValueError("Refined target trust differs from its original observation")
    index = evidence["armIndex"]
    if index is None:
        observation, rows, boundary = label["refinement"]["root"], label["input"]["rows"], origin["bagBoundary"]
    else:
        arms = label["refinement"]["arms"]
        if index >= len(arms) or arms[index]["candidateWitnessId"] != evidence["candidateWitnessId"]:
            raise ValueError("Refined target points to a different local arm")
        witness = next((w for w in label["reachableWholeBagWitnesses"] if w["id"] == evidence["candidateWitnessId"]), None)
        if witness is None or witness["bags"] != 1:
            raise ValueError("Refined arm has no actual first-bag starting board")
        observation, rows, boundary = arms[index]["observation"], witness["endRows"], origin["bagBoundary"] + 1
    if (record["input"]["rows"] != rows or source["bagBoundary"] != boundary or record["targets"] != observation["targets"]
            or record["measurementScope"]["allocatedWork"] != observation["allocatedWork"]
            or any(observation.get(key) != value for key, value in record["measured"].items())):
        raise ValueError("Refined target must preserve the exact measured local scope and physical board")
    return record


def refined_selection_key(record):
    return (canonical_rows(record["input"]["rows"]), record["policy"],
            json.dumps(record["config"], sort_keys=True, separators=(",", ":")),
            json.dumps(record["measurementScope"], sort_keys=True, separators=(",", ":")))


def refined_rank(record):
    measured = record["measured"]
    return (record["targets"]["observedSpins"], measured["deepestReachedPlacements"],
            measured["completedWholeBagCandidates"], measured["distinctWholeBagCandidates"])


def read_proofs(path):
    records = read_jsonl(path, kind="proofs")
    by_id = {}
    for record in records:
        exact_object(record, ("id", "label"), "observation index")
        identity = record["id"]
        if not isinstance(identity, str) or not HASH.fullmatch(identity) or identity in by_id:
            raise ValueError("Proof index identities must be unique immutable observation digests")
        validate_label(record["label"])
        if record["label"]["schemaVersion"] not in (4, 5):
            raise ValueError("The refinement proof index contains a different observation task")
        by_id[identity] = record["label"]
    return by_id


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
    precision = value.get("schemaVersion") == 5
    refined = precision or value.get("schemaVersion") == 4
    next_bag = refined or value.get("schemaVersion") == 3
    if (next_bag and (value.get("task") != (PRECISION_TASK if precision else REFINED_TASK if refined else NEXT_BAG_TASK) or value.get("targetHorizonBags") != 1)
            or not next_bag and (value.get("schemaVersion") != 2 or value.get("task") != TASK)):
        raise ValueError("Model schema, task and prediction horizon disagree")
    if refined and value.get("targetDerivation") != (PRECISION_TARGET_DERIVATION if precision else TARGET_DERIVATION):
        raise ValueError("Refined models must name their separate replayed one-bag target derivation")
    if precision and value.get("trainingObjective") != PRECISION_TRAINING_OBJECTIVE:
        raise ValueError("V5 must retain its asymmetric spin-underestimation objective")
    if precision and (any(type(value["trainingObjective"][key]) is not int for key in
                         ("schemaVersion", "underestimateWeight", "overestimateWeight", "countScale"))
            or any(type(value["targetDerivation"][key]) is not int for key in
                   ("schemaVersion", "sourceObservationVersion", "targetHorizonBags"))):
        raise ValueError("V5 objective and derivation versions/counts must retain integer semantics")
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
    algorithms = {"sha256-raw-jsonl-v1", "sha256-neyra-canonical-projected-records-v1"}
    if refined:
        algorithms.add(REFINED_DATA_HASH)
    if value.get("datasetHashAlgorithm", "sha256-raw-jsonl-v1") not in algorithms:
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
    config = validate_config(value.get("labelConfig"), 5 if precision else 4 if refined else 3 if next_bag else 2)
    if value.get("horizonBags") != config["horizonBags"] or value.get("workBudget") != config["workBudget"]:
        raise ValueError("Model scope disagrees with its labeling policy")
    if not next_bag and value.get("targetHorizonBags", config["horizonBags"]) != config["horizonBags"]:
        raise ValueError("Retained v2 outputs cannot be reinterpreted as one-bag predictions")
    return value


def prediction_horizon(model):
    """Keep measurement depth distinct from the model's supervised horizon."""
    return 1 if model["schemaVersion"] in (3, 4, 5) else model["horizonBags"]


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
        if kind == "labels":
            validate_label(value)
        elif kind == "refined" or kind == "training" and isinstance(value, dict) and value.get("kind") == "refined-next-bag-target":
            validate_refined_record(value)
        elif kind == "training":
            validate_label(value)
        elif kind != "proofs":
            validate_sample(value)
        records.append(value)
    return records
