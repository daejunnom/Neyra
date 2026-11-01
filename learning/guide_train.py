#!/usr/bin/env python3
"""Train the optional fixed-bag guide with NumPy CPU or PyTorch CUDA.

This is an offline, owner-operated learning tool, with no network transport.
It consumes completed verifier measurements, not predicted success labels.
Generated datasets, checkpoints and models belong outside the source tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

INPUTS, HIDDEN, OUTPUTS = 547, 256, 3
PARAMETER_COUNT = 272643
SCHEMA = "neyra-guide-model-v1"
FEATURE_SCHEMA = "neyra-guide-fixed-bag547-v1"
RULES = "neyra-sonic-srsplus-24-clear-hold-first-reflection-v1"
ARCHITECTURE = "dense547x256-residual256x2-head3-f32-v1"
HOLD_POLICY = "ActiveFirst;EmptyBagHold;QuietLocksUseCurrent;DeferOnlyProductiveClearCandidates;HeldLocksClear;TerminalUnusedSwapAllowed;EndFlushRequiresAllNextSpawns"
PIECES = "IJLOSTZ"
SHAPES = {
    "inputWeight": (256, 547), "inputBias": (256,),
    "residualInWeight": (256, 256), "residualInBias": (256,),
    "residualOutWeight": (256, 256), "residualOutBias": (256,),
    "headWeight": (3, 256), "headBias": (3,),
}
MAX_BYTES = 128 * 1024 * 1024
np = None  # Imported after setting the requested CPU thread limit.


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class WireFloat(float):
    """Retain Rust's emitted number token for content-addressed GA evidence.

    Python and serde_json use different exponent formatting. Re-encoding a
    parsed float with json.dumps can therefore change an otherwise intact hash.
    """
    def __new__(cls, token):
        value = super().__new__(cls, token)
        if not math.isfinite(value):
            raise ValueError("Nonfinite GA evidence number")
        value.token = token
        return value


def parse_json(text, preserve_numbers=False):
    return json.loads(text, parse_float=WireFloat if preserve_numbers else float,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON {x}")))


def load_json(path, preserve_numbers=False):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError(f"Input exceeds 128 MiB: {path}")
    return parse_json(path.read_text(), preserve_numbers)


def evidence_bytes(value):
    """Sorted UTF-8 JSON with the original Rust float tokens, before expansion."""
    if isinstance(value, WireFloat):
        return value.token.encode()
    if isinstance(value, dict):
        return b"{" + b",".join(evidence_bytes(k) + b":" + evidence_bytes(value[k]) for k in sorted(value)) + b"}"
    if isinstance(value, list):
        return b"[" + b",".join(evidence_bytes(v) for v in value) + b"]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def expand_ga_evidence(value):
    """Validate then resolve the v3 proof -> episode -> state DAG, never cycles."""
    state = value.get("state")
    if not isinstance(state, dict):
        return value
    if "gaEvidence" not in state:
        pending = [state]
        while pending:
            node = pending.pop()
            if isinstance(node, dict):
                if {"$gaEpisode", "$gaProof"} & node.keys():
                    raise ValueError("GA evidence references have no table")
                pending.extend(v for v in node.values() if isinstance(v, (dict, list)))
            elif isinstance(node, list):
                pending.extend(v for v in node if isinstance(v, (dict, list)))
        return value
    if value.get("schemaVersion") != "neyra-ga-checkpoint-v1" or value.get("sourcePolicy") != "neyra-ga-sonic-full-locks-fixed-plan-v3":
        raise ValueError("Content-addressed GA evidence requires a v3 checkpoint")
    if value.get("checksum") != hashlib.sha256(evidence_bytes(state)).hexdigest():
        raise ValueError("GA checkpoint checksum mismatch; use the original native export")
    table = state["gaEvidence"]
    if (not isinstance(table, dict) or set(table) != {"schemaVersion", "episodes", "proofs"}
            or table["schemaVersion"] != "neyra-ga-evidence-v1"
            or not isinstance(table["episodes"], dict) or len(table["episodes"]) > 1024
            or not isinstance(table["proofs"], dict) or len(table["proofs"]) > 16384):
        raise ValueError("Invalid GA evidence schema or capacity")
    expanded_bytes = 0

    def resolve(node, episodes, proofs):
        nonlocal expanded_bytes
        if isinstance(node, list):
            return [resolve(v, episodes, proofs) for v in node]
        if not isinstance(node, dict):
            return node
        tags = {"$gaEpisode", "$gaProof"} & node.keys()
        if tags:
            if len(node) != 1:
                raise ValueError("Invalid GA evidence reference shape")
            tag = next(iter(tags))
            key = node[tag]
            entries = episodes if tag == "$gaEpisode" else proofs
            if not isinstance(key, str) or key not in entries:
                raise ValueError("Missing or nested GA evidence reference")
            item, size = entries[key]
            expanded_bytes += size
            if expanded_bytes > MAX_BYTES:
                raise ValueError("Expanded GA evidence exceeds 128 MiB")
            return item
        return {k: resolve(v, episodes, proofs) for k, v in node.items()}

    def checked_entries(entries, required, episodes, proofs):
        result = {}
        for key, item in entries.items():
            if (not isinstance(item, dict) or not required <= item.keys()
                    or key != hashlib.sha256(evidence_bytes(item)).hexdigest()):
                raise ValueError("Invalid content-addressed GA evidence")
            resolved = resolve(item, episodes, proofs)
            result[key] = (resolved, len(evidence_bytes(resolved)))
        return result

    proofs = checked_entries(table["proofs"], {"canonicalCosts", "effectiveCosts", "orderCodes", "representativeFrames", "targets", "bagIndex"}, {}, {})
    episodes = checked_entries(table["episodes"], {"bagProofs", "placements", "rawSpins", "rawPieces", "qualifiedFrames", "seed"}, {}, proofs)
    return {**value, "state": resolve({k: v for k, v in state.items() if k != "gaEvidence"}, episodes, proofs)}


def proof_array(value, name, maximum):
    """Validate bounded lossless proof arrays; old plain arrays stay supported."""
    if isinstance(value, dict):
        if set(value) != {"encoding", "runs"} or value["encoding"] != "rle-v1" or not isinstance(value["runs"], list) or len(value["runs"]) > 5040:
            raise ValueError(f"Invalid {name} encoding")
        decoded = []
        for run in value["runs"]:
            if not isinstance(run, list) or len(run) != 2:
                raise ValueError(f"Invalid {name} run")
            count = integer(run[0], f"{name} run length", 1, 5040 - len(decoded))
            decoded.extend([integer(run[1], name, 0, maximum)] * count)
        value = decoded
    if not isinstance(value, list) or len(value) != 5040:
        raise ValueError(f"A complete {name} needs 5040 entries")
    return [integer(x, name, 0, maximum) for x in value]


def save_json(path, value):
    """Replace one complete artifact atomically; never expose a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", delete=False) as f:
            temporary = Path(f.name)
            json.dump(value, f, separators=(",", ":"), allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return value


def normalized_input(value):
    required = {"startRows", "endRows", "targets", "startSpins", "endSpins", "startPieces", "terminal"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Guide input has missing or unknown fields")
    out = {}
    for key in ("startRows", "endRows"):
        rows = value[key]
        if not isinstance(rows, list) or len(rows) != 24:
            raise ValueError(f"{key} must have 24 rows")
        out[key] = [integer(x, key, 0, 1022) for x in rows]
    start = integer(value["startSpins"], "startSpins", 0, 39)
    end = integer(value["endSpins"], "endSpins", start, min(start + 6, 40))
    placed = integer(value["startPieces"], "startPieces", 0, 125)
    if placed % 7 or type(value["terminal"]) is not bool or value["terminal"] != (end == 40):
        raise ValueError("Invalid bag boundary or terminal marker")
    targets = value["targets"]
    if not isinstance(targets, list) or not 1 <= len(targets) <= 7 or (not value["terminal"] and len(targets) != 7):
        raise ValueError("A completed nonterminal plan needs all seven typed targets")
    occupied, labels, normalized = set(), set(), []
    for target in targets:
        if not isinstance(target, dict) or set(target) != {"piece", "cells"}:
            raise ValueError("Invalid typed target")
        piece, cells = target["piece"], target["cells"]
        if not isinstance(piece, str) or piece not in PIECES or len(piece) != 1 or piece in labels:
            raise ValueError("Invalid or repeated piece label")
        if not isinstance(cells, list) or len(cells) != 4:
            raise ValueError("A target needs four logical cells")
        converted = []
        for cell in cells:
            if not isinstance(cell, list) or len(cell) != 2:
                raise ValueError("Invalid target cell")
            cell = (integer(cell[0], "x", 0, 9), integer(cell[1], "y", 0, 31))
            if cell in occupied:
                raise ValueError("Repeated or overlapping target cell")
            occupied.add(cell)
            converted.append(list(cell))
        labels.add(piece)
        normalized.append({"piece": piece, "cells": sorted(converted)})
    out.update(targets=sorted(normalized, key=lambda t: PIECES.index(t["piece"])),
               startSpins=start, endSpins=end, startPieces=placed, terminal=value["terminal"])
    return out


def identities(value):
    # Alternate representatives of the same typed plan stay in one split.
    plan = {k: v for k, v in value.items() if k not in ("endRows", "endSpins")}
    context = {"rulesProfile": RULES, "holdPolicy": HOLD_POLICY}
    return digest({**context, "input": value}), digest({**context, "plan": plan})


def features(value):
    value = normalized_input(value)
    result = np.zeros(INPUTS, dtype=np.float32)
    for offset, key in ((0, "startRows"), (240, "endRows")):
        for y, row in enumerate(value[key]):
            for x in range(10):
                result[offset + y * 10 + x] = (row >> x) & 1
    for target in value["targets"]:
        p = PIECES.index(target["piece"])
        for i, (x, y) in enumerate(target["cells"]):
            result[480 + p * 8 + i * 2] = np.float32(x) / np.float32(9)
            result[480 + p * 8 + i * 2 + 1] = np.float32(y) / np.float32(31)
        result[536 + p] = 1
    result[543:] = [np.float32(value["startSpins"]) / np.float32(40),
                   np.float32(value["endSpins"]) / np.float32(40),
                   np.float32(value["startPieces"]) / np.float32(126), int(value["terminal"])]
    return result


def measured_record(raw):
    if not isinstance(raw, dict) or type(raw.get("checked")) is not int or raw.get("checked") != 5040:
        return None
    if raw.get("graphOverflow") is True or raw.get("status") in ("graph-capacity", "pending"):
        return None
    if raw.get("holdPolicy") != HOLD_POLICY:
        return None
    if "input" in raw:
        if raw.get("graphOverflow") is not False:
            return None
        value = normalized_input(raw["input"])
    else:
        if raw.get("status") not in ("complete", "quiet-policy-rejected"):
            return None
        if not all(k in raw for k in ("startRows", "endRows", "targets", "startSpins", "endSpins", "terminal", "bagIndex")):
            return None
        value = normalized_input({k: raw[k] for k in ("startRows", "endRows", "targets", "startSpins", "endSpins", "terminal")}
                                 | {"startPieces": (integer(raw["bagIndex"], "bagIndex", 1, 18) - 1) * 7})
        for key, maximum in (("canonicalCosts", 255), ("effectiveCosts", 255), ("orderCodes", 2**32 - 1)):
            if key in raw:
                proof_array(raw[key], key, maximum)
    quiet = integer(raw.get("quietSafeSupplies"), "quietSafeSupplies", 0, 5040)
    success = integer(raw.get("success"), "success", 0, 5040)
    if raw.get("successUpper", success) != success:
        return None
    if success > quiet:
        raise ValueError("Success count exceeds quiet-safe coverage")
    gain = value["endSpins"] - value["startSpins"]
    if raw.get("bagSpins", gain) != gain:
        raise ValueError("Measured bag spin count disagrees with input")
    sample, group = identities(value)
    return {"sampleId": sample, "groupId": group, "input": value,
            "checked": 5040, "graphOverflow": False, "quietSafeSupplies": quiet,
            "success": success, "bagSpins": gain, "holdPolicy": HOLD_POLICY}


def walk_records(value):
    stack = [value]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            record = measured_record(node)
            if record is not None:
                yield record
                continue
            stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
        elif isinstance(node, list):
            stack.extend(v for v in node if isinstance(v, (dict, list)))


def make_dataset(paths):
    records, sources, duplicates = {}, [], 0
    for path in paths:
        value = load_json(path, preserve_numbers=True)
        source_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        # The private workbench's download API wraps the opaque JSON string.
        # Parse only this named contract, never arbitrary strings in a file.
        if isinstance(value, dict) and isinstance(value.get("checkpointJson"), str):
            if len(value["checkpointJson"].encode()) > 32 * 1024 * 1024:
                raise ValueError("Wrapped GA checkpoint exceeds 32 MiB")
            value = parse_json(value["checkpointJson"], preserve_numbers=True)
        schema = value.get("schemaVersion") if isinstance(value, dict) else None
        if schema in ("neyra-guide-dataset-v1", "neyra-guide-measurements-v1"):
            if value.get("rulesProfile") != RULES:
                raise ValueError("Source dataset uses another rules profile")
        elif schema in ("neyra-ga-checkpoint-v1", "neyra-ga-model-v1", "neyra-ga-evaluation-v1", "neyra-ga-training-report-v1"):
            if value.get("sourcePolicy") not in ("neyra-ga-sonic-full-locks-fixed-plan-v2", "neyra-ga-sonic-full-locks-fixed-plan-v3"):
                raise ValueError("Unsupported GA source policy; re-export measurements with the current verifier")
        else:
            raise ValueError("Expected a versioned GA checkpoint/model or guide measurement dataset")
        value = expand_ga_evidence(value)
        count = 0
        for record in walk_records(value):
            count += 1
            old = records.get(record["sampleId"])
            if old is not None:
                if old != record:
                    raise ValueError("Conflicting completed measurements for one exact typed context")
                duplicates += 1
            else:
                records[record["sampleId"]] = record
        sources.append({"sha256": source_hash, "completeRecords": count})
    rows = sorted(records.values(), key=lambda r: r["sampleId"])
    result = {"schemaVersion": "neyra-guide-dataset-v1", "rulesProfile": RULES,
              "records": rows, "sources": sources, "duplicateRecords": duplicates,
              "evidenceTrust": "Owner-supplied completed verifier records; checksums are not signatures and do not independently authenticate proofs."}
    result["datasetSha256"] = digest({"rulesProfile": RULES, "records": rows})
    return result


def validate_dataset(value):
    if value.get("schemaVersion") != "neyra-guide-dataset-v1" or value.get("rulesProfile") != RULES:
        raise ValueError("Incompatible dataset contract")
    records = []
    seen = set()
    for raw in value.get("records", []):
        record = measured_record(raw)
        if record is None:
            raise ValueError("Dataset contains an incomplete, overflowing, or incompatible measurement")
        if raw.get("sampleId", record["sampleId"]) != record["sampleId"] or raw.get("groupId", record["groupId"]) != record["groupId"]:
            raise ValueError("Dataset typed-context identity mismatch")
        if record["sampleId"] in seen:
            raise ValueError("Dataset is not deduplicated")
        seen.add(record["sampleId"])
        records.append(record)
    records.sort(key=lambda r: r["sampleId"])
    data_hash = digest({"rulesProfile": RULES, "records": records})
    if value.get("datasetSha256", data_hash) != data_hash:
        raise ValueError("Dataset content checksum mismatch")
    if len({r["groupId"] for r in records}) < 2:
        raise ValueError("Need at least two distinct typed-plan groups for disjoint training and validation")
    return records, data_hash


def weights_hash(parameters):
    h = hashlib.sha256()
    for key in SHAPES:
        h.update(np.asarray(parameters[key], dtype="<f4").tobytes(order="C"))
    return h.hexdigest()


def validate_parameters(parameters):
    if set(parameters) != set(SHAPES):
        raise ValueError("Unexpected weight arrays")
    for name, shape in SHAPES.items():
        a = parameters[name]
        if a.shape != shape or not np.isfinite(a).all() or (np.abs(a) > 32).any():
            raise ValueError(f"Invalid shape or nonfinite/out-of-range weight: {name}")


def model_parameters(model):
    if (model.get("schemaVersion") != SCHEMA or model.get("featureSchema") != FEATURE_SCHEMA
            or model.get("rulesProfile") != RULES or model.get("architecture") != ARCHITECTURE
            or model.get("parameterCount") != PARAMETER_COUNT or model.get("learningComplete") is not False
            or model.get("proofModel") is not False or set(model.get("weights", {})) != set(SHAPES)):
        raise ValueError("Incompatible guide model")
    parameters = {key: np.asarray(model["weights"][key], dtype=np.float32).reshape(shape) for key, shape in SHAPES.items()}
    validate_parameters(parameters)
    h = weights_hash(parameters)
    if model.get("weightsSha256") != h or model.get("modelId") != "guide-" + h[:20]:
        raise ValueError("Guide weight checksum mismatch")
    metadata = model.get("training", {})
    steps = integer(metadata.get("steps"), "steps", 0, 2**53 - 1)
    unique = integer(metadata.get("uniquePlans"), "uniquePlans", 0, 2**31 - 1)
    train = integer(metadata.get("trainGroups"), "trainGroups", 0, unique)
    valid = integer(metadata.get("validationGroups"), "validationGroups", 0, unique)
    if train + valid != unique or model.get("trained") != (steps > 0) or (steps > 0 and min(train, valid) == 0):
        raise ValueError("Invalid model training metadata")
    data_hash = metadata.get("datasetSha256")
    if (steps > 0 and data_hash is None) or (data_hash is not None and
            (not isinstance(data_hash, str) or len(data_hash) != 64 or any(c not in "0123456789abcdefABCDEF" for c in data_hash))):
        raise ValueError("Invalid training dataset checksum")
    return parameters


def initializer(seed):
    # Same xorshift32 and f32 arithmetic as GuideModel::initializer.
    state = max(seed & 0xffffffff, 1)
    parameters = {}
    for key, shape in SHAPES.items():
        if len(shape) == 1:
            parameters[key] = np.zeros(shape, dtype=np.float32)
            continue
        factor = 0.1 if key in ("residualOutWeight", "headWeight") else 1.0
        scale = np.sqrt(np.float32(6) / np.float32(sum(shape))) * np.float32(factor)
        values = np.empty(math.prod(shape), dtype=np.float32)
        for i in range(len(values)):
            state ^= (state << 13) & 0xffffffff
            state ^= state >> 17
            state ^= (state << 5) & 0xffffffff
            values[i] = np.float32(state / 4294967296.0 * 2 - 1) * scale
        parameters[key] = values.reshape(shape)
    return parameters


def numpy_forward(p, x):
    z0 = x @ p["inputWeight"].T + p["inputBias"]
    h0 = np.maximum(z0, 0)
    z1 = h0 @ p["residualInWeight"].T + p["residualInBias"]
    h1 = np.maximum(z1, 0)
    z2 = h0 + h1 @ p["residualOutWeight"].T + p["residualOutBias"]
    h2 = np.maximum(z2, 0)
    logits = h2 @ p["headWeight"].T + p["headBias"]
    predicted = np.exp(-np.logaddexp(np.float32(0), -logits))
    return predicted, (x, z0, h0, z1, h1, z2, h2, logits)


def numpy_gradient(p, x, y):
    predicted, (x, z0, h0, z1, h1, z2, h2, logits) = numpy_forward(p, x)
    loss = np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits))))
    d = (predicted - y) / np.float32(y.size)
    g = {"headWeight": d.T @ h2, "headBias": d.sum(axis=0)}
    d2 = (d @ p["headWeight"]) * (z2 > 0)
    g.update(residualOutWeight=d2.T @ h1, residualOutBias=d2.sum(axis=0))
    d1 = (d2 @ p["residualOutWeight"]) * (z1 > 0)
    g.update(residualInWeight=d1.T @ h0, residualInBias=d1.sum(axis=0))
    d0 = (d2 + d1 @ p["residualInWeight"]) * (z0 > 0)
    g.update(inputWeight=d0.T @ x, inputBias=d0.sum(axis=0))
    return float(loss), g


class Backend:
    """GPU work is speculative; canonical weights/Adam state stay on the CPU.

    A failed GPU forward/backward cannot partially advance the optimizer. The
    identical batch is recomputed on NumPy and the CPU search can keep running.
    """
    def __init__(self, requested):
        self.name, self.torch, self.events = "numpy-cpu", None, []
        self.gpu_steps, self.gpu_predictions = 0, 0
        if requested == "cpu":
            return
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable")
            torch.empty(1, device="cuda")
            torch.cuda.synchronize()
            self.torch, self.name = torch, "pytorch-cuda"
        except (ImportError, RuntimeError, OSError) as error:
            self.events.append({"kind": "cpu-fallback", "reason": str(error), "beforeStep": 0})

    def cuda(self, p, x, y=None):
        t = self.torch
        # CPU canonical state is not mutated. A device loss can safely discard
        # this whole transient computation without losing an optimizer step.
        params = {k: t.tensor(v, device="cuda", requires_grad=y is not None) for k, v in p.items()}
        tx = t.tensor(x, device="cuda")
        h = t.relu(tx @ params["inputWeight"].T + params["inputBias"])
        r = t.relu(h @ params["residualInWeight"].T + params["residualInBias"])
        h = t.relu(h + r @ params["residualOutWeight"].T + params["residualOutBias"])
        logits = h @ params["headWeight"].T + params["headBias"]
        if y is None:
            result = t.sigmoid(logits).detach().cpu().numpy()
            if not np.isfinite(result).all():
                raise FloatingPointError("Nonfinite CUDA prediction")
            self.gpu_predictions += 1
            return result
        loss = t.nn.functional.binary_cross_entropy_with_logits(logits, t.tensor(y, device="cuda"))
        loss.backward()
        result = {k: v.grad.detach().cpu().numpy().copy() for k, v in params.items()}
        number = float(loss.detach().cpu())
        if not math.isfinite(number) or any(not np.isfinite(g).all() for g in result.values()):
            raise FloatingPointError("Nonfinite CUDA loss or gradient")
        self.gpu_steps += 1
        return number, result

    def fallback(self, error, step):
        self.events.append({"kind": "cpu-fallback", "reason": str(error), "beforeStep": step})
        self.name, self.torch = "numpy-cpu", None

    def gradient(self, p, x, y, step):
        if self.torch is not None:
            try:
                return self.cuda(p, x, y)
            except (RuntimeError, FloatingPointError, OSError) as error:
                self.fallback(error, step)
        return numpy_gradient(p, x, y)

    def predict(self, p, x):
        if self.torch is not None:
            try:
                return self.cuda(p, x)
            except (RuntimeError, FloatingPointError, OSError) as error:
                self.fallback(error, 0)
        output = numpy_forward(p, x)[0]
        if not np.isfinite(output).all():
            raise FloatingPointError("Nonfinite CPU prediction")
        return output


def adam_candidate(p, m, v, gradient, step, rate):
    if any(not np.isfinite(g).all() for g in gradient.values()):
        raise FloatingPointError("Nonfinite gradient; canonical optimizer state is unchanged")
    norm = math.sqrt(sum(float(np.sum(g.astype(np.float64) ** 2)) for g in gradient.values()))
    scale = np.float32(min(1, 1 / max(norm, 1e-12)))
    new_p, new_m, new_v = {}, {}, {}
    for key in SHAPES:
        g = gradient[key] * scale
        new_m[key] = np.float32(0.9) * m[key] + np.float32(0.1) * g
        new_v[key] = np.float32(0.999) * v[key] + np.float32(0.001) * (g * g)
        mean = new_m[key] / np.float32(1 - 0.9**step)
        variance = new_v[key] / np.float32(1 - 0.999**step)
        new_p[key] = p[key] - np.float32(rate) * mean / (np.sqrt(variance) + np.float32(1e-8))
    validate_parameters(new_p)
    if any(not np.isfinite(a).all() for a in (*new_m.values(), *new_v.values())):
        raise FloatingPointError("Nonfinite optimizer update; prior state is unchanged")
    return new_p, new_m, new_v


def export_model(p, step, groups, validation_groups, data_hash):
    h = weights_hash(p)
    return {"schemaVersion": SCHEMA, "featureSchema": FEATURE_SCHEMA, "rulesProfile": RULES,
            "architecture": ARCHITECTURE, "parameterCount": PARAMETER_COUNT,
            "modelId": "guide-" + h[:20], "weightsSha256": h,
            "trained": step > 0, "learningComplete": False, "proofModel": False,
            "training": {"steps": step, "uniquePlans": groups, "trainGroups": groups - validation_groups,
                         "validationGroups": validation_groups, "datasetSha256": data_hash},
            "weights": {k: p[k].ravel().tolist() for k in SHAPES}}


def train(args):
    records, data_hash = validate_dataset(load_json(args.data))
    backend = Backend(args.device)
    defaults = {"seed": 1, "batchSize": 32, "learningRate": 0.001, "validationPercent": 20}
    supplied = {"seed": args.seed, "batchSize": args.batch_size, "learningRate": args.learning_rate,
                "validationPercent": args.validation_percent}
    resumed = load_json(args.resume) if args.resume else None
    if resumed:
        if resumed.get("schemaVersion") != "neyra-guide-training-checkpoint-v1" or resumed.get("datasetSha256") != data_hash:
            raise ValueError("Resume requires the identical dataset and checkpoint contract")
        config = resumed["config"]
        if set(config) != set(defaults) or any(v is not None and v != config[k] for k, v in supplied.items()):
            raise ValueError("Resume cannot change saved optimizer, split, or batch settings")
    else:
        config = {k: defaults[k] if v is None else v for k, v in supplied.items()}
    integer(config["seed"], "seed", 0, 2**32 - 1)
    integer(config["batchSize"], "batchSize", 1, 4096)
    if not math.isfinite(config["learningRate"]) or not 0 < config["learningRate"] <= 0.1:
        raise ValueError("learningRate must be in (0,0.1]")
    integer(config["validationPercent"], "validationPercent", 1, 50)
    groups = sorted({r["groupId"] for r in records}, key=lambda g: hashlib.sha256(f"{config['seed']}:{g}".encode()).digest())
    count = min(len(groups) - 1, max(1, math.ceil(len(groups) * config["validationPercent"] / 100)))
    validation = set(groups[:count])
    train_indices = np.asarray([i for i, r in enumerate(records) if r["groupId"] not in validation], dtype=np.int64)
    valid_indices = np.asarray([i for i, r in enumerate(records) if r["groupId"] in validation], dtype=np.int64)
    x = np.stack([features(r["input"]) for r in records])
    y = np.asarray([[r["quietSafeSupplies"] / 5040, r["success"] / 5040, r["bagSpins"] / 6] for r in records], dtype=np.float32)
    rng = np.random.default_rng(config["seed"])
    if resumed:
        p = model_parameters(resumed["model"])
        step = resumed["model"]["training"]["steps"]
        meta = resumed["model"]["training"]
        if (resumed.get("validationGroups") != sorted(validation) or meta["datasetSha256"] != data_hash
                or meta["uniquePlans"] != len(groups) or meta["validationGroups"] != count
                or meta["trainGroups"] != len(groups) - count):
            raise ValueError("Resume split mismatch")
        m = {k: np.asarray(resumed["adamM"][k], dtype=np.float32).reshape(s) for k, s in SHAPES.items()}
        v = {k: np.asarray(resumed["adamV"][k], dtype=np.float32).reshape(s) for k, s in SHAPES.items()}
        if any(not np.isfinite(a).all() for a in (*m.values(), *v.values())) or any((a < 0).any() for a in v.values()):
            raise ValueError("Invalid saved Adam state")
        rng.bit_generator.state = resumed["rngState"]
        permutation = np.asarray([integer(i, "permutation index", 0, len(records) - 1)
                                  for i in resumed["permutation"]], dtype=np.int64)
        cursor = integer(resumed["cursor"], "cursor", 0, len(permutation))
        if sorted(permutation.tolist()) != sorted(train_indices.tolist()):
            raise ValueError("Saved training order does not match the dataset split")
        epoch = integer(resumed["epoch"], "epoch", 0, 2**53 - 1)
    else:
        p = model_parameters(load_json(args.model)) if args.model else initializer(config["seed"])
        m = {k: np.zeros(s, dtype=np.float32) for k, s in SHAPES.items()}
        v = {k: np.zeros(s, dtype=np.float32) for k, s in SHAPES.items()}
        step, epoch, cursor = 0, 0, 0
        permutation = rng.permutation(train_indices)
    start_step, start = step, time.monotonic()
    history = []

    def metrics(indices):
        totals = np.zeros(3, dtype=np.float64)
        for begin in range(0, len(indices), 256):
            selected = indices[begin:begin + 256]
            prediction = numpy_forward(p, x[selected])[0]
            totals += np.abs(prediction - y[selected]).sum(axis=0)
        return (totals / len(indices)).tolist()

    initial_train_mae, initial_validation_mae = metrics(train_indices), metrics(valid_indices)
    train_mean = y[train_indices].mean(axis=0)
    mean_baseline_validation = np.abs(y[valid_indices] - train_mean).mean(axis=0).tolist()

    def coverage(indices):
        bags = {}
        for i in indices:
            bag = str(records[int(i)]["input"]["startPieces"] // 7 + 1)
            bags[bag] = bags.get(bag, 0) + 1
        return {"startBagCounts": bags,
                "quietSafe5040": sum(records[int(i)]["quietSafeSupplies"] == 5040 for i in indices),
                "success5040": sum(records[int(i)]["success"] == 5040 for i in indices),
                "bagSpinCounts": {str(g): sum(records[int(i)]["bagSpins"] == g for i in indices) for g in range(7)}}

    def save():
        model = export_model(p, step, len(groups), count, data_hash)
        checkpoint = {"schemaVersion": "neyra-guide-training-checkpoint-v1", "datasetSha256": data_hash,
                      "config": config, "model": model, "adamM": {k: a.ravel().tolist() for k, a in m.items()},
                      "adamV": {k: a.ravel().tolist() for k, a in v.items()}, "rngState": rng.bit_generator.state,
                      "permutation": permutation.tolist(), "cursor": cursor, "epoch": epoch,
                      "validationGroups": sorted(validation)}
        # The resumable checkpoint is the authority; the inference model is a
        # derivative. A crash between replacements cannot corrupt the checkpoint.
        save_json(args.checkpoint, checkpoint)
        save_json(args.output, model)
        report = {"schemaVersion": "neyra-guide-training-report-v1", "datasetSha256": data_hash,
                  "modelId": model["modelId"], "parameterCount": PARAMETER_COUNT, "steps": step,
                  "newSteps": step - start_step, "records": len(records), "planGroups": len(groups),
                  "trainRecords": len(train_indices), "validationRecords": len(valid_indices),
                  "trainGroups": len(groups) - count, "validationGroups": count,
                  "backend": backend.name, "backendEvents": backend.events, "gpuExecuted": backend.gpu_steps > 0,
                  "gpuSuccessfulSteps": backend.gpu_steps,
                  "trainMae": metrics(train_indices), "validationMae": metrics(valid_indices),
                  "initialTrainMae": initial_train_mae, "initialValidationMae": initial_validation_mae,
                  "trainMeanTarget": train_mean.tolist(), "trainMeanBaselineValidationMae": mean_baseline_validation,
                  "trainCoverage": coverage(train_indices), "validationCoverage": coverage(valid_indices),
                  "headOrder": ["quietSafeFraction", "constructionFraction", "bagSpinsFraction"],
                  "elapsedMs": (time.monotonic() - start) * 1000, "history": history,
                  "learningComplete": False, "proof": False,
                  "limitation": "Grouped held-out label prediction is not a 40L completion, calibrated guarantee, or search-strength measurement."}
        if args.report:
            save_json(args.report, report)
        return report

    try:
        while step - start_step < args.steps and (args.seconds == 0 or time.monotonic() - start < args.seconds):
            if cursor == len(permutation):
                permutation, cursor, epoch = rng.permutation(train_indices), 0, epoch + 1
            end = min(cursor + config["batchSize"], len(permutation))
            batch = permutation[cursor:end]
            loss, gradients = backend.gradient(p, x[batch], y[batch], step)
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; previous optimizer state retained")
            candidate = adam_candidate(p, m, v, gradients, step + 1, config["learningRate"])
            # One atomic logical commit after all weights and moments validated.
            p, m, v = candidate
            step, cursor = step + 1, end
            if step % args.save_every == 0:
                history.append({"step": step, "loss": loss})
                save()
    except (KeyboardInterrupt, Exception):
        save()
        raise
    report = save()
    print(json.dumps({k: report[k] for k in ("modelId", "steps", "newSteps", "records", "planGroups", "backend", "backendEvents", "trainMae", "validationMae", "elapsedMs", "proof")}, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="Deduplicate completed GA/Search verifier measurements")
    extract.add_argument("--source", action="append", required=True)
    extract.add_argument("--output", required=True)
    fit = sub.add_parser("train", help="Fit a guide; CUDA failures recompute the batch on NumPy CPU")
    fit.add_argument("--data", required=True)
    fit.add_argument("--model")
    fit.add_argument("--resume")
    fit.add_argument("--output", required=True)
    fit.add_argument("--checkpoint", required=True)
    fit.add_argument("--report")
    fit.add_argument("--steps", type=int, default=100)
    fit.add_argument("--seconds", type=float, default=0)
    fit.add_argument("--seed", type=int)
    fit.add_argument("--batch-size", type=int)
    fit.add_argument("--learning-rate", type=float)
    fit.add_argument("--validation-percent", type=int)
    fit.add_argument("--save-every", type=int, default=100)
    predict = sub.add_parser("predict", help="Run identical trained weights using optional CUDA or CPU")
    predict.add_argument("--model", required=True)
    predict.add_argument("--input", required=True)
    predict.add_argument("--output")
    for p in (fit, predict):
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        p.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    threads = getattr(args, "cpu_threads", 2)
    integer(threads, "cpuThreads", 1, 64)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)
    global np
    import numpy as np
    if args.command == "extract":
        result = make_dataset(args.source)
        save_json(args.output, result)
        print(json.dumps({"records": len(result["records"]), "groups": len({r["groupId"] for r in result["records"]}),
                          "duplicates": result["duplicateRecords"], "datasetSha256": result["datasetSha256"]}))
    elif args.command == "train":
        integer(args.steps, "steps", 1, 10000000)
        integer(args.save_every, "saveEvery", 1, 1000000)
        if not math.isfinite(args.seconds) or args.seconds < 0 or (args.resume and args.model):
            raise ValueError("Invalid runtime budget or model/resume combination")
        train(args)
    else:
        model = load_json(args.model)
        p, backend = model_parameters(model), Backend(args.device)
        output = backend.predict(p, features(load_json(args.input))[None, :])[0]
        result = {"modelId": model["modelId"], "quietSafeFraction": float(output[0]),
                  "constructionFraction": float(output[1]), "bagSpinsFraction": float(output[2]),
                  "priority": float(2 * float(output[0]) + float(output[1]) + 0.25 * float(output[2])),
                  "trained": model["trained"], "proof": False, "backend": backend.name, "backendEvents": backend.events}
        if args.output:
            save_json(args.output, result)
        else:
            print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"guide_train: {error}", file=sys.stderr)
        sys.exit(1)
