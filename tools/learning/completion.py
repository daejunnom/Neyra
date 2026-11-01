"""Initialize, verify and train the distinct progress-conditioned completion CNN.

CPU only. Every training target is rederived with Neyra's actual Rust input and
hold replay. Genuine completed paths are required; no censor-to-exact conversion.
No command overwrites the public model or automatically enables search ordering.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from completion_contract import (ARCHITECTURE, COST_SCALE, POLICY, RULE_PROFILE, TASK,
                                 bounded_json, canonical, digest, initialize, jsonl, selection_key,
                                 validate_completion_model, validate_target, weights_hash)
from spin_contract import MAX_FILE_BYTES, canonical_rows, read_proofs, validate_model


def write_new(path, value, *, lines=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = ("\n".join(canonical(row) for row in value) if lines else canonical(value)) + "\n"
    with path.open("x", encoding="utf-8") as output:
        output.write(encoded)


def verified_targets(paths, verifier):
    if not paths:
        raise ValueError("No completed-path evidence; do not train a completion model")
    payload = "\n".join(canonical(path) for path in paths) + "\n"
    if len(payload.encode()) > MAX_FILE_BYTES:
        raise ValueError("Combined completion proof data exceeds the bounded limit")
    result = subprocess.run([str(verifier), "--verify-paths"], input=payload, text=True,
                            capture_output=True, timeout=120, check=False)
    if result.returncode:
        raise ValueError("Completion replay failed: " + result.stderr[:2000])
    records = [validate_target(json.loads(line)) for line in result.stdout.splitlines() if line.strip()]
    if len(records) != len(paths):
        raise ValueError("Completion replay returned an incomplete result")
    audit_splits(records, [])
    unique = {}
    assignments = {}
    for record in records:
        key = selection_key(record)
        split = record["input"]["source"]["split"]
        if assignments.setdefault(key[0], split) != split:
            raise ValueError("Identical/reflected completion inputs cross held-out splits")
        previous = unique.get(key)
        ordering = lambda r: (r["targets"]["observedRemainingLocks"], r["evidence"]["proofSha256"])
        if previous is None or ordering(record) < ordering(previous):
            unique[key] = record
    # One best genuinely replayed path per input, progress and exact policy.
    return [unique[key] for key in sorted(unique)]


def extract_paths(args):
    value = bounded_json(args.input)
    source = bounded_json(args.source)
    if isinstance(source, dict) and "source" in source:
        source = source["source"]
    if (not isinstance(source, dict) or source.get("kind") != "search"
            or not source.get("family") or not source.get("group")
            or source.get("split") not in {"train", "validation", "test"}):
        raise ValueError("Use a retained source assignment; never choose a new split for existing evidence")
    candidate = value.get("candidate") or {}
    rules = value.get("rules") or {}
    expected_rules = {"width": 10, "height": 24, "rotation": "SRS+", "spin": "All-mini+", "oSpin": False,
                      "ceiling": "ActualCellDeath24", "movement": "SonicDropOnly", "supply": "UniformP7ClearOnlyHold"}
    if (value.get("format") != "neyra" or value.get("schemaVersion") != 3
            or not all(rules.get(key) == expected for key, expected in expected_rules.items())
            or any(candidate.get(key) is not True for key in ("complete", "qualified", "verified"))
            or candidate.get("spins") != 40 or not args.source_revision or len(args.source_revision) > 128):
        raise ValueError("Only a complete qualified Neyra 40-Spin-Single export with the fixed rules can seed stage two")
    frames, bags = candidate.get("frames", []), value.get("bags", [])
    if not isinstance(frames, list) or not isinstance(bags, list) or len(frames) != candidate.get("placed"):
        raise ValueError("A completion export needs every actual representative frame")
    policy = digest({"rules": expected_rules, "config": value.get("config"),
                     "sourceRevision": args.source_revision, "learnedOrdering": value.get("learnedOrdering")})
    encoded_bags, cursor = [], 0
    for bag in bags:
        inputs = bag.get("representativeInputs", [])
        selected = frames[cursor:cursor + len(inputs)]
        if not inputs or len(inputs) != len(selected):
            raise ValueError("Completion export is missing representative inputs")
        encoded_bags.append({"queue": bag["representativeQueue"],
                             "moves": [{"piece": frame["piece"], "inputs": path} for frame, path in zip(selected, inputs)]})
        cursor += len(inputs)
    if cursor != len(frames):
        raise ValueError("Completion frames and bag witnesses disagree")
    raw_hash = hashlib.sha256(args.input.read_bytes()).hexdigest()
    paths = []
    for boundary in range(2, len(bags)):
        bag = bags[boundary]
        if bag.get("startSpins", 40) >= 40 or not any(bag.get("startRows", [])):
            continue
        paths.append({"schemaVersion": 1, "task": TASK, "policy": POLICY, "rulesProfile": RULE_PROFILE,
                      "sourceRevision": args.source_revision, "collectionPolicySha256": policy,
                      "input": {"id": "completion:" + raw_hash[:32] + ":" + str(boundary), "rows": bag["startRows"],
                                "source": {**source, "bagBoundary": boundary, "reference": "neyra:completed-export:" + raw_hash}},
                      "completedSpins": bag["startSpins"], "bags": encoded_bags[boundary:]})
    verified_targets(paths, args.verifier)
    write_new(args.output, paths, lines=True)
    print(canonical({"pathProofs": len(paths), "sourceSha256": raw_hash, "collectionPolicySha256": policy}))


def audit_splits(records, historical):
    groups, families, boards = {}, {}, {}
    for record in [*historical, *records]:
        source = record["input"]["source"]
        split = source["split"]
        group = source.get("group", source["family"])
        board = canonical_rows(record["input"]["rows"])
        if (groups.setdefault(group, split) != split or families.setdefault(source["family"], split) != split
                or boards.setdefault(board, split) != split):
            raise ValueError("Transferred features or completion inputs leak across family/group/board splits")


def parent_history(parent, paths, proofs_paths, ancestor_paths):
    # Feature transfer retains earlier training dependencies. Require each
    # declared source model and every exact dataset, including v2 ancestors.
    from train import read_data
    if sum(path.stat().st_size for path in [*paths, *proofs_paths, *ancestor_paths]) > MAX_FILE_BYTES:
        raise ValueError("Combined parent lineage exceeds the bounded limit")
    proofs = {}
    for path in proofs_paths:
        for key, value in read_proofs(path).items():
            if key in proofs and proofs[key] != value:
                raise ValueError("Conflicting immutable first-stage proof identity")
            proofs[key] = value
    ancestors = {}
    for path in ancestor_paths:
        ancestor = validate_model(bounded_json(path))
        key = (ancestor["modelId"], ancestor["weightsSha256"])
        if key in ancestors and ancestors[key] != ancestor:
            raise ValueError("Conflicting feature ancestor model")
        ancestors[key] = ancestor
    models, seen = [], set()
    current = parent
    while current is not None:
        key = (current["modelId"], current["weightsSha256"])
        if key in seen:
            raise ValueError("Cyclic feature ancestry")
        seen.add(key)
        models.append(current)
        initialization = current.get("initialization")
        current = None
        if initialization:
            source = (initialization.get("sourceModelId"), initialization.get("sourceWeightsSha256"))
            if source not in ancestors:
                raise ValueError("Supply the retained feature-source model with --feature-ancestor-model; a new task cannot erase its prior data dependencies")
            current = ancestors[source]
    loaded = []
    for path in paths:
        records, _ = read_data(path, proofs, tensorize=False)
        loaded.append((hashlib.sha256(path.read_bytes()).hexdigest(), digest(records), records))
    used = set()
    for model in models:
        algorithm = model.get("datasetHashAlgorithm", "sha256-raw-jsonl-v1")
        if algorithm not in {"sha256-raw-jsonl-v1", "sha256-neyra-refined-records-v1"}:
            raise ValueError("Unsupported first-stage dataset hashing contract")
        lineage = set(model.get("trainingDatasetLineage", [model["datasetSha256"]]))
        matched = set()
        for index, (raw_hash, projected_hash, records) in enumerate(loaded):
            key = raw_hash if algorithm == "sha256-raw-jsonl-v1" else projected_hash
            if key not in lineage:
                continue
            if any(r["task"] != model["task"] or r["config"] != model["labelConfig"] for r in records):
                raise ValueError("Transferred source lineage does not match its parent policy")
            matched.add(key)
            used.add(index)
        if matched != lineage:
            raise ValueError("Supply every exact feature-ancestor trainingDatasetLineage draw using --parent-labels")
    if used != set(range(len(loaded))):
        raise ValueError("Unreferenced parent dataset; provide only the exact retained ancestry")
    return [record for index in sorted(used) for record in loaded[index][2]]


def network():
    import torch
    from torch import nn

    class CompletionCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1, self.conv2, self.conv3 = nn.Conv2d(1, 24, 3, padding=1), nn.Conv2d(24, 32, 3, padding=1), nn.Conv2d(32, 48, 3, padding=1)
            self.pool = nn.AvgPool2d((6, 5))
            self.hidden, self.output = nn.Linear(385, 32), nn.Linear(32, 1)
        def forward(self, x, remaining):
            x = self.pool(torch.relu(self.conv3(torch.relu(self.conv2(torch.relu(self.conv1(x))))))).flatten(1)
            return self.output(torch.relu(self.hidden(torch.cat([x, remaining], dim=1))))

    return CompletionCNN()


def train(args):
    import torch
    from torch import nn
    from train import encode

    parent = validate_model(bounded_json(args.parent_model))
    artifact = initialize(parent)
    paths = jsonl(args.paths)
    records = verified_targets(paths, args.verifier)
    historical = parent_history(parent, args.parent_labels, args.parent_proofs, args.feature_ancestor_model)
    audit_splits(records, historical)
    policies = {record["collectionPolicySha256"] for record in records}
    if len(policies) != 1:
        raise ValueError("One completion model must use one immutable collection policy")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = network()
    if sum(p.numel() for p in model.parameters()) != ARCHITECTURE["parameters"]:
        raise ValueError("Completion architecture parameter mismatch")
    model.load_state_dict({name: torch.tensor(artifact["weights"][name], dtype=torch.float32).reshape(tensor.shape)
                           for name, tensor in model.state_dict().items()}, strict=True)
    # Reuse a finished first-stage occupancy extractor without overwriting it.
    for name, parameter in model.named_parameters():
        if name.startswith("conv"):
            parameter.requires_grad_(False)
    data, counts = {}, {}
    for split in ("train", "validation", "test"):
        selected = [r for r in records if r["input"]["source"]["split"] == split]
        groups = {r["input"]["source"]["group"] for r in selected}
        if len(selected) < (4 if split == "train" else 2) or len(groups) < (2 if split == "train" else 1):
            raise ValueError("Stage-two pilot fitting needs >=4 train inputs in >=2 groups, >=2 validation and >=2 test inputs in independent groups; these are fitting floors, not completion criteria")
        x = torch.tensor([encode(r["input"]["rows"]) for r in selected], dtype=torch.float32)
        remaining = torch.tensor([[r["remainingSpins"] / 40] for r in selected], dtype=torch.float32)
        y = torch.tensor([[r["targets"]["observedRemainingLocks"] / COST_SCALE] for r in selected], dtype=torch.float32)
        data[split] = (x, remaining, y)
        counts[split] = {"uniqueInputs": len(selected), "sourceGroups": len(groups)}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.01)
    criterion = nn.SmoothL1Loss()
    x, remaining, y = data["train"]
    val_x, val_remaining, val_y = data["validation"]
    with torch.inference_mode():
        best_loss = criterion(model(val_x, val_remaining), val_y).item()
    best_state, best_epoch, history = copy.deepcopy(model.state_dict()), 0, []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in torch.randperm(len(x)).split(8):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[batch], remaining[batch]), y[batch])
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite completion training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation = criterion(model(val_x, val_remaining), val_y).item()
        if not math.isfinite(validation):
            raise ValueError("Nonfinite completion validation loss")
        history.append({"epoch": epoch, "validationLoss": validation})
        if validation < best_loss - 1e-7:
            best_loss, best_epoch, best_state = validation, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    weights = {name: value.detach().flatten().tolist() for name, value in model.state_dict().items()}
    metrics = {}
    baseline = float(data["train"][2].mean()) * COST_SCALE
    with torch.inference_mode():
        for split, (x, remaining, y) in data.items():
            estimate = (model(x, remaining) * COST_SCALE).clamp(0, 700)
            truth = y * COST_SCALE
            metrics[split] = {**counts[split], "locksMAE": float((estimate - truth).abs().mean()),
                              "locksRMSE": float((estimate - truth).square().mean().sqrt()),
                              "trainingMeanMAE": float((truth - baseline).abs().mean())}
    dataset_hash, weight_hash = digest(records), weights_hash(weights)
    artifact.update({"trained": best_epoch > 0, "weights": weights, "weightsSha256": weight_hash,
                     "modelId": "naga-completion-v1-" + weight_hash[:12] + "-" + dataset_hash[:12],
                     "datasetSha256": dataset_hash, "datasetHashAlgorithm": "sha256-neyra-completion-records-v1",
                     "collectionPolicySha256": next(iter(policies)),
                     "sourceTrainingDatasetLineage": parent.get("trainingDatasetLineage", [parent["datasetSha256"]]),
                     "scope": "Experimental best-observed completed-path cost conditioned on current occupancy, remaining target and fixed rules; not optimal cost or a pruning bound"})
    validate_completion_model(artifact)
    report = {"schemaVersion": 1, "task": TASK, "modelId": artifact["modelId"],
              "datasetSha256": dataset_hash, "weightsSha256": weight_hash,
              "counts": counts, "metrics": metrics, "trained": artifact["trained"], "learningComplete": False,
              "training": {"device": "cpu", "seed": args.seed, "epochs": len(history), "selectedEpoch": best_epoch,
                           "selection": "validation only; unchanged initialization eligible; test evaluated once after selection",
                           "frozen": ["conv1", "conv2", "conv3"], "seconds": time.monotonic() - started, "history": history},
              "sourceParent": {"modelId": parent["modelId"], "weightsSha256": parent["weightsSha256"]},
              "proofCount": len(paths), "selectedProofSha256": [r["evidence"]["proofSha256"] for r in records],
              "limitations": ["Pilot fitting is not model completion or generalization evidence.",
                              "Completed routes are selection-biased; no probability or optimality claim.",
                              "Current private eligibility/exclusions must be checked before choosing the parent.",
                              "No unfinished paths are converted into exact or lower-bound labels.",
                              "No automatic model promotion or search activation."]}
    write_new(args.model_output, artifact)
    write_new(args.report, report)
    print(canonical({"modelId": artifact["modelId"], "trained": artifact["trained"], "learningComplete": False,
                     "counts": counts, "metrics": metrics}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Transfer a selected trained v5 parent; reset progress and completion heads")
    init.add_argument("--parent-model", type=Path, required=True)
    init.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Physically replay completed-path proofs; choose one best observed cost per input/progress/policy")
    prepare.add_argument("--paths", type=Path, required=True)
    prepare.add_argument("--verifier", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    extract = commands.add_parser("extract", help="Extract replayable stage-two paths from an actual completed Neyra export")
    extract.add_argument("--input", type=Path, required=True)
    extract.add_argument("--source", type=Path, required=True, help="Retained source metadata with unchanged family/group/split")
    extract.add_argument("--source-revision", required=True)
    extract.add_argument("--verifier", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("train", help="Fit the CPU completion head with audited first-stage lineage and genuine completions")
    fit.add_argument("--parent-model", type=Path, required=True)
    fit.add_argument("--parent-labels", type=Path, action="append", required=True)
    fit.add_argument("--parent-proofs", type=Path, action="append", required=True)
    fit.add_argument("--feature-ancestor-model", type=Path, action="append", default=[],
                     help="Retained feature-source checkpoints named in parent initialization, including v2 ancestors")
    fit.add_argument("--paths", type=Path, required=True)
    fit.add_argument("--verifier", type=Path, required=True)
    fit.add_argument("--model-output", type=Path, required=True)
    fit.add_argument("--report", type=Path, required=True)
    fit.add_argument("--threads", type=int, default=2)
    fit.add_argument("--epochs", type=int, default=300)
    fit.add_argument("--patience", type=int, default=40)
    fit.add_argument("--seed", type=int, default=20261102)
    args = parser.parse_args()
    if args.command == "init":
        model = initialize(bounded_json(args.parent_model))
        write_new(args.output, validate_completion_model(model))
        print(canonical({"modelId": model["modelId"], "trained": False, "parameters": ARCHITECTURE["parameters"]}))
    elif args.command == "extract":
        extract_paths(args)
    elif args.command == "prepare":
        records = verified_targets(jsonl(args.paths), args.verifier)
        write_new(args.output, records, lines=True)
        print(canonical({"uniqueTargets": len(records), "learningComplete": False}))
    else:
        if not 1 <= args.threads <= 16 or not 1 <= args.epochs <= 5000 or not 1 <= args.patience <= 5000:
            parser.error("Invalid training limits")
        if args.model_output.exists() or args.report.exists() or args.model_output.resolve() == args.report.resolve():
            parser.error("Use distinct new paths for the model and report; input artifacts are immutable")
        train(args)


if __name__ == "__main__":
    main()
