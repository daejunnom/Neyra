"""Train an occupancy-only Naga Eyes spin CNN on CPU.

V2 accepts fresh bounded-search observations only. Legacy board teachers, opener
priors and path-count objectives are rejected, including in source metadata.
"""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
from spin_contract import (ARCHITECTURE, MAX_FILE_BYTES, MAX_RECORD_BYTES, POLICY, TASK, WEIGHTS_HASH_ALGORITHM,
                           canonical_rows, read_jsonl, reject_constant, unique_keys,
                           validate_model, weights_hash)

HEIGHT, WIDTH = 24, 10


class BoardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, 3, padding=1)
        self.conv2 = nn.Conv2d(24, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)
        self.pool = nn.AvgPool2d((6, 5))
        self.hidden = nn.Linear(384, 32)
        self.output = nn.Linear(32, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x).flatten(1)
        return self.output(torch.relu(self.hidden(x)))


def encode(rows):
    rows = canonical_rows(rows)
    return [[[float(bool(rows[y] & (1 << x))) for x in range(WIDTH)] for y in range(HEIGHT)]]


def read_data(path):
    records = read_jsonl(path)
    if not records:
        raise ValueError("No freshly measured v2 labels")
    groups, families, boards, protocols = {}, {}, set(), set()
    for record in records:
        if record["policy"] != POLICY:
            raise ValueError("Exploration observations are excluded from fixed-policy fitting; measure their certified descendant boards with the fixed collector first")
        source = record["input"]["source"]
        split = source.get("split")
        if split not in ("train", "validation", "test"):
            raise ValueError("An explicit grouped train/validation/test split is required")
        group = source.get("group", source["family"])
        if groups.setdefault(group, split) != split or families.setdefault(source["family"], split) != split:
            raise ValueError("A source family/run leaks across splits")
        board = canonical_rows(record["input"]["rows"])
        if board in boards:
            raise ValueError("Duplicate or reflected board; deduplicate before fitting")
        boards.add(board)
        protocols.add(json.dumps(record["config"], sort_keys=True, separators=(",", ":")))
    if len(protocols) != 1:
        raise ValueError("One model must use one fixed labeling policy")
    data = {}
    for split in ("train", "validation", "test"):
        chosen = [r for r in records if r["input"]["source"]["split"] == split]
        if not chosen:
            raise ValueError(f"Missing independent {split} split")
        xs = [encode(r["input"]["rows"]) for r in chosen]
        ys = [[float(r["targets"]["foundSpin"]), r["targets"]["observedSpins"] / 6.0] for r in chosen]
        data[split] = (torch.tensor(xs, dtype=torch.float32, device="cpu"),
                       torch.tensor(ys, dtype=torch.float32, device="cpu"), chosen)
    return records, data


def initialize_from_checkpoint(model, checkpoint_path, original_labels_path, records, history_paths=()):
    """Warm-start only with demonstrated compatible policy and split history."""
    with checkpoint_path.open("rb") as stream:
        raw = stream.read(MAX_RECORD_BYTES + 1)
    if len(raw) > MAX_RECORD_BYTES:
        raise ValueError("Initial checkpoint exceeds the bounded record limit")
    checkpoint = validate_model(json.loads(raw, object_pairs_hook=unique_keys, parse_constant=reject_constant))
    if checkpoint.get("datasetHashAlgorithm", "sha256-raw-jsonl-v1") != "sha256-raw-jsonl-v1":
        raise ValueError("Offline warm-start currently requires raw-JSONL training artifacts. Browser canonical projected-record hashes are distinct; retain the exported datasetSelection history and use the private Site to continue that candidate.")
    paths = [original_labels_path, *history_paths]
    if sum(path.stat().st_size for path in paths) > MAX_FILE_BYTES:
        raise ValueError("Combined historical labels exceed the bounded learning input limit")
    if hashlib.sha256(original_labels_path.read_bytes()).hexdigest() != checkpoint["datasetSha256"]:
        raise ValueError("Initial labels must match the checkpoint's exact dataset SHA-256")
    lineage = checkpoint.get("trainingDatasetLineage", [checkpoint["datasetSha256"]])
    provided, previous = {}, []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest not in provided:
            provided[digest] = path
            historical, _ = read_data(path)
            previous.extend(historical)
    if set(provided) != set(lineage):
        raise ValueError("Provide every checkpoint trainingDatasetLineage artifact using --initial-labels and --history-labels")
    policy = records[0]["config"]
    if checkpoint["labelConfig"] != policy or any(r["config"] != policy for r in previous):
        raise ValueError("Warm-start requires the identical fixed observation policy")
    families, groups, boards = {}, {}, {}
    for record in previous + records:
        source = record["input"]["source"]
        split = source["split"]
        group = source.get("group", source["family"])
        board = canonical_rows(record["input"]["rows"])
        if (families.setdefault(source["family"], split) != split
                or groups.setdefault(group, split) != split or boards.setdefault(board, split) != split):
            raise ValueError("Warm-start cannot move a historical family, group or reflected board between splits")
    state = {key: torch.tensor(checkpoint["weights"][key], dtype=torch.float32, device="cpu").reshape(value.shape)
             for key, value in model.state_dict().items()}
    model.load_state_dict(state, strict=True)
    return {"modelId": checkpoint["modelId"], "weightsSha256": checkpoint["weightsSha256"],
            "datasetSha256": checkpoint["datasetSha256"],
            "trainingDatasetLineage": lineage,
            "splitAudit": "Historical and new family/group/board assignments agree; only train rows receive gradients."}


def criterion(pred, target):
    found = nn.functional.binary_cross_entropy_with_logits(pred[:, 0], target[:, 0])
    count = nn.functional.smooth_l1_loss(pred[:, 1], target[:, 1])
    return 0.25 * found + count


def metrics(pred, target, horizon):
    probability = pred[:, 0].sigmoid()
    count = (pred[:, 1] * 6).clamp(0, horizon * 6)
    truth = target[:, 1] * 6
    return {
        "foundSpinBrier": (probability - target[:, 0]).square().mean().item(),
        "foundSpinLogLoss": nn.functional.binary_cross_entropy_with_logits(pred[:, 0], target[:, 0]).item(),
        "observedSpinsMAE": (count - truth).abs().mean().item(),
        "observedSpinsRMSE": (count - truth).square().mean().sqrt().item(),
        "selectionLoss": criterion(pred, target).item(),
        "foundSpinPrevalence": target[:, 0].mean().item(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--initial-model", type=Path,
                   help="Optional compatible private v2 checkpoint; never automatically promoted")
    p.add_argument("--initial-labels", type=Path,
                   help="Exact original JSONL dataset matching the initial model hash; required to audit held-out isolation")
    p.add_argument("--history-labels", type=Path, action="append", default=[],
                   help="Earlier lineage JSONL artifacts for repeated warm-starts; may be supplied more than once")
    a = p.parse_args()
    if not 1 <= a.threads <= 16 or not 1 <= a.epochs <= 5000 or a.patience < 1:
        p.error("Invalid CPU training limits")
    if bool(a.initial_model) != bool(a.initial_labels):
        p.error("--initial-model and --initial-labels must be supplied together")
    if a.history_labels and not a.initial_model:
        p.error("--history-labels requires --initial-model")
    protected = [a.labels, a.initial_model, a.initial_labels, *a.history_labels]
    if any(path is not None and a.model_output.resolve() == path.resolve() for path in protected):
        p.error("The output checkpoint must not replace any input artifact")
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    records, data = read_data(a.labels)
    model = BoardCNN().to(device="cpu")
    assert sum(x.numel() for x in model.parameters()) == ARCHITECTURE["parameters"]
    initial = initialize_from_checkpoint(model, a.initial_model, a.initial_labels, records, a.history_labels) if a.initial_model else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best_loss, best_epoch, best_state, history = math.inf, 0, None, []
    started = time.monotonic()
    train_x, train_y, _ = data["train"]
    val_x, val_y, _ = data["validation"]
    if initial:
        model.eval()
        with torch.inference_mode():
            best_loss = criterion(model(val_x), val_y).item()
        best_epoch, best_state = 0, copy.deepcopy(model.state_dict())
        initial["validationLoss"] = best_loss
    for epoch in range(1, a.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), device="cpu")
        for batch in permutation.split(8):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_x[batch]), train_y[batch])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation = criterion(model(val_x), val_y).item()
        history.append({"epoch": epoch, "validationLoss": validation})
        if validation < best_loss - 1e-7:
            best_loss, best_epoch = validation, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= a.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    config = records[0]["config"]
    horizon = config["horizonBags"]
    evaluation, predictions = {}, []
    mean = train_y.mean(0)
    probability = mean[0].clamp(1e-6, 1-1e-6)
    baseline = torch.stack([torch.logit(probability), mean[1]])
    with torch.inference_mode():
        for split, (x, y, chosen) in data.items():
            pred = model(x)
            evaluation[split] = {"count": len(x), "cnn": metrics(pred, y, horizon),
                                 "trainingMeanBaseline": metrics(baseline.expand_as(y), y, horizon)}
            for r, output, target in zip(chosen, pred.tolist(), y.tolist()):
                predictions.append({"id": r["input"]["id"], "split": split,
                                    "group": r["input"]["source"].get("group", r["input"]["source"]["family"]),
                                    "rows": r["input"]["rows"], "rawOutput": output, "target": target})
    weights = {key: value.detach().cpu().flatten().tolist() for key, value in model.state_dict().items()}
    weight_hash = weights_hash(weights)
    dataset_hash = hashlib.sha256(a.labels.read_bytes()).hexdigest()
    runtime = {
        "schemaVersion": 2, "task": TASK,
        # A retained epoch-zero checkpoint can be evaluated against a new
        # dataset without overwriting the original immutable model artifact.
        "modelId": "naga-spin-v2-" + weight_hash[:12] + ("-" + dataset_hash[:12] if initial else ""),
        "architecture": ARCHITECTURE, "trained": True, "device": "cpu",
        "datasetSha256": dataset_hash, "weightsSha256": weight_hash,
        "datasetHashAlgorithm": "sha256-raw-jsonl-v1",
        "weightsSha256Algorithm": WEIGHTS_HASH_ALGORITHM,
        "horizonBags": horizon, "workBudget": config["workBudget"], "labelConfig": config,
        "weights": weights, "outputNames": ["foundSpinLogit", "observedSpinsScaled"],
        "countScale": 6, "sourceInputs": "Canonical current occupancy only; no heuristic score channels.",
        "scope": "Predictions of censored bounded-policy spin observations, not construction probability or exhaustive feasibility.",
        "searchOrderingEnabled": False,
    }
    if initial:
        runtime["trainingDatasetLineage"] = sorted(set(initial["trainingDatasetLineage"]) | {dataset_hash})
    validate_model(runtime)
    a.model_output.parent.mkdir(parents=True, exist_ok=True)
    a.model_output.write_text(json.dumps(runtime, ensure_ascii=False, separators=(",", ":")) + "\n")
    a.output.mkdir(parents=True, exist_ok=True)
    training = {
        "device": "cpu", "torch": torch.__version__, "seed": a.seed, "threads": a.threads,
        "initialization": "Compatible v2 checkpoint; policy and historical splits audited" if initial else "Fresh random weights; no v1 checkpoint or score supervision",
        "initialCheckpoint": initial,
        "epochsRun": len(history), "selectedEpoch": best_epoch,
        "selection": "Validation loss only; held-out test evaluated once after fitting.",
        "seconds": time.monotonic() - started, "optimizer": "AdamW",
        "lr": 0.001, "weightDecay": 0.01, "batchSize": 8,
        "loss": "0.25 * BCEWithLogits(foundSpin) + SmoothL1(observedSpins / 6, beta=1)",
        "promotion": "Private candidate only; no deployed model is replaced. Warm-start retains epoch zero if validation does not improve.",
        "history": history,
    }
    report = {
        "schemaVersion": 2, "task": TASK, "modelId": runtime["modelId"],
        "datasetSha256": dataset_hash, "weightsSha256": weight_hash,
        "datasetHashAlgorithm": runtime["datasetHashAlgorithm"],
        "weightsSha256Algorithm": WEIGHTS_HASH_ALGORITHM, "architecture": ARCHITECTURE,
        "training": training, "metrics": evaluation,
        "limitations": [
            "Small grouped pilot, not proof of stronger search or general calibration.",
            "A missing spin under limited search is censored absence, not impossibility.",
            "No legacy Terrain values, published-opener priors or path-count targets are mixed into v2.",
            "Public completion search does not use this model for ordering.",
        ],
    }
    (a.output / "training-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (a.output / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + "\n")
    (a.output / "model-card.json").write_text(json.dumps({
        **{key: value for key, value in runtime.items() if key != "weights"},
        "training": training, "metrics": evaluation, "weightsIncluded": False,
    }, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "training"}, ensure_ascii=False))
    print(json.dumps({key: value for key, value in training.items() if key != "history"}))


if __name__ == "__main__":
    main()
