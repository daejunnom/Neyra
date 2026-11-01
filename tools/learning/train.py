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
from spin_contract import (ARCHITECTURE, TASK, WEIGHTS_HASH_ALGORITHM,
                           canonical_rows, read_jsonl, validate_model, weights_hash)

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
    a = p.parse_args()
    if not 1 <= a.threads <= 16 or not 1 <= a.epochs <= 5000 or a.patience < 1:
        p.error("Invalid CPU training limits")
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    records, data = read_data(a.labels)
    model = BoardCNN().to(device="cpu")
    assert sum(x.numel() for x in model.parameters()) == ARCHITECTURE["parameters"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best_loss, best_epoch, best_state, history = math.inf, 0, None, []
    started = time.monotonic()
    train_x, train_y, _ = data["train"]
    val_x, val_y, _ = data["validation"]
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
        "schemaVersion": 2, "task": TASK, "modelId": "naga-spin-v2-" + weight_hash[:12],
        "architecture": ARCHITECTURE, "trained": True, "device": "cpu",
        "datasetSha256": dataset_hash, "weightsSha256": weight_hash,
        "weightsSha256Algorithm": WEIGHTS_HASH_ALGORITHM,
        "horizonBags": horizon, "workBudget": config["workBudget"], "labelConfig": config,
        "weights": weights, "outputNames": ["foundSpinLogit", "observedSpinsScaled"],
        "countScale": 6, "sourceInputs": "Canonical current occupancy only; no heuristic score channels.",
        "scope": "Predictions of censored bounded-policy spin observations, not construction probability or exhaustive feasibility.",
        "searchOrderingEnabled": False,
    }
    validate_model(runtime)
    a.model_output.parent.mkdir(parents=True, exist_ok=True)
    a.model_output.write_text(json.dumps(runtime, ensure_ascii=False, separators=(",", ":")) + "\n")
    a.output.mkdir(parents=True, exist_ok=True)
    training = {
        "device": "cpu", "torch": torch.__version__, "seed": a.seed, "threads": a.threads,
        "initialization": "Fresh random weights; no v1 checkpoint or score supervision",
        "epochsRun": len(history), "selectedEpoch": best_epoch,
        "selection": "Validation loss only; held-out test evaluated once after fitting.",
        "seconds": time.monotonic() - started, "optimizer": "AdamW",
        "lr": 0.001, "weightDecay": 0.01, "batchSize": 8,
        "loss": "0.25 * BCEWithLogits(foundSpin) + SmoothL1(observedSpins / 6, beta=1)",
        "history": history,
    }
    report = {
        "schemaVersion": 2, "task": TASK, "modelId": runtime["modelId"],
        "datasetSha256": dataset_hash, "weightsSha256": weight_hash,
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
