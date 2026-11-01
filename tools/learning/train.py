"""Train and export Neyra's small current-board CNN on CPU only.

Measured targets are observations of a fixed, censored search policy. Reference
shape preference is a separate weak prior, never an invented spin witness.
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
from store import digest as canonical_digest, read_records

HEIGHT, WIDTH = 24, 10
ARCHITECTURE = {'name': 'neyra-board-cnn-v1', 'input': [2, HEIGHT, WIDTH],
                'channels': [6, 8], 'kernel': 3, 'padding': 1,
                'poolOutput': [4, 2], 'poolKernel': [6, 5],
                'hidden': 16, 'heads': 3, 'parameters': 1645}


class BoardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 6, 3, padding=1)
        self.conv2 = nn.Conv2d(6, 8, 3, padding=1)
        self.pool = nn.AvgPool2d((6, 5))
        self.hidden = nn.Linear(64, 16)
        self.output = nn.Linear(16, 3)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.output(torch.relu(self.hidden(x)))


def canonical(rows):
    reflected = [int(f'{row:010b}'[::-1], 2) for row in rows]
    return min(tuple(rows), tuple(reflected))


def encode(rows):
    if len(rows) != HEIGHT or not any(rows) or any(type(r) is not int or not 0 <= r < 1023 for r in rows):
        raise ValueError('Input must be a nonempty post-clear 24-row board')
    rows = canonical(rows)
    occupancy = [[float(bool(rows[y] & (1 << x))) for x in range(WIDTH)] for y in range(HEIGHT)]
    holes = [[0.0] * WIDTH for _ in range(HEIGHT)]
    for x in range(WIDTH):
        above = False
        for y in range(HEIGHT - 1, -1, -1):
            if occupancy[y][x]:
                above = True
            elif above:
                holes[y][x] = 1.0
    return [occupancy, holes]


def read_data(path):
    records = read_records(path, 'labels')
    if not records:
        raise ValueError('No measured labels')
    groups, families, boards, protocols = {}, {}, {}, set()
    for record in records:
        inp, measured = record['input'], record['measured']
        source = inp['source']
        split = source['split']
        if split not in ('train', 'validation', 'test') or source['bagBoundary'] < 1:
            raise ValueError('An explicit grouped split and completed boundary are required')
        group = source.get('group', source['family'])
        if groups.setdefault(group, split) != split:
            raise ValueError('A source group leaks across splits')
        if families.setdefault(source['family'], split) != split:
            raise ValueError('A source family leaks across splits')
        board = canonical(inp['rows'])
        if board in boards:
            raise ValueError('Duplicate or reflected board; deduplicate before fitting')
        boards[board] = split
        if measured.get('censored') is not True:
            raise ValueError('Bounded-search observations must retain censoring')
        cfg = record['config']
        protocols.add(json.dumps(cfg, sort_keys=True, separators=(',', ':')))
        terrain = record['currentBoardTeacher']['terrainScore']
        if not math.isfinite(terrain):
            raise ValueError('Nonfinite current-board teacher')
        if source['kind'] == 'opener':
            prior = source.get('referencePrior')
            if type(prior) not in (int, float) or not math.isfinite(prior) or not -1.0 <= prior <= 1.0:
                raise ValueError('Opener preference must be finite and in [-1, 1]')
    if len(protocols) != 1:
        raise ValueError('One model must use one fixed labeling policy/configuration')
    split_data = {}
    for split in ('train', 'validation', 'test'):
        chosen = [r for r in records if r['input']['source']['split'] == split]
        if not chosen:
            raise ValueError(f'Missing independent {split} split')
        xs, ys, weights = [], [], []
        for r in chosen:
            source, measured = r['input']['source'], r['measured']
            terrain = r['currentBoardTeacher']['terrainScore']
            # The opener prior supervises only shape preference. It cannot
            # alter either observed search target or certified-success fields.
            shape = terrain / (120.0 + abs(terrain))
            weight = [1.0, 1.0, 0.5]
            if source['kind'] == 'opener':
                shape = float(source['referencePrior'])
                weight[0] = 0.5
            spins = measured['maxReachableObservedSpins']
            choices = measured['productiveGeometricWholeBagChains']
            if type(spins) is not int or spins < 0 or type(choices) is not int or choices < 0:
                raise ValueError('Observed spin and chain counts must be nonnegative integers')
            xs.append(encode(r['input']['rows']))
            ys.append([shape, spins / 6.0, math.log1p(choices) / 4.0])
            weights.append(weight)
        split_data[split] = (torch.tensor(xs, dtype=torch.float32, device='cpu'),
                             torch.tensor(ys, dtype=torch.float32, device='cpu'),
                             torch.tensor(weights, dtype=torch.float32, device='cpu'), chosen)
    return records, split_data


def metrics(pred, target):
    delta = pred - target
    return {'shapeMAE': delta[:, 0].abs().mean().item(),
            'observedSpinsMAE': (delta[:, 1] * 6).abs().mean().item(),
            'log1pChoicesMAE': (delta[:, 2] * 4).abs().mean().item(),
            'normalizedMSE': delta.square().mean().item()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--labels', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--model-output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=40)
    p.add_argument('--seed', type=int, default=20261101)
    p.add_argument('--threads', type=int, default=2)
    a = p.parse_args()
    if not 1 <= a.threads <= 16 or not 1 <= a.epochs <= 5000 or a.patience < 1:
        p.error('Invalid CPU training limits')
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.random.default_generator.manual_seed(a.seed)
    torch.use_deterministic_algorithms(True)
    records, data = read_data(a.labels)
    model = BoardCNN().to(device='cpu')
    assert sum(x.numel() for x in model.parameters()) == ARCHITECTURE['parameters']
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    loss_fn = nn.SmoothL1Loss(reduction='none')
    best_loss, best_epoch, best_state, history = math.inf, 0, None, []
    start = time.monotonic()
    train_x, train_y, train_weights, _ = data['train']
    val_x, val_y, val_weights, _ = data['validation']
    for epoch in range(1, a.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_x), device='cpu')
        for batch in permutation.split(16):
            optimizer.zero_grad(set_to_none=True)
            loss = (loss_fn(model(train_x[batch]), train_y[batch]) * train_weights[batch]).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation = (loss_fn(model(val_x), val_y) * val_weights).mean().item()
        history.append({'epoch': epoch, 'validationLoss': validation})
        if validation < best_loss - 1e-7:
            best_loss, best_epoch = validation, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= a.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    evaluation, predictions = {}, []
    train_mean = train_y.mean(0)
    with torch.inference_mode():
        for split, (x, y, _, chosen) in data.items():
            pred = model(x)
            evaluation[split] = {'count': len(x), 'cnn': metrics(pred, y),
                                 'trainingMeanBaseline': metrics(train_mean.expand_as(y), y)}
            for r, output, target in zip(chosen, pred.tolist(), y.tolist()):
                predictions.append({'id': r['input']['id'], 'split': split,
                                    'group': r['input']['source'].get('group', r['input']['source']['family']),
                                    'rows': r['input']['rows'], 'rawOutput': output, 'target': target})
    dataset_hash = hashlib.sha256(a.labels.read_bytes()).hexdigest()
    weights = {k: v.detach().cpu().flatten().tolist() for k, v in model.state_dict().items()}
    weights_hash = canonical_digest(weights)
    config = records[0]['config']
    runtime = {'schemaVersion': 1, 'modelId': 'naga-cnn-pilot-' + weights_hash[:12],
               'architecture': ARCHITECTURE, 'trained': True, 'device': 'cpu',
               'datasetSha256': dataset_hash, 'weightsSha256': weights_hash,
               'weightsSha256Algorithm': 'sha256-canonical-json-numeric-v1',
               'horizonBags': config['horizonBags'], 'workBudget': config['workBudget'],
               'labelConfig': config, 'weights': weights,
               'scope': 'Small-sample fixed-policy observations, not spin feasibility, exhaustive choices, or supply probability.',
               'defaultSearchEnabled': False, 'maxBoundaryAdjustment': 12.0}
    a.model_output.parent.mkdir(parents=True, exist_ok=True)
    a.model_output.write_text(json.dumps(runtime, ensure_ascii=False, separators=(',', ':')) + '\n')
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/'predictions.json').write_text(json.dumps(predictions, ensure_ascii=False, indent=2) + '\n')
    report = {'schemaVersion': 1, 'modelId': runtime['modelId'], 'datasetSha256': dataset_hash,
              'weightsSha256': weights_hash, 'architecture': ARCHITECTURE,
              'weightsSha256Algorithm': 'sha256-canonical-json-numeric-v1',
              'training': {'device': 'cpu', 'torch': torch.__version__, 'seed': a.seed,
                           'threads': a.threads, 'epochsRun': len(history), 'selectedEpoch': best_epoch,
                           'selection': 'Validation loss only; held-out test evaluated after fitting.',
                           'seconds': time.monotonic()-start, 'optimizer': 'AdamW', 'lr': 0.003,
                           'weightDecay': 0.01, 'history': history},
              'metrics': evaluation,
              'limitations': ['Tiny grouped pilot; no general strength or completion claim.',
                              'Outcome heads predict this bounded search policy, not true maximum spins or exhaustive path counts.',
                              'Opener shape targets are weak priors, separate from observed spin/choice labels.',
                              'Certified supply counts are retained in data but are not imitated as probability estimates.']}
    (a.output/'training-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (a.output/'model-envelope.json').write_text(json.dumps({
        'schemaVersion': 1, 'modelId': runtime['modelId'], 'datasetSha256': dataset_hash,
        'weightsSha256': weights_hash, 'architecture': ARCHITECTURE,
        'weightsSha256Algorithm': 'sha256-canonical-json-numeric-v1',
        'training': report['training'], 'metrics': evaluation, 'weights': weights,
        'runtime': {k: v for k, v in runtime.items() if k != 'weights'}}, ensure_ascii=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'training'}, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report['training'].items() if k != 'history'}))


if __name__ == '__main__':
    main()
