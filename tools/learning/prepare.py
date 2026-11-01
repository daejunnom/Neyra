"""Import published Naga residues and recorded whole-bag search boundaries.

Outputs contain current boards and provenance, never future outcomes inferred
from opener names. Reference preference is a separate, explicitly weak prior.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def canonical(rows):
    reflected = [int(f'{row:010b}'[::-1], 2) for row in rows]
    return min(tuple(rows), tuple(reflected))


def board_rows(values):
    if not isinstance(values, list) or not 1 <= len(values) <= 24:
        raise ValueError('Expected at most 24 bottom-up rows')
    if any(type(row) is not int or not 0 <= row <= 1023 for row in values):
        raise ValueError('Invalid row bitmask')
    rows = values + [0] * (24 - len(values))
    if not any(rows) or 1023 in rows:
        raise ValueError('Empty first-bag starts and uncleared full rows are excluded')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-analysis', type=Path, required=True)
    p.add_argument('--search-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-search-per-seed', type=int, default=24)
    a = p.parse_args()
    if not 1 <= a.max_search_per_seed <= 10000:
        p.error('max-search-per-seed must be 1..10000')
    references = json.loads(a.reference_analysis.read_text())
    raw = []
    for key, name, split in [('sao', 'SAO', 'train'), ('sarcelle_naga', 'Sarcelle Naga Eyes', 'test')]:
        for bag in references[key]['bags']:
            # Use completed residues before the final two published bags, so
            # an opener preference concerns a field with remaining play.
            if not 1 <= bag['bag'] <= 15:
                continue
            rows = board_rows(bag['residue_row_masks_bottom_up'])
            source = {'kind': 'opener', 'family': name, 'group': f'opener:{name}',
                      'reference': 'https://harddrop.com/wiki/' + ('SAO' if key == 'sao' else 'Sarcelle'),
                      'bagBoundary': bag['bag'], 'split': split,
                      'referencePrior': 0.65, 'priorIsMeasuredOutcome': False,
                      'sourceFileSha256': hashlib.sha256(a.reference_analysis.read_bytes()).hexdigest(),
                      'scope': 'Published completed residue. No supply/hold compatibility is presumed.'}
            raw.append({'id': f'{key}-after-bag-{bag["bag"]}', 'rows': rows, 'source': source})
    source_files = []
    by_seed = {}
    for path in sorted(a.search_results.glob('*-p*-s*.json')):
        d = json.loads(path.read_text())
        if d.get('seed') not in (1, 11, 23):
            continue
        seed = d['seed']
        split = {1: 'train', 11: 'validation', 23: 'test'}[seed]
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        source_files.append({'path': str(path), 'sha256': file_hash})
        routes = [e['route'] for e in d.get('history', {}).get('events', []) if e.get('route')]
        for key in ('bestSpinCertificates', 'deepestCertificates', 'bestZeroStartCertificates'):
            for i, cert in enumerate(d.get('history', {}).get(key, [])):
                routes.append({'rows': cert['endRows'], 'bags': i + 1, 'id': cert['id']})
        candidates = by_seed.setdefault(seed, {})
        for route in routes:
            try:
                rows = board_rows(route['rows'])
                bag = route.get('bags', route.get('bagCount'))
                if type(bag) is not int or bag < 1:
                    continue
            except (KeyError, ValueError):
                continue
            source = {'kind': 'search', 'family': f'search-seed-{seed}',
                      'group': f'search-seed-{seed}', 'reference': path.name,
                      'sourceFileSha256': file_hash, 'bagBoundary': bag, 'split': split,
                      'profile': d['profile'], 'seed': seed, 'variant': d['label'],
                      'scope': 'Actual previously certified whole-bag boundary; continuation labels are recomputed.'}
            record = {'id': f'search-{seed}-{digest(rows)[:16]}', 'rows': rows, 'source': source}
            candidates.setdefault(canonical(rows), record)
    for seed, candidates in sorted(by_seed.items()):
        raw.extend(sorted(candidates.values(), key=lambda v: digest(v['rows']))[:a.max_search_per_seed])
    # Purge identical and reflected boards across source groups. The independent
    # test split has precedence, then validation; no duplicate leaks into train.
    priority = {'test': 0, 'validation': 1, 'train': 2}
    unique = {}
    for record in sorted(raw, key=lambda x: (priority[x['source']['split']], x['source']['kind'] != 'opener', x['id'])):
        unique.setdefault(canonical(record['rows']), record)
    records = sorted(unique.values(), key=lambda x: (x['source']['split'], x['id']))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(''.join(json.dumps(x, ensure_ascii=False, separators=(',', ':')) + '\n' for x in records))
    priors = [{'schemaVersion': 1, 'input': x, 'prior': {
        'score': x['source']['referencePrior'], 'reference': x['source']['reference'],
        'reason': 'Dedicated published Naga Eyes residue; weak shape preference, not a measured spin/probability label.'}}
        for x in records if x['source']['kind'] == 'opener']
    a.output.with_suffix('.priors.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in priors))
    manifest = {'schemaVersion': 1, 'boards': len(records), 'beforeDeduplication': len(raw),
                'splitCounts': {s: sum(x['source']['split'] == s for x in records) for s in priority},
                'familyCounts': {f: sum(x['source']['family'] == f for x in records) for f in sorted({x['source']['family'] for x in records})},
                'inputSha256': hashlib.sha256(a.output.read_bytes()).hexdigest(),
                'referenceAnalysisSha256': hashlib.sha256(a.reference_analysis.read_bytes()).hexdigest(),
                'searchFiles': source_files,
                'splitPolicy': 'Entire opener families and search seeds stay in one split. Canonical-reflection duplicates purged before training; test then validation take precedence.',
                'emptyFirstBagStarts': 0, 'futureLabelsFromReferenceAnnotations': False}
    a.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in manifest.items() if k != 'searchFiles'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
