# Second-stage completion-cost CNN

This is a reusable implementation contract, not a trained-model release or a
40-line completion report. No second-stage checkpoint is bundled or selected by
public search. The retained first-stage checkpoint and its one-bag task are not
reinterpreted.

## Separate prediction task

The first-stage v5 network has 33,442 parameters and receives only current board
occupancy. Its two outputs predict bounded one-bag spin discovery and observed
spin count. The second stage has a different task, `naga-completion-cost-v1`:

- Input: the same canonical current occupancy plus explicit
  `remainingSpins = 40 - completedSpins`.
- Rules: the immutable `neyra-srs-plus-24-spin-single-clear-hold-v1` profile.
  Inference must supply this exact profile. A model for another rule profile is
  a different artifact, not another unlabeled numeric feature.
- Target: the number of additional locks on an actually completed representative
  path. When several replayed paths have the same canonical input, remaining
  target and collection policy, preparation retains the lowest observed cost
  and its exact proof. The raw proofs remain separate.
- Meaning: best **observed** completed-path cost under the recorded collection
  policy. It is not the globally shortest path, an all-supplies completion cost,
  construction probability, or a permissible pruning bound.

An unfinished search does not provide this target. Its spent locks are neither
an exact completion cost nor a valid lower bound on the shortest completion:
a different continuation might finish sooner. Version one therefore rejects
unfinished proofs instead of fitting a fabricated censoring objective. Genuine
censored objectives require their own future contract.

## Architecture and feature transfer

| Part | Contract |
| --- | --- |
| Occupancy | One 24 × 10 channel, reflected to the same canonical orientation as stage one |
| Convolutions | 1→24→32→48 channels, 3 × 3 kernels, padding one, ReLU |
| Pool | 6 × 5 average pooling; 384 flattened occupancy features |
| Progress | Append one scalar, `remainingSpins / 40`, after the occupancy features |
| Hidden | 385→32, ReLU |
| Output | 32→1, `observedRemainingLocksScaled`; scale 140 locks |
| Parameters | 33,441 total; 21,056 convolution parameters stay frozen during the supplied CPU trainer |
| Trainable parameters | 12,385 in the progress-aware hidden layer and completion output |

Initialization requires an explicitly selected trained v5 artifact. It copies
`conv1`, `conv2`, `conv3`, the 384 occupancy columns of the hidden layer, and the
hidden bias. The new progress column and output weights/bias are zeroed. The
artifact records the exact source model ID and float32 weight digest. The first
model is never overwritten.

The initializer is `trained: false`, `learningComplete: false` and
`searchOrderingEnabled: false`. Rust/WASM inference returns `null` for it. A
trained stage-two candidate remains a separate experimental artifact;
`trained: true` means that fitting selected an updated checkpoint, not that the
model has passed completion, generalization, or search-improvement criteria.

## Actual path evidence

`neyra-completion --verify-paths` reads JSONL objects with these fields:

| Field | Meaning |
| --- | --- |
| `schemaVersion`, `task`, `policy` | `1`, `naga-completion-cost-v1`, `input-replayed-completion-cost-v1` |
| `rulesProfile` | Exact fixed profile above |
| `sourceRevision` | Source revision of the collecting search |
| `collectionPolicySha256` | Immutable identity of the collector settings, source and evaluator selection |
| `input` | Existing `BoardSample`: physical rows and retained search family/group/split/reference/bag boundary |
| `completedSpins` | Actual progress at that boundary, below 40 |
| `bags` | Ordered bag objects, each with its seven-label `queue` and actual `moves` |
| Each move | `piece` and the full ordered `inputs`, ending in `hard_drop` |

The verifier checks a nonempty boundary of at least two completed bags and
material consistency, `28 × bagBoundary - 10 × completedSpins = occupiedCells`.
It then replays every lock through Neyra's SRS+ physics. Only quiet locks and
actual non-O Spin Singles are admitted. Intermediate bags use every label
exactly once; only the final goal-reaching bag may be partial. Extra moves after
40 spins, broken input paths, repeated pieces, and inconsistent progress fail.

Each bag's declared representative supply is also executed through the existing
clear-only hold language. No new hold interpretation or handwritten kick table
is introduced. This checks that representative route; it does not re-prove all
5,040 supplies. The target explicitly records
`representativeSupplyVerified: true` and `allSuppliesVerified: false`.

Each target keeps the original physical input, progress, collection-policy
identity, source revision and SHA-256 of the canonical complete proof. The native
trainer reruns this verifier rather than trusting a stored `verified` flag.
Source family/group/split assignment must come from retained admission evidence;
it is not created or shuffled by the completion trainer. Future continuations
are measured from that supplied source, whose earlier provenance remains the
responsibility of the admission record.

## Native workflow

Keep all models, datasets and reports outside the repository. Every output path
must be new. None of the commands promotes a model or changes a running session.
Build the binaries using the repository's pinned toolchain and lockfile:

```sh
cargo build --release --locked -p neyra-core --bins
python tools/learning/completion.py --help
```

### Initialize after the first model is ready for transfer

```sh
python tools/learning/completion.py init \
  --parent-model /private/models/selected-v5.json \
  --output /private/models/completion-v1-initial.json
```

The selected parent must still have an eligible lineage in the current private
corpus. A syntactically valid checkpoint cannot establish that later exclusions
have not invalidated its training history. Select/export an eligible parent in
the private workflow before running this command.

### Extract genuine completed paths

An actual completed Neyra export can be converted using its retained source
assignment. `source.json` is the admitted source object, or an object containing
it under `source`; it contains the unchanged family, group and split.

```sh
python tools/learning/completion.py extract \
  --input /private/completions/completed-neyra.json \
  --source /private/completions/retained-source.json \
  --source-revision <collecting-commit-or-source-hash> \
  --verifier target/release/neyra-completion \
  --output /private/completions/paths.jsonl
```

Extraction requires a qualified, verified, 40-Spin-Single export with all actual
frames and representative inputs. It records the collector configuration and
model selection in the collection-policy digest. It derives suffixes beginning
at eligible nonempty boundaries, preserving their common source group, then
replays them before writing. An incomplete best-so-far export cannot supply
stage-two labels.

Already recorded proofs can be prepared without an exported whole route:

```sh
python tools/learning/completion.py prepare \
  --paths /private/completions/paths.jsonl \
  --verifier target/release/neyra-completion \
  --output /private/completions/targets.jsonl
```

This removes duplicate/reflected training inputs only within the same remaining
target and collection policy, choosing one best replayed cost. It preserves all
raw input proofs and rejects conflicting split assignments.

### Fit a separate CPU candidate

```sh
python tools/learning/completion.py train \
  --parent-model /private/models/selected-v5.json \
  --parent-labels /private/lineage/exact-v5-draw.jsonl \
  --parent-proofs /private/lineage/observations-index.jsonl \
  --paths /private/completions/paths.jsonl \
  --verifier target/release/neyra-completion \
  --model-output /private/models/completion-v1-candidate.json \
  --report /private/reports/completion-v1-training.json \
  --epochs 300 --threads 2
```

Supply each exact first-stage lineage dataset with another `--parent-labels`.
Raw-JSONL and browser refined-record digests remain distinct. For a browser
candidate, reconstruct its exact ordered draw from the retained training run;
the entire current corpus is not a substitute for that historical draw.

If the v5 model reused earlier features, also pass each checkpoint named by its
initialization using `--feature-ancestor-model`, and include those models' exact
lineage datasets and needed proofs. This includes the retained v2 feature source
when applicable. Missing ancestors, altered dataset hashes or cross-split
family/group/reflected-board collisions stop training. A different output task
does not erase earlier feature-training dependencies.

The fitting floor is four unique training inputs in at least two source groups,
plus two validation and two test inputs in independent groups. These are only
minimum conditions for executing a pilot fit, not evidence that the model is
finished. Duplicate suffixes from one run do not create independent groups.

The supplied trainer freezes convolutions, fits the hidden/output layers with
SmoothL1 loss on locks divided by 140, and selects by validation loss. The
unchanged initializer remains eligible at epoch zero. Test data receives no
gradients and is measured only after selection. Reports contain exact proof
identities, parent lineage, counts, CPU execution metadata, selected epoch and
MAE/RMSE alongside a training-mean baseline. They keep
`learningComplete: false` and disable automatic promotion.

## Rust, WASM and inference

`CompletionModel::from_json` checks task, architecture, progress semantics,
source initialization, rule profile, tensor sizes and the actual float32
little-endian SHA-256 digest (`sha256-f32le-neyracompletion-v1`).

Native inference reads JSONL inputs with `rows`, integer `remainingSpins` in
1–40, and `rulesProfile`:

```sh
target/release/neyra-completion --model /private/models/completion-v1-candidate.json
```

WASM exposes `completion_model_contract()` and a reusable
`CompletionEvaluator`. Call `evaluate(rowsJson, remainingSpins, rulesProfile)`
and release the evaluator with `free()` when replacing it. A trained prediction
returns `observedRemainingLocks`; an untrained model or unsupported board returns
`null`, while invalid progress/rule input is rejected. Predictions are bounded
to the artifact's 0–700 reporting range, never used as admissible search bounds.

The existing `NeyraEngine.with_model` path selects only a first-stage trained v5
model for experimental completed-boundary ordering. Stage-two inference does not
activate that path, replace the user's mino limit, or alter physics, P7 proofs,
probability admission or public model selection.

## Remaining evidence before adoption

No actual second-stage training completion is claimed by these tools. Adoption
still needs diverse independently grouped genuine completions, known source
eligibility, validation and held-out test results across remaining-target ranges,
comparison against simple cost baselines, Rust/WASM numerical parity, and matched
search comparisons. Observed completions are selected successes; this version
does not estimate the failure probability of arbitrary fields or generalize a
finite successful route into guaranteed 40-line completion.
