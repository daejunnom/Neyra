# Small CPU board evaluator

Neyra includes an experimental current-board CNN and reusable tools for collecting
limited continuation observations. The structural evaluator is the default.
The **소형 CNN 평가 (실험)** setting adds at most 12 points in either direction to
a certified nonterminal boundary's quality before its usual bounded spin rank.
It does not change actual spins, supply coverage, admission or movement rules.

## Inference

The model has 1,645 parameters. Its input is a 2×24×10 tensor: occupied cells and
empty cells covered by an occupied cell in the same column. Rows run bottom-up.
The lexicographically smaller board/reflection is used for geometry invariance;
this is not a claim that mirrored physical inputs always have identical coverage.
Opener name, bag number, previous placements and forecasts are not network inputs.

Two 3×3 convolutions (2→6→8), ReLU, 6×5 average pooling, a 64→16 layer and three
output heads run as ordinary CPU arithmetic in Rust, including browser WASM.
Python is needed only for training. No GPU, CUDA connection, ONNX server or
downloaded runtime model is required by the website.

Outputs are shape preference, observed follow-up spins and observed productive
fixed-plan choices under the model's recorded labeling policy. These are estimates
of what that policy found with a limited budget, not true maximum spins, exhaustive
choice counts or construction probabilities. They are displayed separately from
verified results. `models/naga-cnn-v1.json` carries the policy and content hashes.

## Current-board teacher

The teacher uses `BoardFacts::terrain_score(None)`, excluding pursuit-specific
goals, previous-field differences, continuation forecasts and historical debt.
The current structural teacher strengthens upper-field height costs above 16 and
20 rows; lower boards keep their prior height cost. The principle was checked in
the inspectable current-board parts of [Cold Clear](https://github.com/MinusKelvin/cold-clear/blob/279edd7c3177ff8077f6a930193397814b281f27/bot/src/evaluation/standard.rs)
and [Fusion](https://github.com/MosaicTETR/fusion/blob/1eb879f34a77b868ef3ef0025714e0094bf06eca/src/eval.rs).
Neyra's coefficients are independent choices. The full Cold Clear evaluator also
uses move-dependent and hypothetical continuation terms, so its total score is
not used as a current-board target. Reference source implementations are not copied.

Published SAO and Sarcelle Naga Eyes completed residues receive a separate weak
shape-preference prior. Their source annotations do not replace measured spin or
supply labels. A source build may rely on hold or movement conventions different
from Neyra; the collector must validate its own continuation from that board.

## Collecting observations

The browser's **학습 필드 저장** exports actual retained, certified whole-bag
boundaries as JSONL. Each input contains `id`, 24 bottom-up `rows` and `source`
provenance with `kind`, `family`, `reference` and `bagBoundary >= 1`. An empty
first-bag start or an uncleared full row is rejected. A completed first bag is a
valid starting point for the second bag; the setup itself is not the label task.

Build the native collector with the committed Rust lockfile:

```sh
cargo build --locked --release -p neyra-core --bin neyra-label
target/release/neyra-label --input boards.jsonl --output labels.jsonl --config label-config.json
```

The config declares `horizonBags` (2 or 3), `workBudget`, `beamWidth`, `profile`,
`seed`, `maxNodes`, `attemptsPerNode` and `sourceRevision`. The collector starts
with a fresh seven-piece inventory at each actual boundary and uses existing
legal-placement and clear-only-hold verification. Only certified parent nodes
are extended into later bags. It does not apply the live 40-spin pace target or
hide an extra continuation forecast outside the declared horizon.

Labels retain actual work, early-stop reason, censored status, per-depth counts,
reachable witnesses and compact exact-supply witnesses. Geometry and certified
coverage remain distinct. A distinct choice is a fixed-target chain, not each key
sequence or supply permutation. Productive choice counts require a positive
observed spin total; all tilings remain available separately for auditing.
If no certificate is found, the certified maximum is null. Search exhaustion
under beam/attempt limits is still censored, not a proof of impossibility.

## CPU training and export

Use the CPU package index and pinned version in `tools/learning/requirements-cpu.txt`.
The training script fixes `device='cpu'` and CPU thread limits; it does not discover
or initialize a GPU. PyTorch's tensor model can be extended later without adding
CUDA infrastructure to this pilot.

Before fitting, assign `source.split` (`train`, `validation`, `test`) and keep a
whole source family/run in one `source.group`. Identical and reflected boards must
be deduplicated across every split. The importer `tools/learning/prepare.py` supports
the preserved Naga-reference analysis and actual native search result files; it
keeps entire opener families and search seeds separate.

```sh
python -m pip install -r tools/learning/requirements-cpu.txt
python tools/learning/train.py --labels labels.jsonl --output learning-output \
  --model-output crates/neyra-core/models/naga-cnn-v1.json
```

The architecture is fixed; its checkpoint is selected by validation loss, with
the independent test split evaluated after fitting. The output includes weights, provenance,
constant-baseline comparisons and predictions for native/Python parity checks.
Raw datasets, checkpoints, reports and new verification scripts belong outside
the repository. A low fitting error on a small sample does not establish better
search or justify replacing the default evaluator.

## Private remote retention

`supabase/learning-schema.sql` is a ready-to-apply private schema definition, not
an already-applied migration. It separates boards, inputs, labeling policies,
measured labels, reference priors and model versions. Tables have RLS and no
anonymous/client access. The agent can use authorized SQL access; the website
contains no privileged database key and does not upload data automatically.

`tools/learning/store.py` validates JSONL, emits transactional SQL or uses a
configured `psql` connection. It verifies content hashes on actual readback after
a push. Credentials are read from `NEYRA_DATABASE_URL`, never command arguments
or frontend code; verified TLS is required. The `pull` command preserves full
provenance for subsequent training. Use `--help` for the exact kind/action options.
An inactive or unselected project is not reported as working remote storage.

Transformer work is deferred until substantially more independent, diverse
observations and stronger evaluation data justify the additional capacity.
