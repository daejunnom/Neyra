# CPU Naga Eyes spin evaluator

Neyra separates public search from private CNN learning. Public GitHub Pages uses
the existing structural search evaluator. The developer-only learning Site uses
fresh, bounded continuation observations to train an occupancy-only CNN.

## Public search

- A fresh nonzero Web Crypto seed is generated on every page entry. A manually
  entered seed then remains reproducible for the run; saved seeds are not restored.
- Learning-data contribution defaults **on**, with an opt-out and upload status.
  Public learning is **off**. Old saved `cnn_evaluation` values do not enable it.
- Only retained, verified completed-bag records are submitted. Inputs contain the
  current rows, run and candidate identity, bag depth, seed, rules and search
  configuration. They contain no heuristic teacher or invented continuation label.
- The bounded queue batches at most eight fields, prefers deeper fields, retries
  temporarily unavailable uploads, and stops/clears pending work on opt-out.
- The public endpoint validates input and applies quotas. A client certificate
  claim is untrusted metadata; the private collector measures its own labels.

## Private developer learning

[Neyra Learning Studio](https://neyra-learning-studio.daejun0311.chatgpt.site)
is owner-private. Data provision and learning are fixed **on**; there are no mode
selectors. Start/pause controls limit CPU work without changing those modes.
The browser must remain open for its CPU worker to run.

The worker prioritizes nonempty boards after at least two completed bags. It
observes the next two or three bags under one declared policy, rather than
restarting the empty first bag or searching for an entire completion. A session
is bounded to three cycles, each with up to four label jobs and 20 training epochs.
The initial model and offline collector use the same schema and can be run by an
agent without relying on an open browser tab.

Sites D1 stores boards, source groups, fixed splits, observation indices, model
indices and training runs. R2 stores the full observation witnesses and model
JSON. A same-origin authenticated API supports collection and model uploads.
The Site's existing service credential also permits agent uploads and readback;
it must never be copied into public code, command arguments, files or logs.

## Occupancy-only inference

The model has **33,442 parameters**, including **21,056** in the convolutional
extractor. Input is a `1 × 24 × 10` occupancy tensor, bottom row first.
The lexicographically smaller board/reflection is used for geometric invariance;
this does not claim identical mirrored physical supply coverage.

| Layer | Shape |
| --- | --- |
| 3×3 convolution + ReLU | 1 → 24, padding 1 |
| 3×3 convolution + ReLU | 24 → 32, padding 1 |
| 3×3 convolution + ReLU | 32 → 48, padding 1 |
| Average pooling | 6×5 → 48×4×2 |
| Dense + ReLU | 384 → 32 |
| Output | 32 → 2 |

The two outputs are `foundSpinLikelihood` and `observedSpins`. They estimate
discovery under the model's fixed finite observation policy. They are not P7
construction probabilities, proven maxima or impossibility judgments.

Rust evaluates the model using ordinary CPU arithmetic, including in WASM.
Offline learning uses CPU PyTorch; the private Site uses TensorFlow.js core,
layers and its CPU backend. No CUDA, WebGPU or WebGL backend is connected.
Neither public search ordering nor its archive exports automatically invoke CNN
evaluation. `learnedOrdering.enabled` remains false for compatibility.

## Fresh observation contract

Schema 2 has task `naga-spin-observations-v2` and policy
`neyra-fixed-bag-continuation-v2`. It accepts only current occupancy and provenance
from boundaries of at least two completed bags. Existing board scores, Terrain
scores, opener preferences, shape objectives and path-count targets are rejected.
The v2 model starts from fresh weights instead of a teacher-trained v1 checkpoint.

The labels are measured discovery of at least one spin and the observed count.
Input-reachable prefixes and exact P7/quiet-safe completed chains remain separate
in the evidence. Exhausted search is censored; an unobserved certificate maximum
stays unknown. Work, nodes, attempts, seed, horizon and labeling-source hash are
recorded so incomparable policies are not silently mixed.

Whole opener families and search runs remain in one train/validation/test split.
Descendants inherit their source group. Identical and reflected occupancy is
deduplicated globally, while each stored physical witness retains its original
orientation. Checkpoint choice uses validation only; test records do not enter
gradient updates or browser checkpoint selection.

The browser dataset has a bounded recent-plus-coverage sample per split, with
explicit total and selected counts. New observations can therefore enter training
after the dataset grows beyond the per-session capacity.

## Reusable command-line tools

Build the release binaries with the repository's locked toolchain, then run the
commands with data and reports outside the repository:

```sh
cargo build --release --locked -p neyra-core --bins
python tools/learning/prepare.py --help
target/release/neyra-label --help
python tools/learning/train.py --help
python tools/learning/remote.py --help
```

`prepare.py` imports only deep occupancy/provenance, groups sources and removes
reflected duplicates. `neyra-label` collects new bounded continuations.
`spin_contract.py` validates observations, model dimensions and hashes.
`train.py` fits the CPU network and emits its model plus evaluation report.
`remote.py` uploads labels/models and checks the stored content by actual readback.
Use `NEYRA_SITE_URL` and `NEYRA_SITE_TOKEN` in the process environment for private
uploads. HTTPS certificate verification is required and redirects are refused.

The selected runtime checkpoint is
`crates/neyra-core/models/naga-spin-cnn-v2.json`. Its weight digest is SHA-256 over
the fixed sequence of float32 little-endian tensors, identified by
`sha256-f32le-neyraspin-v2`. JSON number formatting does not change that digest.
WASM exposes `evaluate_board`, `evaluate_board_model`, `spin_model_json` and
`label_board` for the private learning worker.

## Public queue and private persistence

The public Pages origin cannot receive a private Site service credential.
`supabase/contribution-schema.sql` and the `neyra-contribute` Edge Function form
a bounded bridge: Pages submits input-only fields, and the private Site pulls
them with a separate server-only consumer secret.

All queue/settings/quota tables enable RLS and deny anonymous/authenticated table
access. Only the ingress service can admit fields; only the private consumer can
pull or acknowledge them. Quotas bound batch size, daily volume and pending rows.
Deduplication promotes an existing occupancy only when a deeper record arrives.
Receipts include both board key and depth so an old acknowledgement cannot hide
a newly promoted field. Supabase is the transient queue; D1/R2 is the durable
learning store. First-bag fields are explicitly excluded by the private importer.

The Site runtime needs `NEYRA_QUEUE_URL`, `NEYRA_QUEUE_ANON_KEY` and
`NEYRA_QUEUE_SECRET`. The public build contains only the endpoint and public anon
key. The consumer secret is stored privately and only its hash is retained in
the Supabase settings table. No database service-role key enters public code.

Datasets, new tests, one-time experiments and measured reports stay outside the
product repository. Reusable collection/training tools and the selected model
are versioned. Transformer work remains deferred until substantially larger,
independent data and evaluation evidence justify it.
