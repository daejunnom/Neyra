# Neyra development and delivery

- Keep the user's fixed-bag, Sonic Drop, adjustable success threshold, and
  minimum-lock objective. The default 119-piece goal is an aspiration, not a rejection
  threshold. The v2 design remains implementation work until its acceptance
  criteria are demonstrated by real search results.
- The SRS+ authority is the ordered data and first-origin rotation behavior in
  TETR.IO's official tetrio.js. Clearra's corrected Rust profile is canonical;
  Neyra consumes the same reviewed fixture through scripts/generate-srs-plus.py.
  Do not maintain a second handwritten kick table or silently change the
  interpretation of O rotations and I half turns.
- Use focused verification for the changed behavior. Check generated rule data
  with `python3 scripts/generate-srs-plus.py --check` and use the committed
  lockfiles when testing Rust and building the Svelte/WASM site.
- The user has authorized the assistant to deploy completed requested Neyra
  changes. After verification, publish the normal non-force main update, follow
  the existing Test and deploy Pages workflow, and verify the deployed result.
  Do not leave deployment for the user to trigger manually, use `[skip ci]` for
  a delivery that is expected to publish, or report a source push as a successful
  deployment. Existing GitHub checks and access controls still apply.
- Keep source research, implementation status, and measured search results
  distinct. Do not copy private reference code into this public repository.
- Exclude O spins in this build-search profile. Keep physical rotation data
  separate from search-state equivalence: an origin-only rotation need not
  become a separate geometric search candidate.
- Do not commit performance experiment scripts, benchmark harnesses, raw runs,
  or benchmark result reports. Keep these outside the repository. Evaluate
  additional computation sharing with matched-work ABBA measurements before
  adopting it; worker-local caches do not imply a global shared search DAG.
- Use explicit static board features and structural obligations for evaluation.
  Do not introduce NNUE, training data dependencies, or learned evaluation.
- The overall construction-success threshold defaults to 100%; preserve
  explicit saved user settings. The first-bag floor is separate, as below.
  Independent worker profiles must keep separate certificates.
  Combining their displayed results does not combine their P7 success sets.
- Keep proposal ordering distinct from proven bounds. An open passage or an
  unfinished preparation goal is not an executed spin witness. A canonical
  terminal plan can contain seven targets while each supply needs fewer locks;
  never prune it using that canonical placement count as its minimum cost.
- Within the same bag depth, search ranking must preserve the bounded spin contract: field quality
  can reverse a one-spin difference, never a two-spin difference. Keep raw
  evaluation separate and normalize once; no bonuses after normalization.
  Temporary scheduling cooldowns must not change the archived best route.
- The displayed partial record uses spins/max(7, placed), then more bags and
  more total spins. For two one-bag records with equal spins, higher first-bag
  construction probability precedes raw field quality. Otherwise use raw field
  quality, fewer locks and higher effective probability for remaining ties.
  Include the setup bag in every public efficiency value; only internal
  scheduling may exclude it. Keep the total-spin maximum as separate progress.
  Reset only for a new run; compare records identically in Rust and TypeScript.
  Same-ID quality improvements must replace old fields and archived exports.
- First-bag probability exclusion defaults on and is conditional on a
  constructible first bag. Preserve its separate chance and explicit saved
  settings; zero successful starts never qualify. The independent first-bag
  minimum defaults to 80%; 0 disables that separate floor. Compute the allowed
  first-bag symmetry whenever the first bag is counted OR this floor is enabled.
  Reflection swaps J/L and S/Z together in the canonical supply results.
  The user accepts rare kick/spawn asymmetry: do not add reflected physical
  validation or a second P7 pass. Keep the chosen orientation for later bags,
  export the symmetry assumption when it changes coverage or cost, and retain
  canonical bits/costs/input evidence without claiming reflected-input proof.
  Traverse the first bag's 2,520 reflection pairs directly. A proven pair cost
  may skip its other canonical query; skipped raw entries remain unknown.
  Distinguish effective model coverage from actual canonical query counts.
- Share only monotonic verified same-depth spin benchmarks between workers.
  Imported hints never create local progress or supply certificates. New
  benchmarks revoke old scheduling leases; backpropagation must not refill
  ancestor budgets. Reserve exploration for fresh starts, untried foundations
  and older eligible deeper nodes, including when replacing full-tree leaves.
- Productivity, pace and inaccessible-mass cuts are selective heuristics, not
  impossibility proofs or exact P7 failures. Preserve real terminal prefixes,
  widen a stalled soft budget, and do not cache heuristic misses as failures.
  Static air clearance is not exact mino reachability; actual spin witnesses
  may override its roof/access penalties. Keep code and documents explicit.
- Same-depth spin/quality disadvantage is a selective futility criterion,
  never mathematical dominance between different boards. Retain its verified
  quality references across horizon widening; do not renew a dominated branch's
  budget just because a larger cost allowance is available. A strictly promising
  short continuation may earn one narrow initial probe. Verified incumbents and
  genuine terminal prefixes must not be erased by scheduler heuristics.
  Pace slack decreases with distance to the chosen horizon; first-bag setup
  protection must not become a permanent seven-lock discount in that cut.
  Ordinary scheduling follows the preferred piece goal. Longer-horizon probes
  must stay narrow and infrequent and must not bypass quality disadvantage.
- Keep one bag's remaining inventories distinct inside the proposal frontier.
  Public nonterminal paths and certified bestPartial are whole-bag boundaries;
  a real terminal prefix may consume only part of its last bag.
- Board equality does not imply the same fixed tiling. Candidate deduplication
  must include piece-to-target assignment and execution context. Logical rows
  through 31 remain valid after clears despite the 24-row physical ceiling.
  Collapse identical post-lock geometry only in proposal lists; retain exact
  orientation queries and ordered-kick replay for verification.
- Worker lanes partition first-placement proposals, not every complete build.
  Preserve independent P7 certificates, compact progress, correct replay
  ownership, manual worker preferences and all-worker pause/cancel semantics.
- Default to the latest retained route and render a monochrome board without
  column numbers or per-bag spin-gain decoration. Playback steps through the
  actual representative piece frames; do not imply that order is universal.
  Keep desktop controls within an ordinary Chrome viewport and label an exact
  100% threshold as 100%, without an "at least" suffix.
  CTK3/Fumen documents remain bag diagrams. Recover their colors only from
  recorded pre-clear cells and actual line compression, never mask differences.
  Preserve all 24 physical rows in CTK3 and reject Fumen exports with an
  occupied 24th row instead of cropping or using its separate garbage row.
  Report clipboard success only after the browser write succeeds.
