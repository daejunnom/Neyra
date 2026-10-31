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
- New settings default to 100% construction success; preserve explicit saved
  user settings. Independent worker profiles must keep separate certificates.
  Combining their displayed results does not combine their P7 success sets.
- Keep proposal ordering distinct from proven bounds. An open passage or an
  unfinished preparation goal is not an executed spin witness. A canonical
  terminal plan can contain seven targets while each supply needs fewer locks;
  never prune it using that canonical placement count as its minimum cost.
- Within the same bag depth, search ranking must preserve the bounded spin contract: field quality
  can reverse a one-spin difference, never a two-spin difference. Keep raw
  evaluation separate and normalize once; no bonuses after normalization.
  Temporary scheduling cooldowns must not change the archived best route.
- The displayed partial record uses spins/max(7, placed-7), then more total
  spins, fewer locks and higher effective probability. Keep the total-spin
  maximum as separate progress. Reset only for a new run; preserve certificate
  ownership and compare records with the same function in Rust and TypeScript.
- First-bag probability exclusion is opt-in and conditional on a constructible
  first bag. Preserve its actual chance; zero successful starts never qualify.
  First-bag reflection swaps J/L and S/Z together. Ordered 180 kicks and spawn
  are not fully symmetric: certify shared reflected edges and keep the chosen
  orientation for later bags. Restricted-policy failures are not a proof of
  physical impossibility. Keep ordinary and reflection-safe policies distinct.
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
