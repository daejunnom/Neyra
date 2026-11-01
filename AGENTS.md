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
- Use focused verification for the changed behavior and committed lockfiles.
  Repeat SRS+ authority checks only when the fixture, generator, or rotation
  semantics change. Keep one-time audits, new experiment scripts and their
  results outside the repository; do not run unchanged physics/supply suites
  for a search-order, terrain-weight or display-only edit. The Pages gate keeps
  one actual search execution check plus type checking and the production build.
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
- Each fixed plan must be quiet-safe for all 5,040 model supplies, regardless
  of either probability floor: every supply needs a legal strategy that reaches
  success or a forced clearing-supply blockage without a failed quiet placement.
  Start with the first piece active and an empty bag-local hold. Quiet locks use
  the current piece; only candidates on a productive clearing continuation may
  be deferred, and releasing a held piece must clear one line. A goal-ending
  held clear may leave an unused current piece in hold. An undeclared piece in
  a terminal plan is a clearing-supply blockage, not a quiet-placement failure.
  End-of-bag held release requires every next-piece type to be spawnable.
  Never restore a free initial hold or treat a quiet failure as tolerated chance.
  A completed nonterminal tiling may identify certainly quiet targets from its
  full logical rows only when those rows match the representative deleted-row
  mask. Use that fact for necessary first-supply checks; uncertainty and genuine
  terminal prefixes must bypass this gate. Preserve first-bag reflection.
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
  minimum defaults to 80%; 0 disables that separate floor. Always compute the
  empty first bag's allowed symmetry, including when its probability is excluded
  and its independent floor is zero; quiet coverage still requires that model.
  Reflection swaps J/L and S/Z together in the canonical supply results.
  The user accepts rare kick/spawn asymmetry: do not add reflected physical
  validation or a second P7 pass. Keep the chosen orientation for later bags,
  export the symmetry assumption when it changes success, quiet coverage or cost, and retain
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
- Apply necessary material checks at each placement. The sum of the cheapest
  current row deficits bounds additional singles from the remaining cells;
  O supplies cells but is never a finisher. Do not confuse that optimistic
  capacity with proof that any spin is reachable. Within-bag pace compares the
  optimistic bag-end total, not an interpolated quota on the current lock.
  Selective prefix spin/quality comparisons must use the same exact parent
  and next piece, with no weaker supply mask or optimistic spin capacity.
  Inventory buckets remain bounded-beam capacity limits, not cross-parent
  dominance proofs. Begin every bag with all seven labels available; a prior
  continuation may order placements, never consume a mandatory prefix.
  Keep necessary cost cuts, selective pace/rank cuts and bounded-beam admission
  distinct. None of the latter proves a board impossible or a P7 supply failed.
  Pace uses the lower SAO/Sarcelle envelope through bag 9 and the continuous SAO
  route from bag 10, scaled to the selected goal and horizon, with slack
  1 + remaining/horizon. Sarcelle's loop rebuild must not grant additional
  preparation allowance. The diagrams are progress references, not certificates
  for the clear-only hold policy.
- Four rounds without improvement reduce a non-root boundary's next lease
  to one narrow generation. Failed retries alone must not widen its frontier;
  productive or improved boundaries retain larger budgets. Fresh root
  foundations keep their separate exploration allowance.
- Increase unexplained-hole cost only as a static ordering preference. Only an
  actual spin witness may exempt its required-empty cells from the extra 0.5
  cost; a speculative goal cannot claim that exemption. Preserve open channels
  and the witnessed roof, and never turn this score into a hard impossibility.
- Prefer dense side banks beside accessible low working space, without fixing
  the center column or rewarding absolute stack height. A capped prepared-roof
  preference must refer to an existing roof over a concrete single goal; it is
  not a spin witness and cannot cancel hole, access or debt obligations.
- Continuation forecasts may span two bag inventories and seek six coherent
  Spin Singles. Consume all seven labels before resetting inventory and its
  bag-local start board, deleted rows and supply mask. Keep at most three
  distinct alternative paths, including piece-to-target assignment; never sum
  their spins, debt payments or supply success. First-forecast-bag spins and
  obstruction/debt relief stay separate from discounted deeper preparation.
  These are bounded geometric move-ordering hints, not P7 certificates or a
  mandatory prefix. Actual progress and public best routes count executed,
  verified bags only. Failure to find a forecast is unknown, not impossibility.
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
- Default to the latest retained route and render Clearra piece colors without
  column numbers or per-bag spin-gain decoration. Playback steps through the
  actual representative piece frames; do not imply that order is universal.
  Keep desktop controls within an ordinary Chrome viewport and label an exact
  100% threshold as 100%, without an "at least" suffix.
  CTK3/Fumen documents remain bag diagrams. Recover their colors only from
  recorded pre-clear cells and actual line compression, never mask differences.
  Preserve all 24 physical rows in CTK3 and reject Fumen exports with an
  occupied 24th row instead of cropping or using its separate garbage row.
  Report clipboard success only after the browser write succeeds.
