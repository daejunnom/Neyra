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
- Keep af8dac63a9a9155fc28e480246fd617bc855bfb1 as the fixed strongest-source
  reference. Build candidate algorithm changes from that source and preserve
  the current probability/transport contracts. Improvement over the shipped
  version alone does not establish parity with this fixed reference. Compare
  matched rules, settings and work; keep benchmark tools and results outside
  the repository. Preserve existing test modules and do not commit new tests.
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
  explicit saved user settings. The independent per-bag floor defaults to 100%
  and applies from bag two; the first-bag floor is separate, as below.
  Default the hard search cap to 126 pieces and keep the preferred goal at 119.
  Admission must satisfy every bag floor and the exact cumulative product,
  including the terminal cost CDF. Remaining probability requirements must
  increase after probability has already been consumed.
  Independent worker profiles must keep separate certificates.
  Combining their displayed results does not combine their P7 success sets.
- From bag two, permit a non-100% construction certificate only when that
  fixed plan executes at least `risk_min_spins_per_bag` Spin Singles in the bag.
  Default this configurable integer to 3; allow 0 through 6, with 0 disabling
  the extra condition. Below it, require all 5,040 supplies, including the
  terminal cost CDF. Use actual plan gain, never forecast or alternate-path
  spins. Preserve explicit saved zero values. The first bag keeps its separate
  construction floor and exclusion setting.
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
  A completed nonterminal zero-clear tiling whose full logical union proves
  every target quiet may also use a necessary two-prefix geometry check before
  expensive refinement. Check both spawns and one coherent orientation for
  each ordered first/next pair; uncertainty bypasses it. Geometry cannot replace
  actual input reachability or certify quiet coverage.
- Keep proposal ordering distinct from proven bounds. An open passage or an
  unfinished preparation goal is not an executed spin witness. A canonical
  terminal plan can contain seven targets while each supply needs fewer locks;
  never prune it using that canonical placement count as its minimum cost.
- Treat every nonterminal first bag as a foundation, including mixed rounds
  and prefixes that incidentally clear. Compare its workspace, side banks,
  preparation and dependency quality without an actual-spin rank bonus. Do not
  apply first-bag peer-spin, same-depth spin-gap or efficiency cuts. Retain real
  spins, exact certificates, necessary material/cost bounds and terminal priority;
  this exception does not ban first-bag spins or suppress small-goal completion.
- After the first bag, within the same bag depth and probability resource, search ranking must preserve the bounded spin contract: field quality
  can reverse a one-spin difference, never a two-spin difference. Keep composite
  forecast/terrain quality separate from the actual-spin rank; apply that rank
  normalization once and add no bonuses afterwards.
  Normalize scheduling quality in spin units before dividing by productive
  locks, so its influence does not grow with bag depth. Explicit exploration
  age/attempt preferences remain scheduling choices, not field evidence.
  Temporary scheduling cooldowns must not change the archived best route.
- The displayed partial record first prefers certified paths of at least two
  bags over one-bag foundations. Compare two foundations by finite raw field
  quality, first-bag probability, counted probability, actual spins, then fewer
  locks. Missing quality remains unknown. Display foundation quality as its main
  metric. For two later paths, use spins*countedProbability/max(7, placed),
  then higher counted probability, more bags and more total spins.
  This is a selection score, not an expected completion-time certificate.
  Scheduling uses R=spins*7/L + 0.2*quality/(120+abs(quality))*7/L,
  where L=max(7,placed-7), then (R+shift)*countedProbability-shift with
  shift=0.2*7/L. Keep quality and its risk shift on the same depth scale.
  At probability one, return this R expression exactly. Pending plans
  use their verified parent's probability, never invented future coverage.
  Use raw field quality, fewer locks and higher effective probability for
  remaining later-path ties. First-bag scheduling sets productive spins to zero;
  it does not remove actual spins from certificates or later-path efficiency.
  Include the setup bag in every public efficiency value; only internal
  scheduling may exclude it. Keep the total-spin maximum as separate progress.
  Reset only for a new run; compare records identically in Rust and TypeScript.
  Same-ID quality improvements must replace old fields and archived exports.
- Retain two qualified, verified whole-bag partial routes per placed bag count,
  independently of live-tree replacement. Keep distinct plan IDs, complete
  certificate chains and representative frames; never combine alternatives'
  success sets. `bestByBag` remains compact even in full/export snapshots.
  Export a retained route through its exact candidate ID; a missing ID must
  return no candidate instead of another route. In the browser, scope IDs by
  worker and run, retain the global two per depth plus one explicit selection,
  and capture new records before advertising them. A selected route stays pinned
  through new progress, pause, cancellation or worker failure, until another
  selection or a new run. Serialize its full proof only on a record change or
  selected export, not on every current-field detail update. Keep terminal
  completions separate from these whole-bag partial records.
- Under a relaxed overall floor, compare verified probability-resource use
  with uniform logarithmic consumption to a fixed selected horizon. Add
  `24*clamp(pacedFraction-spentFraction,-1,1)` only to temporary scheduling
  quality. Ahead of pace is positive; early overspending is negative. Pending
  plans use their verified parent's counts. Horizon widening must not refill
  the resource, and 100% disables this adjustment. Export the projection and
  its meaning without changing stored field quality, probability, or admission.
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
  Keep supply reflection separate from optional root geometry reduction.
  The adopted af8-based generator partitions first-placement proposals among
  workers without the later chronological reflected-prefix cut. Preserve
  piece-to-target assignment, deletion history and execution context; board
  occupancy alone must not merge distinct fixed tilings.

- Share only monotonic verified same-depth spin benchmarks between workers.
  Enable those raw cross-path spin/quality cuts and peer hints only when the
  overall success threshold is 100%. Under a relaxed threshold, keep raw maxima
  as progress without letting riskier paths prune safer probability resources.
  Physical/material cuts, absolute pace, bounded beam and same-parent comparison
  remain separate. Never combine worker success sets into one certificate.
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
- Body-clearance connectivity is an ordering approximation, not an input or
  supply proof. In the selected low connected working cavity, retain a terrain
  residual of 0.5*holes + 0.05*cover + 0.5*buriedSplits. Charge that geometry
  once instead of stacking ordinary hole, historical debt and pursuit penalties
  on the same connected cells. Outside that region retain the ordinary cavity,
  cover and repair costs plus historical rates 4 in prefixes and 12 at boundaries.
  Keep the full raw debt mask for row mapping. Opening a roof is not repayment;
  current classification may make a cell neutral without deleting its history.
  Forecast repayment begins only from initially charged history and counts
  actual fills/deleted rows in one first-bag continuation. Never credit initially
  uncharged working cells as newly repaid debt.
  Use only a zero-preparation, actually replayed first spin as immediate_anchor
  on the original board. A prepared/later anchor is route evidence only.
  Residual cover/splits and the ordinary far-hole base do not all vanish merely
  because an immediate footprint explains a hole. The retained cavity_roof_tops
  allowance remains a static connected-cavity roof model, not a new input proof.
- Analyze all connected workspace-depth branches and break static board ties
  in a canonical reflected orientation. On a self-reflecting board, alternatives
  are not simultaneous resources: use one goal-aligned workspace for terrain
  and only their shared connected region for neutral historical cost.
  Keep finite-budget forecast differences distinct from static reflection.
  Gate the small unresolved-preparation bonus with preserved_after: withhold
  that bonus when a non-target finisher row would close. Do not imply this guard
  filters every quiet/support proposal. The retained accepts_support checks
  footprint emptiness and useful support; ready() and actual execution must
  still clear exactly one row. An arbitrary proposed quiet alternative is not
  a certificate that the intended finisher remains a valid Spin Single.

- Prefer dense side banks beside accessible low working space, without fixing
  the center column or rewarding absolute stack height. Give an existing roof
  over a concrete single goal a strong capped preference, independently of
  whether the rest of the field is damaged or its total Terrain is negative.
  A static access warning does not suppress this shape reward, but all access,
  hole and debt costs remain separate; the roof is not an input/spin witness.
  Preserve deep one-column repair dependencies and excess towers above a roof.
  Every root prefix keeps foundation-dependency weighting in mixed rounds too.
  Use `k*setup_score + weight*structural`, with weights .06 for locks 1–3,
  then .12/.18/.24/.30 for locks 4–7, and k=1 or .35 for a relaxed setup round.
  Use at most two points of ordering jitter. A first-bag pursuit may refer to a
  next-bag finisher already spent in this inventory if its footprint remains
  preserved; actual clears release it. This is ordering, not permission to clear.
- Keep the af8 preparation gradient: the selected low workspace uses explicit
  width/depth preference and weaker-side occupancy density, plus the increased
  concrete prepared-roof value recorded in STRUCTURAL_EVALUATION.md. Do not
  multiply these positive preferences down merely because the finishing spin
  has not yet been executed. They remain shape preferences and cannot add actual
  spins or manufacture probability. Lower the width-two workspace preference;
  a shallow low width-one notch earns at most 0.8 and avoids its generic narrow
  gap charge only with a floor immediately below and body clearance above.
  This notch is not body clearance itself and does not erase hole/access/debt
  obligations; long one-column shafts and edge I spaces are not such notches.
  Do not claim ordinary bank preference uses the later wall-attachment blend
  or split-channel penalty, or that an unconditional outward slope is active.
  Only an actual immediate
  input witness enables directional bank cost: min(4,0.15*sum(inwardRise)
  +0.10*sum(min(max(inwardRise-1,0),4)^2)). Use its nearby selected workspace
  or actual footprint; leave unfinished prefix ordering unchanged. Only when
  that immediate goal lies more than two rows above the inherited lower floor
  may the goal-relative active-roof formula replace the ordinary roof reward,
  using the same increased cap. Keep inherited cells, raw debt and probability
  resources; a higher workspace never resets the board.
- Charge every adjacent column edge for height dependence, independently of
  selected goal/workspace protection: difference two is mild; three and higher
  have a strong increasing cost. Only actual suspended occupancy over useful
  working air reduces its explained portion to 25%; height above that roof
  keeps its full cost. A detached roof after clears may qualify for this relief;
  arbitrary floating blocks cannot. Replace the old cliff subtotal rather than
  charge it twice. Sum reflected edge pairs in a fixed order so rounding cannot
  change a preparation-threshold branch under reflection.
- Quiet-placement ordering applies to all piece types. Reward a newly formed
  actual working roof directly; if new height dependence is large, compare a
  single remaining mino's roof-making placement with ordinary leveling. Reserve
  the intended finisher, preserve its empty cells, reject hypothetical clears
  and unrelated unsupported gaps, and take one best alternative. Retain a
  dependency cost for consuming that specific piece. Keep this bounded geometry
  pass separate from actual input/P7 proof; do not perform new flood analyses
  or recursive search for each counterfactual placement.
- Add bounded curl potential and straight-wall cost around the selected low
  workspace, within two rows of the inherited lower floor. Follow at most five
  rows of actual empty intervals, width 2–6, between occupied sidewalls. Adjacent
  rows must share at least two adjacent body-clearance cells. Keep coherent
  paths separate; alternative branches must not donate lip columns to each other.
  Curl potential uses the increased explicit per-column reward and cap in
  STRUCTURAL_EVALUATION.md, taking one best coherent path, with L=1/.6/.3 at
  levels 0/1/2. Count inward steps only from a gap at least
  three cells wide, with occupied cells continuously attaching the new lip to
  the previous sidewall. This goal-free potential is separate from the
  concrete prepared-roof reward. Floating islands, closed lids, repeated lip
  columns and height outside the five-row band add no curl preference.
- Add the worst connected straight segment's cost to terrain defects D:
  min(8,3*max(rows-2-explained,0)). An unchanged gap over 3/4/5 rows costs
  3/6/8; a lateral shift or changed width ends that segment. Do not sum costs
  from alternative segments or apply the curl's level weight to this cost.
  Only an exact immediate spin witness whose clearing footprint uses that
  segment may set explained=1; every other segment keeps its charge. A curl
  above a straight segment earns its own potential but does not explain a
  straight window. Preserve one-column notches and edge I spaces outside this
  2–6-cell rule. Terrain gains curl potential minus straight cost. These terms
  do not change access exemptions, raw hole/debt masks, repayment, actual spins,
  input reachability or P7 probability.

- Continuation forecasts span two bag inventories and seek six coherent
  Spin Singles. Open the second inventory only after all seven first-bag labels
  have been placed and at least two actual forecast spins have occurred.
  Reset the bag-local start board, deleted rows and supply mask at that real
  boundary. Keep at most six distinct typed paths and six preparation recipes;
  interleave finisher types and target positions before repeating one slot.
  Never sum alternatives'
  spins, debt payments or supply success. First-bag gain/debt/obstruction remains
  separate from the total. With first/second gains g1/g2, first-bag quiet count q,
  paid initially charged cells p and terrain change dT, use
  A1=-10*q+6*p+0.25*dT, forecastLine=60*g1+18*g2+A1, and select lines with
  g1+0.3*g2+0.45*A1/(120+abs(A1)). Add the best readiness and independence
  evidence once. Boundary quality is 2*terrain+forecast-12*unpaidHistory.
  Keep this direct raw composition; do not reintroduce 120*totalForecast or
  an additional 48-point saturation that suppresses all secondary differences.
  Never assume an unprobed prefix already has forecast spins. Extra work needs
  actual productive completed inventory, not open sky or an unfinished goal.
  Keep the base/second-bag work caps 64/96 and 320/352, with reserved evidence
  checks sharing the cap. Stop deeper expansion at six spins.
  Forecasts are move-ordering hints, not P7 certificates or mandatory prefixes.
  Public progress counts executed verified bags. A missing forecast is unknown.
- Continuation preparation may use up to three actual zero-line placements,
  retaining both preserved goals and fresh alternatives. Allow one bounded
  goal-free bank/cavity preparation after a spin or at the initial continuation
  state; reserve an available ordinary placement inside the working cavity as
  well as a general bank candidate. Every target-specific support must preserve
  its goal. Keep input checks, inventory use and existing work caps; readiness
  credit still requires at most two preparations. Diversify final paths by
  finisher and target, and keep at most one independently replayed recipe pair.
- Reserve one phase of every fourth exploration cycle for the deepest eligible
  certified boundary of at least two bags. Retain stale, age, material, pace and
  quality gates and verify waiting plans first. Protect a bounded set of untried
  deep leaves during full-tree replacement, in addition to fresh foundations.
  This schedules real continuations; it is not completion evidence or extra spins.

- Retain forecast count, two-bag distribution and placement count in full,
  compact and exported summaries. Preserve the fixed plan's supply proof,
  actual frames and probability. This source does not implement automatic
  forecast-only refresh of an already known P7 key. A future refresh must prove
  matching tiling, boards, actual counts, debt and certificate context first.
  Temporary scheduling uses the fading positive forecast plus predicted paid
  history, not a rewritten archived field quality or probability.

- Keep pending-spin readiness and order independence as separate capped field
  features. Use 15/(1+quiet) for an actually replayed first spin after at most
  two quiet preparations; an unresolved preserved goal with missing<=4 uses
  5/((1+missing)*(1+quiet)). Take the best preparation evidence, never a sum.
  A disjoint pair that works in both orders scores 20/(1+quietA+quietB), at most
  one best pair. The combined preparation/independence cap is 35. Use one best
  concrete preparation/first-spin witness and one best
  disjoint recipe pair checked in both orders with actual single clears and a
  common end board. Do not sum alternate slots, count a repeated finisher twice,
  or turn pairwise geometric order tolerance into a P7 probability. Extra target
  checks share the existing continuation work cap; retain diverse finishers.
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
  Show the best and runner-up for each placed bag count with actual spins per
  bag, counted chance and field quality. Explicit selection controls replay,
  the shown bag probabilities and exports together; returning to the recommended
  route resumes automatic best selection. At equal chance, six spins in three
  bags outranks nine in five, while both depths remain separately inspectable.
  Keep playback on the retained best route; the current-search tab shows only
  its final board and has no autoplay or replay controls. Show all certified
  per-bag construction probabilities on the best route, including 100%, and
  identify the first bag when excluded from the counted cumulative probability.
  The compact best-route field evaluation displays that route's final raw
  fieldQuality, not normalized rankingScore or the currently replayed frame.
  Same-ID quality improvements must update it; missing values remain unknown.
  Keep desktop controls within an ordinary Chrome viewport and label an exact
  100% threshold as 100%, without an "at least" suffix.
  CTK3/Fumen documents remain bag diagrams. Recover their colors only from
  recorded pre-clear cells and actual line compression, never mask differences.
  Preserve all 24 physical rows in CTK3 and reject Fumen exports with an
  occupied 24th row instead of cropping or using its separate garbage row.
  Report clipboard success only after the browser write succeeds.
