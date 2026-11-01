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
- Develop from the latest accepted best source. The user's November 2026
  instruction retires the frozen af8 source baseline; do not restore or maintain
  a parallel historical implementation. Accept changes against the immediately
  preceding accepted source using matched rules, settings and work, preserving
  probability and transport contracts. Keep benchmark tools and results outside
  the repository. Preserve existing test modules and do not commit new tests.
- Exclude O spins in this build-search profile. Keep physical rotation data
  separate from search-state equivalence: an origin-only rotation need not
  become a separate geometric search candidate.
- Do not commit performance experiment scripts, benchmark harnesses, raw runs,
  or benchmark result reports. Keep these outside the repository. Evaluate
  additional computation sharing with matched-work ABBA measurements before
  adopting it; worker-local caches do not imply a global shared search DAG.
- Public Pages always uses the existing structural search evaluator. Accept the
  old cnn_evaluation setting for compatibility but ignore it in ranking. Public
  learning-data sharing defaults off and requires explicit versioned consent;
  legacy default-on settings are not consent. Learning stays off. Generate a
  fresh nonzero random seed on page entry, then honor the user's manual seed.
  Preserve physics, probability, necessary bounds and supply certificates.
- The CPU CNN learns current occupancy only and predicts bounded-search spin
  discovery and observed spin count. Do not mix in structural teachers, Terrain
  scores, shape preferences, opener priors or path-count targets. Preserve the
  released two-bag v2 checkpoint. Explicit v3 initialization may reuse its
  convolution/hidden features only with a reset output head and recorded source
  provenance; it must remain marked untrained until actual v3-only fitting.
  Keep history, inventory, forecasts and opener identity out of inference inputs.
  Inspect actual accessible source before using another engine's board features;
  never infer closed-source evaluation or copy private reference implementation.
  The portable Rust/WASM CNN uses no GPU connection or runtime ML dependency.
- Learning observations begin at nonempty completed boundaries of at least two
  bags and use two or three new inventories with a fixed declared work/node/
  attempt budget. Prioritize deeper fields instead of empty first-bag or full
  completion searches. Keep input-reachable observations distinct from exact
  P7/quiet-safe chains. Limited search is censored: a miss is not impossibility,
  and the discovery head is not construction probability. Split whole opener
  families and search runs; isolate identical and reflected boards across splits.
  New v3 labels predict only the first new bag, with targetHorizonBags=1, while
  the collector still attempts at least two new inventories. Retain the actual
  reached depth, zero-spin endpoints, physical input witnesses and censored
  capacity/work limits. Do not apply peer score, completion pace, sibling
  futility or all-P7-parent gates in this learning-only path. Public certified
  search keeps those separate rules. Reserve capacity for deeper continuation.
  Commit reusable collection/training/storage tools and the selected runtime
  model, but keep datasets, new tests, one-time experiments and result reports
  outside this repository. Record dataset, labeling-source and model hashes.
- The developer learning Site stays owner-private with sharing and learning on,
  without mode selectors. Analyze received fixed-policy observations first;
  do not rerun the collector to consume an already supplied observation. Only
  when both the received analysis queue and relay are empty may the versioned
  collector continue from an actual received result board or a recorded
  input-reachable descendant. Do not generate empty first-bag/bootstrap searches.
  Sites D1/R2 holds observations and model artifacts; Supabase is the bounded
  public relay. Received labels remain client-reported even after structural
  validation; independent private measurements retain their separate trust.
  Keep depth-qualified trace receipts and identity-qualified observation receipts,
  acknowledge only stored or explicitly excluded inputs, and preserve source
  groups across descendants. Never expose privileged keys in browser bundles,
  logs or commits, or allow anonymous access to private labels or model writes.
- Capture public search traces only after explicit consent, using existing
  retained frames and worker totals. Do not retrospectively upload old traces.
  With consent on, use a verified boundary of at least two bags as an origin
  for the v3 fixed-policy collector, then prioritize deeper replayed result
  fields, including zero-spin boundaries. Existing retained boards may seed new jobs.
  Use the selected worker count, a bounded global frontier, and parent delivery
  acknowledgement before dispatching descendants. Keep queued observations
  under backpressure rather than evicting parents. With consent off, discard
  new collector outputs, stop new collection/upload and resume ordinary search.
  Per run, cap each shallow origin depth (bags two and three) at two broad roots
  and four total jobs; then let ordinary search find deeper eligible origins.
  Preserve these limits through consent toggles, and refund unstarted jobs when
  a worker completes before dispatch. Do not invent certified deeper fields.
  Keep heuristic scores and bounded path observations
  in separate untrusted trace records, never in the CNN's supervised targets.
  Reject old default-on ingress envelopes. Deduplicate trace identity separately
  from training occupancy so different runs of a known board retain evidence.
- Keep queued frontier keys separate from dispatched-job history. Evicted or
  unstarted proposals must remain eligible; admission limits still apply.
  Unassigned public workers continue broad search during focused collection
  and acknowledgement backpressure. Immutable wire payloads may be shared,
  but consent generations and parent acknowledgements must remain independent.
- Private parallel claims are atomic and lease-qualified. Received acceptance
  and its label insertion must commit together; late work cannot bypass
  quarantine or release another worker's lease. Reuse exact validated model
  weights within each worker, with private engine instances and memory.
  Private CPU capacity follows the selected hardware thread count up to 128;
  candidate training shares at most 20 attempted epochs per cycle, at most ten
  per candidate and two concurrent GPU candidates. Preserve the parent as a
  validation candidate and record actual allocation and backend metadata.
  Load the training runtime only for training work. Interrupted recorded child
  admission must be repairable from the immutable parent proof, with a bounded
  repair batch after received-work priority checks; do not rerun collection.
- Preserve the released v2 checkpoint and legacy collector semantics. V3 uses
  an explicitly different policy, model task and one-bag output contract; never
  relabel a two-bag prediction or mix versions in a training dataset. Private seeded
  exploration changes proposal ordering only, within the same declared budget;
  store its seed, ordinal and distinct policy. Its replayed child boards may
  enter fixed-policy collection, but stochastic labels must not silently mix
  with the fixed-policy training set. Pin the served champion independently of
  newer private candidates; training never promotes a checkpoint automatically.
- Retire old opener/bootstrap/perturbed examples from the active corpus. Keep
  immutable historical proofs/exports only for previous model provenance.
  Use corpus-qualified canonical board identity so old labels cannot silently
  enter the new public-results training set. Preserve all physical witness
  orientations and per-link trust when attaching ancestor feedback.
  Keep historical geometry/run split constraints even after corpus retirement.
  Assign a durable source group before admitting its first board. A new group
  first inherits an unambiguous existing geometry component's split; hash a new
  component only when no prior constraint exists. Record duplicate-only source
  occurrences and exact orientations independently of canonical training rows.
  Never move historical groups/geometry between splits or clear old exclusions.
  Quarantine a genuine cross-split component conflict and its descendants.
  V3 proofs keep their actual measurement group, physical orientation and source;
  an excluded first canonical owner must not poison a later independent group
  that inherits one unambiguous historical split. Keep immutable proofs per
  measurement group and select one earliest eligible representative per canonical
  board and policy for fitting. Legacy v2 eligibility remains unchanged. Full
  exports retain evidence; derived training files honor manifest exclusions.
  Excluded raw aliases in the active corpus without qualifying prior observations
  cannot impose new split constraints.
  Keep legacy corpus memberships authoritative. An excluded active canonical
  owner anchors historical membership only when that board had an observation
  from its own measurement group, or a legacy NULL measurement slot, no later
  than its exclusion. Another group's observation does not authorize that owner.
  Otherwise retain the alias as raw provenance. V3 dataset split selection
  follows the actual measurement group's durable assignment. Its own observation
  no later than exclusion also preserves that group's historical membership when
  another source owns the canonical alias.
- CPU remains the default. Optional private GPU learning must finish an active
  unit before switching off and retry failed GPU work from retained CPU weights
  and optimizer state. Native training uses epoch boundaries; ORT inference
  uses batch boundaries. The Site uses bounded parallel candidate workers and
  actual backend metadata, with validation-only selection and the unchanged
  parent eligible. A portable model's device=cpu does not describe training
  hardware. ONNX export must verify CPU parity without replacing the frozen
  JSON checkpoint. Rust placement search remains CPU; do not label it GPU search.
- Limit private CNN decision capture to the supplied board and one selected
  successor. Keep predicted values, allocated work, consumed work and measured
  outcomes distinct. Ancestor feedback needs an exact physical recorded bridge
  and witness references; it is a censored path lower bound, not a replacement
  for the original versioned label. One-bag targets and coherent measured
  two-bag path totals remain separate. Do not fabricate remaining-to-40/completion
  cost labels from unfinished searches. Adaptive budgets and new cost heads need
  separate policy evidence before activation. Keep the current mino controls.
- Export complete immutable private snapshots separately from bounded training
  samples, with cursor/count/hash checks and atomic local latest-pointer updates.
  Preserve earlier snapshots. Fine-tuning requires the retained parent model and
  parent dataset provenance, stable family/reflection splits, and validation-only
  checkpoint selection with the unchanged parent as an eligible candidate.
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
  The accepted generator partitions first-placement proposals among
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
  A bounded exact TT may reuse proven quiet-policy rejection only after complete
  relevant execution proof. Compare full oriented context bytes, typed logical
  target assignments, goal and symmetry context. Never cache heuristic misses,
  interrupted work or probability cutoffs as failures. Shared immutable batches
  must not block workers, echo imports, cross run generations or merge success
  sets. Mirror deduplication is for display/training identity only; actual physics
  and certificates keep their original orientation. Retain two distinct final
  field identities per bag before truncation, plus the explicitly pinned route.
- Default to the latest retained route and render Clearra piece colors without
  column numbers or per-bag spin-gain decoration. Playback steps through the
  actual representative piece frames; do not imply that order is universal.
  Show the best and runner-up for each placed bag count beside the board's lower
  right, with actual spins per bag and field quality. Keep the selector compact
  and the desktop controls free of an internal vertical scroller. Omit public
  construction-probability displays while retaining settings and exact proofs.
  Explicit selection controls replay and exports together; returning to the recommended
  route resumes automatic best selection. At equal chance, six spins in three
  bags outranks nine in five, while both depths remain separately inspectable.
  Keep playback on the retained best route; the current-search tab shows only
  its final board and has no autoplay or replay controls. A focused-learning
  view shows its real origin board and next-bag target separately from deeper
  observed continuations, never
  inventing cumulative progress toward 40 clears from that isolated board.
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
