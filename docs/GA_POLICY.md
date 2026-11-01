# GA placement policy and model capacity

The placement policy ranks actual legal Sonic Drop locks. It does not predict a
construction certificate. The physical GA experiment separates this policy from
the exact bag-supply checker used by later search. A physically replayable
40-Spin-Single trace is a candidate for that checker, not evidence that every
random bag order can construct the same typed plan.

## Two explicit capacities

| Policy | Evolved coefficients | Feature schema | Purpose |
| --- | ---: | --- | --- |
| Linear | 24 | `neyra-ga-features-v1` | Retained small policy and comparison baseline |
| Contextual residual | 256 | `neyra-ga-context256-v1` | Add observable action and bag context while preserving a linear starting point |

The physical configuration's `physical.contextFeatures` selects the capacity.
Model imports validate the configuration, feature names, coefficient count,
finite range, and digest together. The separate learned verification guide has
its own architecture and training data; its parameter count is not the GA
placement policy's parameter count.

The contextual policy is linear in its evolved coefficients and nonlinear in
its observed state through feature products. It is not a CNN or a dense neural
network. The deliberately limited genome keeps CPU and browser-worker training
usable while allowing a direct capacity comparison.

## Observable inputs

The first 24 inputs are the unchanged geometry/action measurements. They include
actual Spin Single gain, height, holes, wells, row deficits, contacts, and spawn
obstruction. The additional 232 inputs are:

| Group | Inputs | Meaning |
| --- | ---: | --- |
| Proposed piece and geometry | 84 | Seven piece identities multiplied by twelve selected geometry measurements |
| Remaining bag inventory and geometry | 56 | Seven availability flags after the proposed move multiplied by eight geometry measurements |
| Phase and geometry | 36 | Current bag progress, remaining lock fraction, and actual spin progress, each multiplied by twelve geometry measurements |
| Local landing patch | 48 | A six-column, eight-row occupancy patch of the board before the move, anchored to the placement's leftmost and lowest cells |
| Pose and interactions | 8 | Four rotation flags, landing position, and two height-dependent geometry products |

The inventory mask describes pieces already placed in the current bag. It is
not the probability of a future supply and does not expose a future queue. Spin
and placement counts are measured before the proposed move. The remaining-lock
feature uses the configured hard cap; it does not turn the preferred piece goal
into a rejection threshold.

The local patch includes the real walls and floor through the physics board's
occupancy query. It describes geometry only. Neither a patch nor a scalar
feature establishes body clearance, spin legality, or a legal execution order.
The canonical input generator and replay remain responsible for those facts.

## Compatible initialization and explicit migration

Expansion copies all 24 fitted coefficients and initializes the 232 residual
coefficients to zero. The initial expanded score therefore equals the original
linear score with the same accumulation order. Subsequent evolutionary updates
can learn context-dependent corrections without deliberately discarding the
previous policy at initialization.

Legacy runs retain their original configuration, RNG, model representation, and
selection rules on exact resume. Changing the training objective or model
capacity starts an explicit new experiment. Reducing a fitted 256-coefficient
model to 24 coefficients is rejected instead of silently discarding its learned
context. A separate, explicitly evaluated projection or distillation would be
needed for that operation.

The model identifier hashes the coefficient vector. Model-policy/configuration
validation remains necessary even when two files have the same coefficients.
A coefficient digest is an integrity check, not evidence of playing strength.

## Mutation and the meaning of learning rate

This GA does not use the Adam/SGD learning rate of the separate learned guide.
Its mutation probability, mutation scale, parent selection, continuation
schedule, and number of evaluated rollouts determine how training explores.
Changing more coefficients at once can disrupt a useful prefix; raising a
mutation scale is not a guarantee of faster improvement.

Unit-valued local occupancy inputs can change an action score more strongly
than a heavily normalized global feature. The physical evolution path therefore
uses a bounded coordinate mutation budget and smaller residual steps. This is
a simple stabilization mechanism, not an implementation of gradient-based safe
mutation or a proof that behaviors remain unchanged after mutation.

## When a larger policy is justified

The 24-input policy cannot condition a choice on information it never receives.
Additional context can distinguish such situations. That representational
difference alone does not establish better decisions: mirrored or otherwise
distinct actions can legitimately have equal value, and a larger genome can
take more evaluations to tune.

Capacity comparisons must keep physical rules, training objective, inherited
starting weights, search settings, and declared work or wall-time budget
matched. Record both physical completion frequency and the quality of the
retained complete traces. Freeze weights for comparisons on previously unused
exploration seeds. Those seeds change the search, not the underlying game, so
they do not establish broad out-of-distribution generalization.

Increase capacity when the larger policy improves those measured outcomes.
Retain the smaller policy when selection, lost prefixes, or insufficient
continuation search explain the bottleneck more directly. Neither a configured
generation count nor a single high raw-spin result establishes learning
completion, random-bag robustness, or 40L reliability.

## Compressing a discovered physical route

A later fitting stage can use an actually replayed candidate as its own source
of action preferences. At each state on that route, enumerate the same legal
actions and compare their feature vectors with the chosen action's vector.
Because both policy variants are linear in their coefficients, bounded linear
constraints can test whether those choices can all receive a positive ranking
margin. This tests representational capacity for that particular route.

Such fitting is an optional way to compress a discovered behavior into the
policy. It must be identified separately from genetic evolution, and the fitted
coefficients must be checked by fresh physical rollout. Equivalent placements
need not be strictly ordered. A zero margin for one trace does not prove that
every 40-spin policy is impossible at that capacity; a positive margin does not
prove that unseen states or random supply orders will work.

Repeated reproduction of one trace is useful execution evidence, but it does
not create new typed plans or establish supply robustness. Keep the subsequent
construction analysis and probability-aware learning stage separate. Private
route data, fitted coefficients, feasibility experiments, and measured results
belong outside the public source repository.

See [GA_TRAINING.md](GA_TRAINING.md) for the staged training contract and
[FRONTIER_ENGINE.md](FRONTIER_ENGINE.md) for exact later-stage probability.
