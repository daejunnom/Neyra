# Movement categories and completion bounds

The canonical core now exposes `MovementMode::SonicOnly` and
`MovementMode::SoftDrop`. Existing APIs remain SonicOnly. The additional
`reachable_locks_with_mode`, `replay_with_mode`, `apply_with_mode` and
`ga::score_placement_with_mode` APIs require an explicit category.

SonicOnly is the existing fast, no-wait movement profile. SoftDrop permits the
same inputs plus a legal one-cell `soft_drop` descent before lock. It does not
disable the Sonic key. A successful downward translation clears rotation spin
evidence. Rotation data, kick order, collision and spin predicates are unchanged.

This is an explicit new movement category, not a claim about historical UI
settings. It does not model frame timing, gravity, DAS/ARR or lock delay, nor
prove a ranking time. Keep movement category, legal replay and speed evidence
separate. Legacy replay rejects a soft-drop witness so old certificates do not
silently acquire new rules. Public fixed-plan search and supply certificates
continue to use their existing SonicOnly contract; opting the future planner
into SoftDrop requires its own mode-aware certification path.

## 100 pieces cannot finish 40 Spin Singles

For an empty 10-column board with no external cells, after N locks and S single
line clears the remaining cells are `C = 4N - 10S`. At N=100 and S=40, C=0.
Immediately before the last clear, every occupied cell, including all four
cells of the final mino, would have to lie in the one cleared row. Thus the
last mino must be a horizontal I. Every other row is empty, leaving upward
translation unblocked. The canonical I immobility spin predicate is false.
The fatal ceiling is not material and cannot supply a spin roof.

Before the last lock, S=39 and every already-cleared board row has at most9
occupied cells. Hence `4(N-1)-390 <= 24*9`, giving N<=152. The necessary range
is therefore **101<=N<=152**; this does not establish feasibility at101.
Do not implement the derived upper bound as an artificial game-over rule.

An exact100% fixed plan establishes certainty for its own declared probability
model. A minimum-certain-cost goal may stop only after all smaller budgets are
ruled out. A best-found record is not a global minimality proof, and a first-bag
reflection assumption is not independent mirrored input verification. Lower-cost
partial-probability records and input-speed records are separate optional goals.

