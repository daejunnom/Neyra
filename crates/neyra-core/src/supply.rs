//! Finite see-7 supply languages. Runtime policy starts with an active first
//! piece and an empty bag-local hold. The unrestricted normalized evaluator is
//! retained as a reference; it is not the build certificate's hold policy.

use std::collections::{HashSet, VecDeque};
use std::sync::OnceLock;

pub const P7_COUNT: usize = 5040;
pub const ALL_PIECES: u8 = 0x7f;
pub const NO_PIECE: u8 = 7;

#[derive(Clone, Debug)]
pub struct LanguageNode {
    /// Piece labels, not operation indices; a bag contains at most one of each.
    pub edges: Vec<(u8, usize)>,
    pub accepting: bool,
    /// Pieces which can spawn before a hold action at this board.
    pub spawn_mask: u8,
}

pub const CLEAR_ONLY_HOLD_POLICY: &str = "ActiveFirst;EmptyBagHold;QuietLocksUseCurrent;DeferOnlyProductiveClearCandidates;HeldLocksClear;TerminalUnusedSwapAllowed;EndFlushRequiresAllNextSpawns";

#[derive(Clone, Copy, Debug)]
pub struct ClearingEdge {
    pub piece: u8,
    pub child: usize,
    pub clears: bool,
}

#[derive(Clone, Debug)]
pub struct ClearingNode {
    pub edges: Vec<ClearingEdge>,
    pub accepting: bool,
    pub spawn_mask: u8,
    pub used: u8,
}

/// Immutable execution facts reused by all supplies of one fixed target plan.
/// A piece may clear in one order and lock quietly in another. Deferral uses
/// a productive clearing continuation; the actual held release must clear.
pub struct ClearOnlyLanguage {
    nodes: Vec<ClearingNode>,
    targets: u8,
    productive: Vec<bool>,
    future_clears: Vec<u8>,
}

impl ClearOnlyLanguage {
    pub fn new(nodes: Vec<ClearingNode>, targets: u8) -> Self {
        let mut productive = vec![false; nodes.len()];
        let mut future_clears = vec![0; nodes.len()];
        // Edges add exactly one used target. Using the rank rather than index
        // order also supports converging execution graphs and test languages.
        for depth in (0..=7).rev() {
            for (index, node) in nodes.iter().enumerate() {
                if node.used.count_ones() != depth { continue; }
                if node.accepting { productive[index] = true; continue; }
                for edge in &node.edges {
                    debug_assert_eq!(nodes[edge.child].used, node.used | (1 << edge.piece));
                    if !productive[edge.child] { continue; }
                    productive[index] = true;
                    future_clears[index] |= future_clears[edge.child]
                        | if edge.clears { 1 << edge.piece } else { 0 };
                }
            }
        }
        Self { nodes, targets, productive, future_clears }
    }

    fn quiet_only(&self, queue: &[u8; 7]) -> SupplyOutcome {
        let mut index = 0;
        let mut order = 0;
        for depth in 0..=7 {
            let node = &self.nodes[index];
            if node.accepting { return SupplyOutcome::success(depth as u8, order); }
            if depth == 7 { return SupplyOutcome::blocked(false); }
            let piece = queue[depth];
            let required = self.targets & !node.used & (1 << piece) != 0;
            if node.spawn_mask & (1 << piece) == 0 {
                return SupplyOutcome::blocked(!required);
            }
            let edge = node.edges.iter().find(|edge| edge.piece == piece
                && !edge.clears && self.productive[edge.child]);
            let Some(edge) = edge else { return SupplyOutcome::blocked(!required); };
            order |= u32::from(piece) << (depth * 3);
            index = edge.child;
        }
        unreachable!()
    }
}

/// A quiet-safe failure is a forced clear/supply blockage, never a failed
/// ordinary placement. All supplies must be quiet-safe independently of the
/// requested success threshold. Only accepting finishes contribute chance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SupplyOutcome {
    pub finish: Option<(u8, u32)>,
    pub quiet_safe: bool,
}

impl SupplyOutcome {
    fn success(cost: u8, code: u32) -> Self {
        Self { finish: Some((cost, code)), quiet_safe: true }
    }
    fn blocked(quiet_safe: bool) -> Self { Self { finish: None, quiet_safe } }
}

/// Exact OR over legal strategies for this visible queue. The caller supplies
/// the outer AND over all queues (or over first-bag reflection pairs).
/// Zero-lock empty-hold transitions are explored explicitly: a blocked clear
/// cannot hide the next quiet piece's failure by stopping before that hold.
pub fn clearing_only_finish(language: &ClearOnlyLanguage, queue: &[u8; 7]) -> SupplyOutcome {
    if language.nodes.is_empty() { return SupplyOutcome::blocked(false); }
    if language.future_clears[0] == 0 { return language.quiet_only(queue); }
    let mut pending = VecDeque::from([(0_usize, 0_u8, NO_PIECE, true, 0_u32)]);
    let mut seen = HashSet::from([(0_usize, 0_u8, NO_PIECE, true)]);
    let mut clear_blocked = false;
    while let Some((index, cursor, held, hold_ready, order)) = pending.pop_front() {
        let node = &language.nodes[index];
        let depth = node.used.count_ones() as u8;
        if node.accepting { return SupplyOutcome::success(depth, order); }
        if depth >= 7 { continue; }
        let current = queue.get(cursor as usize).copied().unwrap_or(NO_PIECE);
        let required = current != NO_PIECE
            && language.targets & !node.used & (1 << current) != 0;
        let can_clear = current != NO_PIECE
            && language.future_clears[index] & (1 << current) != 0;
        let ordinary = required && !can_clear;
        // After consuming all seven supplies, releasing the last held clear
        // exchanges the next bag's unknown active piece. Every type must spawn.
        let spawnable = if current == NO_PIECE {
            held != NO_PIECE && node.spawn_mask == ALL_PIECES
        } else { node.spawn_mask & (1 << current) != 0 };
        if !spawnable {
            clear_blocked |= !ordinary;
            continue;
        }
        let mut moved = false;
        for edge in &node.edges {
            if !language.productive[edge.child] { continue; }
            let next_hold = if edge.piece == current {
                held
            } else if edge.piece == held && hold_ready && edge.clears
                && (current == NO_PIECE || language.nodes[edge.child].accepting
                    || language.future_clears[edge.child] & (1 << current) != 0)
            {
                // A goal-ending held clear may leave an unused current in hold.
                // There is no ordinary placement after that absorbing finish.
                current
            } else { continue; };
            moved = true;
            let next_cursor = cursor + u8::from(current != NO_PIECE);
            let next = (edge.child, next_cursor, next_hold, true);
            if seen.insert(next) {
                pending.push_back((next.0, next.1, next.2, next.3,
                    order | (u32::from(edge.piece) << (depth * 3))));
            }
        }
        if held == NO_PIECE && hold_ready && can_clear && cursor < 7 {
            moved = true;
            let next = (index, cursor + 1, current, false);
            if seen.insert(next) {
                // The state rank has not increased: 0-1 BFS retains minimum
                // lock cost while still checking the newly active spawn.
                pending.push_front((next.0, next.1, next.2, next.3, order));
            }
        }
        if !moved { clear_blocked |= !ordinary; }
    }
    SupplyOutcome::blocked(clear_blocked)
}

pub fn permutation(mut rank: usize) -> [u8; 7] {
    const FACT: [usize; 7] = [720, 120, 24, 6, 2, 1, 1];
    assert!(rank < P7_COUNT);
    let mut available = vec![0, 1, 2, 3, 4, 5, 6];
    let mut result = [0; 7];
    for i in 0..7 {
        let index = rank / FACT[i];
        rank %= FACT[i];
        result[i] = available.remove(index);
    }
    result
}

/// Inverse of `permutation` in the same I J L O S T Z lexicographic order.
/// Callers supply one complete bag, with each label present exactly once.
pub fn permutation_rank(queue: &[u8; 7]) -> usize {
    const FACT: [usize; 7] = [720, 120, 24, 6, 2, 1, 1];
    let mut available = ALL_PIECES;
    let mut rank = 0;
    for (index, &piece) in queue.iter().enumerate() {
        assert!(piece < 7 && available & (1 << piece) != 0);
        rank += (available & ((1 << piece) - 1)).count_ones() as usize * FACT[index];
        available &= !(1 << piece);
    }
    rank
}

/// Relabel a supply under horizontal reflection. J/L and S/Z swap together;
/// I/O/T and the next-bag exchange placeholder are unchanged. This mapping
/// alone does not certify a reflected physical movement or kick sequence.
pub const fn mirror_piece(piece: u8) -> u8 {
    match piece {
        1 => 2,
        2 => 1,
        4 => 6,
        6 => 4,
        _ => piece,
    }
}

pub fn mirror_queue(queue: &[u8; 7]) -> [u8; 7] {
    queue.map(mirror_piece)
}

pub fn mirror_rank(rank: usize) -> usize {
    static RANKS: OnceLock<[u16; P7_COUNT]> = OnceLock::new();
    RANKS.get_or_init(|| {
        std::array::from_fn(|index| {
            permutation_rank(&mirror_queue(&permutation(index))) as u16
        })
    })[rank] as usize
}

pub fn mirror_mask(mask: u8) -> u8 {
    (0..7).fold(0, |result, piece| {
        result | (((mask >> piece) & 1) << mirror_piece(piece))
    })
}

/// Directly enumerate the 2,520 two-queue reflection orbits. A caller resolves
/// one orbit only after its minimum accepting cost is known: an observed cost
/// at a proven language-wide lower bound can skip the partner query. Otherwise
/// both canonical queues must be evaluated. No canonical result is inferred.
#[derive(Clone, Debug)]
pub struct SymmetryOrbits {
    cursor: usize,
    checked: u16,
    success: u16,
    costs: Vec<u8>,
}

impl Default for SymmetryOrbits {
    fn default() -> Self {
        Self { cursor: 0, checked: 0, success: 0, costs: vec![0; P7_COUNT] }
    }
}

impl SymmetryOrbits {
    /// Each rank appears in exactly one pair. Reuse the existing reflection
    /// map; no second rank table or board cache is shared between workers.
    pub fn next_pair(&mut self) -> Option<[usize; 2]> {
        while self.cursor < P7_COUNT {
            let rank = self.cursor;
            self.cursor += 1;
            let mirrored = mirror_rank(rank);
            if rank < mirrored { return Some([rank, mirrored]); }
        }
        None
    }

    pub fn resolve(&mut self, pair: [usize; 2], cost: u8) {
        assert!((1..=7).contains(&cost) || cost == 255);
        debug_assert_eq!(mirror_rank(pair[0]), pair[1]);
        debug_assert!(pair[0] < pair[1]);
        debug_assert_eq!(self.costs[pair[0]], 0);
        debug_assert_eq!(self.costs[pair[1]], 0);
        self.costs[pair[0]] = cost;
        self.costs[pair[1]] = cost;
        self.checked += 2;
        if cost != 255 { self.success += 2; }
    }

    pub fn checked(&self) -> u16 { self.checked }
    pub fn success(&self) -> u16 { self.success }
    pub fn upper(&self) -> u16 { self.success + P7_COUNT as u16 - self.checked }
    pub fn costs(&self) -> &[u8] { &self.costs }
}

/// Incremental coverage of canonical supplies and their reflected labels.
/// Each orbit contains two distinct full-bag permutations. This is a supply
/// calculation only: the caller must explicitly state its physical symmetry
/// assumption, including later bags in the orientation chosen at the start.
#[derive(Clone, Debug)]
pub struct SymmetryCoverage {
    /// Low bits: checked orbit members. High bits: successful members.
    states: Vec<u8>,
    success: u16,
    failed: u16,
}

impl Default for SymmetryCoverage {
    fn default() -> Self {
        Self {
            states: vec![0; P7_COUNT],
            success: 0,
            failed: 0,
        }
    }
}

impl SymmetryCoverage {
    /// Observe one exact canonical result. Repeating the same result is safe;
    /// changing an already observed result is a caller error.
    pub fn observe(&mut self, rank: usize, successful: bool) {
        let mirrored = mirror_rank(rank);
        let index = rank.min(mirrored);
        let member = if rank == index { 1 } else { 2 };
        let state = &mut self.states[index];
        if *state & member != 0 {
            assert_eq!(*state & (member << 2) != 0, successful);
            return;
        }
        let was_successful = *state & 12 != 0;
        *state |= member;
        if successful {
            *state |= member << 2;
            if !was_successful {
                self.success += 2;
            }
        } else if *state & 3 == 3 && !was_successful {
            self.failed += 2;
        }
    }

    pub fn success(&self) -> u16 {
        self.success
    }

    /// Unknown members stay possible; one failed member never fails its orbit.
    pub fn upper(&self) -> u16 {
        P7_COUNT as u16 - self.failed
    }
}

/// Shortest costs in the supply-label reflection model, after a complete
/// canonical check. This arithmetic does not verify reflected input paths.
/// Orientation may be chosen once from the first bag's visible queue. The
/// original costs and codes must remain available for the canonical witness.
/// Values are 1..=7 or 255; unknown entries cannot become certified minima.
pub fn symmetry_union_costs(costs: &[u8]) -> Vec<u8> {
    assert_eq!(costs.len(), P7_COUNT);
    assert!(costs.iter().all(|cost| (1..=7).contains(cost) || *cost == 255));
    (0..P7_COUNT)
        .map(|rank| costs[rank].min(costs[mirror_rank(rank)]))
        .collect()
}

/// Returns the shortest accepting prefix and its packed piece order.
/// Every edge locks one piece. Layered BFS therefore certifies minimum cost.
pub fn minimum_finish(nodes: &[LanguageNode], queue: &[u8; 7]) -> Option<(u8, u32)> {
    if nodes.is_empty() {
        return None;
    }
    let mut pending = VecDeque::from([(0_usize, 1_u8, queue[0], 0_u32)]);
    let mut seen = HashSet::from([(0_usize, 1_u8, queue[0])]);
    while let Some((node_index, cursor, held, order)) = pending.pop_front() {
        let depth = cursor - 1;
        let node = &nodes[node_index];
        if node.accepting {
            return Some((depth, order));
        }
        if depth >= 7 {
            continue;
        }
        let current = if cursor < 7 {
            queue[cursor as usize]
        } else {
            NO_PIECE
        };
        // The next bag's first piece is an identity-blind exchange placeholder.
        if current == NO_PIECE {
            if node.spawn_mask != ALL_PIECES {
                continue;
            }
        } else if node.spawn_mask & (1 << current) == 0 {
            continue;
        }
        for (piece, target) in &node.edges {
            let next_hold = if *piece == current {
                held
            } else if *piece == held {
                current
            } else {
                continue;
            };
            let next = (*target, cursor + 1, next_hold);
            if seen.insert(next) {
                pending.push_back((
                    next.0,
                    next.1,
                    next.2,
                    order | ((*piece as u32) << (depth * 3)),
                ));
            }
        }
    }
    None
}

pub fn decode_order(code: u32, length: u8) -> Vec<u8> {
    (0..length).map(|i| ((code >> (3 * i)) & 7) as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn language_for_orders(orders: &[Vec<u8>]) -> Vec<LanguageNode> {
        let mut nodes = vec![LanguageNode {
            edges: vec![],
            accepting: false,
            spawn_mask: ALL_PIECES,
        }];
        for order in orders {
            let mut index = 0;
            for piece in order {
                let existing = nodes[index]
                    .edges
                    .iter()
                    .find(|(p, _)| p == piece)
                    .map(|(_, n)| *n);
                index = if let Some(next) = existing {
                    next
                } else {
                    let next = nodes.len();
                    nodes.push(LanguageNode {
                        edges: vec![],
                        accepting: false,
                        spawn_mask: ALL_PIECES,
                    });
                    nodes[index].edges.push((*piece, next));
                    next
                };
            }
            nodes[index].accepting = true;
        }
        nodes
    }

    fn count(nodes: &[LanguageNode]) -> usize {
        (0..P7_COUNT)
            .filter(|rank| minimum_finish(nodes, &permutation(*rank)).is_some())
            .count()
    }

    #[test]
    fn all_permutations_are_distinct() {
        let values: HashSet<_> = (0..P7_COUNT).map(permutation).collect();
        assert_eq!(values.len(), P7_COUNT);
        assert_eq!(permutation(0), [0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(permutation(5039), [6, 5, 4, 3, 2, 1, 0]);
    }

    #[test]
    fn hold_cannot_pick_arbitrary_visible_pieces() {
        let both = language_for_orders(&[vec![4, 6], vec![6, 4]]);
        let fixed = language_for_orders(&[vec![4, 6]]);
        assert_eq!(count(&both), 720);
        assert_eq!(count(&fixed), 480);
    }

    #[test]
    fn fixed_seven_piece_order_has_64_incoming_queues() {
        let nodes = language_for_orders(&[vec![0, 1, 2, 3, 4, 5, 6]]);
        assert_eq!(count(&nodes), 64);
    }

    #[test]
    fn unconstrained_subset_language_covers_all_p7() {
        let nodes: Vec<_> = (0_u8..128)
            .map(|mask| LanguageNode {
                edges: (0..7)
                    .filter(|p| mask & (1 << p) == 0)
                    .map(|p| (p, (mask | (1 << p)) as usize))
                    .collect(),
                accepting: mask == ALL_PIECES,
                spawn_mask: ALL_PIECES,
            })
            .collect();
        assert_eq!(count(&nodes), 5040);
    }

    #[test]
    fn goal_is_absorbing_before_an_unused_next_spawn() {
        let mut nodes = language_for_orders(&[vec![4]]);
        nodes[1].spawn_mask = 0;
        assert_eq!(minimum_finish(&nodes, &[4, 0, 1, 2, 3, 5, 6]), Some((1, 4)));
    }

    #[test]
    fn current_piece_spawn_cannot_be_skipped_by_hold() {
        let mut nodes = language_for_orders(&[vec![4]]);
        nodes[0].spawn_mask &= !(1 << 0);
        assert_eq!(minimum_finish(&nodes, &[4, 0, 1, 2, 3, 5, 6]), None);
    }
}
