//! Finite normalized see-7 supply language. The first piece is already held.
//! This is an independent small product evaluator, not a queue oracle.

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

/// Incremental coverage of canonical supplies and their reflected labels.
/// Each orbit contains two distinct full-bag permutations. This is a supply
/// calculation only: callers must first establish a reflection-safe movement
/// policy, including every later bag that continues in the chosen orientation.
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

/// Exact shortest costs within a fully checked reflection-safe policy.
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
