//! Bounded-board, no-wait TETR.IO-style movement for Neyra.
//!
//! The locked board is 10 × 24. The active piece may briefly occupy the empty
//! sky above it; that search envelope is never treated as a solid ceiling.
//! `Pose` uses the top-left corner of the SRS matrix, with positive y upwards.
//! Thus a matrix cell (dx, dy) occupies (pose.x + dx, pose.y - dy).
//!
//! Ordered JLSTZ 90/180 and I 90 kick data are adapted from Clearra's
//! core-c/src/rules/srs_kicks.c at 5c98563332706a718330bc7ae8e403d2583afe9d
//! (MIT, Copyright (c) 2026 Clearra contributors). I 180 uses the two-candidate
//! SRS+ transitions in the pinned Spilink/Triangle engine, not Clearra's longer
//! I 180 sequences. See THIRD_PARTY_NOTICES.md for source pins and licenses.
//! The movement graph deliberately has no one-row down or gravity-wait edge.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

pub const WIDTH: usize = 10;
pub const HEIGHT: usize = 24;
pub const FULL_ROW: u16 = (1 << WIDTH) - 1;
const SKY_CELLS: i8 = 4;
const MAX_POSE_Y: i8 = HEIGHT as i8 + SKY_CELLS + 2;
const MIN_POSE_X: i8 = -3;
const X_COUNT: usize = WIDTH + 3;
const STATE_COUNT: usize = 4 * X_COUNT * (MAX_POSE_Y as usize + 1) * 2;
const NO_PARENT: usize = usize::MAX;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
#[repr(u8)]
pub enum Piece {
    I,
    J,
    L,
    O,
    S,
    T,
    Z,
}

impl Piece {
    pub const ALL: [Self; 7] = [
        Self::I,
        Self::J,
        Self::L,
        Self::O,
        Self::S,
        Self::T,
        Self::Z,
    ];

    pub const fn spawn(self) -> Pose {
        Pose {
            piece: self,
            x: if matches!(self, Self::O) { 4 } else { 3 },
            y: 22,
            rotation: 0,
        }
    }

    pub const fn bit(self) -> u8 {
        1 << self as u8
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Board {
    pub rows: [u16; HEIGHT],
}

impl Default for Board {
    fn default() -> Self {
        Self { rows: [0; HEIGHT] }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct Pose {
    pub piece: Piece,
    pub x: i8,
    pub y: i8,
    pub rotation: u8,
}

impl Pose {
    pub fn cells(self) -> [(i8, i8); 4] {
        let shape = SHAPES[self.piece as usize][(self.rotation & 3) as usize];
        shape.map(|(dx, dy)| (self.x + dx, self.y - dy))
    }

    fn shifted(self, dx: i8, dy: i8) -> Self {
        Self {
            x: self.x + dx,
            y: self.y + dy,
            ..self
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Input {
    Left,
    Right,
    Cw,
    Ccw,
    Half,
    Sonic,
    HardDrop,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    pub pose: Pose,
    /// All-mini+ spin evidence, evaluated before the line clear.
    pub spin: bool,
    /// Replayable input sequence from spawn, including the final hard drop.
    pub path: Vec<Input>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct LockResult {
    pub board: Board,
    pub lines: u8,
    pub spin: bool,
    /// Bit y denotes a cleared physical row in the board before compaction.
    pub cleared_rows: u32,
    pub cells: [(i8, i8); 4],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
struct State {
    pose: Pose,
    spin: bool,
}

#[derive(Clone, Copy)]
struct Node {
    state: State,
    parent: usize,
    input: Input,
}

impl Board {
    pub fn is_valid(&self) -> bool {
        self.rows.iter().all(|row| row & !FULL_ROW == 0)
    }
    pub fn mask(&self) -> [u16; HEIGHT] {
        self.rows
    }
    pub fn cell_count(&self) -> u16 {
        self.rows.iter().map(|r| r.count_ones() as u16).sum()
    }
    pub fn height(&self) -> u8 {
        self.rows
            .iter()
            .rposition(|row| *row != 0)
            .map_or(0, |y| y as u8 + 1)
    }

    /// Walls/floor are occupied; the sky above the 24 locked rows is empty.
    pub fn occupied(&self, x: i8, y: i8) -> bool {
        if x < 0 || x >= WIDTH as i8 || y < 0 {
            return true;
        }
        y < HEIGHT as i8 && self.rows[y as usize] & (1 << x) != 0
    }

    fn legal_physical(&self, pose: Pose) -> bool {
        pose.rotation < 4
            && pose.x >= MIN_POSE_X
            && pose.x < WIDTH as i8
            && pose.y >= 0
            && pose.cells().iter().all(|&(x, y)| !self.occupied(x, y))
    }

    fn in_envelope(pose: Pose) -> bool {
        pose.x >= MIN_POSE_X
            && pose.x < WIDTH as i8
            && pose.y >= 0
            && pose.y <= MAX_POSE_Y
            && pose
                .cells()
                .iter()
                .all(|&(_, y)| y < HEIGHT as i8 + SKY_CELLS)
    }

    pub fn legal(&self, pose: Pose) -> bool {
        Self::in_envelope(pose) && self.legal_physical(pose)
    }
    pub fn can_spawn(&self, piece: Piece) -> bool {
        self.is_valid() && self.legal(piece.spawn())
    }
    pub fn grounded(&self, pose: Pose) -> bool {
        self.legal_physical(pose) && !self.legal_physical(pose.shifted(0, -1))
    }

    fn lockable(&self, pose: Pose) -> bool {
        self.legal(pose)
            && self.grounded(pose)
            && pose
                .cells()
                .iter()
                .all(|&(_, y)| y >= 0 && y < HEIGHT as i8)
    }

    pub fn sonic_pose(&self, mut pose: Pose) -> Option<Pose> {
        if !self.legal(pose) {
            return None;
        }
        while self.legal_physical(pose.shifted(0, -1)) {
            pose.y -= 1;
        }
        Some(pose)
    }

    /// First physically successful kick is authoritative. A candidate that
    /// leaves our finite sky envelope is discarded, never replaced by a later
    /// kick; otherwise the envelope could manufacture a different rotation.
    pub fn rotate(&self, pose: Pose, input: Input) -> Option<Pose> {
        let delta = rotation_delta(input)?;
        if !self.legal(pose) {
            return None;
        }
        let to = ((pose.rotation as i8 + delta).rem_euclid(4)) as u8;
        for &(dx, dy) in kicks(pose.piece, pose.rotation, to) {
            let candidate = Pose {
                x: pose.x + dx,
                y: pose.y + dy,
                rotation: to,
                ..pose
            };
            if self.legal_physical(candidate) {
                return Self::in_envelope(candidate).then_some(candidate);
            }
        }
        None
    }

    /// Whether a successful rotation ending here earns any All-mini+ spin.
    /// Mini/full distinctions do not change Naga Eyes' spin-single objective.
    pub fn spin_at(&self, pose: Pose) -> bool {
        if !self.legal(pose) || !self.grounded(pose) {
            return false;
        }
        if pose.piece == Piece::T {
            let cx = pose.x + 1;
            let cy = pose.y - 1;
            let corners = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
                .iter()
                .filter(|&&(dx, dy)| self.occupied(cx + dx, cy + dy))
                .count();
            if corners >= 3 {
                return true;
            }
        }
        [(-1, 0), (1, 0), (0, 1), (0, -1)]
            .iter()
            .all(|&(dx, dy)| !self.legal_physical(pose.shifted(dx, dy)))
    }

    fn advance(&self, state: State, input: Input) -> Option<State> {
        match input {
            Input::Left | Input::Right => {
                let pose = state
                    .pose
                    .shifted(if input == Input::Left { -1 } else { 1 }, 0);
                self.legal(pose).then_some(State { pose, spin: false })
            }
            Input::Cw | Input::Ccw | Input::Half => {
                let pose = self.rotate(state.pose, input)?;
                Some(State {
                    pose,
                    spin: self.spin_at(pose),
                })
            }
            Input::Sonic | Input::HardDrop => {
                let pose = self.sonic_pose(state.pose)?;
                Some(State {
                    pose,
                    spin: state.spin && pose.y == state.pose.y,
                })
            }
        }
    }

    /// Replays a lock witness. Failed movement inputs are omitted by the
    /// solver; an explicitly supplied failed input makes a witness invalid.
    pub fn replay(&self, piece: Piece, path: &[Input]) -> Option<Placement> {
        if !self.can_spawn(piece) || path.last() != Some(&Input::HardDrop) {
            return None;
        }
        let mut state = State {
            pose: piece.spawn(),
            spin: false,
        };
        for (i, &input) in path.iter().enumerate() {
            if input == Input::HardDrop && i + 1 != path.len() {
                return None;
            }
            state = self.advance(state, input)?;
        }
        self.lockable(state.pose).then(|| Placement {
            pose: state.pose,
            spin: state.spin,
            path: path.to_vec(),
        })
    }

    /// Cheap geometric placement for structural analysis. It provides no spin
    /// evidence: use `apply` for an accepted/replayable placement.
    pub fn place_geometry(&self, pose: Pose) -> Option<LockResult> {
        if !self.is_valid() || !self.lockable(pose) {
            return None;
        }
        let cells = pose.cells();
        let mut rows = self.rows;
        for &(x, y) in &cells {
            rows[y as usize] |= 1 << x;
        }
        let mut board = Self::default();
        let mut cleared_rows = 0_u32;
        let mut write = 0;
        for (y, row) in rows.into_iter().enumerate() {
            if row == FULL_ROW {
                cleared_rows |= 1 << y;
            } else {
                board.rows[write] = row;
                write += 1;
            }
        }
        Some(LockResult {
            board,
            lines: cleared_rows.count_ones() as u8,
            spin: false,
            cleared_rows,
            cells,
        })
    }

    /// Validates the complete witness before assigning spin evidence.
    pub fn apply(&self, placement: &Placement) -> Option<LockResult> {
        let replayed = self.replay(placement.pose.piece, &placement.path)?;
        if replayed.pose != placement.pose || replayed.spin != placement.spin {
            return None;
        }
        let mut result = self.place_geometry(placement.pose)?;
        result.spin = placement.spin;
        Some(result)
    }

    /// Forward reference enumerator. Geometry with spin and without spin are
    /// separate states so a short non-spin route cannot hide a legal spin.
    pub fn reachable_locks(&self, piece: Piece) -> Vec<Placement> {
        if !self.can_spawn(piece) {
            return Vec::new();
        }
        let (nodes, _) = self.forward_nodes(piece, true);
        nodes
            .iter()
            .enumerate()
            .filter_map(|(i, node)| {
                if !self.lockable(node.state.pose) {
                    return None;
                }
                let mut path = forward_path(&nodes, i);
                path.push(Input::HardDrop);
                Some(Placement {
                    pose: node.state.pose,
                    spin: node.state.spin,
                    path,
                })
            })
            .collect()
    }

    /// Reverse target reachability, joining a goal to actual no-drop entry
    /// presets. Every inverse rotation is checked by the ordered forward
    /// rotation; every inverse sonic edge must fall all the way to its target.
    /// `wants_spin=false` accepts either spin class.
    pub fn can_reach_target(
        &self,
        piece: Piece,
        pose: Pose,
        wants_spin: bool,
    ) -> Option<Placement> {
        if pose.piece != piece || !self.can_spawn(piece) || !self.lockable(pose) {
            return None;
        }
        let can_spin = self.spin_at(pose);
        if wants_spin && !can_spin {
            return None;
        }
        let (entries, entry_index) = self.forward_nodes(piece, false);
        let mut nodes = Vec::new();
        let mut seen = vec![false; STATE_COUNT];
        for spin in [false, true] {
            if (wants_spin && !spin) || (spin && !can_spin) {
                continue;
            }
            let state = State { pose, spin };
            seen[state_index(state)] = true;
            nodes.push(Node {
                state,
                parent: NO_PARENT,
                input: Input::HardDrop,
            });
        }
        let mut cursor = 0;
        while cursor < nodes.len() {
            let node = nodes[cursor];
            let entry = entry_index[state_index(node.state)];
            if entry != NO_PARENT {
                let mut path = forward_path(&entries, entry);
                let mut i = cursor;
                while nodes[i].parent != NO_PARENT {
                    path.push(nodes[i].input);
                    i = nodes[i].parent;
                }
                path.push(Input::HardDrop);
                let result = self.replay(piece, &path)?;
                if result.pose == pose && (!wants_spin || result.spin) {
                    return Some(result);
                }
                // A disagreement would be an implementation bug. Never emit
                // a route whose replay does not prove the requested target.
                debug_assert!(
                    false,
                    "inverse reachability witness disagrees with forward replay"
                );
                return None;
            }
            self.predecessors(node.state, |state, input| {
                let key = state_index(state);
                if !seen[key] {
                    seen[key] = true;
                    nodes.push(Node {
                        state,
                        parent: cursor,
                        input,
                    });
                }
            });
            cursor += 1;
        }
        None
    }

    fn forward_nodes(&self, piece: Piece, allow_sonic: bool) -> (Vec<Node>, Vec<usize>) {
        let root = State {
            pose: piece.spawn(),
            spin: false,
        };
        let mut nodes = vec![Node {
            state: root,
            parent: NO_PARENT,
            input: Input::HardDrop,
        }];
        let mut index = vec![NO_PARENT; STATE_COUNT];
        index[state_index(root)] = 0;
        let mut queue = VecDeque::from([0]);
        let actions = [
            Input::Sonic,
            Input::Left,
            Input::Right,
            Input::Cw,
            Input::Ccw,
            Input::Half,
        ];
        while let Some(parent) = queue.pop_front() {
            for input in actions {
                if !allow_sonic && input == Input::Sonic {
                    continue;
                }
                if let Some(state) = self.advance(nodes[parent].state, input) {
                    let key = state_index(state);
                    if index[key] == NO_PARENT {
                        index[key] = nodes.len();
                        queue.push_back(nodes.len());
                        nodes.push(Node {
                            state,
                            parent,
                            input,
                        });
                    }
                }
            }
        }
        (nodes, index)
    }

    fn predecessors(&self, state: State, mut emit: impl FnMut(State, Input)) {
        let mut add = |pose: Pose, input: Input| {
            if !self.legal(pose) {
                return;
            }
            for spin in [false, true] {
                if spin && !self.spin_at(pose) {
                    continue;
                }
                let prior = State { pose, spin };
                if self.advance(prior, input) == Some(state) {
                    emit(prior, input);
                }
            }
        };
        if !state.spin {
            add(state.pose.shifted(1, 0), Input::Left);
            add(state.pose.shifted(-1, 0), Input::Right);
        }
        if state.spin == self.spin_at(state.pose) {
            for input in [Input::Cw, Input::Ccw, Input::Half] {
                let delta = rotation_delta(input).unwrap();
                let from = ((state.pose.rotation as i8 - delta).rem_euclid(4)) as u8;
                for &(dx, dy) in kicks(state.pose.piece, from, state.pose.rotation) {
                    add(
                        Pose {
                            x: state.pose.x - dx,
                            y: state.pose.y - dy,
                            rotation: from,
                            ..state.pose
                        },
                        input,
                    );
                }
            }
        }
        if !state.spin && self.grounded(state.pose) {
            for y in state.pose.y + 1..=MAX_POSE_Y {
                let pose = Pose { y, ..state.pose };
                // A collision separates the column into drop components.
                // No source above this obstruction can sonic through it.
                if !self.legal(pose) {
                    break;
                }
                add(pose, Input::Sonic);
            }
        }
    }
}

fn state_index(state: State) -> usize {
    (((state.pose.y as usize * X_COUNT + (state.pose.x - MIN_POSE_X) as usize) * 4
        + state.pose.rotation as usize)
        * 2)
        + usize::from(state.spin)
}

fn forward_path(nodes: &[Node], mut i: usize) -> Vec<Input> {
    let mut path = Vec::new();
    while nodes[i].parent != NO_PARENT {
        path.push(nodes[i].input);
        i = nodes[i].parent;
    }
    path.reverse();
    path
}

fn rotation_delta(input: Input) -> Option<i8> {
    match input {
        Input::Cw => Some(1),
        Input::Ccw => Some(-1),
        Input::Half => Some(2),
        _ => None,
    }
}

// All kick coordinates below are x-right, y-up; origin is tested first.
const ORIGIN: &[(i8, i8)] = &[(0, 0)];
const J_A: &[(i8, i8)] = &[(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)];
const J_B: &[(i8, i8)] = &[(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)];
const J_C: &[(i8, i8)] = &[(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)];
const J_D: &[(i8, i8)] = &[(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)];
const J_02: &[(i8, i8)] = &[(0, 0), (0, 1), (1, 1), (-1, 1), (1, 0), (-1, 0)];
const J_20: &[(i8, i8)] = &[(0, 0), (0, -1), (-1, -1), (1, -1), (-1, 0), (1, 0)];
const J_13: &[(i8, i8)] = &[(0, 0), (1, 0), (1, 2), (1, 1), (0, 2), (0, 1)];
const J_31: &[(i8, i8)] = &[(0, 0), (-1, 0), (-1, 2), (-1, 1), (0, 2), (0, 1)];
const I_01: &[(i8, i8)] = &[(0, 0), (1, 0), (-2, 0), (-2, -1), (1, 2)];
const I_10: &[(i8, i8)] = &[(0, 0), (-1, 0), (2, 0), (-1, -2), (2, 1)];
const I_12: &[(i8, i8)] = &[(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)];
const I_21: &[(i8, i8)] = &[(0, 0), (-2, 0), (1, 0), (-2, 1), (1, -2)];
const I_23: &[(i8, i8)] = &[(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)];
const I_32: &[(i8, i8)] = &[(0, 0), (1, 0), (-2, 0), (1, 2), (-2, -1)];
const I_30: &[(i8, i8)] = &[(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)];
const I_03: &[(i8, i8)] = &[(0, 0), (-1, 0), (2, 0), (2, -1), (-1, 2)];
const I_02: &[(i8, i8)] = &[(0, 0), (0, 1)];
const I_13: &[(i8, i8)] = &[(0, 0), (1, 0)];
const I_20: &[(i8, i8)] = &[(0, 0), (0, -1)];
const I_31: &[(i8, i8)] = &[(0, 0), (-1, 0)];

fn kicks(piece: Piece, from: u8, to: u8) -> &'static [(i8, i8)] {
    if piece == Piece::O {
        return ORIGIN;
    }
    if piece == Piece::I {
        return match (from, to) {
            (0, 1) => I_01,
            (1, 0) => I_10,
            (1, 2) => I_12,
            (2, 1) => I_21,
            (2, 3) => I_23,
            (3, 2) => I_32,
            (3, 0) => I_30,
            (0, 3) => I_03,
            (0, 2) => I_02,
            (1, 3) => I_13,
            (2, 0) => I_20,
            (3, 1) => I_31,
            _ => &[],
        };
    }
    match (from, to) {
        (0, 1) | (2, 1) => J_A,
        (1, 0) | (1, 2) => J_B,
        (2, 3) | (0, 3) => J_C,
        (3, 2) | (3, 0) => J_D,
        (0, 2) => J_02,
        (2, 0) => J_20,
        (1, 3) => J_13,
        (3, 1) => J_31,
        _ => &[],
    }
}

const fn three_by_three(base: [(i8, i8); 4]) -> [[(i8, i8); 4]; 4] {
    let mut result = [base; 4];
    let mut r = 1;
    while r < 4 {
        let mut b = 0;
        while b < 4 {
            let (x, y) = result[r - 1][b];
            result[r][b] = (2 - y, x);
            b += 1;
        }
        r += 1;
    }
    result
}

const SHAPES: [[[(i8, i8); 4]; 4]; 7] = [
    [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
    ],
    three_by_three([(0, 0), (0, 1), (1, 1), (2, 1)]),
    three_by_three([(2, 0), (0, 1), (1, 1), (2, 1)]),
    [[(0, 0), (1, 0), (0, 1), (1, 1)]; 4],
    three_by_three([(1, 0), (2, 0), (0, 1), (1, 1)]),
    three_by_three([(1, 0), (0, 1), (1, 1), (2, 1)]),
    three_by_three([(0, 0), (1, 0), (1, 1), (2, 1)]),
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn board(rows: &[u16]) -> Board {
        let mut board = Board::default();
        board.rows[..rows.len()].copy_from_slice(rows);
        board
    }

    #[test]
    fn empty_field_locks_are_replayable_and_never_spins() {
        let board = Board::default();
        for piece in Piece::ALL {
            let locks = board.reachable_locks(piece);
            assert!(!locks.is_empty());
            for lock in locks {
                assert!(!lock.spin);
                let result = board.apply(&lock).expect("forward witness replays");
                assert_eq!(result.board.cell_count(), 4);
                assert_eq!(result.lines, 0);
                assert_eq!(board.sonic_pose(lock.pose), Some(lock.pose));
            }
        }
    }

    #[test]
    fn official_matrix_origins_and_floor_drop() {
        let board = Board::default();
        assert_eq!(
            Piece::T.spawn().cells(),
            [(4, 22), (3, 21), (4, 21), (5, 21)]
        );
        assert_eq!(
            Piece::I.spawn().cells(),
            [(3, 21), (4, 21), (5, 21), (6, 21)]
        );
        assert_eq!(
            Piece::O.spawn().cells(),
            [(4, 22), (5, 22), (4, 21), (5, 21)]
        );
        for piece in Piece::ALL {
            let pose = board.sonic_pose(piece.spawn()).unwrap();
            assert_eq!(pose.cells().iter().map(|cell| cell.1).min(), Some(0));
        }
    }

    #[test]
    fn line_compaction_and_cell_conservation() {
        let board = board(&[FULL_ROW ^ (15 << 3), 1 << 8, 1 << 9]);
        let result = board
            .apply(&board.replay(Piece::I, &[Input::HardDrop]).unwrap())
            .unwrap();
        assert_eq!(result.lines, 1);
        assert_eq!(result.cleared_rows, 1);
        assert_eq!(result.board.rows[0], 1 << 8);
        assert_eq!(result.board.rows[1], 1 << 9);
        assert_eq!(
            result.board.cell_count() as i32,
            board.cell_count() as i32 + 4 - 10
        );
        assert!(!result.spin);
    }

    #[test]
    fn twenty_four_rows_do_not_make_a_solid_ceiling() {
        let mut board = Board::default();
        let pose = Pose {
            piece: Piece::O,
            x: 4,
            y: 23,
            rotation: 0,
        };
        board.rows[21] = (1 << 4) | (1 << 5);
        board.rows[22] = (1 << 3) | (1 << 6);
        assert!(board.lockable(pose));
        assert!(board.legal_physical(pose.shifted(0, 1)));
        assert!(
            !board.spin_at(pose),
            "the empty sky must not award an O spin"
        );
        assert!(
            !board.lockable(pose.shifted(0, 1)),
            "row 24 cannot be locked"
        );
        assert!(!board.occupied(4, 24));
    }

    #[test]
    fn i_half_turn_uses_the_short_tetrio_sequence() {
        assert_eq!(kicks(Piece::I, 0, 2), &[(0, 0), (0, 1)]);
        assert_eq!(kicks(Piece::I, 1, 3), &[(0, 0), (1, 0)]);
        let board = Board::default();
        let pose = Pose {
            piece: Piece::I,
            x: 3,
            y: 1,
            rotation: 0,
        };
        assert_eq!(
            board.rotate(pose, Input::Half),
            Some(Pose {
                y: 2,
                rotation: 2,
                ..pose
            })
        );
    }

    #[test]
    fn first_successful_kick_cannot_be_skipped() {
        let board = Board::default();
        let pose = Piece::T.spawn();
        let target = board.rotate(pose, Input::Cw).unwrap();
        assert_eq!((target.x, target.y), (pose.x, pose.y));
        // Later offsets are legal too, but inverse rotation must reject them.
        let later = Pose {
            x: target.x - 1,
            ..target
        };
        assert!(board.legal(later));
        let prior = State { pose, spin: false };
        let mut inverse = Vec::new();
        board.predecessors(
            State {
                pose: later,
                spin: false,
            },
            |state, input| inverse.push((state, input)),
        );
        assert!(!inverse.contains(&(prior, Input::Cw)));
    }

    #[test]
    fn pinned_triangle_rotation_regressions() {
        // Expected results were produced by performKick in the pinned
        // Triangle/Spilink engine, independently of this implementation.
        // A wider development audit compared 62,331 legal-pose rotations.
        let board = board(&[23, 600, 993, 917, 147, 849, 763]);
        let cases = [
            (Piece::I, 0, 8, 0, Input::Ccw, -1, 10, 3),
            (Piece::I, 1, 8, 0, Input::Cw, 2, 10, 1),
            (Piece::J, 1, 8, 1, Input::Half, 2, 10, 3),
            (Piece::L, 0, 2, 0, Input::Half, 1, 3, 2),
            (Piece::T, 0, 2, 0, Input::Half, 1, 3, 2),
            (Piece::T, 0, 8, 0, Input::Cw, -1, 9, 1),
            (Piece::T, 2, 8, 0, Input::Ccw, 3, 9, 3),
        ];
        for (piece, x, y, rotation, input, tx, ty, tr) in cases {
            let source = Pose {
                piece,
                x,
                y,
                rotation,
            };
            let expected = Pose {
                piece,
                x: tx,
                y: ty,
                rotation: tr,
            };
            assert_eq!(board.rotate(source, input), Some(expected));
        }
    }

    #[test]
    fn actual_t_and_s_spin_single_witnesses() {
        let t_board = board(&[FULL_ROW ^ (1 << 4), 0b0001000111, 0b0000001111]);
        let t = t_board
            .replay(
                Piece::T,
                &[Input::Cw, Input::Sonic, Input::Cw, Input::HardDrop],
            )
            .unwrap();
        assert!(t.spin);
        let cleared = t_board.apply(&t).unwrap();
        assert_eq!(cleared.lines, 1);
        assert!(cleared.spin);
        assert!(t_board.can_reach_target(Piece::T, t.pose, true).is_some());

        // The S enters a low pocket via the fourth ordered 0->R kick.
        // Its upward translation is blocked, so All-mini+ awards a mini.
        let s_board = board(&[FULL_ROW ^ (1 << 5), 0, (1 << 3) | (1 << 5), 0, 1 << 3]);
        let s = s_board
            .replay(
                Piece::S,
                &[
                    Input::Right,
                    Input::Sonic,
                    Input::Left,
                    Input::Cw,
                    Input::HardDrop,
                ],
            )
            .unwrap();
        assert_eq!(
            s.pose,
            Pose {
                piece: Piece::S,
                x: 3,
                y: 2,
                rotation: 1
            }
        );
        assert!(s.spin);
        let cleared = s_board.apply(&s).unwrap();
        assert_eq!(cleared.lines, 1);
        assert!(cleared.spin);
        assert!(s_board.can_reach_target(Piece::S, s.pose, true).is_some());
    }

    #[test]
    fn drop_preserves_spin_only_when_distance_is_zero() {
        // A corner-qualified T can be translated upward, but positive falling
        // always erases rotation evidence; the classifier itself is not enough.
        let board = board(&[0b0000101000, 0, 0b0000001000]);
        let pose = Pose {
            piece: Piece::T,
            x: 3,
            y: 2,
            rotation: 2,
        };
        assert!(board.legal(pose));
        assert!(board.grounded(pose));
        assert!(board.spin_at(pose));
        let state = State { pose, spin: true };
        assert!(board.advance(state, Input::Sonic).unwrap().spin);
        assert!(board.advance(state, Input::HardDrop).unwrap().spin);
        let high = State {
            pose: Piece::T.spawn(),
            spin: true,
        };
        assert!(!board.advance(high, Input::HardDrop).unwrap().spin);
    }

    #[test]
    fn forged_spin_or_empty_witness_is_rejected() {
        let board = Board::default();
        let mut lock = board.replay(Piece::T, &[Input::HardDrop]).unwrap();
        lock.spin = true;
        assert!(board.apply(&lock).is_none());
        lock.spin = false;
        lock.path.clear();
        assert!(board.apply(&lock).is_none());
    }

    #[test]
    fn reverse_matches_forward_and_replays_in_caves() {
        // Ragged stacks and overhangs exercise floor kicks, spin histories,
        // sonic components, and physically legal but unreachable cavity poses.
        let boards = [
            Board::default(),
            board(&[0b1110011111, 0b1100011111, 0b1000011111]),
            board(&[0b1111101111, 0b1110001111, 0b1100011111, 0b0000000111]),
            board(&[0b1100000011, 0b1000000001, 0b1010000101, 0b0000000100]),
        ];
        for board in boards {
            for piece in Piece::ALL {
                let forward: HashSet<_> = board
                    .reachable_locks(piece)
                    .into_iter()
                    .map(|p| (p.pose, p.spin))
                    .collect();
                for rotation in 0..4 {
                    for x in -2..10 {
                        for y in 0..=7 {
                            let pose = Pose {
                                piece,
                                x,
                                y,
                                rotation,
                            };
                            if !board.lockable(pose) {
                                continue;
                            }
                            for wants_spin in [false, true] {
                                let expected = forward.contains(&(pose, true))
                                    || (!wants_spin && forward.contains(&(pose, false)));
                                let reverse = board.can_reach_target(piece, pose, wants_spin);
                                assert_eq!(
                                    reverse.is_some(),
                                    expected,
                                    "piece={piece:?} pose={pose:?} spin={wants_spin} rows={:?}",
                                    board.rows
                                );
                                if let Some(lock) = reverse {
                                    assert!(board.apply(&lock).is_some());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn inverse_sonic_cannot_cross_an_obstruction() {
        let board = board(&[0, 0, 0, 0, 1 << 4]);
        let target = Pose {
            piece: Piece::O,
            x: 4,
            y: 1,
            rotation: 0,
        };
        assert!(board.lockable(target));
        let mut sources = Vec::new();
        board.predecessors(
            State {
                pose: target,
                spin: false,
            },
            |state, input| {
                if input == Input::Sonic {
                    sources.push(state.pose.y);
                }
            },
        );
        assert!(sources.contains(&2));
        assert!(sources.iter().all(|&y| y < 4));
    }
}
