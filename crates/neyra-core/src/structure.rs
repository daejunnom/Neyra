//! Cheap structural facts and bounded Spin Single goals.
//!
//! A goal describes one no-clear preparation stage. Its occupied clauses are a
//! relaxation of a real move: `ready` still needs an exact rotation/Sonic witness
//! before it can be accepted. Shortlists and defect scores order the search;
//! neither a missing goal nor an unpaid debt proves a board impossible.

use crate::physics::{Board, Piece, Pose, FULL_ROW, HEIGHT, WIDTH};

pub(crate) type CellMask = [u16; HEIGHT];

pub(crate) fn mask_count(mask: &CellMask) -> u16 {
    mask.iter().map(|row| row.count_ones() as u16).sum()
}

/// Map all rows simultaneously, using the pre-clear physical row indices.
pub(crate) fn map_after_clear(mask: &CellMask, cleared_rows: u32) -> CellMask {
    let mut result = [0; HEIGHT];
    let mut write = 0;
    for (y, &row) in mask.iter().enumerate() {
        if cleared_rows & (1 << y) == 0 {
            result[write] = row;
            write += 1;
        }
    }
    result
}

/// An opened roof does not pay a debt. Only filling or deleting its cell does.
pub(crate) fn remaining_cells(mask: &CellMask, board: &Board) -> CellMask {
    std::array::from_fn(|y| mask[y] & !board.rows[y] & FULL_ROW)
}

#[derive(Clone, Debug)]
pub(crate) struct BoardFacts {
    pub holes: CellMask,
    pub hole_count: u16,
    /// Four-neighbour enclosed empty cells; kicks may still reach them.
    pub sealed: CellMask,
    pub sealed_count: u16,
    pub components: u16,
    pub sealed_components: u16,
    pub cover_depth: u16,
    pub row_gap_runs: u16,
    pub split_gaps: u16,
    pub heights: [u8; WIDTH],
    pub roughness: u16,
    pub height: u8,
    pub cells: u16,
}

impl BoardFacts {
    pub(crate) fn new(board: &Board) -> Self {
        let mut facts = Self {
            holes: [0; HEIGHT],
            hole_count: 0,
            sealed: [0; HEIGHT],
            sealed_count: 0,
            components: 0,
            sealed_components: 0,
            cover_depth: 0,
            row_gap_runs: 0,
            split_gaps: 0,
            heights: [0; WIDTH],
            roughness: 0,
            height: board.height(),
            cells: board.cell_count(),
        };
        for x in 0..WIDTH {
            let bit = 1 << x;
            let mut above = 0;
            for y in (0..HEIGHT).rev() {
                if board.rows[y] & bit != 0 {
                    if above == 0 {
                        facts.heights[x] = y as u8 + 1;
                    }
                    above += 1;
                } else if above != 0 {
                    facts.holes[y] |= bit;
                    facts.hole_count += 1;
                    facts.cover_depth += above;
                }
            }
        }
        for &row in &board.rows {
            if row == 0 {
                continue;
            }
            let gaps = FULL_ROW & !row;
            let runs = (gaps & !(gaps << 1)).count_ones() as u16;
            facts.row_gap_runs += runs;
            facts.split_gaps += runs.saturating_sub(1);
        }
        facts.roughness = facts
            .heights
            .windows(2)
            .map(|w| w[0].abs_diff(w[1]) as u16)
            .sum();
        facts.find_components(board);
        facts
    }

    // Flood only buried cells. Any adjacent empty non-hole cell has an open
    // vertical path to the ceiling, so a second whole-board air flood is unneeded.
    fn find_components(&mut self, board: &Board) {
        let mut visited = [0_u16; HEIGHT];
        let mut queue = [(0_u8, 0_u8); WIDTH * HEIGHT];
        for y in 0..HEIGHT {
            let mut starts = self.holes[y] & !visited[y];
            while starts != 0 {
                let x = starts.trailing_zeros() as usize;
                self.components += 1;
                visited[y] |= 1 << x;
                queue[0] = (x as u8, y as u8);
                let (mut read, mut len) = (0, 1);
                let mut open = false;
                while read < len {
                    let (x, y) = queue[read];
                    read += 1;
                    for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let (nx, ny) = (x as i8 + dx, y as i8 + dy);
                        if nx < 0 || nx >= WIDTH as i8 || ny < 0 {
                            continue;
                        }
                        if ny >= HEIGHT as i8 {
                            open = true;
                            continue;
                        }
                        let (nx, ny) = (nx as usize, ny as usize);
                        let bit = 1 << nx;
                        if board.rows[ny] & bit != 0 {
                            continue;
                        }
                        if self.holes[ny] & bit == 0 {
                            open = true;
                        } else if visited[ny] & bit == 0 {
                            visited[ny] |= bit;
                            queue[len] = (nx as u8, ny as u8);
                            len += 1;
                        }
                    }
                }
                if !open {
                    self.sealed_components += 1;
                    self.sealed_count += len as u16;
                    for &(x, y) in &queue[..len] {
                        self.sealed[y as usize] |= 1 << x;
                    }
                }
                starts = self.holes[y] & !visited[y];
            }
        }
    }

    pub(crate) fn unplanned_holes(&self, owned: &CellMask) -> u16 {
        (0..HEIGHT)
            .map(|y| (self.holes[y] & !owned[y]).count_ones() as u16)
            .sum()
    }

    /// Exclude the chosen working columns, rather than assuming column five.
    pub(crate) fn side_roughness(&self, working_columns: u16) -> u16 {
        (0..WIDTH - 1)
            .filter(|&x| working_columns & (3 << x) == 0)
            .map(|x| self.heights[x].abs_diff(self.heights[x + 1]) as u16)
            .sum()
    }

    /// Defect ordering only. Enclosures and almost-full rows earn no reward.
    pub(crate) fn score(&self) -> f64 {
        -(self.hole_count as f64 * 2.0
            + self.sealed_count as f64 * 0.5
            + self.sealed_components as f64
            + self.cover_depth as f64 * 0.25
            + self.split_gaps as f64 * 2.0
            + self.roughness as f64 * 0.16
            + (self.height as f64).powi(2) * 0.06
            + self.cells as f64 * 0.04)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SingleGoal {
    pub pose: Pose,
    pub clear_row: u8,
    pub required_filled: CellMask,
    pub required_empty: CellMask,
    /// Each mask is an OR clause: at least one cell must be occupied.
    /// Different goals for one pose represent alternative spin supports.
    pub support_any: Vec<CellMask>,
}

impl SingleGoal {
    /// Ordering units, not a minimum number of cells, pieces or future spins.
    pub(crate) fn remaining(&self, board: &Board) -> u16 {
        debt(board, &self.required_filled, &self.support_any)
    }

    /// A quiet placement must preserve the final shape and pay some obligation.
    /// The caller reserves the finisher, checks no clear, and replays the move.
    pub(crate) fn accepts_support(&self, board: &Board, pose: Pose) -> bool {
        if !board.legal(pose) || intersects(&board.rows, &self.required_empty) {
            return false;
        }
        let Some(cells) = pose_mask(pose) else {
            return false;
        };
        if intersects(&cells, &self.required_empty) {
            return false;
        }
        (0..HEIGHT).any(|y| cells[y] & self.required_filled[y] & !board.rows[y] != 0)
            || self
                .support_any
                .iter()
                .any(|clause| !intersects(clause, &board.rows) && intersects(clause, &cells))
    }

    /// Necessary geometric readiness; not a spin-reachability certificate.
    pub(crate) fn ready(&self, board: &Board) -> bool {
        if intersects(&board.rows, &self.required_empty)
            || (0..HEIGHT).any(|y| self.required_filled[y] & !board.rows[y] != 0)
            || self
                .support_any
                .iter()
                .any(|clause| !intersects(clause, &board.rows))
            || !board.spin_at(self.pose)
        {
            return false;
        }
        board
            .place_geometry(self.pose)
            .is_some_and(|result| result.lines == 1 && result.cleared_rows == 1 << self.clear_row)
    }

    pub(crate) fn working_columns(&self) -> u16 {
        self.required_empty
            .iter()
            .fold(0, |columns, row| columns | row)
    }
}

fn intersects(a: &CellMask, b: &CellMask) -> bool {
    a.iter().zip(b).any(|(a, b)| a & b != 0)
}

fn debt(board: &Board, filled: &CellMask, clauses: &[CellMask]) -> u16 {
    let missing = (0..HEIGHT)
        .map(|y| (filled[y] & !board.rows[y]).count_ones() as u16)
        .sum::<u16>();
    missing
        + clauses
            .iter()
            .filter(|clause| !(0..HEIGHT).any(|y| clause[y] & (board.rows[y] | filled[y]) != 0))
            .count() as u16
}

fn pose_mask(pose: Pose) -> Option<CellMask> {
    let mut mask = [0; HEIGHT];
    for (x, y) in pose.cells() {
        if x < 0 || x >= WIDTH as i8 || y < 0 || y >= HEIGHT as i8 {
            return None;
        }
        mask[y as usize] |= 1 << x;
    }
    Some(mask)
}

/// Enumerate final poses from a row footprint, before any movement search.
/// Prepared poses need no present floor support; later quiet placements may add it.
fn row_poses(
    board: &Board,
    inventory: u8,
    prepared: bool,
    mut visit: impl FnMut(Pose, u8, CellMask),
) {
    let end = if prepared {
        (board.height() as usize + 1).max(2).min(HEIGHT)
    } else {
        HEIGHT
    };
    for row in 0..end {
        let missing = FULL_ROW & !board.rows[row];
        if missing == 0 || (!prepared && missing.count_ones() > 4) {
            continue;
        }
        for piece in Piece::ALL {
            if piece == Piece::O || inventory & piece.bit() == 0 {
                continue;
            }
            for rotation in 0..4 {
                let offsets = Pose {
                    piece,
                    x: 0,
                    y: 0,
                    rotation,
                }
                .cells();
                let min_x = offsets.iter().map(|p| p.0).min().unwrap();
                let max_x = offsets.iter().map(|p| p.0).max().unwrap();
                for (i, &(_, offset_y)) in offsets.iter().enumerate() {
                    if offsets[..i].iter().any(|p| p.1 == offset_y) {
                        continue;
                    }
                    let footprint = offsets
                        .iter()
                        .filter(|p| p.1 == offset_y)
                        .fold(0_u16, |mask, p| mask | (1 << p.0));
                    for x in -min_x..WIDTH as i8 - max_x {
                        let row_cells = if x < 0 {
                            footprint >> -x
                        } else {
                            footprint << x
                        };
                        if row_cells & !missing != 0 || (!prepared && row_cells != missing) {
                            continue;
                        }
                        let pose = Pose {
                            piece,
                            x,
                            y: row as i8 - offset_y,
                            rotation,
                        };
                        if !board.legal(pose) {
                            continue;
                        }
                        let Some(mask) = pose_mask(pose) else {
                            continue;
                        };
                        // An already inevitable second clear cannot be repaired
                        // by adding blocks during this no-clear preparation stage.
                        if (0..HEIGHT).any(|y| y != row && board.rows[y] | mask[y] == FULL_ROW) {
                            continue;
                        }
                        visit(pose, row as u8, mask);
                    }
                }
            }
        }
    }
}

/// All immediate geometric Spin Single targets. Exact witnesses remain lazy.
pub(crate) fn single_targets(board: &Board, inventory: u8) -> Vec<Pose> {
    let mut targets = Vec::new();
    row_poses(board, inventory, false, |pose, row, _| {
        if board.spin_at(pose)
            && board
                .place_geometry(pose)
                .is_some_and(|r| r.lines == 1 && r.cleared_rows == 1 << row)
        {
            targets.push(pose);
        }
    });
    targets
}

// Four shift blockers, or six corner-pair clauses plus floor support. Keep
// drafts on the stack; only shortlisted goals allocate their clause vectors.
#[derive(Clone)]
struct Clauses {
    masks: [CellMask; 7],
    len: usize,
}

impl Clauses {
    fn new() -> Self {
        Self {
            masks: [[0; HEIGHT]; 7],
            len: 0,
        }
    }
    fn slice(&self) -> &[CellMask] {
        &self.masks[..self.len]
    }

    fn add(
        &mut self,
        mut cells: CellMask,
        board: &Board,
        filled: &CellMask,
        empty: &CellMask,
    ) -> bool {
        for y in 0..HEIGHT {
            cells[y] &= !empty[y];
        }
        if intersects(&cells, &board.rows) || intersects(&cells, filled) {
            return true;
        }
        if cells.iter().all(|row| *row == 0) {
            return false;
        }
        if self
            .slice()
            .iter()
            .any(|old| (0..HEIGHT).all(|y| old[y] & !cells[y] == 0))
        {
            return true;
        }
        let mut i = 0;
        while i < self.len {
            if (0..HEIGHT).all(|y| cells[y] & !self.masks[i][y] == 0) {
                self.len -= 1;
                self.masks[i] = self.masks[self.len];
            } else {
                i += 1;
            }
        }
        self.masks[self.len] = cells;
        self.len += 1;
        self.masks[..self.len].sort_unstable();
        true
    }

    fn block_shift(
        &mut self,
        pose: Pose,
        dx: i8,
        dy: i8,
        board: &Board,
        filled: &CellMask,
        empty: &CellMask,
    ) -> bool {
        self.occupied_any(
            &pose.cells().map(|(x, y)| (x + dx, y + dy)),
            board,
            filled,
            empty,
        )
    }

    fn occupied_any(
        &mut self,
        points: &[(i8, i8)],
        board: &Board,
        filled: &CellMask,
        empty: &CellMask,
    ) -> bool {
        let mut cells = [0; HEIGHT];
        for &(x, y) in points {
            if x < 0 || x >= WIDTH as i8 || y < 0 {
                return true; // Real wall or floor, never the ceiling.
            }
            if y < HEIGHT as i8 {
                cells[y as usize] |= 1 << x;
            }
        }
        self.add(cells, board, filled, empty)
    }
}

fn support_height(clauses: &[CellMask], heights: &[u8; WIDTH]) -> u16 {
    clauses
        .iter()
        .map(|clause| {
            let mut nearest = HEIGHT as u16;
            for (y, &row) in clause.iter().enumerate() {
                let mut cells = row;
                while cells != 0 {
                    let x = cells.trailing_zeros() as usize;
                    nearest = nearest.min((y as u16 + 1).saturating_sub(heights[x] as u16));
                    cells &= cells - 1;
                }
            }
            nearest
        })
        .sum()
}

type GoalRank = (u16, u16, u8, Pose);

/// Select a few inexpensive support contracts. This is deliberately incomplete:
/// rejected shortlist entries are not physical or probability failures.
pub(crate) fn single_goals(board: &Board, inventory: u8, limit: usize) -> Vec<SingleGoal> {
    let finishers = (inventory & 127 & !Piece::O.bit()).count_ones() as usize;
    if limit == 0 || finishers == 0 {
        return Vec::new();
    }
    let per_piece = limit.div_ceil(finishers).max(2);
    let mut pools: [Vec<(GoalRank, SingleGoal)>; 7] = std::array::from_fn(|_| Vec::new());
    let heights = std::array::from_fn(|x| {
        board
            .rows
            .iter()
            .rposition(|row| row & (1 << x) != 0)
            .map_or(0, |y| y as u8 + 1)
    });
    row_poses(board, inventory, true, |pose, clear_row, empty| {
        let mut filled = [0; HEIGHT];
        filled[clear_row as usize] = FULL_ROW & !empty[clear_row as usize];
        let mut keep = |clauses: Clauses| {
            let pool = &mut pools[pose.piece as usize];
            if pool.iter().any(|(_, g)| {
                g.pose == pose && g.clear_row == clear_row && g.support_any == clauses.slice()
            }) {
                return;
            }
            let rank = (
                debt(board, &filled, clauses.slice()),
                support_height(clauses.slice(), &heights),
                clear_row,
                pose,
            );
            let mut same_count = 0;
            let mut same_worst = None;
            for (i, (_, g)) in pool.iter().enumerate() {
                if g.pose == pose && g.clear_row == clear_row {
                    same_count += 1;
                    same_worst = Some(i); // The pool is sorted by rank.
                }
            }
            // Preserve alternative supports without one orientation consuming
            // the entire shortlist. This cap is a heuristic, not a proof.
            let replace = if same_count >= 2 {
                same_worst
            } else if pool.len() == per_piece {
                Some(pool.len() - 1)
            } else {
                None
            };
            if let Some(i) = replace {
                if rank >= pool[i].0 {
                    return;
                }
                pool.remove(i);
            }
            pool.push((
                rank,
                SingleGoal {
                    pose,
                    clear_row,
                    required_filled: filled,
                    required_empty: empty,
                    support_any: clauses.slice().to_vec(),
                },
            ));
            pool.sort_by_key(|(rank, _)| *rank);
        };

        let mut immobile = Clauses::new();
        if [(-1, 0), (1, 0), (0, 1), (0, -1)]
            .into_iter()
            .all(|(dx, dy)| immobile.block_shift(pose, dx, dy, board, &filled, &empty))
        {
            keep(immobile);
        }

        if pose.piece == Piece::T {
            // At least three of four corners means every pair has at least
            // one occupied corner. Preserve the alternatives as OR clauses.
            let corners = [(-1, -1), (1, -1), (-1, 1), (1, 1)]
                .map(|(dx, dy)| (pose.x + 1 + dx, pose.y - 1 + dy));
            let mut clauses = Clauses::new();
            let mut possible = clauses.block_shift(pose, 0, -1, board, &filled, &empty);
            for i in 0..4 {
                for j in i + 1..4 {
                    possible &=
                        clauses.occupied_any(&[corners[i], corners[j]], board, &filled, &empty);
                }
            }
            if possible {
                keep(clauses);
            }
        }
    });
    // First keep one contract per available finisher when the limit permits;
    // otherwise a broad I footprint can hide every J/L/S/T/Z preparation goal.
    let mut first = Vec::new();
    let mut rest = Vec::new();
    for mut pool in pools {
        if !pool.is_empty() {
            first.push(pool.remove(0));
            rest.extend(pool);
        }
    }
    first.sort_by_key(|(rank, _)| *rank);
    rest.sort_by_key(|(rank, _)| *rank);
    first
        .into_iter()
        .chain(rest)
        .take(limit)
        .map(|(_, goal)| goal)
        .collect()
}
