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
    /// Empty obligations without nearby two-cell body clearance. This is a
    /// geometric warning, not an exact reachability or permanence claim.
    pub access_deficit: u16,
    /// Occupied cells on rows tied to those low-clearance obligations.
    /// Clears and kicks can recover them; never use this as a proven bound.
    pub obstructed_mass: u16,
    /// A connected working opening chosen from the board, never fixed columns.
    pub workspace_columns: u16,
    pub workspace_width: u8,
    pub workspace_depth: u8,
    body_air: CellMask,
    service_air: CellMask,
    obstructed_cells: CellMask,
    workspace_roof_top: u8,
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
            access_deficit: 0,
            obstructed_mass: 0,
            workspace_columns: 0,
            workspace_width: 0,
            workspace_depth: 0,
            body_air: [0; HEIGHT],
            service_air: [0; HEIGHT],
            obstructed_cells: [0; HEIGHT],
            workspace_roof_top: 0,
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
        facts.find_working_air(board);
        facts
    }

    /// Flood a two-by-two clearance stencil, not a point and not a mino.
    /// The small fringe accommodates final spin tips and necessary roofs.
    /// A stencil miss only lowers heuristic quality: rotation kicks and later
    /// clears are deliberately left to the exact move/continuation verifier.
    fn find_working_air(&mut self, board: &Board) {
        let free: CellMask = std::array::from_fn(|y| FULL_ROW & !board.rows[y]);
        let pairs: CellMask = std::array::from_fn(|y| free[y] & (free[y] >> 1));
        let slots: CellMask = std::array::from_fn(|y| {
            if y + 1 < HEIGHT {
                pairs[y] & pairs[y + 1]
            } else {
                0
            }
        });
        let mut reached = [0_u16; HEIGHT];
        let mut queue = [0_u8; WIDTH * HEIGHT];
        let (mut read, mut len) = (0, 0);
        let top = HEIGHT - 2;
        if self.hole_count == 0 {
            // With no overhang every free column extends to the sky, so every
            // fitting stencil has a direct vertical route. Avoid the flood.
            reached = slots;
        } else {
            reached[top] = slots[top];
        }
        if self.hole_count != 0 && reached[top] != 0 {
            queue[len] = top as u8;
            len += 1;
        }
        while read < len {
            let y = queue[read] as usize;
            read += 1;
            let mut row = reached[y];
            loop {
                let expanded = row | ((row << 1 | row >> 1) & slots[y]);
                if row == expanded {
                    break;
                }
                row = expanded;
            }
            reached[y] = row;
            for adjacent in [y.checked_sub(1), (y + 1 < HEIGHT - 1).then_some(y + 1)] {
                if let Some(next) = adjacent {
                    let added = row & slots[next] & !reached[next];
                    if added != 0 {
                        reached[next] |= added;
                        // Every enqueue adds at least one of the <240 slots.
                        queue[len] = next as u8;
                        len += 1;
                    }
                }
            }
        }
        for (y, &row) in reached.iter().enumerate().take(HEIGHT - 1) {
            let cells = row | row << 1;
            self.body_air[y] |= cells;
            self.body_air[y + 1] |= cells;
        }
        self.service_air = self.body_air;
        for _ in 0..2 {
            let previous = self.service_air;
            for y in 0..HEIGHT {
                let vertical = if y > 0 { previous[y - 1] } else { 0 }
                    | if y + 1 < HEIGHT { previous[y + 1] } else { 0 };
                self.service_air[y] =
                    (previous[y] | previous[y] << 1 | previous[y] >> 1 | vertical)
                        & free[y]
                        & !self.sealed[y];
            }
        }
        for y in 0..self.height as usize {
            // Buried cavities and long, narrow shafts both need a specific
            // access plan. Broad empty space beside one bank is not bad mass.
            let constrained = self.holes[y]
                | if board.rows[y].count_ones() >= 6 {
                    free[y]
                } else {
                    0
                };
            let deficit = constrained & !self.service_air[y];
            self.access_deficit += deficit.count_ones() as u16;
            if deficit != 0 {
                self.obstructed_cells[y] = board.rows[y];
                self.obstructed_mass += board.rows[y].count_ones() as u16;
            }
        }
        let mut best = f64::NEG_INFINITY;
        for y in 0..self.height as usize {
            let mut air = self.body_air[y];
            while air != 0 {
                let left = air.trailing_zeros() as usize;
                let width = (air >> left).trailing_ones() as usize;
                let columns = ((1_u16 << width) - 1) << left;
                air &= !columns;
                if !(2..=6).contains(&width) || !two_banks(board.rows[y], columns) {
                    continue;
                }
                let mut connected = columns;
                let mut depth = 1;
                for above in y + 1..self.height as usize {
                    let mut alternatives = self.body_air[above];
                    let mut next = 0;
                    while alternatives != 0 {
                        let start = alternatives.trailing_zeros() as usize;
                        let span = (alternatives >> start).trailing_ones() as usize;
                        let run = ((1_u16 << span) - 1) << start;
                        alternatives &= !run;
                        if span <= 6
                            && (run & connected).count_ones() >= 2
                            && two_banks(board.rows[above], run)
                        {
                            next = run;
                            break;
                        }
                    }
                    if next == 0 {
                        break;
                    }
                    connected = next;
                    depth += 1;
                }
                // A four-cell mouth is only slightly preferred to the common
                // three-cell spin pocket. Wider is not unboundedly better.
                let value =
                    depth.min(4) as f64 * 2.0 - opening_width_cost(width as u8) - y as f64 * 0.05;
                if value > best {
                    best = value;
                    self.workspace_columns = columns;
                    self.workspace_width = width as u8;
                    self.workspace_depth = depth as u8;
                    self.workspace_roof_top = (y + 4).min(HEIGHT) as u8;
                }
            }
        }
    }

    fn goal_access_cost(&self, goal: &CellMask) -> u16 {
        (0..HEIGHT)
            .map(|y| (goal[y] & !self.service_air[y]).count_ones() as u16)
            .sum()
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

    /// Defect ordering only. Enclosures and almost-full rows earn no reward.
    pub(crate) fn score(&self) -> f64 {
        self.terrain_score(None)
    }

    /// Evaluate the two banks separately from a concrete working pocket.
    /// A low central passage is not a rough side stack. Conversely, selecting
    /// a low goal must not hide arbitrary towers in the same columns: the
    /// exemption ends one support cell above the goal's highest actual cell.
    /// This is an ordering preference, never a reachability certificate.
    pub(crate) fn terrain_score(&self, goal: Option<&SingleGoal>) -> f64 {
        self.terrain_score_inner(goal, false)
    }

    /// Only an actually executed spin may override the clearance relaxation.
    /// Its required roof is preserved even when no 2x2 stencil can reach it.
    pub(crate) fn terrain_score_witnessed(&self, goal: Option<&SingleGoal>) -> f64 {
        self.terrain_score_inner(goal, true)
    }

    fn terrain_score_inner(&self, goal: Option<&SingleGoal>, witnessed: bool) -> f64 {
        let witnessed_access = if witnessed {
            goal.map_or(0, |g| self.goal_access_cost(&g.required_empty))
        } else {
            0
        };
        let witnessed_mass = if witnessed {
            goal.map_or(0, |g| {
                self.obstructed_cells[g.clear_row as usize].count_ones() as u16
            })
        } else {
            0
        };
        let defects = self.hole_count as f64 * 2.0
            + self.sealed_count as f64 * 0.5
            + self.sealed_components as f64
            + self.cover_depth as f64 * 0.25
            + self.split_gaps as f64 * 2.0
            + (self.height as f64).powi(2) * 0.06
            + self.cells as f64 * 0.04
            + self.access_deficit.saturating_sub(witnessed_access) as f64 * 3.0
            + self.obstructed_mass.saturating_sub(witnessed_mass) as f64 * 0.35;
        let usable_goal =
            goal.filter(|g| witnessed || self.goal_access_cost(&g.required_empty) == 0);
        let dynamic = (self.workspace_columns != 0).then_some((
            (self.workspace_columns | self.workspace_columns << 1 | self.workspace_columns >> 1)
                & FULL_ROW,
            self.workspace_roof_top,
        ));
        let Some((working, roof_top)) = usable_goal.and_then(working_region).or(dynamic) else {
            return -defects - self.roughness as f64 * 0.16;
        };
        let footprint = usable_goal.map_or(self.workspace_columns, SingleGoal::working_columns);
        let mut linear = 0_u16;
        let mut squared = 0_u16;
        for x in 0..WIDTH - 1 {
            if working & (3 << x) == 0 {
                let change = self.heights[x].abs_diff(self.heights[x + 1]) as u16;
                linear += change;
                squared += change * change;
            }
        }
        let excess = self.heights.iter().enumerate().fold(0_u16, |sum, (x, h)| {
            // The rim can be a tall legitimate bank alongside a low pocket.
            // Only towers over the final mino's own columns are capped here.
            let above = if footprint & (1 << x) != 0 {
                h.saturating_sub(roof_top) as u16
            } else {
                0
            };
            sum + above * above
        });
        let opening_cost = if self.workspace_width == 0 {
            0.0
        } else {
            opening_width_cost(self.workspace_width)
                + 3_u8.saturating_sub(self.workspace_depth) as f64 * 0.6
        };
        -defects
            - linear as f64 * 0.20
            - squared as f64 * 0.045
            - excess as f64 * 0.10
            - opening_cost
    }
}

fn two_banks(row: u16, opening: u16) -> bool {
    let left = opening.trailing_zeros();
    let right = 16 - opening.leading_zeros();
    row & ((1_u16 << left) - 1) != 0 && row & (FULL_ROW << right) != 0
}

fn opening_width_cost(width: u8) -> f64 {
    match width {
        4 => 0.0,
        3 => 0.6,
        5 => 1.8,
        2 => 3.6,
        _ => (width.abs_diff(4) as f64).powi(2) * 1.35,
    }
}

fn working_region(goal: &SingleGoal) -> Option<(u16, u8)> {
    let cells = goal.pose.cells();
    if cells
        .iter()
        .any(|&(x, y)| x < 0 || x >= WIDTH as i8 || y < 0 || y >= HEIGHT as i8)
    {
        return None;
    }
    let footprint = cells.iter().fold(0, |mask, &(x, _)| mask | (1 << x));
    // The one-column rim contains lateral blockers and the space occupied
    // by neighbouring orientations during the final rotation. Treating only
    // a vertical T's two final columns as its working area would penalize the
    // bank along the adjacent entry pocket as an accidental cliff.
    let columns = (footprint | footprint << 1 | footprint >> 1) & FULL_ROW;
    let highest = cells.iter().map(|&(_, y)| y).max()? as u8;
    // Height is the first empty row: highest + 2 includes the row immediately
    // above the final mino, where a necessary roof blocker can be located.
    Some((columns, highest.saturating_add(2).min(HEIGHT as u8)))
}

fn ridge_pressure(heights: &[u8; WIDTH], column: usize, working: u16) -> u16 {
    let mut pressure = 0;
    for neighbour in [column - 1, column + 1] {
        if working & ((1 << column) | (1 << neighbour)) == 0 {
            let rise = heights[column].saturating_sub(heights[neighbour]) as u16;
            pressure += rise * rise;
        }
    }
    pressure
}

/// Quiet-placement ordering for an interior vertical I. Horizontal and edge I
/// placements, cavity insertions that do not raise the skyline, and filling an
/// existing valley receive no extra penalty. A remaining tetromino that can
/// geometrically level the new ridge reduces the penalty; it is not claimed
/// to be reachable or independently placeable until the normal verifier runs.
/// Existing BoardFacts are passed in to avoid another full-board analysis.
pub(crate) fn placement_bias(
    before: &BoardFacts,
    after: &BoardFacts,
    board_after: &Board,
    pose: Pose,
    remaining: u8,
    goal: Option<&SingleGoal>,
) -> f64 {
    if pose.piece != Piece::I {
        return 0.0;
    }
    let cells = pose.cells();
    let column = cells[0].0;
    if column <= 0 || column >= WIDTH as i8 - 1 || cells.iter().any(|&(x, _)| x != column) {
        return 0.0;
    }
    let column = column as usize;
    if after.heights[column] <= before.heights[column] {
        return 0.0;
    }
    let working = goal.and_then(working_region).map_or(0, |r| r.0);
    let prior = ridge_pressure(&before.heights, column, working);
    let current = ridge_pressure(&after.heights, column, working);
    let introduced = current.saturating_sub(prior);
    if introduced == 0 {
        return 0.0;
    }
    let mut best = current;
    // The intended finisher cannot also be spent as a quiet repair in this
    // same proposal. Geometry here is an optimistic, single-use resource hint.
    let available = remaining & !goal.map_or(0, |g| g.pose.piece.bit());
    for piece in Piece::ALL {
        if available & piece.bit() == 0 {
            continue;
        }
        let rotations = match piece {
            Piece::O => 1,
            Piece::I | Piece::S | Piece::Z => 2,
            _ => 4,
        };
        for rotation in 0..rotations {
            let shape = Pose {
                piece,
                rotation,
                x: 0,
                y: 0,
            }
            .cells();
            let min_x = shape.iter().map(|p| p.0).min().unwrap();
            let max_x = shape.iter().map(|p| p.0).max().unwrap();
            // A useful repair must touch the ridge or either neighbour.
            let start = (-min_x).max(column as i8 - 1 - max_x);
            let end = (WIDTH as i8 - max_x).min(column as i8 + 2 - min_x);
            for x in start..end {
                let mut bottoms = [i8::MAX; WIDTH];
                for &(dx, dy) in &shape {
                    let at = (x + dx) as usize;
                    bottoms[at] = bottoms[at].min(dy);
                }
                let y = (0..WIDTH)
                    .filter(|&at| bottoms[at] != i8::MAX)
                    .map(|at| after.heights[at] as i8 - bottoms[at])
                    .max()
                    .unwrap();
                // Every used column is supported without creating a new
                // buried gap. Necessary roofs are handled by goal generation,
                // not by pretending these arbitrary gaps are free repairs.
                if (0..WIDTH)
                    .any(|at| bottoms[at] != i8::MAX && y + bottoms[at] != after.heights[at] as i8)
                {
                    continue;
                }
                let repair = Pose {
                    piece,
                    rotation,
                    x,
                    y,
                };
                if !board_after.legal(repair) {
                    continue;
                }
                let mut heights = after.heights;
                let mut filled = [0_u16; HEIGHT];
                let mut preserves = true;
                for (at, row) in repair.cells() {
                    let (at, row) = (at as usize, row as usize);
                    if goal.is_some_and(|g| g.required_empty[row] & (1 << at) != 0) {
                        preserves = false;
                        break;
                    }
                    heights[at] = heights[at].max(row as u8 + 1);
                    filled[row] |= 1 << at;
                }
                if !preserves
                    || filled.iter().enumerate().any(|(row, cells)| {
                        *cells != 0 && board_after.rows[row] | cells == FULL_ROW
                    })
                {
                    continue;
                }
                best = best.min(ridge_pressure(&heights, column, working));
                if current.saturating_sub(best) >= introduced {
                    return -(introduced as f64) * 0.20;
                }
            }
        }
    }
    // Retain a small cost even for an optimistic repair: spending one specific
    // remaining piece is a dependency, not unconditional terrain flexibility.
    let recoverable = current.saturating_sub(best).min(introduced);
    -(introduced as f64 - recoverable as f64 * 0.75) * 0.8
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

fn goal_profile_cost(pose: Pose, profile: u8, opening: u16) -> u16 {
    let cells = pose.cells();
    let left = cells.iter().map(|p| p.0).min().unwrap();
    let right = cells.iter().map(|p| p.0).max().unwrap();
    let center_twice = left + right;
    let (center, left_bias, right_bias) = if opening == 0 {
        (9, 5, 13)
    } else {
        let start = opening.trailing_zeros() as i8;
        let end = 15 - opening.leading_zeros() as i8;
        (start + end, start * 2, end * 2)
    };
    match profile % 4 {
        // Preferences participate before per-piece shortlisting. Applying
        // these after a shortlist dominated by wall-supported goals would
        // leave the workers exploring essentially the same geometry.
        0 => center_twice.abs_diff(center) as u16 * 2,
        1 => center_twice.abs_diff(left_bias) as u16 * 2,
        2 => center_twice.abs_diff(right_bias) as u16 * 2,
        // A wider working pocket, including horizontal I use, with a weak
        // central preference. No lane bans a displaced or narrower fallback.
        _ => (4 - (right - left + 1)) as u16 * 5 + center_twice.abs_diff(center) as u16 / 3,
    }
}

/// Portfolio lanes prefer different working regions before truncating each
/// finisher's pool. The region may move as the board evolves; it is not a fixed
/// well, a mirrored probability union, or a restriction on physical legality.
/// Rejected shortlist entries are not physical or probability failures.
pub(crate) fn single_goals_for(
    board: &Board,
    inventory: u8,
    limit: usize,
    profile: u8,
) -> Vec<SingleGoal> {
    let finishers = (inventory & 127 & !Piece::O.bit()).count_ones() as usize;
    if limit == 0 || finishers == 0 {
        return Vec::new();
    }
    let per_piece = limit.div_ceil(finishers).max(2);
    let mut pools: [Vec<(GoalRank, SingleGoal)>; 7] = std::array::from_fn(|_| Vec::new());
    let facts = BoardFacts::new(board);
    let heights = facts.heights;
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
                debt(board, &filled, clauses.slice()) * 4
                    + facts.goal_access_cost(&empty) * 8
                    + goal_profile_cost(pose, profile, facts.workspace_columns),
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
