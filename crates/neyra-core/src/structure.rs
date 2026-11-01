//! Cheap structural facts and bounded Spin Single goals.
//!
//! A goal describes one no-clear preparation stage. Its occupied clauses are a
//! relaxation of a real move: `ready` still needs an exact rotation/Sonic witness
//! before it can be accepted. Shortlists and defect scores order the search;
//! neither a missing goal nor an unpaid debt proves a board impossible.

use crate::physics::{Board, Piece, Pose, FULL_ROW, HEIGHT, WIDTH};

pub(crate) type CellMask = [u16; HEIGHT];

// A smooth skyline is a weak convenience, not evidence of a usable spin
// sequence. Accessible side cavities are work space, while inaccessible
// cavities, severe one-column debt and excess towers keep their own costs.
const OUTWARD_BANK_LIMIT: f64 = 4.0;
const SKYLINE_PREFERENCE_LIMIT: f64 = 2.0;
const BANK_DENSITY_LIMIT: f64 = 8.0;
// Only cavities outside the central body's local service area incur this
// burden. An actual spin witness can additionally explain its own cells.
const UNEXPLAINED_HOLE_COST: f64 = 0.5;
// Prepared roofs may justify some local surface damage, but never acquire the
// value of an executed spin or cancel the board's hole/access obligations.
const PREPARED_ROOF_LIMIT: f64 = 6.0;
// A turning side wall can prepare a roof without already choosing a finisher.
// It remains a small shape preference, separate from witnessed goal readiness.
const CURLED_ROOF_COLUMN_REWARD: f64 = 0.5;
const CURLED_ROOF_PREFERENCE_LIMIT: f64 = 2.0;
// Repeated identical outlines need new preparation before they offer a turn.
// Inspect a bounded local band; neither more sky nor an arbitrarily tall wall
// creates additional preference beyond these limits.
const WORKSPACE_SHAPE_LEVELS: usize = 5;
const STRAIGHT_WORKSPACE_WINDOW_COST: f64 = 3.0;
const STRAIGHT_WORKSPACE_LIMIT: f64 = 8.0;

#[derive(Clone, Copy, Debug, Default)]
struct StraightWorkspace {
    columns: u16,
    first_row: u8,
    rows: u8,
}

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
    /// Fragmented row gaps that each contain a buried obligation. Sky-open
    /// surface steps are only a weak skyline preference, not cavity debt.
    pub buried_split_gaps: u16,
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
    is_symmetric_board: bool,
    body_air: CellMask,
    service_air: CellMask,
    central_air: CellMask,
    charged_hole_count: u16,
    charged_cover_depth: u16,
    charged_buried_splits: u16,
    charged_surface_splits: u16,
    cavity_roof_tops: [u8; WIDTH],
    obstructed_cells: CellMask,
    workspace_roof_top: u8,
    workspace_row: u8,
    lower_floor: u8,
    lower_narrow: [u16; 3],
    lower_slits: [u16; 3],
    workspace_reward: f64,
    bank_reward: f64,
    curled_roof_reward: f64,
    straight_workspaces: [StraightWorkspace; 16],
    straight_workspace_count: usize,
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
            buried_split_gaps: 0,
            heights: [0; WIDTH],
            roughness: 0,
            height: board.height(),
            cells: board.cell_count(),
            access_deficit: 0,
            obstructed_mass: 0,
            workspace_columns: 0,
            workspace_width: 0,
            workspace_depth: 0,
            is_symmetric_board: board.rows.iter().all(|&row| {
                row == row.reverse_bits() >> (u16::BITS as usize - WIDTH)
            }),
            body_air: [0; HEIGHT],
            service_air: [0; HEIGHT],
            central_air: [0; HEIGHT],
            charged_hole_count: 0,
            charged_cover_depth: 0,
            charged_buried_splits: 0,
            charged_surface_splits: 0,
            cavity_roof_tops: [0; WIDTH],
            obstructed_cells: [0; HEIGHT],
            workspace_roof_top: 0,
            workspace_row: 0,
            lower_floor: board
                .rows
                .iter()
                .position(|row| *row != 0 && *row != FULL_ROW)
                .unwrap_or(0) as u8,
            lower_narrow: [0; 3],
            lower_slits: [0; 3],
            workspace_reward: 0.0,
            bank_reward: 0.0,
            curled_roof_reward: 0.0,
            straight_workspaces: [StraightWorkspace::default(); 16],
            straight_workspace_count: 0,
        };
        for x in 0..WIDTH {
            let bit = 1 << x;
            let mut above = 0;
            for y in (0..facts.height as usize).rev() {
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
        facts.roughness = facts
            .heights
            .windows(2)
            .map(|w| w[0].abs_diff(w[1]) as u16)
            .sum();
        facts.find_components(board);
        facts.find_working_air(board);
        facts.find_workspace_shape(board);
        facts.summarize_cavity_costs(board);
        facts
    }

    /// Follow actual empty row intervals from the selected accessible pocket.
    /// At most five levels and one coherent path contribute roof potential;
    /// alternative branches never donate their lip columns to one another.
    /// A lip must grow from the previous occupied side wall, so a floating
    /// middle island and a completely closed lid do not earn this preference.
    fn find_workspace_shape(&mut self, board: &Board) {
        if self.workspace_columns == 0
            || self.workspace_row > self.lower_floor.saturating_add(2)
        {
            return;
        }
        let start = self.workspace_row as usize;
        let top = (start + WORKSPACE_SHAPE_LEVELS).min(self.height as usize);
        let weight = lower_weight(start.saturating_sub(self.lower_floor as usize));
        // A width<=6 interval can overlap at most two disjoint >=2-cell
        // successors. Five levels therefore need at most 31 path entries.
        let mut pending = [(0_u8, 0_u16, 0_u16, 0_u8, 0_u8); 64];
        let mut count = 0;
        let mut gaps = FULL_ROW & !board.rows[start];
        while gaps != 0 {
            let left = gaps.trailing_zeros() as usize;
            let width = (gaps >> left).trailing_ones() as usize;
            let columns = ((1_u16 << width) - 1) << left;
            gaps &= !columns;
            let connected = columns & self.workspace_columns & self.body_air[start];
            if left > 0 && left + width < WIDTH && (2..=6).contains(&width)
                && connected & (connected << 1) != 0
            {
                pending[count] = (start as u8, columns, 0, start as u8, 1);
                count += 1;
            }
        }
        while count != 0 {
            count -= 1;
            let (row, columns, lips, straight_start, straight_rows) = pending[count];
            self.curled_roof_reward = self.curled_roof_reward.max(
                (lips.count_ones() as f64 * CURLED_ROOF_COLUMN_REWARD)
                    .min(CURLED_ROOF_PREFERENCE_LIMIT) * weight,
            );
            if straight_rows >= 3 {
                if let Some(prior) = self.straight_workspaces[..self.straight_workspace_count]
                    .iter_mut().find(|segment| segment.columns == columns
                        && segment.first_row == straight_start)
                {
                    prior.rows = prior.rows.max(straight_rows);
                } else {
                    // At most three bounded >=2-cell intervals per row,
                    // over the fixed five-row band: 16 slots cover all of them.
                    debug_assert!(self.straight_workspace_count < self.straight_workspaces.len());
                    self.straight_workspaces[self.straight_workspace_count] = StraightWorkspace {
                        columns, first_row: straight_start, rows: straight_rows,
                    };
                    self.straight_workspace_count += 1;
                }
            }
            let y = row as usize;
            if y + 1 >= top { continue; }
            let left = columns.trailing_zeros() as usize;
            let right = (u16::BITS - columns.leading_zeros()) as usize;
            let occupied = board.rows[y + 1];
            let mut above = FULL_ROW & !occupied;
            while above != 0 {
                let next_left = above.trailing_zeros() as usize;
                let width = (above >> next_left).trailing_ones() as usize;
                let next_right = next_left + width;
                let next = ((1_u16 << width) - 1) << next_left;
                above &= !next;
                let shared = columns & next & self.body_air[y] & self.body_air[y + 1];
                if next_left == 0 || next_right == WIDTH || !(2..=6).contains(&width)
                    || shared & (shared << 1) == 0
                {
                    continue;
                }
                let mut added = 0;
                if right - left >= 3 {
                    if next_left > left {
                        let attached = ((1_u16 << next_left) - 1)
                            & !((1_u16 << (left - 1)) - 1);
                        if occupied & attached == attached {
                            added |= columns & attached;
                        }
                    }
                    if next_right < right {
                        let attached = ((1_u16 << (right + 1)) - 1)
                            & !((1_u16 << next_right) - 1);
                        if occupied & attached == attached {
                            added |= columns & attached;
                        }
                    }
                }
                let (first, rows) = if next == columns {
                    (straight_start, straight_rows + 1)
                } else { (row + 1, 1) };
                debug_assert!(count < pending.len());
                pending[count] = (row + 1, next, lips | added, first, rows);
                count += 1;
            }
        }
    }

    /// Three identical rows form one straight window; another unchanged row
    /// adds one window. Charge the worst connected segment rather than adding
    /// alternative pockets. A current exact Single may explain only one window
    /// of the segment containing its clearing footprint, not another segment.
    fn straight_workspace_cost(&self, witness: Option<&SingleGoal>) -> f64 {
        let reflected = self.is_symmetric_board && witness.is_some_and(|goal| {
            self.active_workspace(Some(goal)) != self.workspace_columns
        });
        self.straight_workspaces[..self.straight_workspace_count].iter()
            .map(|segment| {
                let columns = if reflected {
                    segment.columns.reverse_bits() >> (u16::BITS as usize - WIDTH)
                } else { segment.columns };
                let explained = witness.is_some_and(|goal| {
                    goal.clear_row >= segment.first_row
                        && goal.clear_row < segment.first_row + segment.rows
                        && goal.required_empty[goal.clear_row as usize] & columns != 0
                });
                let windows = segment.rows.saturating_sub(2)
                    .saturating_sub(u8::from(explained));
                (f64::from(windows) * STRAIGHT_WORKSPACE_WINDOW_COST)
                    .min(STRAIGHT_WORKSPACE_LIMIT)
            })
            .fold(0.0, f64::max)
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
        // Every row above the skyline is empty. Mark its stencils together
        // and start the flood at the skyline, avoiding a repeated traversal
        // of the unchanged sky. Near the real ceiling keep the original seed.
        let top = (self.height as usize).min(HEIGHT - 2);
        if self.hole_count == 0 {
            // With no overhang every free column extends to the sky, so every
            // fitting stencil has a direct vertical route. Avoid the flood.
            reached = slots;
        } else {
            reached[top..HEIGHT - 1].copy_from_slice(&slots[top..HEIGHT - 1]);
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
            let level = y.saturating_sub(self.lower_floor as usize);
            if y >= self.lower_floor as usize && level < 3 {
                let mut gaps = free[y];
                while gaps != 0 {
                    let left = gaps.trailing_zeros();
                    let width = (gaps >> left).trailing_ones();
                    let run = ((1_u16 << width) - 1) << left;
                    gaps &= !run;
                    if width <= 2 && two_banks(board.rows[y], run) {
                        self.lower_narrow[level] |= run;
                        if width == 1 {
                            self.lower_slits[level] |= run;
                        }
                    }
                }
            }
        }
        // Equal workspaces follow a canonical board orientation, not the
        // first physical x-run. This does not merge separate openings.
        let reflection_preferred = board.rows.iter().find_map(|&row| {
            let reflected = row.reverse_bits() >> (u16::BITS as usize - WIDTH);
            (row != reflected).then_some(row > reflected)
        }).unwrap_or(false);
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
                // Only the first four connected levels affect ordering.
                for above in y + 1..(y + 4).min(self.height as usize) {
                    let mut alternatives = self.body_air[above];
                    let mut next = 0;
                    while alternatives != 0 {
                        let start = alternatives.trailing_zeros() as usize;
                        let span = (alternatives >> start).trailing_ones() as usize;
                        let run = ((1_u16 << span) - 1) << start;
                        alternatives &= !run;
                        if span <= 6
                            && ((run & connected) & ((run & connected) << 1)) != 0
                            && two_banks(board.rows[above], run)
                        {
                            // Every connected branch contributes; reflection must not
                            // select a different first branch.
                            next |= run;
                        }
                    }
                    if next == 0 {
                        break;
                    }
                    connected = next;
                    depth += 1;
                }
                // A wide mouth high above an unresolved bottom is not a
                // usable low working pocket. Reward only body-clearance
                // connected openings near the lowest unfinished row.
                let level = y.saturating_sub(self.lower_floor as usize);
                let reward = lower_weight(level)
                    * (opening_width_reward(width as u8) + (depth - 1) as f64 * 0.6);
                // Count the density of each bank, not its absolute mass or
                // height. A solid pair of banks beside usable low air earns
                // more than isolated blocks around the same opening. The
                // weaker side limits this small bonus without demanding
                // equal bank widths, heights, or a fixed central column.
                let bank_reward = if width >= 3 && level < 3 {
                    let right = left + width;
                    let left_cells = (board.rows[y] & ((1_u16 << left) - 1)).count_ones();
                    let right_cells = (board.rows[y] & (FULL_ROW << right)).count_ones();
                    let density = (left_cells as f64 / left as f64)
                        .min(right_cells as f64 / (WIDTH - right) as f64);
                    lower_weight(level) * density * BANK_DENSITY_LIMIT
                } else {
                    0.0
                };
                let value = reward + bank_reward - level as f64 * 0.4;
                let canonical_left = if reflection_preferred {
                    WIDTH - left - width
                } else { left };
                let old_left = if self.workspace_columns == 0 { WIDTH }
                    else if reflection_preferred {
                        WIDTH - self.workspace_columns.trailing_zeros() as usize
                            - self.workspace_width as usize
                    } else { self.workspace_columns.trailing_zeros() as usize };
                let tie = (y, canonical_left);
                let old_tie = (self.workspace_row as usize, old_left);
                if value > best || (value == best && tie < old_tie) {
                    best = value;
                    self.workspace_columns = columns;
                    self.workspace_width = width as u8;
                    self.workspace_depth = depth as u8;
                    self.workspace_roof_top = (y + 4).min(HEIGHT) as u8;
                    self.workspace_row = y as u8;
                    self.workspace_reward = reward;
                    self.bank_reward = bank_reward;
                }
            }
        }
        self.find_central_air(&free, &slots, &mut queue);
        // Two equal, disconnected openings on a self-reflecting board are
        // alternatives. Only their shared core may relax unowned cavity or
        // history debt; an arbitrary first x-run must not waive one side.
        if self.is_symmetric_board {
            for row in &mut self.central_air {
                *row &= row.reverse_bits() >> (u16::BITS as usize - WIDTH);
            }
        }
    }

    /// Reuse the clearance slots and queue for a local component rooted in
    /// the selected low workspace. Global sky access alone does not make an
    /// unrelated side shaft part of the central working cavity. The top is
    /// only four rows above the workspace, so a high mouth cannot wash away
    /// a blocked bottom. This is still a stencil, not exact mino reachability.
    fn find_central_air(
        &mut self,
        free: &CellMask,
        slots: &CellMask,
        queue: &mut [u8; WIDTH * HEIGHT],
    ) {
        if self.workspace_columns == 0
            || self.workspace_row > self.lower_floor.saturating_add(2)
        {
            return;
        }
        let start = self.workspace_row as usize;
        let top = (start + 4).min(HEIGHT);
        let pairs = self.workspace_columns & (self.workspace_columns >> 1);
        let mut reached = [0_u16; HEIGHT];
        let (mut read, mut len) = (0, 0);
        for row in [start.checked_sub(1), Some(start)].into_iter().flatten() {
            if row + 1 < top {
                reached[row] = slots[row] & pairs;
                if reached[row] != 0 {
                    queue[len] = row as u8;
                    len += 1;
                }
            }
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
            for next in [y.checked_sub(1), (y + 2 < top).then_some(y + 1)]
                .into_iter()
                .flatten()
            {
                let added = row & slots[next] & !reached[next];
                if added != 0 {
                    reached[next] |= added;
                    queue[len] = next as u8;
                    len += 1;
                }
            }
        }
        for (y, &row) in reached.iter().enumerate().take(top - 1) {
            let cells = row | row << 1;
            self.central_air[y] |= cells;
            self.central_air[y + 1] |= cells;
        }
        for (y, row) in self.central_air.iter_mut().enumerate().take(top) {
            // One lateral tip can sit under a roof alongside the body. Do
            // not propagate down one-cell shafts or through a long slit.
            *row = (*row | *row << 1 | *row >> 1) & free[y] & !self.sealed[y];
        }
    }

    /// Raw cell history remains intact. Accessible working cavities have no
    /// current debt cost, but opening their roof never counts as filling them.
    pub(crate) fn penalized_debt(&self, cells: &CellMask) -> CellMask {
        std::array::from_fn(|y| cells[y] & !self.central_air[y] & FULL_ROW)
    }

    fn summarize_cavity_costs(&mut self, board: &Board) {
        self.charged_hole_count = self.hole_count;
        self.charged_cover_depth = self.cover_depth;
        let mut relieved_columns = 0_u16;
        for (y, &row) in board.rows.iter().enumerate().take(self.height as usize) {
            let relieved = self.holes[y] & self.central_air[y];
            self.charged_hole_count -= relieved.count_ones() as u16;
            relieved_columns |= relieved;
            if y + 1 < HEIGHT {
                let mut roofs = relieved & board.rows[y + 1];
                while roofs != 0 {
                    let x = roofs.trailing_zeros() as usize;
                    roofs &= roofs - 1;
                    self.cavity_roof_tops[x] = y as u8 + 2;
                }
            }
            if row == 0 {
                continue;
            }
            let gaps = FULL_ROW & !row;
            let runs = (gaps & !(gaps << 1)).count_ones() as u16;
            self.row_gap_runs += runs;
            self.split_gaps += runs.saturating_sub(1);
            // Outside the local workspace an uncovered row has only the
            // ordinary surface splits; no gap-by-gap classification is needed.
            if self.central_air[y] == 0 && self.holes[y] == 0 {
                self.charged_surface_splits += runs.saturating_sub(1);
                continue;
            }
            let mut remaining = if runs > 1 { gaps } else { 0 };
            let (mut buried, mut charged, mut charged_buried) = (0_u16, 0_u16, 0_u16);
            while remaining != 0 {
                let left = remaining.trailing_zeros();
                let width = (remaining >> left).trailing_ones();
                let run = ((1_u16 << width) - 1) << left;
                remaining &= !run;
                buried += u16::from(run & self.holes[y] != 0);
                let unserved = run & !self.central_air[y];
                charged += u16::from(unserved != 0);
                charged_buried += u16::from(unserved & self.holes[y] != 0);
            }
            self.buried_split_gaps += buried.saturating_sub(1);
            let buried_splits = charged_buried.saturating_sub(1);
            self.charged_buried_splits += buried_splits;
            self.charged_surface_splits += charged.saturating_sub(1) - buried_splits;
        }
        // Only columns with an exempt cavity need another cover scan. This
        // preserves raw cover facts and avoids a second complete board pass.
        while relieved_columns != 0 {
            let x = relieved_columns.trailing_zeros() as usize;
            relieved_columns &= relieved_columns - 1;
            let bit = 1 << x;
            let mut above = 0;
            for y in (0..self.heights[x] as usize).rev() {
                if board.rows[y] & bit != 0 {
                    above += 1;
                } else if self.central_air[y] & bit != 0 {
                    self.charged_cover_depth -= above;
                }
            }
        }
    }

    fn goal_access_cost(&self, goal: &CellMask) -> u16 {
        (0..HEIGHT)
            .map(|y| (goal[y] & !self.service_air[y]).count_ones() as u16)
            .sum()
    }

    fn lower_gap_cost(&self, witnessed_goal: Option<&SingleGoal>) -> f64 {
        (0..3)
            .map(|level| {
                let y = self.lower_floor as usize + level;
                let protected = self.central_air.get(y).copied().unwrap_or(0) | witnessed_goal
                    .filter(|_| y < HEIGHT)
                    .map_or(0, |goal| goal.required_empty[y]);
                let narrow = self.lower_narrow[level] & !protected;
                let slit = self.lower_slits[level] & !protected;
                lower_weight(level)
                    * (narrow.count_ones() as f64 * 0.8 + slit.count_ones() as f64 * 1.2)
            })
            .sum()
    }

    /// Recover occupancy from already computed column tops and buried holes.
    /// Only the selected goal's row needs this, so no board copy or new flood
    /// is necessary for roof preparation ordering.
    fn occupied_row(&self, row: usize) -> u16 {
        self.heights.iter().enumerate().fold(0, |mask, (x, &height)| {
            mask | if row < height as usize && self.holes[row] & (1 << x) == 0 {
                1 << x
            } else {
                0
            }
        })
    }

    /// Select one actual workspace for this concrete goal. On a symmetric
    /// board the other opening is an alternative, never simultaneous space.
    /// Geometric goal orientation resolves equal overlap without any physical
    /// movement or probability claim. Raw debt uses the shared core separately.
    fn active_workspace(&self, goal: Option<&SingleGoal>) -> u16 {
        let primary = self.workspace_columns;
        if !self.is_symmetric_board { return primary; }
        let reflected = primary.reverse_bits() >> (u16::BITS as usize - WIDTH);
        if primary == reflected { return primary; }
        let Some(goal) = goal else { return primary; };
        let columns = goal.working_columns();
        let overlap = (columns & primary).count_ones();
        let other_overlap = (columns & reflected).count_ones();
        if overlap != other_overlap {
            return if other_overlap > overlap { reflected } else { primary };
        }
        let working = working_region(goal).map_or(columns, |region| region.0);
        let overlap = (working & primary).count_ones();
        let other_overlap = (working & reflected).count_ones();
        if overlap != other_overlap {
            return if other_overlap > overlap { reflected } else { primary };
        }
        if columns != 0 {
            let center = |mask: u16| {
                mask.trailing_zeros() + (u16::BITS - mask.leading_zeros()) - 1
            };
            let target = center(columns);
            let distance = center(primary).abs_diff(target);
            let other_distance = center(reflected).abs_diff(target);
            if distance != other_distance {
                return if other_distance < distance { reflected } else { primary };
            }
        }
        for &row in &goal.required_empty {
            let mirror = row.reverse_bits() >> (u16::BITS as usize - WIDTH);
            if row != mirror { return if row > mirror { reflected } else { primary }; }
        }
        primary
    }

    /// A currently present overhang is useful only in the context of a low,
    /// accessible single goal with an unoccupied finisher footprint and some
    /// of its target row already built. This is capped preparation value, not
    /// a future-spin count, a reachability witness, or a hole/debt exemption.
    fn prepared_roof_reward(&self, goal: &SingleGoal) -> f64 {
        let row = goal.clear_row as usize;
        let workspace = self.active_workspace(Some(goal));
        if row >= HEIGHT
            || workspace == 0
            || self.workspace_row > self.lower_floor.saturating_add(2)
            || row > self.lower_floor as usize + 2
        {
            return 0.0;
        }
        let cells = goal.pose.cells();
        let mut contacts = 0;
        let mut columns = 0_u16;
        for &(x, y) in &cells {
            if x < 0 || x >= WIDTH as i8 || y < 0 || y >= HEIGHT as i8 {
                return 0.0;
            }
            let (x, y) = (x as usize, y as usize);
            let bit = 1 << x;
            columns |= bit;
            if y < self.heights[x] as usize && self.holes[y] & bit == 0 {
                return 0.0;
            }
            if y + 1 < HEIGHT
                && y + 1 < self.heights[x] as usize
                && self.holes[y + 1] & bit == 0
            {
                contacts += 1;
            }
        }
        if contacts == 0 || columns & workspace == 0 {
            return 0.0;
        }
        let required = goal.required_filled[row];
        let needed = required.count_ones();
        let present = (required & self.occupied_row(row)).count_ones();
        if needed == 0 || present * 2 < needed {
            return 0.0;
        }
        let readiness = present as f64 / needed as f64;
        let low = match row.saturating_sub(self.lower_floor as usize) {
            0 => 1.0,
            1 => 0.75,
            _ => 0.5,
        };
        (2.0 + f64::from(contacts.min(2)) * 1.5 + self.bank_reward * 0.1875)
            .min(PREPARED_ROOF_LIMIT)
            * readiness
            * low
    }

    fn active_roof_reward(&self, goal: &SingleGoal, witnessed: bool) -> f64 {
        let row = goal.clear_row as usize;
        if row >= HEIGHT {
            return 0.0;
        }
        let cells = goal.pose.cells();
        let mut contacts = 0;
        let mut columns = 0_u16;
        let mut floor = HEIGHT;
        let mut top = 0;
        for &(x, y) in &cells {
            if x < 0 || x >= WIDTH as i8 || y < 0 || y >= HEIGHT as i8 {
                return 0.0;
            }
            let (x, y) = (x as usize, y as usize);
            let bit = 1 << x;
            columns |= bit;
            floor = floor.min(y);
            top = top.max(y);
            if y < self.heights[x] as usize && self.holes[y] & bit == 0 {
                return 0.0;
            }
            // Count only the actual layer immediately above the finisher.
            // Extra height over this blocker cannot increase roof readiness.
            if y + 1 < HEIGHT
                && y + 1 < self.heights[x] as usize
                && self.holes[y + 1] & bit == 0
            {
                contacts += 1;
            }
        }
        if contacts == 0 {
            return 0.0;
        }
        // A real immediate witness may justify a narrow kick entry. A merely
        // prepared goal still needs local body air; sky far above the goal is
        // insufficient. The caller separately checks its full service access.
        let local_body = (floor.saturating_sub(1)..=(top + 1).min(HEIGHT - 1))
            .any(|y| self.body_air[y] & columns != 0);
        if !witnessed && !local_body {
            return 0.0;
        }
        let required = goal.required_filled[row];
        let needed = required.count_ones();
        let present = (required & self.occupied_row(row)).count_ones();
        if needed == 0 || present * 2 < needed {
            return 0.0;
        }
        let readiness = present as f64 / needed as f64;
        let local = match row.saturating_sub(floor) {
            0 => 1.0,
            1 => 0.75,
            _ => 0.5,
        };
        let bank = bank_density(self.occupied_row(row), columns) * BANK_DENSITY_LIMIT;
        (2.0 + f64::from(contacts.min(2)) * 1.5 + bank * 0.1875)
            .min(PREPARED_ROOF_LIMIT)
            * readiness
            * local
    }

    // Flood only buried cells. Any adjacent empty non-hole cell has an open
    // vertical path to the ceiling, so a second whole-board air flood is unneeded.
    fn find_components(&mut self, board: &Board) {
        let mut visited = [0_u16; HEIGHT];
        let mut queue = [(0_u8, 0_u8); WIDTH * HEIGHT];
        for y in 0..self.height as usize {
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
            .map(|y| (self.holes[y] & !self.central_air[y] & !owned[y]).count_ones() as u16)
            .sum()
    }

    /// Static workspace and defect ordering, not a spin-feasibility claim.
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
        let unexplained_holes = if witnessed {
            goal.map_or(self.charged_hole_count, |g| self.unplanned_holes(&g.required_empty))
        } else {
            self.charged_hole_count
        };
        // A reachable body stencil is only a preparation hint. Keep a
        // residual cost for its roofed cells; the old baseline exempted
        // these cells entirely from geometry, debt and pursuit together.
        // Charge the geometric obligation once rather than each history.
        let connected = (0..HEIGHT).map(|y| {
            let owned = if witnessed { goal.map_or(0, |g| g.required_empty[y]) } else { 0 };
            (self.holes[y] & self.central_air[y] & !owned).count_ones() as u16
        }).sum::<u16>();
        let connected_cover = self.cover_depth.saturating_sub(self.charged_cover_depth);
        let connected_splits = self.buried_split_gaps.saturating_sub(self.charged_buried_splits);
        let residual = f64::from(connected) * 0.5
            + f64::from(connected_cover) * 0.05
            + f64::from(connected_splits) * 0.5;
        let defects = residual + self.charged_hole_count as f64 * 2.0
            + unexplained_holes as f64 * UNEXPLAINED_HOLE_COST
            + self.sealed_count as f64 * 0.5
            + self.sealed_components as f64
            + self.charged_cover_depth as f64 * 0.25
            + self.charged_buried_splits as f64 * 2.0
            + (self.height as f64).powi(2) * 0.06
            + self.cells as f64 * 0.04
            + self.access_deficit.saturating_sub(witnessed_access) as f64 * 3.0
            + self.obstructed_mass.saturating_sub(witnessed_mass) as f64 * 0.35
            + self.lower_gap_cost(if witnessed { goal } else { None })
            + self.straight_workspace_cost(if witnessed { goal } else { None });
        let usable_goal =
            goal.filter(|g| witnessed || self.goal_access_cost(&g.required_empty) == 0);
        let workspace = self.active_workspace(usable_goal);
        let dynamic = (workspace != 0).then_some((
            (workspace | workspace << 1 | workspace >> 1)
                & FULL_ROW,
            self.workspace_roof_top,
        ));
        let surface_fragmentation = self.charged_surface_splits as f64 * 0.0625;
        let Some((working, roof_top)) = usable_goal.and_then(working_region).or(dynamic) else {
            let (_, _, repair_pressure) = self.bank_skyline(0, 0);
            return -defects
                - (self.roughness as f64 * 0.0125 + surface_fragmentation)
                    .min(SKYLINE_PREFERENCE_LIMIT)
                - repair_pressure;
        };
        let footprint = usable_goal.map_or(workspace, SingleGoal::working_columns);
        // The dynamic opening does not prove that its neighbouring bank
        // needs a roof. Reserve the full rim only for a concrete usable goal.
        let repair_protected = if usable_goal.is_some() {
            // A final mino may occupy only one side of a wider connected
            // opening. Keep its other side out of bank repair pressure too.
            working
                | if workspace & working != 0 {
                    workspace
                } else {
                    0
                }
        } else {
            footprint
        };
        // Use the actual selected opening when it belongs to this goal's
        // active band, otherwise the concrete footprint. The expanded kick
        // rim must not hide inward bank steps immediately beside the opening.
        let direction_workspace = if usable_goal.is_none_or(|goal| {
            workspace & footprint != 0
                && self.workspace_row.abs_diff(goal.clear_row) <= 2
        }) {
            workspace
        } else {
            footprint
        };
        // Apply the directional tie-break only around a present input-witnessed
        // spin. Unfinished first-bag foundations keep their original ordering.
        let directional = if witnessed && usable_goal.is_some() {
            self.outward_bank_cost(direction_workspace)
        } else { 0.0 };
        let (linear, squared, repair_pressure) = self.bank_skyline(working, repair_protected);
        let rim_pressure = usable_goal.map_or(0, |_| self.roof_rim_pressure(footprint, roof_top));
        let excess = self.heights.iter().enumerate().fold(0_u16, |sum, (x, h)| {
            // The rim can be a tall legitimate bank alongside a low pocket.
            // Only towers over the final mino's own columns are capped here.
            let above = if footprint & (1 << x) != 0 {
                h.saturating_sub(roof_top.max(self.cavity_roof_tops[x])) as u16
            } else {
                0
            };
            sum + above * above
        });
        let skyline = (linear as f64 * 0.02 + squared as f64 * 0.003 + surface_fragmentation)
            .min(SKYLINE_PREFERENCE_LIMIT);
        -defects - skyline - repair_pressure - rim_pressure as f64 * 0.45 - excess as f64 * 0.10
            - directional
            + self.workspace_reward
            + self.bank_reward
            + self.curled_roof_reward
            + usable_goal.map_or(0.0, |goal| {
                if witnessed && goal.clear_row > self.lower_floor.saturating_add(2) {
                    // A real higher spin can work above inherited residue.
                    // The residue and raw debt are retained without a reset.
                    self.active_roof_reward(goal, true)
                } else { self.prepared_roof_reward(goal) }
            })
    }

    fn outward_bank_cost(&self, workspace: u16) -> f64 {
        if workspace == 0 {
            return 0.0;
        }
        let left = workspace.trailing_zeros() as usize;
        let right = (u16::BITS - workspace.leading_zeros()) as usize;
        let mut linear = 0_u16;
        let mut severe = 0_u16;
        let mut add = |rise: u8| {
            let rise = u16::from(rise);
            linear += rise;
            let excess = rise.saturating_sub(1).min(4);
            severe += excess * excess;
        };
        for x in 1..left {
            add(self.heights[x].saturating_sub(self.heights[x - 1]));
        }
        for x in right..WIDTH - 1 {
            add(self.heights[x].saturating_sub(self.heights[x + 1]));
        }
        (linear as f64 * 0.15 + severe as f64 * 0.10).min(OUTWARD_BANK_LIMIT)
    }

    /// Ordinary bank steps only affect the capped skyline preference. Large
    /// cliffs and deep, one-column wells restrict the next bag's useful piece
    /// choices, so their excess receives a separate repair-pressure cost.
    /// This is not a feasibility test: no piece order or future clear is known.
    /// Reuse the nine bank edges. Normal skyline texture may use the generous
    /// pocket rim; severe bank defects only inherit that rim from a usable
    /// goal. In either case the movable central opening itself is protected.
    fn bank_skyline(&self, working: u16, repair_protected: u16) -> (u16, u16, f64) {
        let mut linear = 0_u16;
        let mut squared = 0_u16;
        let mut cliffs = 0_u16;
        let mut narrow_wells = 0_u16;
        for x in 0..WIDTH - 1 {
            // The roof immediately above a usable side cavity is not bank
            // damage. Charge only height above that roof; a low cavity must
            // never protect an unrelated tall spike in the same column.
            let (higher, lower) = if self.heights[x] >= self.heights[x + 1] {
                (x, x + 1)
            } else {
                (x + 1, x)
            };
            let change = self.heights[higher]
                .saturating_sub(self.heights[lower].max(self.cavity_roof_tops[higher])) as u16;
            if working & (3 << x) == 0 {
                linear += change;
                squared += change * change;
            }
            if repair_protected & (3 << x) == 0 {
                // Up to three rows often match an ordinary J/L or I surface.
                // Taller differences must not disappear when skyline caps.
                let surplus = change.saturating_sub(3);
                cliffs += repair_severity(surplus);
                if x > 0 && repair_protected & (1 << (x - 1)) == 0 {
                    let depth = self.heights[x - 1]
                        .min(self.heights[x + 1])
                        .saturating_sub(self.heights[x]);
                    // Two-deep notches remain ordinary surface texture.
                    // Deeper interior shafts introduce specific-piece debt;
                    // an edge well remains eligible for the normal edge I.
                    let surplus = depth.saturating_sub(2) as u16;
                    narrow_wells += repair_severity(surplus);
                }
            }
        }
        (
            linear,
            squared,
            cliffs as f64 * 0.45 + narrow_wells as f64 * 0.60,
        )
    }

    /// A goal's rim may contain its necessary roof or a tall, continuous bank.
    /// It may not conceal an isolated spike far above both that roof and the
    /// outer bank. Only the excess above both is charged, never the low pocket
    /// beside a tall bank. There are at most two rim-to-bank edges to inspect.
    fn roof_rim_pressure(&self, footprint: u16, roof_top: u8) -> u16 {
        let left = footprint.trailing_zeros() as usize;
        let right = (u16::BITS - footprint.leading_zeros()) as usize;
        let neighbours = [
            (left.checked_sub(1), left.checked_sub(2)),
            (
                (right < WIDTH).then_some(right),
                (right + 1 < WIDTH).then_some(right + 1),
            ),
        ];
        neighbours.into_iter().fold(0, |pressure, pair| {
            let (Some(rim), Some(bank)) = pair else {
                return pressure;
            };
            let rise = self.heights[rim].saturating_sub(
                roof_top.max(self.heights[bank]).max(self.cavity_roof_tops[rim]),
            );
            pressure + repair_severity(rise.saturating_sub(3) as u16)
        })
    }
}

/// Preserve a growing cost instead of another saturation plateau. The first
/// tetromino-height of excess is quadratic; beyond it the marginal cost stays
/// linear, so a ceiling-scale defect cannot inflate the squared term forever.
fn repair_severity(excess: u16) -> u16 {
    let local = excess.min(4);
    local * local + excess.saturating_sub(4) * 8
}

fn bank_density(row: u16, opening: u16) -> f64 {
    if opening == 0 { return 0.0; }
    let left = opening.trailing_zeros() as usize;
    let right = (u16::BITS - opening.leading_zeros()) as usize;
    if left == 0 || right >= WIDTH { return 0.0; }
    let left_cells = (row & ((1_u16 << left) - 1)).count_ones();
    let right_cells = (row & (FULL_ROW << right)).count_ones();
    let reflected = (row.reverse_bits() >> (u16::BITS as usize - WIDTH)) & FULL_ROW;
    let left_edge = row.trailing_ones().min(left as u32);
    let right_edge = reflected.trailing_ones().min((WIDTH - right) as u32);
    ((left_cells + left_edge) as f64 / (2 * left) as f64)
        .min((right_cells + right_edge) as f64 / (2 * (WIDTH - right)) as f64)
}

fn two_banks(row: u16, opening: u16) -> bool {
    let left = opening.trailing_zeros();
    let right = 16 - opening.leading_zeros();
    row & ((1_u16 << left) - 1) != 0 && row & (FULL_ROW << right) != 0
}

fn lower_weight(level: usize) -> f64 {
    match level {
        0 => 1.0,
        1 => 0.6,
        2 => 0.3,
        _ => 0.0,
    }
}

fn opening_width_reward(width: u8) -> f64 {
    match width {
        4 => 4.8,
        3 => 4.2,
        5 => 3.2,
        6 => 2.4,
        2 => -2.4,
        _ => 0.0,
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

    pub(crate) fn preserved_after(&self, board: &Board, pose: Option<Pose>) -> bool {
        if self.clear_row as usize >= HEIGHT { return false; }
        let added = if let Some(pose) = pose {
            if !board.legal(pose) { return false; }
            let Some(mask) = pose_mask(pose) else { return false; };
            mask
        } else { [0; HEIGHT] };
        (0..HEIGHT).all(|y| {
            let row = board.rows[y] | added[y];
            let empty = self.required_empty[y];
            row & empty == 0
                && (y == self.clear_row as usize || empty == 0 || row | empty != FULL_ROW)
        })
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
                    + u16::from(clear_row.saturating_sub(facts.lower_floor.saturating_add(1))) * 3
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
