//! Anytime de-novo fixed-bag construction search.
//! Geometry proposals are bounded and heuristic; canonical P7 certificates are exact.
//! Optional first-bag supply reflection is an explicitly exported approximation.
//! Clearra's separation of logical-row operation language and supply language is
//! retained here, with a 24-row SonicDropOnly physics backend.

use crate::physics::{Board, Input, Piece, Placement, Pose, HEIGHT};
use crate::structure::{
    map_after_clear, mask_count, placement_bias, remaining_cells, single_goals_for, single_targets,
    BoardFacts, CellMask, SingleGoal,
};
use crate::supply::{
    decode_order, minimum_finish, mirror_rank, permutation, symmetry_union_costs,
    LanguageNode, SymmetryCoverage, ALL_PIECES, P7_COUNT,
};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

const TARGET_SPINS: u8 = 40;
const GRAPH_LIMIT: usize = 4096;
const SPIN_DELTA: u8 = 3;
const RESIDUE_MARGIN: u16 = 12;
const BUDGET_STALE_ROUNDS: u64 = 64;
const EXPLORATION_INTERVAL: u64 = 8;
const QUALITY_RESEARCH_INTERVAL: u64 = 8;
const PACE_NEAR_SLACK: u16 = 2;
const PACE_DISTANCE_SLACK: u16 = 6;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    #[serde(alias = "minSuccessPercent")]
    pub min_success_percent: f64,
    #[serde(alias = "ignoreFirstBagProbability")]
    pub ignore_first_bag_probability: bool,
    /// Independent start filter. Zero removes this floor, never the need for
    /// at least one constructible supply. Later bags use min_success_percent.
    #[serde(alias = "firstBagMinSuccessPercent")]
    pub first_bag_min_success_percent: f64,
    #[serde(alias = "pieceGoal")]
    pub piece_goal: u16,
    #[serde(alias = "maxPieces")]
    pub max_pieces: u16,
    #[serde(alias = "budgetStart")]
    pub budget_start: u16,
    pub seed: u64,
    #[serde(alias = "beamWidth")]
    pub beam_width: usize,
    /// Independent geometric portfolio lane, never a supply/physics rule.
    #[serde(alias = "searchProfile")]
    pub search_profile: u8,
    #[serde(alias = "searchLane")]
    pub search_lane: u16,
    #[serde(alias = "searchLanes")]
    pub search_lanes: u16,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            min_success_percent: 100.0,
            ignore_first_bag_probability: true,
            first_bag_min_success_percent: 80.0,
            piece_goal: 119,
            max_pieces: 140,
            budget_start: 0,
            seed: 1,
            beam_width: 64,
            search_profile: 0,
            search_lane: 0,
            search_lanes: 1,
        }
    }
}

impl SearchConfig {
    fn validate(&self) -> Result<(), String> {
        if !self.min_success_percent.is_finite()
            || self.min_success_percent <= 0.0
            || self.min_success_percent > 100.0
        {
            return Err("최소 성공률은 0% 초과 100% 이하여야 합니다.".into());
        }
        if (self.min_success_percent * 100.0 - (self.min_success_percent * 100.0).round()).abs()
            > 1e-7
        {
            return Err("성공률은 소수 둘째 자리까지 입력해 주세요.".into());
        }
        if !self.first_bag_min_success_percent.is_finite()
            || !(0.0..=100.0).contains(&self.first_bag_min_success_percent)
            || (self.first_bag_min_success_percent * 100.0
                - (self.first_bag_min_success_percent * 100.0).round()).abs() > 1e-7
        {
            return Err("첫 가방 최소 성공률은 0–100%, 소수 둘째 자리까지 입력해 주세요. 0은 제한 없음입니다.".into());
        }
        if self.max_pieces == 0 || self.max_pieces > 700 || self.piece_goal == 0 {
            return Err("미노 상한은 1–700, 목표는 1 이상이어야 합니다.".into());
        }
        if self.budget_start > self.max_pieces
            || !(4..=256).contains(&self.beam_width)
            || self.search_profile > 3
            || self.search_lanes == 0
            || self.search_lanes > 128
            || self.search_lane >= self.search_lanes
        {
            return Err("탐색 시작 예산 또는 탐색 폭을 확인해 주세요.".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DisplayFrame {
    pub rows: [u16; HEIGHT],
    pub piece: String,
    /// Actual physical cells immediately before this lock's line clears.
    pub placed_cells: [(u8, u8); 4],
    pub spins: u8,
    pub placed: u16,
    pub lines: u8,
    pub spin: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CdfPoint {
    pub pieces: u16,
    pub probability: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateSummary {
    pub id: String,
    pub rows: [u16; HEIGHT],
    pub spins: u8,
    pub placed: u16,
    pub probability: f64,
    pub lower_probability: f64,
    pub upper_probability: f64,
    pub first_bag_probability: Option<f64>,
    pub first_bag_symmetry: bool,
    pub first_bag_symmetry_assumed: bool,
    pub complete: bool,
    pub qualified: bool,
    pub verified: bool,
    pub checked: u16,
    pub success: u16,
    pub failed: u16,
    pub bag_count: u16,
    pub budget_pieces: Option<u16>,
    pub cdf: Vec<CdfPoint>,
    pub ranking_score: f64,
    /// Raw boundary evaluation, separate from bounded internal spin ranking.
    pub field_quality: f64,
    pub frame_count: u16,
    pub frames: Vec<DisplayFrame>,
}

impl CandidateSummary {
    fn for_snapshot(&self, include_frames: bool) -> Self {
        Self {
            id: self.id.clone(),
            rows: self.rows,
            spins: self.spins,
            placed: self.placed,
            probability: self.probability,
            lower_probability: self.lower_probability,
            upper_probability: self.upper_probability,
            first_bag_probability: self.first_bag_probability,
            first_bag_symmetry: self.first_bag_symmetry,
            first_bag_symmetry_assumed: self.first_bag_symmetry_assumed,
            complete: self.complete,
            qualified: self.qualified,
            verified: self.verified,
            checked: self.checked,
            success: self.success,
            failed: self.failed,
            bag_count: self.bag_count,
            budget_pieces: self.budget_pieces,
            cdf: self.cdf.clone(),
            ranking_score: self.ranking_score,
            field_quality: self.field_quality,
            frame_count: self.frame_count,
            frames: if include_frames {
                self.frames.clone()
            } else {
                vec![]
            },
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressSnapshot {
    pub phase: String,
    pub expanded: u64,
    pub candidates: u64,
    pub verified_candidates: u64,
    pub qualified_bags: u16,
    pub qualified_spins: u8,
    pub pruned: u64,
    pub deepest_spins: u8,
    pub bags: u16,
    pub checked: u16,
    pub success: u16,
    pub failed: u16,
    pub frontier: usize,
    pub finished: bool,
    pub current: Option<CandidateSummary>,
    pub best: Option<CandidateSummary>,
    pub best_partial: Option<CandidateSummary>,
    pub duplicates: u64,
    /// Selective productivity/terrain cuts, never exact P7 failures.
    pub selective_pruned: u64,
    /// Verified same-depth maxima; peers may share these scheduling hints only.
    pub best_spins_by_bag: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TargetOperation {
    pub piece: Piece,
    /// Logical rows are local to this bag, before row compression.
    pub cells: [(u8, u8); 4],
}

#[derive(Clone)]
struct SeedState {
    board: Board,
    spins: u8,
    used: u8,
    deleted: u32,
    targets: Vec<TargetOperation>,
    target_key: [u64; 7],
    frames: Vec<DisplayFrame>,
    score: f64,
    quality: f64,
    pursuit: Option<Rc<SingleGoal>>,
    /// Newly buried cells created by quiet setup placements. Row deletion and
    /// actual filling pay this debt; merely opening a roof does not.
    debt: CellMask,
    future: Continuation,
    /// Optimistic first-lock choices before this bag's row deletions.
    first_mask: u8,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Plan {
    id: u64,
    start: Board,
    start_spins: u8,
    start_pieces: u16,
    expected: Board,
    expected_spins: u8,
    targets: Vec<TargetOperation>,
    seed_frames: Vec<DisplayFrame>,
    terminal: bool,
    /// Supply-label reflection at the empty start, or its inherited orientation
    /// assumption. Physical execution always uses the canonical SRS+ graph.
    assume_reflection: bool,
    /// Reuse the completed proposal's raw boundary evaluation in cheap cuts.
    #[serde(skip)]
    quality: f64,
    /// Periodic reduced re-search; selective comparisons are not proofs.
    #[serde(skip)]
    quality_escape: bool,
    /// A bounded, exact geometric continuation, not a supply certificate.
    #[serde(skip)]
    future: Continuation,
    #[serde(skip)]
    debt: CellMask,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct ExecutionKey {
    board: Board,
    used: u8,
    deleted: u32,
    spins: u8,
}

#[derive(Clone)]
struct ExecutionEdge {
    piece: u8,
    child: usize,
    placement: Placement,
    lines: u8,
}

struct ExecutionNode {
    key: ExecutionKey,
    edges: Vec<ExecutionEdge>,
    accepting: bool,
    spawn_mask: u8,
}

struct ExecutionGraph {
    nodes: Vec<ExecutionNode>,
    indices: HashMap<ExecutionKey, usize>,
    cursor: usize,
    overflow: bool,
    all_targets: u8,
}

#[derive(Default)]
struct PhysicsCache {
    locks: HashMap<(Board, Piece), CachedLocks>,
    targets: HashMap<(Board, Pose, bool), Option<Placement>>,
}

struct CachedLocks {
    all: Rc<Vec<Placement>>,
    proposals: Rc<Vec<Placement>>,
}

impl PhysicsCache {
    fn proposals(&mut self, board: Board, piece: Piece, broad: bool) -> Rc<Vec<Placement>> {
        if broad {
            self.locks(board, piece);
            return Rc::clone(&self.locks[&(board, piece)].proposals);
        }
        let mut placements = board.surface_locks(piece);
        for pose in single_targets(&board, piece.bit()) {
            let key = lock_geometry(pose, true);
            if placements
                .iter()
                .any(|p| p.spin && lock_geometry(p.pose, true) == key)
            {
                continue;
            }
            if let Some(placement) = self.target(board, pose, true) {
                // Only a successful witness claims this geometry. Another
                // orientation must still be tried if the first one failed.
                placements.push(placement);
            }
        }
        Rc::new(placements)
    }
    fn locks(&mut self, board: Board, piece: Piece) -> Rc<Vec<Placement>> {
        if let Some(locks) = self.locks.get(&(board, piece)) {
            return Rc::clone(&locks.all);
        }
        if self.locks.len() >= 512 {
            self.locks.clear();
        }
        let locks = Rc::new(board.search_locks(piece));
        let proposals = if matches!(piece, Piece::I | Piece::S | Piece::Z) {
            let mut seen = HashSet::with_capacity(locks.len());
            Rc::new(
                locks
                    .iter()
                    .filter(|p| seen.insert(lock_geometry(p.pose, p.spin)))
                    .cloned()
                    .collect(),
            )
        } else {
            Rc::clone(&locks)
        };
        self.locks.insert(
            (board, piece),
            CachedLocks {
                all: Rc::clone(&locks),
                proposals,
            },
        );
        locks
    }

    fn target(&mut self, board: Board, pose: Pose, spin: bool) -> Option<Placement> {
        // The search list has only O's orientation-zero representative.
        // Preserve exact-pose queries in other orientations through reverse
        // reachability rather than treating their missing cache entry as failure.
        if pose.piece != Piece::O || pose.rotation == 0 {
            if let Some(locks) = self.locks.get(&(board, pose.piece)) {
                return locks
                    .all
                    .iter()
                    .find(|p| p.pose == pose && (!spin || p.spin))
                    .cloned();
            }
        }
        let key = (board, pose, spin);
        if let Some(placement) = self.targets.get(&key) {
            return placement.clone();
        }
        if self.targets.len() >= 8192 {
            self.targets.clear();
        }
        let placement = board.can_reach_target(pose.piece, pose, spin);
        self.targets.insert(key, placement.clone());
        placement
    }
}

/// Equality only after lock: every following mino starts at spawn. The exact
/// verifier still preserves all projected orientations and their kick paths.
fn lock_geometry(pose: Pose, spin: bool) -> u64 {
    let mut cells = pose.cells().map(|(x, y)| (y as u16 * 10 + x as u16) as u8);
    cells.sort_unstable();
    u32::from_le_bytes(cells) as u64 | (u64::from(spin) << 32)
}

fn piece_name(piece: Piece) -> String {
    ["I", "J", "L", "O", "S", "T", "Z"][piece as usize].into()
}

fn spawn_mask(board: &Board) -> u8 {
    Piece::ALL.iter().fold(0, |mask, p| {
        if board.can_spawn(*p) {
            mask | (1 << (*p as u8))
        } else {
            mask
        }
    })
}

fn logical_row(physical: u8, deleted: u32) -> Option<u8> {
    let mut current = 0;
    for logical in 0..32_u8 {
        if deleted & (1_u32 << logical) == 0 {
            if current == physical {
                return Some(logical);
            }
            current += 1;
        }
    }
    None
}

fn deleted_after(deleted: u32, cleared: u32) -> Option<u32> {
    let mut result = deleted;
    for y in 0..HEIGHT as u8 {
        if cleared & (1 << y) != 0 {
            result |= 1 << logical_row(y, deleted)?;
        }
    }
    Some(result)
}

fn normalize_target(pose: Pose, deleted: u32) -> Option<TargetOperation> {
    let mut cells = [(0, 0); 4];
    for (index, (x, y)) in pose.cells().iter().copied().enumerate() {
        if x < 0 || y < 0 {
            return None;
        }
        cells[index] = (x as u8, logical_row(y as u8, deleted)?);
    }
    cells.sort_unstable();
    Some(TargetOperation {
        piece: pose.piece,
        cells,
    })
}

/// Logical rows can reach 31 after clears in a 24-row physical field.
/// The array slot identifies the piece; target cells are already sorted.
fn target_code(target: &TargetOperation) -> u64 {
    target
        .cells
        .iter()
        .enumerate()
        .fold(0, |code, (i, &(x, y))| {
            code | ((x as u64 | (y as u64) << 4) << (i * 9))
        })
}

/// All field, dependency and speculative-future terms share this one budget.
/// A one-spin tradeoff is possible; a two-spin advantage always wins.
fn spin_rank(spins: u8, quality: f64) -> f64 {
    spins as f64 + 0.95 * quality / (120.0 + quality.abs())
}

fn projected_poses(target: &TargetOperation, deleted: u32) -> Vec<Pose> {
    let mut cells = [(0_i8, 0_i8); 4];
    for (i, &(x, logical)) in target.cells.iter().enumerate() {
        if logical >= 32 || deleted & (1_u32 << logical) != 0 {
            return vec![];
        }
        let lower = if logical == 0 {
            0
        } else {
            deleted & ((1_u32 << logical) - 1)
        };
        let y = logical as i8 - lower.count_ones() as i8;
        if y < 0 || y >= HEIGHT as i8 {
            return vec![];
        }
        cells[i] = (x as i8, y);
    }
    cells.sort_unstable();
    let min_x = cells.iter().map(|c| c.0).min().unwrap();
    let min_y = cells.iter().map(|c| c.1).min().unwrap();
    let mut poses = vec![];
    let rotations = if target.piece == Piece::O { 1 } else { 4 };
    for rotation in 0..rotations {
        let origin = Pose {
            piece: target.piece,
            x: 0,
            y: 0,
            rotation,
        };
        let offsets = origin.cells();
        let x = min_x - offsets.iter().map(|c| c.0).min().unwrap();
        let y = min_y - offsets.iter().map(|c| c.1).min().unwrap();
        let pose = Pose {
            piece: target.piece,
            x,
            y,
            rotation,
        };
        let mut actual = pose.cells();
        actual.sort_unstable();
        if actual == cells {
            poses.push(pose);
        }
    }
    poses
}

/// Every ordered held/current pair represents 5! queues. Unknown future
/// targets may overestimate this mask; never remove a choice on a heuristic.
fn first_pair_upper(available: u8, spawn: u8, accepting: bool) -> u16 {
    if accepting {
        return P7_COUNT as u16;
    }
    let mut pairs = 0;
    for held in 0..7 {
        for current in 0..7 {
            if held != current
                && spawn & (1 << current) != 0
                && available & ((1 << held) | (1 << current)) != 0
            {
                pairs += 1;
            }
        }
    }
    pairs * 120
}

fn can_be_first(start: &Board, target: &TargetOperation) -> bool {
    projected_poses(target, 0).into_iter().any(|pose| {
        start
            .place_geometry(pose)
            .is_some_and(|result| result.lines == 0 || (result.lines == 1 && start.spin_at(pose)))
    })
}

/// Optimistic two-lock supply bound before a complete bag occupies a scarce
/// refinement slot. Geometry and spin shape are necessary; input reachability
/// is deliberately omitted. Each held/current/next triple represents 4! queues.
fn geometric_two_lock_upper(
    start: &Board,
    targets: &[TargetOperation],
    start_spins: u8,
    goal: u8,
) -> u16 {
    let inventory = targets.iter().fold(0, |mask, target| mask | target.piece.bit());
    // Early acceptance takes precedence over spawn. Leave terminal/partial
    // plans to the exact graph instead of imposing two mandatory locks on them.
    if targets.len() != 7 || inventory != ALL_PIECES || start_spins.saturating_add(2) >= goal {
        return P7_COUNT as u16;
    }
    let mut second = [0_u8; 7];
    let mut next_spawn = [0_u8; 7];
    for first in targets {
        let geometry = projected_poses(first, 0).into_iter().find_map(|pose| {
            start.place_geometry(pose).filter(|result| {
                result.lines == 0 || (result.lines == 1 && start.spin_at(pose))
            })
        });
        let Some(geometry) = geometry else { continue };
        let Some(deleted) = deleted_after(0, geometry.cleared_rows) else {
            return P7_COUNT as u16;
        };
        let index = first.piece as usize;
        next_spawn[index] = spawn_mask(&geometry.board);
        // All valid projected orientations occupy the same cells and leave
        // the same board; union spin eligibility before choosing that board.
        for target in targets {
            if target.piece == first.piece {
                continue;
            }
            if projected_poses(target, deleted).into_iter().any(|pose| {
                geometry.board.place_geometry(pose).is_some_and(|result| {
                    result.lines == 0 || (result.lines == 1 && geometry.board.spin_at(pose))
                })
            }) {
                second[index] |= target.piece.bit();
            }
        }
    }
    let initial_spawn = spawn_mask(start);
    let mut prefixes = 0;
    for held in 0..7 {
        for current in 0..7 {
            if held == current || initial_spawn & (1 << current) == 0 {
                continue;
            }
            for next in 0..7 {
                if next == held || next == current {
                    continue;
                }
                let current_first = next_spawn[current] & (1 << next) != 0
                    && second[current] & ((1 << held) | (1 << next)) != 0;
                let held_first = next_spawn[held] & (1 << next) != 0
                    && second[held] & ((1 << current) | (1 << next)) != 0;
                prefixes += u16::from(current_first || held_first);
            }
        }
    }
    prefixes * 24
}

impl ExecutionGraph {
    fn new(plan: &Plan) -> Self {
        let key = ExecutionKey {
            board: plan.start,
            used: 0,
            deleted: 0,
            spins: plan.start_spins,
        };
        let mut indices = HashMap::new();
        indices.insert(key.clone(), 0);
        let all_targets = plan
            .targets
            .iter()
            .fold(0, |m, t| m | (1 << (t.piece as u8)));
        Self {
            nodes: vec![ExecutionNode {
                spawn_mask: spawn_mask(&key.board),
                key,
                edges: vec![],
                accepting: false,
            }],
            indices,
            cursor: 0,
            overflow: false,
            all_targets,
        }
    }

    fn expand_one(&mut self, plan: &Plan, goal: u8, cache: &mut PhysicsCache) {
        if self.cursor >= self.nodes.len() || self.overflow {
            return;
        }
        let index = self.cursor;
        self.cursor += 1;
        let state = self.nodes[index].key.clone();
        if state.spins >= goal {
            self.nodes[index].accepting = true;
            return;
        }
        if state.used == self.all_targets {
            self.nodes[index].accepting = !plan.terminal
                && state.board == plan.expected
                && state.spins == plan.expected_spins
                && self.nodes[index].spawn_mask == ALL_PIECES;
            return;
        }
        for target in &plan.targets {
            let piece = target.piece as u8;
            if state.used & (1 << piece) != 0 {
                continue;
            }
            let mut valid = None;
            for pose in projected_poses(target, state.deleted) {
                let Some(geometry) = state.board.place_geometry(pose) else {
                    continue;
                };
                if geometry.lines > 1 {
                    continue;
                }
                let Some(placement) = cache.target(state.board, pose, geometry.lines == 1) else {
                    continue;
                };
                let Some(result) = state.board.apply(&placement) else {
                    continue;
                };
                if result.lines == 1 && !result.spin {
                    continue;
                }
                valid = Some((placement, result));
                break;
            }
            let Some((placement, result)) = valid else {
                continue;
            };
            let Some(deleted) = deleted_after(state.deleted, result.cleared_rows) else {
                self.overflow = true;
                return;
            };
            let next = ExecutionKey {
                board: result.board,
                used: state.used | (1 << piece),
                deleted,
                spins: state.spins + result.lines,
            };
            let child = if let Some(child) = self.indices.get(&next) {
                *child
            } else {
                if self.nodes.len() >= GRAPH_LIMIT {
                    self.overflow = true;
                    return;
                }
                let child = self.nodes.len();
                self.indices.insert(next.clone(), child);
                self.nodes.push(ExecutionNode {
                    spawn_mask: spawn_mask(&next.board),
                    key: next,
                    edges: vec![],
                    accepting: false,
                });
                child
            };
            self.nodes[index].edges.push(ExecutionEdge {
                piece,
                child,
                placement,
                lines: result.lines,
            });
        }
    }

    fn complete(&self) -> bool {
        !self.overflow && self.cursor == self.nodes.len()
    }

    fn first_upper(&self) -> u16 {
        if self.overflow || self.cursor == 0 {
            return P7_COUNT as u16;
        }
        let root = &self.nodes[0];
        first_pair_upper(
            root.edges
                .iter()
                .fold(0, |mask, edge| mask | (1 << edge.piece)),
            root.spawn_mask,
            root.accepting,
        )
    }

    fn prefix_possible(&self, queue: &[u8; 7], index: usize, cursor: u8, held: u8) -> bool {
        let node = &self.nodes[index];
        // An unexpanded node is optimistically accepting. Goal acceptance
        // precedes every next-spawn check, including partial terminal bags.
        if node.accepting || index >= self.cursor || cursor >= 3 {
            return true;
        }
        let current = queue[cursor as usize];
        if node.spawn_mask & (1 << current) == 0 {
            return false;
        }
        node.edges.iter().any(|edge| {
            let next_hold = if edge.piece == current {
                held
            } else if edge.piece == held {
                current
            } else {
                return false;
            };
            self.prefix_possible(queue, edge.child, cursor + 1, next_hold)
        })
    }

    fn two_lock_upper(&self) -> u16 {
        if self.overflow {
            return P7_COUNT as u16;
        }
        (0..210)
            .filter(|prefix| {
                let queue = permutation(prefix * 24);
                self.prefix_possible(&queue, 0, 1, queue[0])
            })
            .count() as u16
            * 24
    }

    fn language(&self) -> Vec<LanguageNode> {
        self.nodes
            .iter()
            .map(|node| LanguageNode {
                accepting: node.accepting,
                spawn_mask: node.spawn_mask,
                edges: node.edges.iter().map(|e| (e.piece, e.child)).collect(),
            })
            .collect()
    }

    fn witness(
        &self,
        code: u32,
        length: u8,
        start_pieces: u16,
    ) -> (Vec<DisplayFrame>, Vec<Vec<Input>>) {
        let mut frames = vec![];
        let mut inputs = vec![];
        let mut node = 0;
        for (i, piece) in decode_order(code, length).into_iter().enumerate() {
            let edge = self.nodes[node]
                .edges
                .iter()
                .find(|e| e.piece == piece)
                .expect("certified order edge");
            let target = &self.nodes[edge.child];
            frames.push(DisplayFrame {
                rows: target.key.board.rows,
                piece: piece_name(edge.placement.pose.piece),
                placed_cells: edge.placement.pose.cells().map(|(x, y)| (x as u8, y as u8)),
                spins: target.key.spins,
                placed: start_pieces + i as u16 + 1,
                lines: edge.lines,
                spin: edge.placement.spin,
            });
            inputs.push(edge.placement.path.clone());
            node = edge.child;
        }
        (frames, inputs)
    }


}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BagCertificate {
    id: String,
    start_rows: [u16; HEIGHT],
    end_rows: [u16; HEIGHT],
    start_spins: u8,
    end_spins: u8,
    targets: Vec<TargetOperation>,
    terminal: bool,
    /// The selected first-bag orientation relies on supply-label symmetry.
    /// Raw canonical evidence below never claims to verify reflected inputs.
    first_bag_symmetry_assumed: bool,
    mirrored_inputs_verified: bool,
    /// First-bag only. Canonical costs/codes below remain canonical evidence.
    #[serde(skip_serializing_if = "Option::is_none")]
    symmetry_costs: Option<Vec<u8>>,
    success: u16,
    effective_success: u16,
    checked: u16,
    /// Stable lexicographic P7 rank, I J L O S T Z.
    #[serde(serialize_with = "serialize_bit_words")]
    success_bits: Vec<u64>,
    /// 0=unknown, 1..7=shortest in this language, 255=no accepting policy path.
    costs: Vec<u8>,
    order_codes: Vec<u32>,
    representative_queue: [u8; 7],
    representative_inputs: Vec<Vec<Input>>,
}

fn serialize_bit_words<S: serde::Serializer>(
    words: &[u64],
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let hex: Vec<_> = words.iter().map(|word| format!("0x{word:016x}")).collect();
    hex.serialize(serializer)
}

struct VerificationJob {
    parent: usize,
    plan: Plan,
    graph: ExecutionGraph,
    language: Option<Vec<LanguageNode>>,
    checked: usize,
    success: usize,
    costs: Vec<u8>,
    codes: Vec<u32>,
    bits: Vec<u64>,
    upper: u16,
    prefix_checked: bool,
    symmetry: Option<SymmetryCoverage>,
}

impl VerificationJob {
    fn new(parent: usize, plan: Plan) -> Self {
        let symmetry = (plan.start_pieces == 0 && plan.start == Board::default() && plan.assume_reflection)
            .then(SymmetryCoverage::default);
        Self {
            graph: ExecutionGraph::new(&plan),
            parent,
            plan,
            language: None,
            checked: 0,
            success: 0,
            costs: vec![0; P7_COUNT],
            codes: vec![0; P7_COUNT],
            bits: vec![0; 79],
            upper: P7_COUNT as u16,
            prefix_checked: false,
            symmetry,
        }
    }

    fn advance(&mut self, goal: u8, cache: &mut PhysicsCache) {
        if !self.graph.complete() {
            self.graph.expand_one(&self.plan, goal, cache);
            self.upper = self.upper.min(self.graph.first_upper());
            if !self.prefix_checked
                && !self.graph.overflow
                && self.graph.cursor > 1
                && self.graph.nodes[0]
                    .edges
                    .iter()
                    .all(|e| e.child < self.graph.cursor)
            {
                self.upper = self.upper.min(self.graph.two_lock_upper());
                self.prefix_checked = true;
            }
            return;
        }
        if self.language.is_none() {
            self.language = Some(self.graph.language());
        }
        let end = (self.checked + 64).min(P7_COUNT);
        for rank in self.checked..end {
            if let Some((cost, code)) =
                minimum_finish(self.language.as_ref().unwrap(), &permutation(rank))
            {
                self.costs[rank] = cost;
                self.codes[rank] = code;
                self.bits[rank / 64] |= 1_u64 << (rank % 64);
                self.success += 1;
            } else {
                self.costs[rank] = 255;
            }
            if let Some(symmetry) = &mut self.symmetry {
                symmetry.observe(rank, self.costs[rank] != 255);
            }
        }
        self.checked = end;
        self.upper = self
            .upper
            .min((self.success + P7_COUNT - self.checked) as u16);
    }

    fn coverage_success(&self) -> u16 {
        self.symmetry.as_ref().map_or(self.success as u16, SymmetryCoverage::success)
    }

    fn coverage_upper(&self) -> u16 {
        self.symmetry.as_ref().map_or(self.upper, |symmetry| {
            symmetry.upper().min(self.upper.saturating_mul(2).min(P7_COUNT as u16))
        })
    }
}

#[derive(Clone, Default)]
struct Continuation {
    placements: Vec<Placement>,
    score: f64,
    paid: u16,
    spins: u8,
    anchor: Option<Rc<SingleGoal>>,
    /// Obstruction left by this one replayed continuation, not an exact bound.
    obstructed_mass: Option<u16>,
}

#[derive(Clone)]
struct ProbeState {
    board: Board,
    inventory: u8,
    debt: CellMask,
    quiet: u8,
    quiet_since_spin: u8,
    spins: u8,
    placements: Vec<Placement>,
    pursuit: Option<Rc<SingleGoal>>,
    anchor: Option<Rc<SingleGoal>>,
    deleted: u32,
    first_mask: u8,
}

enum ProbeTask {
    Expand(ProbeState),
    Finish(ProbeState, Pose),
    Support(ProbeState, Rc<SingleGoal>, Piece),
}

/// A small geometric continuation search. It consumes each available piece at
/// most once and never crosses a bag. A miss or exhausted budget is Unknown;
/// it never changes the exact P7 failure language or the cost lower bound.
struct ContinuationProbe {
    tasks: Vec<ProbeTask>,
    seen: Vec<(Board, u8, CellMask, Option<Rc<SingleGoal>>, u32, u8, u8)>,
    work: u16,
    debt_count: u16,
    base_score: f64,
    best: Continuation,
    start: Board,
    minimum_success: u16,
    profile: u8,
    budget: u16,
}

impl ContinuationProbe {
    fn new(board: Board, debt: CellMask, profile: u8, minimum_success: u16) -> Self {
        Self {
            tasks: vec![ProbeTask::Expand(ProbeState {
                board,
                inventory: ALL_PIECES,
                debt,
                quiet: 0,
                quiet_since_spin: 0,
                spins: 0,
                placements: vec![],
                pursuit: None,
                anchor: None,
                deleted: 0,
                first_mask: ALL_PIECES,
            })],
            seen: vec![],
            work: 0,
            debt_count: mask_count(&debt),
            base_score: BoardFacts::new(&board).score(),
            best: Continuation::default(),
            start: board,
            minimum_success,
            profile,
            budget: if profile == 3 { 96 } else { 64 },
        }
    }

    fn finished(&self) -> bool {
        self.work >= self.budget || self.tasks.is_empty()
    }

    fn step(&mut self, cache: &mut PhysicsCache) {
        if self.finished() {
            return;
        }
        self.work += 1;
        let Some(task) = self.tasks.pop() else { return };
        match task {
            ProbeTask::Expand(state) => {
                let lock_limit = if state.spins == 0 { 4 } else { 7 };
                if state.placements.len() >= lock_limit || state.inventory == 0 {
                    return;
                }
                // A one- or two-piece preparation may create the corner,
                // floor, or row cells needed by a final rotation. Looking only
                // for currently grounded spin slots would miss these plans.
                if state.quiet < 4
                    && state.quiet_since_spin < 2
                    && state.placements.len() + 1 < lock_limit
                {
                    let goals = if let Some(goal) = &state.pursuit {
                        vec![Rc::clone(goal)]
                    } else {
                        single_goals_for(&state.board, state.inventory, 12, self.profile)
                            .into_iter()
                            .map(Rc::new)
                            .collect()
                    };
                    for goal in goals.into_iter().rev() {
                        if goal.ready(&state.board) || goal.remaining(&state.board) > 8 {
                            continue;
                        }
                        for piece in Piece::ALL.into_iter().rev() {
                            if state.inventory & piece.bit() != 0 && piece != goal.pose.piece {
                                self.tasks.push(ProbeTask::Support(
                                    state.clone(),
                                    Rc::clone(&goal),
                                    piece,
                                ));
                            }
                        }
                    }
                }
                // Ready goals are generated from the exact missing-row
                // footprint; only these targets pay for reverse reachability.
                for pose in single_targets(&state.board, state.inventory)
                    .into_iter()
                    .rev()
                {
                    self.tasks.push(ProbeTask::Finish(state.clone(), pose));
                }
            }
            ProbeTask::Finish(state, pose) => {
                let Some(placement) = cache.target(state.board, pose, true) else {
                    return;
                };
                let Some(result) = state.board.apply(&placement) else {
                    return;
                };
                if result.lines != 1 || !result.spin {
                    return;
                }
                self.accept(
                    state,
                    placement,
                    result.board,
                    result.cleared_rows,
                    true,
                    None,
                );
            }
            ProbeTask::Support(state, goal, piece) => {
                let locks = cache.proposals(state.board, piece, self.profile == 3);
                let mut best = Vec::<(f64, Placement, Board)>::new();
                for placement in locks.iter() {
                    if !goal.accepts_support(&state.board, placement.pose) {
                        continue;
                    }
                    let Some(result) = state.board.apply(placement) else {
                        continue;
                    };
                    if result.lines != 0 {
                        continue;
                    }
                    let score = BoardFacts::new(&result.board).terrain_score(Some(&goal))
                        - goal.remaining(&result.board) as f64 * 8.0;
                    if best.iter().any(|(_, _, board)| *board == result.board) {
                        continue;
                    }
                    let at = best.iter().position(|b| b.0 < score).unwrap_or(best.len());
                    best.insert(at, (score, placement.clone(), result.board));
                    best.truncate(2);
                }
                for (_, placement, board) in best.into_iter().rev() {
                    self.accept(
                        state.clone(),
                        placement,
                        board,
                        0,
                        false,
                        Some(Rc::clone(&goal)),
                    );
                }
            }
        }
    }

    fn accept(
        &mut self,
        mut state: ProbeState,
        placement: Placement,
        board: Board,
        cleared_rows: u32,
        spin: bool,
        pursuit: Option<Rc<SingleGoal>>,
    ) {
        let Some(target) = normalize_target(placement.pose, state.deleted) else {
            return;
        };
        if !can_be_first(&self.start, &target) {
            state.first_mask &= !placement.pose.piece.bit();
        }
        if first_pair_upper(state.first_mask, spawn_mask(&self.start), false) < self.minimum_success
        {
            return;
        }
        let Some(deleted) = deleted_after(state.deleted, cleared_rows) else {
            return;
        };
        state.deleted = deleted;
        // Quiet support may accidentally enable a different spin than its
        // proposed goal. Only the first *executed* spin owns a terrain region.
        if state.anchor.is_none() && spin {
            let mut empty = [0; HEIGHT];
            for (x, y) in placement.pose.cells() {
                empty[y as usize] |= 1 << x;
            }
            let row = cleared_rows.trailing_zeros() as usize;
            let mut filled = [0; HEIGHT];
            filled[row] = crate::physics::FULL_ROW & !empty[row];
            state.anchor = Some(Rc::new(SingleGoal {
                pose: placement.pose,
                clear_row: row as u8,
                required_filled: filled,
                required_empty: empty,
                support_any: vec![],
            }));
        }
        state.inventory &= !placement.pose.piece.bit();
        state.board = board;
        state.debt = remaining_cells(&map_after_clear(&state.debt, cleared_rows), &board);
        state.spins += u8::from(spin);
        state.quiet += u8::from(!spin);
        state.quiet_since_spin = if spin { 0 } else { state.quiet_since_spin + 1 };
        state.placements.push(placement);
        state.pursuit = pursuit;
        if state.spins != 0 {
            let facts = BoardFacts::new(&board);
            let paid = self.debt_count.saturating_sub(mask_count(&state.debt));
            // Use one coherent route. Do not sum incompatible pockets or
            // spend the same finisher several times for hypothetical rewards.
            let quality = -(state.quiet as f64) * 10.0
                + paid as f64 * 6.0
                + (facts.score() - self.base_score) * 0.25;
            let score = state.spins as f64 * 60.0 + quality;
            if self.best.placements.is_empty()
                || spin_rank(state.spins, quality)
                    > spin_rank(
                        self.best.spins,
                        self.best.score - self.best.spins as f64 * 60.0,
                    )
            {
                self.best = Continuation {
                    placements: state.placements.clone(),
                    score,
                    paid,
                    spins: state.spins,
                    anchor: state.anchor.clone(),
                    obstructed_mass: Some(facts.obstructed_mass),
                };
                // Spend more only after an actual coherent spin witness.
                // The total remains bounded even if many support tasks exist.
                self.budget =
                    (if self.profile == 3 { 96 } else { 64 }) + u16::from(state.spins.min(4)) * 32;
            }
        }
        let key = (
            state.board,
            state.inventory,
            state.debt,
            state.pursuit.clone(),
            state.deleted,
            state.first_mask,
            state.quiet_since_spin,
        );
        if !self.seen.contains(&key) {
            self.seen.push(key);
            self.tasks.push(ProbeTask::Expand(state));
        }
    }
}

struct RefinementJob {
    pending: Vec<SeedState>,
    current: Option<(SeedState, ContinuationProbe)>,
    complete: Vec<SeedState>,
    profile: u8,
    minimum_success: u16,
}

impl RefinementJob {
    fn new(mut pending: Vec<SeedState>, profile: u8, minimum_success: u16) -> Self {
        pending.reverse();
        Self {
            pending,
            current: None,
            complete: vec![],
            profile,
            minimum_success,
        }
    }

    fn step(&mut self, goal: u8, cache: &mut PhysicsCache) {
        if let Some((mut seed, mut probe)) = self.current.take() {
            probe.step(cache);
            if probe.finished() {
                seed.future = probe.best;
                seed.quality = boundary_quality(&seed.board, &seed.debt, &seed.future);
                seed.score = spin_rank(seed.spins, seed.quality);
                insert_seed(&mut self.complete, seed, 8);
            } else {
                self.current = Some((seed, probe));
            }
        } else if let Some(seed) = self.pending.pop() {
            if seed.spins >= goal {
                // The final partial bag needs no useful future field.
                insert_seed(&mut self.complete, seed, 8);
            } else {
                let probe = ContinuationProbe::new(
                    seed.board,
                    seed.debt,
                    self.profile,
                    self.minimum_success,
                );
                self.current = Some((seed, probe));
            }
        }
    }

    fn finished(&self) -> bool {
        self.pending.is_empty() && self.current.is_none()
    }
}

struct GenerationJob {
    parent: usize,
    start: Board,
    start_spins: u8,
    start_pieces: u16,
    frontier: Vec<SeedState>,
    next_by_inventory: [Vec<SeedState>; 128],
    complete: Vec<SeedState>,
    state_index: usize,
    piece_index: usize,
    width: usize,
    depth: u8,
    setup: bool,
    variant: u64,
    refinement: Option<RefinementJob>,
    profile: u8,
    minimum_success: u16,
    broad: bool,
    pruned: u64,
    duplicates: u64,
    initial_used: u8,
    lane: u16,
    lanes: u16,
    root_choice: usize,
    best_bag_spins: u8,
    selective_pruned: u64,
    prune_budget: u16,
    first_bag_symmetry: bool,
    relaxed_setup: bool,
    continuation_minimum_success: u16,
}

impl GenerationJob {
    fn new(
        parent: usize,
        node: &TreeNode,
        width: usize,
        setup: bool,
        variant: u64,
        goal: u8,
        profile: u8,
        minimum_success: u16,
        lane: u16,
        lanes: u16,
    ) -> Self {
        let mut seed = SeedState {
            board: node.board,
            spins: node.spins,
            used: 0,
            deleted: 0,
            targets: vec![],
            target_key: [0; 7],
            frames: vec![],
            score: 0.0,
            quality: 0.0,
            pursuit: None,
            debt: node.debt,
            future: Continuation::default(),
            first_mask: ALL_PIECES,
        };
        // The earlier bounded probe found these placements from this exact
        // boundary. Use that coherent route as a geometric proposal prefix;
        // the final bag still goes through all 5040 normalized hold queues.
        // Other rounds explore independently instead of fixing this prefix.
        if !setup && variant % 3 != 2 {
            for placement in &node.future.placements {
                if seed.spins >= goal || seed.used & placement.pose.piece.bit() != 0 {
                    break;
                }
                let Some(result) = seed.board.apply(placement) else {
                    break;
                };
                if result.lines > 1 || (result.lines == 1 && !result.spin) {
                    break;
                }
                let Some(target) = normalize_target(placement.pose, seed.deleted) else {
                    break;
                };
                let Some(deleted) = deleted_after(seed.deleted, result.cleared_rows) else {
                    break;
                };
                let first_mask = if can_be_first(&node.board, &target) {
                    seed.first_mask
                } else {
                    seed.first_mask & !placement.pose.piece.bit()
                };
                if first_pair_upper(first_mask, spawn_mask(&node.board), false) < minimum_success {
                    break;
                }
                seed.debt = next_debt(&seed, &result.board, result.cleared_rows);
                seed.board = result.board;
                seed.used |= placement.pose.piece.bit();
                seed.deleted = deleted;
                seed.spins += result.lines;
                seed.first_mask = first_mask;
                seed.target_key[target.piece as usize] = target_code(&target);
                seed.targets.push(target);
                seed.frames.push(DisplayFrame {
                    rows: result.board.rows,
                    piece: piece_name(placement.pose.piece),
                    placed_cells: placement.pose.cells().map(|(x, y)| (x as u8, y as u8)),
                    spins: seed.spins,
                    placed: node.pieces + seed.targets.len() as u16,
                    lines: result.lines,
                    spin: result.spin,
                });
            }
        }
        seed.quality = board_score(&seed.board);
        seed.score = spin_rank(seed.spins, seed.quality);
        let depth = seed.targets.len() as u8;
        let initial_used = seed.used;
        let terminal = seed.spins >= goal;
        Self {
            parent,
            start: node.board,
            start_spins: node.spins,
            start_pieces: node.pieces,
            frontier: if terminal { vec![] } else { vec![seed.clone()] },
            next_by_inventory: std::array::from_fn(|_| Vec::new()),
            complete: if terminal { vec![seed] } else { vec![] },
            state_index: 0,
            piece_index: 0,
            width: width.clamp(4, 128),
            depth,
            setup,
            variant,
            refinement: None,
            profile,
            minimum_success,
            broad: variant % 4 == 3,
            pruned: 0,
            duplicates: 0,
            initial_used,
            lane,
            lanes,
            root_choice: 0,
            best_bag_spins: 0,
            selective_pruned: 0,
            prune_budget: u16::MAX,
            first_bag_symmetry: node.pieces == 0,
            relaxed_setup: false,
            continuation_minimum_success: minimum_success,
        }
    }

    fn advance(&mut self, rng: &mut Random, goal: u8, cache: &mut PhysicsCache) -> u64 {
        if let Some(refinement) = &mut self.refinement {
            refinement.step(goal, cache);
            return 0;
        }
        if self.state_index >= self.frontier.len() {
            return 0;
        }
        let state = &self.frontier[self.state_index];
        if delta_lag(
            state.spins,
            self.best_bag_spins,
            remaining_bag_spins(&state.board, state.used),
        ) {
            self.state_index += 1;
            self.piece_index = 0;
            self.selective_pruned += 1;
            return 0;
        }
        let foundation_ready = !self.setup
            || self.variant % 3 == 0
            || (self.variant % 3 == 2
                && self
                    .frontier
                    .get(self.state_index)
                    .is_some_and(|state| state.targets.len() >= 3));
        if self.piece_index == 0
            && foundation_ready
            && self.frontier[self.state_index].used != 0
            && self.frontier[self.state_index].pursuit.is_none()
        {
            let state = &mut self.frontier[self.state_index];
            let inventory = if self.setup {
                ALL_PIECES
            } else {
                ALL_PIECES & !state.used
            };
            let goals = single_goals_for(&state.board, inventory, 12, self.profile);
            if !goals.is_empty() {
                let choice = (self.variant as usize + self.state_index) % goals.len();
                state.pursuit = goals.into_iter().nth(choice).map(Rc::new);
            }
        }
        let state = &self.frontier[self.state_index];
        let piece = Piece::ALL[self.piece_index];
        self.piece_index += 1;
        if self.piece_index == 7 {
            self.piece_index = 0;
            self.state_index += 1;
        }
        if state.used & (1 << (piece as u8)) != 0 {
            return 0;
        }
        let root_choice = self.start_pieces == 0 && state.used == 0;
        // Every lane partitions the same first-lock geometry list. The rest
        // of each bag is built locally; no worker shares a supply certificate.
        let locks = cache.proposals(state.board, piece, self.broad && !root_choice);
        let per_inventory = self.inventory_limit(state.used | piece.bit());
        let before_facts = BoardFacts::new(&state.board);
        let before_holes = before_facts.holes;
        let mut expanded = 0;
        for placement in locks.iter() {
            expanded += 1;
            if root_choice {
                let choice = self.root_choice;
                self.root_choice += 1;
                if (choice + self.variant as usize) % self.lanes as usize != self.lane as usize {
                    continue;
                }
            }
            let Some(result) = state.board.apply(&placement) else {
                continue;
            };
            if result.lines > 1 || (result.lines == 1 && !result.spin) {
                continue;
            }
            // Reserve entire proposal rounds for robust setup bags. Without
            // this stratum, a local spin reward eliminates every quiet setup
            // long before its supply flexibility can be measured.
            if self.setup && result.lines != 0 {
                continue;
            }
            if result.lines == 0 {
                if let Some(pursuit) = &state.pursuit {
                    // Preserve a concrete finisher footprint while preparing
                    // it. Unrelated side-stack placements remain available.
                    if placement
                        .pose
                        .cells()
                        .iter()
                        .any(|&(x, y)| pursuit.required_empty[y as usize] & (1 << x) != 0)
                    {
                        continue;
                    }
                }
            }
            let Some(target) = normalize_target(placement.pose, state.deleted) else {
                continue;
            };
            let first_mask = if can_be_first(&self.start, &target) {
                state.first_mask
            } else {
                state.first_mask & !piece.bit()
            };
            let complete = state.spins + result.lines >= goal;
            let available = if complete {
                first_mask & (state.used | piece.bit())
            } else {
                first_mask
            };
            let first_upper = first_pair_upper(available, spawn_mask(&self.start), self.start_spins >= goal);
            let first_upper = if self.first_bag_symmetry {
                first_upper.saturating_mul(2).min(P7_COUNT as u16)
            } else { first_upper };
            if first_upper < self.minimum_success
            {
                self.pruned += 1;
                continue;
            }
            let Some(deleted) = deleted_after(state.deleted, result.cleared_rows) else {
                continue;
            };
            let spins = state.spins + result.lines;
            let used = state.used | piece.bit();
            if spins < goal
                && (delta_lag(
                    spins,
                    self.best_bag_spins,
                    remaining_bag_spins(&result.board, used),
                ) || (used == ALL_PIECES
                    && selective_reason(
                        result.board.cell_count(),
                        0, // Obstruction is checked after its continuation witness.
                        self.start_pieces + 7,
                        spins,
                        self.best_bag_spins,
                        goal,
                        self.prune_budget,
                    )
                    .is_some()))
            {
                self.selective_pruned += 1;
                continue;
            }
            let mut targets = state.targets.clone();
            let mut target_key = state.target_key;
            target_key[piece as usize] = target_code(&target);
            targets.push(target);
            if used == ALL_PIECES && !complete {
                let upper = geometric_two_lock_upper(&self.start, &targets, self.start_spins, goal);
                let upper = if self.first_bag_symmetry {
                    upper.saturating_mul(2).min(P7_COUNT as u16)
                } else { upper };
                if upper < self.minimum_success {
                    self.pruned += 1;
                    continue;
                }
            }
            let mut frames = state.frames.clone();
            frames.push(DisplayFrame {
                rows: result.board.rows,
                piece: piece_name(piece),
                placed_cells: placement.pose.cells().map(|(x, y)| (x as u8, y as u8)),
                spins,
                placed: self.start_pieces + targets.len() as u16,
                lines: result.lines,
                spin: result.spin,
            });
            let pursuit = if result.lines == 0 {
                state.pursuit.as_ref().and_then(|pursuit| {
                    // Once a non-setup bag spends its intended finisher,
                    // release that footprint and prepare another usable spin.
                    (self.setup || (state.used | piece.bit()) & pursuit.pose.piece.bit() == 0)
                        .then(|| Rc::clone(pursuit))
                })
            } else {
                None
            };
            let facts = BoardFacts::new(&result.board);
            let debt = next_debt_with_holes(
                state,
                &result.board,
                result.cleared_rows,
                &before_holes,
                &facts.holes,
            );
            let mut structural = facts.terrain_score(pursuit.as_deref()) * 2.0;
            structural -= mask_count(&debt) as f64 * 4.0;
            if result.lines == 0 {
                structural += placement_bias(
                    &before_facts,
                    &facts,
                    &result.board,
                    placement.pose,
                    ALL_PIECES & !(state.used | piece.bit()),
                    pursuit.as_deref(),
                );
            }
            if let Some(pursuit) = &pursuit {
                structural -= pursuit.remaining(&result.board) as f64 * 8.0;
                structural -= facts.unplanned_holes(&pursuit.required_empty) as f64 * 4.0;
                // Spending the intended finisher quietly may still be needed
                // to finish a fixed bag. Keep the footprint for a later bag,
                // but do not value it as an available same-bag spin.
                if !self.setup && (state.used | piece.bit()) & pursuit.pose.piece.bit() != 0 {
                    structural -= 16.0;
                }
            }
            let quality = if self.setup {
                // The first bag must first leave independent landing slots
                // for all seven pieces. A strong flatness/slot reward here
                // spends the floor too early and destroys P7 robustness.
                // Continuation refinement compares the completed foundations.
                setup_score(&self.start, &targets) * if self.relaxed_setup { 0.35 } else { 1.0 }
                    + structural * 0.06 + rng.unit() * 6.0
            } else {
                structural + setup_score(&self.start, &targets) * 0.25 + rng.unit() * 2.0
            };
            let score = spin_rank(spins, quality);
            let next = SeedState {
                board: result.board,
                spins,
                used: state.used | (1 << (piece as u8)),
                deleted,
                targets,
                target_key,
                frames,
                score,
                quality,
                pursuit,
                debt,
                future: Continuation::default(),
                first_mask,
            };
            if spins >= goal || next.used == ALL_PIECES {
                self.duplicates += u64::from(insert_seed(&mut self.complete, next, 12));
            } else {
                let inventory = next.used as usize;
                self.duplicates += u64::from(insert_seed(
                    &mut self.next_by_inventory[inventory],
                    next,
                    per_inventory,
                ));
            }
        }
        expanded
    }

    fn next_layer(&mut self) -> bool {
        if self
            .refinement
            .as_ref()
            .is_some_and(RefinementJob::finished)
        {
            self.complete = self.refinement.take().unwrap().complete;
            return true;
        }
        if self.refinement.is_some() {
            return false;
        }
        if self.state_index < self.frontier.len() {
            return false;
        }
        self.frontier = self.finish_inventory_layer();
        self.state_index = 0;
        self.piece_index = 0;
        self.depth += 1;
        if self.frontier.is_empty() || self.depth >= 7 {
            if self.complete.is_empty() {
                true
            } else {
                self.refinement = Some(RefinementJob::new(
                    std::mem::take(&mut self.complete),
                    self.profile,
                    self.continuation_minimum_success,
                ));
                false
            }
        } else {
            false
        }
    }

    fn display_seed(&self) -> Option<&SeedState> {
        if let Some(refinement) = &self.refinement {
            return refinement
                .complete
                .first()
                .or_else(|| refinement.current.as_ref().map(|(seed, _)| seed))
                .or_else(|| refinement.pending.last());
        }
        self.complete.first()
    }

    fn inventory_limit(&self, used: u8) -> usize {
        let n = 7 - self.initial_used.count_ones() as usize;
        let k = (used & !self.initial_used).count_ones() as usize;
        let mut masks = 1;
        for i in 0..k.min(n - k) {
            masks = masks * (n - i) / (i + 1);
        }
        self.width.div_ceil(masks).max(1)
    }

    fn finish_inventory_layer(&mut self) -> Vec<SeedState> {
        // Keep one alternative for every remaining inventory, then spend the
        // remaining total width on quality. Never multiply width by 35 masks.
        let mut next = Vec::new();
        let mut extra = Vec::new();
        for bucket in &mut self.next_by_inventory {
            if bucket.is_empty() {
                continue;
            }
            let mut entries = std::mem::take(bucket).into_iter();
            next.push(entries.next().unwrap());
            extra.extend(entries);
        }
        extra.sort_by(|a, b| b.score.total_cmp(&a.score));
        let remaining = self.width.saturating_sub(next.len());
        next.extend(extra.into_iter().take(remaining));
        next.sort_by(|a, b| b.score.total_cmp(&a.score));
        next
    }
}

fn next_debt(state: &SeedState, board: &Board, cleared_rows: u32) -> CellMask {
    next_debt_with_holes(
        state,
        board,
        cleared_rows,
        &BoardFacts::new(&state.board).holes,
        &BoardFacts::new(board).holes,
    )
}

fn next_debt_with_holes(
    state: &SeedState,
    board: &Board,
    cleared_rows: u32,
    before: &CellMask,
    after: &CellMask,
) -> CellMask {
    let mut debt = remaining_cells(&map_after_clear(&state.debt, cleared_rows), board);
    if cleared_rows == 0 {
        for y in 0..HEIGHT {
            debt[y] |= after[y] & !before[y];
        }
    }
    debt
}

fn insert_seed(list: &mut Vec<SeedState>, seed: SeedState, limit: usize) -> bool {
    // Only identical fixed tilings with the same execution/evaluation context
    // are duplicates. Capacity truncation is a separate selective heuristic.
    let duplicate = list.iter().position(|s| {
        s.board == seed.board
            && s.used == seed.used
            && s.deleted == seed.deleted
            && s.spins == seed.spins
            && s.pursuit == seed.pursuit
            && s.debt == seed.debt
            && s.first_mask == seed.first_mask
            && s.target_key == seed.target_key
    });
    if let Some(index) = duplicate {
        if list[index].score >= seed.score {
            return true;
        }
        list.remove(index);
    }
    let at = list
        .iter()
        .position(|s| s.score < seed.score)
        .unwrap_or(list.len());
    list.insert(at, seed);
    if list.len() > limit {
        list.pop();
    }
    duplicate.is_some()
}

/// A proposal heuristic, deliberately not an impossibility test.
fn board_score(board: &Board) -> f64 {
    BoardFacts::new(board).score() * 2.0
}

/// An intentionally conservative *ordering score*, never an eligibility test.
/// Direct-drop interference and missing base support expose fragile setup bags
/// cheaply. Actual rotation, OR support and hold possibilities remain the exact
/// execution verifier's responsibility.
fn setup_score(base: &Board, targets: &[TargetOperation]) -> f64 {
    let unsupported = targets
        .iter()
        .filter(|target| {
            !projected_poses(target, 0)
                .into_iter()
                .any(|pose| base.place_geometry(pose).is_some())
        })
        .count();
    let mut incoming = [0_u8; 7];
    for a in targets {
        for b in targets {
            if a.piece == b.piece {
                continue;
            }
            if a.cells
                .iter()
                .any(|(ax, ay)| b.cells.iter().any(|(bx, by)| ax == bx && by > ay))
            {
                incoming[b.piece as usize] |= 1 << (a.piece as u8);
            }
        }
    }
    let constrained = incoming.iter().filter(|mask| **mask != 0).count();
    let edges: u32 = incoming.iter().map(|mask| mask.count_ones()).sum();
    let cycles = (0..7)
        .flat_map(|a| (a + 1..7).map(move |b| (a, b)))
        .filter(|(a, b)| incoming[*a] & (1 << b) != 0 && incoming[*b] & (1 << a) != 0)
        .count();
    // Preserve independent foundation slots for pieces not yet placed. On an
    // empty board this favors narrow floor contacts before a low, wide skyline
    // greedily consumes all ten columns and forces several dependent pieces.
    let foundation_used = targets
        .iter()
        .flat_map(|target| target.cells.iter())
        .filter(|(x, y)| *y == 0 || base.occupied(*x as i8, *y as i8 - 1))
        .count();
    let remaining_foundation: usize = Piece::ALL
        .iter()
        .filter(|piece| !targets.iter().any(|target| target.piece == **piece))
        .map(|piece| if *piece == Piece::O { 2 } else { 1 })
        .sum();
    -(unsupported as f64) * 18.0
        - (foundation_used + remaining_foundation) as f64 * 5.0
        - constrained.saturating_sub(1) as f64 * 9.0
        - edges as f64 * 1.0
        - cycles as f64 * 12.0
}

struct TreeNode {
    parent: Option<usize>,
    children: Vec<usize>,
    board: Board,
    spins: u8,
    pieces: u16,
    counts: Vec<u16>,
    certificates: Vec<Rc<BagCertificate>>,
    frames: Vec<DisplayFrame>,
    visits: u64,
    reward: f64,
    future: Continuation,
    debt: CellMask,
    attempts: u32,
    allowance: u32,
    lease_epoch: u64,
    last_selected: u64,
    quality: f64,
    last_improved_attempt: u32,
    best_extension_spins: u8,
    best_extension_quality: f64,
    revisit_after: u64,
    /// Static obstruction discounted by one actual coherent continuation.
    obstructed_mass: u16,
}

#[derive(Clone)]
struct Archived {
    summary: CandidateSummary,
    certificates: Vec<Rc<BagCertificate>>,
    pending_plan: Option<Plan>,
}

struct Random(u64);
impl Random {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / ((1_u64 << 53) as f64)
    }
}

fn probability(counts: &[u16]) -> f64 {
    counts
        .iter()
        .fold(1.0, |p, c| p * *c as f64 / P7_COUNT as f64)
}

fn threshold_met(counts: &[u16], percent: f64) -> bool {
    let threshold = (percent * 100.0).round() as u32;
    let numerator = counts.iter().fold(BigUint::from(10_000_u32), |n, c| n * *c);
    let denominator = counts
        .iter()
        .fold(BigUint::from(threshold), |n, _| n * (P7_COUNT as u32));
    numerator >= denominator
}

fn required_success(counts: &[u16], percent: f64) -> u16 {
    let (mut lo, mut hi) = (0, P7_COUNT as u16);
    let mut prefix = counts.to_vec();
    prefix.push(0);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        *prefix.last_mut().unwrap() = mid;
        if threshold_met(&prefix, percent) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

fn boundary_quality(board: &Board, debt: &CellMask, future: &Continuation) -> f64 {
    boundary_quality_from_facts(&BoardFacts::new(board), debt, future)
}

fn boundary_quality_from_facts(facts: &BoardFacts, debt: &CellMask, future: &Continuation) -> f64 {
    let unexplained = mask_count(debt).saturating_sub(future.paid);
    facts.terrain_score_witnessed(future.anchor.as_deref()) * 2.0 + future.score
        - unexplained as f64 * 12.0
}

fn boundary_obstruction(board: &Board, future: &Continuation) -> u16 {
    let mass = BoardFacts::new(board).obstructed_mass;
    future.obstructed_mass.map_or(mass, |after| mass.min(after))
}

fn compare_probability(a: &[u16], b: &[u16]) -> std::cmp::Ordering {
    let a_num = a.iter().fold(BigUint::from(1_u8), |n, c| n * *c)
        * BigUint::from(P7_COUNT as u32).pow(b.len() as u32);
    let b_num = b.iter().fold(BigUint::from(1_u8), |n, c| n * *c)
        * BigUint::from(P7_COUNT as u32).pow(a.len() as u32);
    a_num.cmp(&b_num)
}

fn cost_lower_bound(board: &Board, spins: u8, goal: u8) -> u16 {
    let remaining = goal.saturating_sub(spins) as u16;
    remaining.max(
        (remaining * 10)
            .saturating_sub(board.cell_count())
            .div_ceil(4),
    )
}

/// An optimistic within-bag opportunity count, including material supplied by O
/// but never crediting O with a spin. Used only to avoid cutting preparations
/// that can still close the requested gap to a certified same-depth result.
fn remaining_bag_spins(board: &Board, used: u8) -> u8 {
    let remaining = ALL_PIECES & !used;
    let finishers = (remaining & !Piece::O.bit()).count_ones() as u16;
    let material = board.cell_count() + remaining.count_ones() as u16 * 4;
    finishers.min(material / 10) as u8
}

fn delta_lag(spins: u8, best_at_bag: u8, optimistic_more: u8) -> bool {
    spins
        .saturating_add(optimistic_more)
        .saturating_add(SPIN_DELTA)
        <= best_at_bag
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SelectiveReason {
    BagDelta,
    Pace,
    Residue,
}

/// User-requested selective pruning. None of these comparisons proves that a
/// board is impossible or changes a fixed plan's exact supply language.
fn selective_reason(
    cells: u16,
    obstructed_mass: u16,
    pieces: u16,
    spins: u8,
    best_at_bag: u8,
    goal: u8,
    budget: u16,
) -> Option<SelectiveReason> {
    // A genuine terminal prefix needs neither a productive future nor a
    // preferred residue. Same-depth delta still compares completed first bags.
    if spins >= goal {
        return None;
    }
    if delta_lag(spins, best_at_bag, 0) {
        return Some(SelectiveReason::BagDelta);
    }
    if pieces <= 7 {
        return None;
    }
    // Use all placed locks, with room to prepare while the horizon is far
    // away. The setup exemption above does not permanently shift the curve.
    // Cross multiplication avoids rounding a quota at each bag boundary.
    let horizon = u32::from(budget.max(1));
    let slack = u32::from(PACE_NEAR_SLACK) * horizon
        + u32::from(PACE_DISTANCE_SLACK) * u32::from(budget.saturating_sub(pieces));
    if u32::from(goal) * u32::from(pieces) >= u32::from(spins) * horizon + slack {
        return Some(SelectiveReason::Pace);
    }
    let remaining_spins = u16::from(goal - spins);
    let residue = (u32::from(cells) + u32::from(budget.saturating_sub(pieces)) * 4)
        .saturating_sub(u32::from(remaining_spins) * 10);
    if u32::from(obstructed_mass) > residue + u32::from(RESIDUE_MARGIN) {
        return Some(SelectiveReason::Residue);
    }
    None
}

fn focused_rounds(last_bag_spins: u8, future_spins: u8, stale: u32) -> u32 {
    let gain = u32::from(last_bag_spins.min(4));
    let witnessed = if stale < 4 {
        u32::from(future_spins.min(3))
    } else {
        0
    };
    let rounds = (1 + gain * gain + witnessed).min(16);
    rounds.div_ceil(1 + (stale / 4).min(7)).max(1)
}

fn productive_locks(pieces: u16) -> u16 {
    pieces.saturating_sub(7).max(7)
}

fn efficiency(spins: u8, pieces: u16) -> f64 {
    f64::from(spins) * 7.0 / f64::from(productive_locks(pieces))
}

fn pace_lag(spins: u8, pieces: u16, goal: u8, budget: u16) -> f64 {
    if pieces <= 7 { return 0.0; }
    f64::from(goal) * f64::from(pieces)
        / f64::from(budget.max(1)) - f64::from(spins)
}

fn pace_slack(pieces: u16, budget: u16) -> f64 {
    f64::from(PACE_NEAR_SLACK)
        + f64::from(PACE_DISTANCE_SLACK) * f64::from(budget.saturating_sub(pieces))
            / f64::from(budget.max(1))
}

fn under_pace(spins: u8, pieces: u16, goal: u8, budget: u16) -> bool {
    pieces > 7 && pace_lag(spins, pieces, goal, budget) >= pace_slack(pieces, budget) * 0.5
}

fn productive_rounds(
    spins: u8, pieces: u16, last_gain: u8, future_spins: u8,
    stale: u32, goal: u8, budget: u16,
) -> u32 {
    // Reduce the budget before reaching the selective cut, but grant the same
    // distance-aware preparation allowance used by the horizon policy.
    if under_pace(spins, pieces, goal, budget) {
        return 1;
    }
    if pieces <= 7 { return 1 + u32::from(future_spins != 0 && stale == 0); }
    focused_rounds(last_gain, future_spins.min(1), stale).min(8)
}

pub struct SearchSession {
    config: SearchConfig,
    goal: u8,
    rng: Random,
    tree: Vec<TreeNode>,
    generation: Option<GenerationJob>,
    waiting_plans: Vec<(usize, Plan)>,
    verification: Option<VerificationJob>,
    seen_plans: HashSet<Vec<u8>>,
    expanded: u64,
    candidates: u64,
    verified_candidates: u64,
    deepest_spins: u8,
    next_id: u64,
    current: Option<Archived>,
    best: Option<Archived>,
    other_complete: Option<Archived>,
    best_partial: Option<Archived>,
    finished: bool,
    last_checked: u16,
    last_success: u16,
    generation_rounds: u64,
    active_budget: u16,
    physics_cache: PhysicsCache,
    focus: Option<usize>,
    pruned: u64,
    qualified_bags: u16,
    qualified_spins: u8,
    preview_steps: u32,
    duplicates: u64,
    schedule_epoch: u64,
    preview_generation: u64,
    best_spins_by_bag: Vec<u8>,
    /// Worker-local quality references from fully qualified fixed bags only.
    /// Per-spin entries retain alternatives instead of one winning board.
    best_quality_by_bag: Vec<[f64; TARGET_SPINS as usize + 1]>,
    quality_pruned: u64,
    selective_pruned: u64,
    last_budget_progress_round: u64,
    benchmark_epoch: u64,
    dispatches: u64,
    focus_since_dispatch: u64,
    last_exploration_dispatch: u64,
    exploration_phase: u8,
}

impl SearchSession {
    pub fn new(config: SearchConfig) -> Result<Self, String> {
        config.validate()?;
        let seed = if config.seed == 0 {
            0x9e37_79b9_7f4a_7c15
        } else {
            config.seed
        };
        let root = TreeNode {
            parent: None,
            children: vec![],
            board: Board::default(),
            spins: 0,
            pieces: 0,
            counts: vec![],
            certificates: vec![],
            frames: vec![],
            visits: 0,
            reward: 0.0,
            future: Continuation::default(),
            debt: [0; HEIGHT],
            attempts: 0,
            allowance: 4,
            lease_epoch: 0,
            last_selected: 0,
            quality: 0.0,
            last_improved_attempt: 0,
            best_extension_spins: 0,
            best_extension_quality: f64::NEG_INFINITY,
            revisit_after: 0,
            obstructed_mass: 0,
        };
        let lower_bound = cost_lower_bound(&Board::default(), 0, TARGET_SPINS);
        let active_budget = if config.budget_start == 0 {
            config.piece_goal.max(lower_bound)
        } else {
            config.budget_start.max(lower_bound)
        }
        .min(config.max_pieces);
        Ok(Self {
            config,
            goal: TARGET_SPINS,
            rng: Random(seed),
            tree: vec![root],
            generation: None,
            waiting_plans: vec![],
            verification: None,
            seen_plans: HashSet::new(),
            expanded: 0,
            candidates: 0,
            verified_candidates: 0,
            deepest_spins: 0,
            next_id: 1,
            current: None,
            best: None,
            other_complete: None,
            best_partial: None,
            finished: false,
            last_checked: 0,
            last_success: 0,
            generation_rounds: 0,
            active_budget,
            physics_cache: PhysicsCache::default(),
            focus: Some(0),
            pruned: 0,
            qualified_bags: 0,
            qualified_spins: 0,
            preview_steps: 0,
            duplicates: 0,
            schedule_epoch: 0,
            preview_generation: u64::MAX,
            best_spins_by_bag: vec![0; 101],
            best_quality_by_bag: vec![[f64::NEG_INFINITY; TARGET_SPINS as usize + 1]; 101],
            quality_pruned: 0,
            selective_pruned: 0,
            last_budget_progress_round: 0,
            benchmark_epoch: 0,
            dispatches: 0,
            focus_since_dispatch: 0,
            last_exploration_dispatch: 0,
            exploration_phase: 0,
        })
    }

    /// One unit either expands a geometric piece choice, one execution node,
    /// or at most 64 finite supply products. No step consumes an entire search.
    pub fn step(&mut self, max_work: u32) {
        for _ in 0..max_work.clamp(1, 64) {
            if self.finished {
                break;
            }
            self.preview_steps = self.preview_steps.wrapping_add(1);
            if let Some(mut job) = self.verification.take() {
                let pieces = job.plan.start_pieces + job.plan.targets.len() as u16;
                if !job.plan.terminal && selective_reason(
                    job.plan.expected.cell_count(), 0, pieces, job.plan.expected_spins,
                    self.best_spins_by_bag[pieces as usize / 7], self.goal, self.active_budget,
                ).is_some() {
                    // A newer peer result may invalidate the search lease,
                    // but unexamined supplies remain unknown, never failures.
                    self.selective_pruned += 1;
                    self.forget_plan(job.parent, &job.plan);
                    self.backpropagate(job.parent, 0.0);
                    continue;
                }
                job.advance(self.goal, &mut self.physics_cache);
                self.last_checked = job.checked as u16;
                self.last_success = job.success as u16;
                if self.preview_steps % 16 == 0 {
                    self.update_verifying_preview(&job);
                }
                if job.graph.overflow {
                    // Incomplete analysis is not inserted into any failure bitset.
                    self.backpropagate(job.parent, 0.0);
                } else if job.checked == P7_COUNT {
                    self.finish_verification(job);
                } else if !self.can_still_qualify(&job) {
                    self.pruned += 1;
                    self.backpropagate(job.parent, 0.0);
                } else {
                    self.verification = Some(job);
                }
                continue;
            }
            if let Some(mut job) = self.generation.take() {
                // Imported or newly verified maxima take effect on the next
                // work slice, including a generation already in flight.
                job.best_bag_spins = self.best_spins_by_bag[(job.start_pieces as usize / 7) + 1];
                job.prune_budget = self.active_budget;
                let prior_pruned = job.pruned;
                let prior_duplicates = job.duplicates;
                let prior_selective = job.selective_pruned;
                self.expanded += job.advance(&mut self.rng, self.goal, &mut self.physics_cache);
                self.pruned += job.pruned - prior_pruned;
                self.duplicates += job.duplicates - prior_duplicates;
                self.selective_pruned += job.selective_pruned - prior_selective;
                if self.preview_steps % 16 == 0 {
                    self.update_generation_preview(&job);
                }
                if job.next_layer() {
                    let parent = job.parent;
                    for state in job.complete.into_iter().rev() {
                        let pieces = job.start_pieces + state.targets.len() as u16;
                        if state.spins < self.goal
                            && selective_reason(
                                state.board.cell_count(),
                                boundary_obstruction(&state.board, &state.future),
                                pieces,
                                state.spins,
                                self.best_spins_by_bag[pieces as usize / 7],
                                self.goal,
                                self.active_budget,
                            )
                            .is_some()
                        {
                            // Do not remember a selective rejection: wider
                            // later budgets may make this proposal eligible.
                            self.selective_pruned += 1;
                            continue;
                        }
                        let id = self.next_id;
                        self.next_id += 1;
                        let plan = Plan {
                            id,
                            start: job.start,
                            start_spins: job.start_spins,
                            start_pieces: job.start_pieces,
                            expected: state.board,
                            expected_spins: state.spins,
                            targets: state.targets,
                            seed_frames: state.frames,
                            terminal: state.spins >= self.goal,
                            assume_reflection: (job.start_pieces == 0 && (!self.config.ignore_first_bag_probability || self.config.first_bag_min_success_percent > 0.0))
                                || self.tree[parent].certificates.first()
                                .is_some_and(|certificate| certificate.first_bag_symmetry_assumed),
                            quality: state.quality,
                            quality_escape: job.variant % QUALITY_RESEARCH_INTERVAL == 0,
                            future: state.future,
                            debt: state.debt,
                        };
                        if self.quality_prune_plan(parent, &plan) {
                            self.quality_pruned += 1;
                            self.selective_pruned += 1;
                            continue;
                        }
                        if self.remember_plan(parent, &plan) {
                            self.candidates += 1;
                            self.waiting_plans.push((parent, plan));
                        } else {
                            self.duplicates += 1;
                        }
                    }
                    if self.waiting_plans.is_empty() {
                        self.backpropagate(parent, 0.0);
                    }
                } else {
                    self.generation = Some(job);
                }
                continue;
            }
            if self.explore_once() { continue; }
            // A lease survives only until a better verified benchmark or a
            // bounded dispatch quantum. Descendants cannot refill ancestors.
            if let Some(parent) = self.focus {
                let lease_valid = self.tree[parent].lease_epoch == self.benchmark_epoch
                    && self.dispatches.saturating_sub(self.focus_since_dispatch) < EXPLORATION_INTERVAL
                    && self.node_selective_reason(parent).is_none();
                if let Some(index) = self.best_waiting_plan(Some(parent)).filter(|_| lease_valid) {
                    let (_, plan) = self.waiting_plans.remove(index);
                    self.begin_verification(parent, plan);
                    continue;
                }
                if self.can_expand(parent)
                    && lease_valid
                    && self.node_selective_reason(parent).is_none()
                    && self.tree[parent].attempts < self.tree[parent].allowance
                {
                    self.begin_generation(parent);
                    continue;
                }
                let stale = self.tree[parent]
                    .attempts
                    .saturating_sub(self.tree[parent].last_improved_attempt);
                self.tree[parent].revisit_after = self
                    .schedule_epoch
                    .saturating_add(1_u64 << (stale / 4).min(8));
                self.focus = None;
            }
            if let Some(index) = self.best_waiting_plan(None) {
                let (parent, plan) = self.waiting_plans.remove(index);
                self.begin_verification(parent, plan);
                continue;
            }
            let parent = loop {
                if let Some(parent) = self.select_node() {
                    break Some(parent);
                }
                if self.active_budget >= self.config.max_pieces {
                    break None;
                }
                // Reconsider retained boundaries under the relaxed horizon.
                self.widen_budget();
            };
            let Some(parent) = parent else {
                self.finished = true;
                break;
            };
            self.focus = Some(parent);
            self.focus_since_dispatch = self.dispatches;
            let stale = self.tree[parent]
                .attempts
                .saturating_sub(self.tree[parent].last_improved_attempt);
            let extra = self.node_rounds(parent, stale);
            self.tree[parent].allowance = self.tree[parent].attempts + extra;
            self.tree[parent].lease_epoch = self.benchmark_epoch;
            self.begin_generation(parent);
        }
    }

    fn begin_verification(&mut self, parent: usize, plan: Plan) {
        let pieces = plan.start_pieces + plan.targets.len() as u16;
        if self.quality_prune_plan(parent, &plan) {
            self.quality_pruned += 1;
            self.selective_pruned += 1;
            self.forget_plan(parent, &plan);
            return;
        }
        if !plan.terminal
            && selective_reason(
                plan.expected.cell_count(),
                boundary_obstruction(&plan.expected, &plan.future),
                pieces,
                plan.expected_spins,
                self.best_spins_by_bag[pieces as usize / 7],
                self.goal,
                self.active_budget,
            ).is_some()
        {
            self.selective_pruned += 1;
            self.forget_plan(parent, &plan);
            return;
        }
        self.verification = Some(VerificationJob::new(parent, plan));
        self.dispatches += 1;
        self.last_checked = 0;
        self.last_success = 0;
    }

    fn cost_cap(&self) -> u16 {
        self.best
            .as_ref()
            .and_then(|b| b.summary.budget_pieces)
            .unwrap_or(self.config.max_pieces)
            .min(self.config.max_pieces)
    }

    fn can_expand(&self, index: usize) -> bool {
        let node = &self.tree[index];
        node.spins < self.goal
            && node.pieces < self.config.max_pieces
            && node.pieces + cost_lower_bound(&node.board, node.spins, self.goal) <= self.cost_cap()
    }

    /// This is a selective futility comparison, not alpha-beta dominance:
    /// different boards can still have very different longer continuations.
    fn quality_dominator(&self, pieces: u16, spins: u8, quality: f64) -> Option<u8> {
        if pieces <= 7 || pieces % 7 != 0 || spins >= self.goal || !quality.is_finite() {
            return None;
        }
        let references = self.best_quality_by_bag.get(pieces as usize / 7)?;
        (spins + 1..=self.goal.min(TARGET_SPINS)).rev()
            .find(|&better_spins| references[better_spins as usize] > quality)
    }

    fn node_quality_dominated(&self, index: usize) -> bool {
        let node = &self.tree[index];
        self.quality_dominator(node.pieces, node.spins, node.quality).is_some()
    }

    fn quality_prune_plan(&self, parent: usize, plan: &Plan) -> bool {
        if plan.terminal || plan.quality_escape { return false; }
        let Some(better_spins) = self.quality_dominator(
            plan.start_pieces + plan.targets.len() as u16, plan.expected_spins, plan.quality,
        ) else { return false; };
        let stale = self.tree[parent].attempts
            .saturating_sub(self.tree[parent].last_improved_attempt);
        // Give a coherent, promising preparation its first reduced probes.
        // Repeated unsupported optimism cannot buy unlimited verification.
        !(stale <= 2 && plan.expected_spins.saturating_add(plan.future.spins) > better_spins)
    }

    fn widen_budget(&mut self) {
        let widened = self.active_budget.saturating_add(7).min(self.config.max_pieces);
        if widened != self.active_budget {
            self.active_budget = widened;
            // Lower-quality preparations can become useful with more room.
            // This resets only selective references, never verified records.
            self.best_quality_by_bag.fill([f64::NEG_INFINITY; TARGET_SPINS as usize + 1]);
            self.benchmark_epoch += 1;
            self.focus = None;
        }
        self.last_budget_progress_round = self.generation_rounds;
    }

    fn node_selective_reason(&self, index: usize) -> Option<SelectiveReason> {
        let node = &self.tree[index];
        selective_reason(
            node.board.cell_count(),
            node.obstructed_mass,
            node.pieces,
            node.spins,
            self.best_spins_by_bag[node.pieces as usize / 7],
            self.goal,
            self.active_budget,
        )
    }

    fn last_bag_spins(&self, index: usize) -> u8 {
        let node = &self.tree[index];
        node.parent.map_or(0, |parent| {
            node.spins.saturating_sub(self.tree[parent].spins)
        })
    }

    fn counted_counts<'a>(&self, counts: &'a [u16]) -> &'a [u16] {
        if self.config.ignore_first_bag_probability && !counts.is_empty() {
            &counts[1..]
        } else { counts }
    }

    fn path_probability(&self, counts: &[u16]) -> f64 {
        if counts.first() == Some(&0) { return 0.0; }
        probability(self.counted_counts(counts))
    }

    fn first_bag_minimum(&self) -> u16 {
        // The integer threshold also makes 0 mean any nonzero coverage.
        let basis_points = (self.config.first_bag_min_success_percent * 100.0).round() as u32;
        (basis_points * P7_COUNT as u32).div_ceil(10_000).max(1) as u16
    }

    fn start_qualifies(&self, counts: &[u16]) -> bool {
        counts.first().is_none_or(|&count| count >= self.first_bag_minimum())
    }

    fn path_qualifies(&self, counts: &[u16]) -> bool {
        self.start_qualifies(counts)
            && threshold_met(self.counted_counts(counts), self.config.min_success_percent)
    }

    fn node_rounds(&self, index: usize, stale: u32) -> u32 {
        let node = &self.tree[index];
        if self.node_quality_dominated(index) { return 1; }
        productive_rounds(node.spins, node.pieces, self.last_bag_spins(index),
            node.future.spins, stale, self.goal, self.active_budget)
    }

    /// Peers share only verified performance benchmarks. These selective hints
    /// never manufacture local progress, a board, or a supply certificate.
    pub fn observe_benchmarks(&mut self, values: &[u8]) {
        let mut changed = false;
        for (depth, &spins) in values.iter().take(self.best_spins_by_bag.len()).enumerate().skip(1) {
            if spins > self.goal || usize::from(spins) * 10 > depth * 28 { continue; }
            if spins > self.best_spins_by_bag[depth] {
                self.best_spins_by_bag[depth] = spins;
                changed = true;
            }
        }
        if changed {
            self.benchmark_epoch += 1;
            self.focus = None;
        }
    }

    fn best_waiting_plan(&self, parent: Option<usize>) -> Option<usize> {
        self.waiting_plans.iter().enumerate()
            .filter(|(_, (p, _))| parent.is_none_or(|parent| parent == *p))
            .max_by(|(_, (_, a)), (_, (_, b))| {
                let priority = |plan: &Plan| {
                    // Reuse the bounded probe result; do not flood every
                    // pending board again whenever the scheduler compares it.
                    let quality = plan.future.score
                        - f64::from(mask_count(&plan.debt).saturating_sub(plan.future.paid)) * 12.0;
                    efficiency(plan.expected_spins, plan.start_pieces + plan.targets.len() as u16)
                        + 0.2 * quality / (120.0 + quality.abs())
                };
                priority(a).total_cmp(&priority(b)).then_with(|| b.id.cmp(&a.id))
            }).map(|(index, _)| index)
    }

    fn explore_once(&mut self) -> bool {
        if self.dispatches.saturating_sub(self.last_exploration_dispatch) < EXPLORATION_INTERVAL {
            return false;
        }
        self.last_exploration_dispatch = self.dispatches;
        let phase = self.exploration_phase;
        self.exploration_phase = (phase + 1) % 3;
        if phase == 0 {
            if let Some(index) = self.best_waiting_plan(Some(0)) {
                let (parent, plan) = self.waiting_plans.remove(index);
                self.focus = None;
                self.begin_verification(parent, plan);
                return true;
            }
        }
        let eligible = |i: usize| self.can_expand(i) && self.node_selective_reason(i).is_none();
        let parent = if phase == 0 && eligible(0) { Some(0) }
        else if phase == 1 {
            (1..self.tree.len()).filter(|&i| eligible(i) && self.tree[i].pieces == 7
                && self.tree[i].attempts == 0)
                .max_by(|&a, &b| self.tree[a].quality.total_cmp(&self.tree[b].quality))
        } else { None };
        // Oldest eligible deeper nodes also get service; a short early burst
        // must not make all lower-efficiency continuations wait forever.
        let parent = parent.or_else(|| (0..self.tree.len()).filter(|&i| eligible(i))
            .min_by_key(|&i| (self.tree[i].last_selected, i)));
        let Some(parent) = parent else { return false; };
        self.focus = Some(parent);
        self.focus_since_dispatch = self.dispatches;
        self.tree[parent].allowance = self.tree[parent].attempts + 1;
        self.tree[parent].lease_epoch = self.benchmark_epoch;
        self.begin_generation(parent);
        true
    }

    fn begin_generation(&mut self, parent: usize) {
        let attempt = self.tree[parent].attempts;
        let minimum = if parent == 0 {
            self.first_bag_minimum().max(if self.config.ignore_first_bag_probability { 1 }
                else { required_success(&[], self.config.min_success_percent) })
        } else { required_success(self.counted_counts(&self.tree[parent].counts), self.config.min_success_percent) };
        // Re-search only selected boundaries more broadly; siblings start cheap.
        let under_pace = under_pace(self.tree[parent].spins, self.tree[parent].pieces,
            self.goal, self.active_budget) || self.node_quality_dominated(parent);
        let stage = if under_pace { 0 }
            else { (attempt / 4 + u32::from(self.last_bag_spins(parent) >= 2)).min(2) };
        let width = (self.config.beam_width >> (2 - stage)).max(4);
        let setup = parent == 0 && attempt % 4 != 3;
        let mut generation = GenerationJob::new(
            parent,
            &self.tree[parent],
            width,
            setup,
            attempt as u64,
            self.goal,
            self.config.search_profile,
            minimum,
            self.config.search_lane,
            self.config.search_lanes,
        );
        generation.best_bag_spins =
            self.best_spins_by_bag[(self.tree[parent].pieces as usize / 7) + 1];
        generation.prune_budget = self.active_budget;
        generation.first_bag_symmetry = parent == 0
            && (!self.config.ignore_first_bag_probability || self.config.first_bag_min_success_percent > 0.0);
        generation.relaxed_setup = parent == 0;
        if parent == 0 {
            generation.continuation_minimum_success = required_success(&[], self.config.min_success_percent);
        }
        self.generation = Some(generation);
        self.tree[parent].attempts += 1;
        self.tree[parent].last_selected = self.dispatches;
        self.dispatches += 1;
        self.generation_rounds += 1;
        self.schedule_epoch += 1;
        if self
            .generation_rounds
            .saturating_sub(self.last_budget_progress_round)
            >= BUDGET_STALE_ROUNDS
        {
            self.widen_budget();
        }
    }

    fn plan_key(&self, parent: usize, plan: &Plan) -> Vec<u8> {
        let mut key = Vec::with_capacity(128);
        for row in plan.start.rows {
            key.extend(row.to_le_bytes());
        }
        key.push(plan.start_spins);
        key.extend(plan.start_pieces.to_le_bytes());
        for row in plan.expected.rows {
            key.extend(row.to_le_bytes());
        }
        key.extend([plan.expected_spins, u8::from(plan.terminal), u8::from(plan.assume_reflection)]);
        // The same bag plan can qualify after a stronger probability prefix
        // even when it was rejected after a weaker route to the same board.
        key.extend((self.tree[parent].counts.len() as u16).to_le_bytes());
        for count in &self.tree[parent].counts {
            key.extend(count.to_le_bytes());
        }
        let mut targets = plan.targets.clone();
        targets.sort_by_key(|t| t.piece as u8);
        for target in targets {
            key.push(target.piece as u8);
            for (x, y) in target.cells {
                key.extend([x, y]);
            }
        }
        key
    }

    fn forget_plan(&mut self, parent: usize, plan: &Plan) {
        self.seen_plans.remove(&self.plan_key(parent, plan));
    }

    fn remember_plan(&mut self, parent: usize, plan: &Plan) -> bool {
        let key = self.plan_key(parent, plan);
        // Bounded duplicate cache; clearing affects cost, never certificates.
        if self.seen_plans.len() >= 8192 {
            self.seen_plans.clear();
        }
        self.seen_plans.insert(key)
    }

    fn select_node(&mut self) -> Option<usize> {
        // Bounded best-first resumption between focused depth-first stretches.
        // The exact mass/cost bound prunes; this priority only schedules effort.
        let earliest = (0..self.tree.len())
            .filter(|&i| self.can_expand(i) && self.node_selective_reason(i).is_none())
            .map(|i| self.tree[i].revisit_after)
            .min()?;
        self.schedule_epoch = self.schedule_epoch.max(earliest);
        (0..self.tree.len())
            .filter(|&i| {
                self.can_expand(i)
                    && self.node_selective_reason(i).is_none()
                    && self.tree[i].revisit_after <= self.schedule_epoch
            })
            .max_by(|&a, &b| {
                let priority = |i: usize| {
                    let node = &self.tree[i];
                    let bound = node.pieces + cost_lower_bound(&node.board, node.spins, self.goal);
                    let age = self
                        .dispatches
                        .saturating_sub(node.last_selected)
                        .min(128);
                    let stale = node.attempts.saturating_sub(node.last_improved_attempt) as f64;
                    // A geometric continuation is not a P7 extension. Its
                    // speculative credit fades when repeated rounds cannot
                    // improve a certified child. Real progress resets the age.
                    let speculative = node.future.score.max(0.0)
                        + f64::from(node.future.paid.min(mask_count(&node.debt))) * 12.0;
                    let quality = node.quality - speculative * stale / (stale + 4.0);
                    // Stored cells can pay for later singles. Charging every
                    // prefix lock again here permanently starves deeper setup
                    // boundaries even when their optimistic total cost is the
                    // same. Compare that total bound below, not raw depth.
                    let quality = quality + age as f64 * 1.2
                            - stale * 6.0
                            - ((node.attempts + 1) as f64).log2() * 32.0
                            - bound.saturating_sub(self.active_budget) as f64 * 2.0
                            - bound.saturating_sub(self.config.piece_goal) as f64 * 0.8;
                    efficiency(node.spins, node.pieces)
                        + 0.2 * quality / (120.0 + quality.abs())
                };
                priority(a).total_cmp(&priority(b))
            })
    }

    fn backpropagate(&mut self, mut index: usize, reward: f64) {
        loop {
            self.tree[index].visits += 1;
            self.tree[index].reward = self.tree[index].reward.max(reward);
            // A descendant's progress is a statistic, not a renewable lease
            // for every ancestor. Only the scheduler allocates generation work.
            let Some(parent) = self.tree[index].parent else {
                break;
            };
            index = parent;
        }
    }

    fn can_still_qualify(&self, job: &VerificationJob) -> bool {
        let mut counts = self.tree[job.parent].counts.clone();
        counts.push(job.coverage_upper());
        self.path_qualifies(&counts)
    }

    fn update_generation_preview(&mut self, job: &GenerationJob) {
        let Some(seed) = job.display_seed() else {
            return;
        };
        // A candidate is a whole fixed bag (or an actual terminal prefix),
        // never one of the internal per-inventory partial states.
        debug_assert!(seed.used == ALL_PIECES || seed.spins >= self.goal);
        if self.preview_generation == self.generation_rounds {
            if let Some(current) = &mut self.current {
                if current.pending_plan.as_ref().is_some_and(|plan| {
                    plan.id == 0 && plan.targets == seed.targets && plan.seed_frames == seed.frames
                }) {
                    current.summary.ranking_score = seed.score;
                    current.summary.field_quality = seed.quality;
                    return;
                }
            }
        }
        self.preview_generation = self.generation_rounds;
        self.deepest_spins = self.deepest_spins.max(seed.spins);
        let parent = &self.tree[job.parent];
        let placed = job.start_pieces + seed.targets.len() as u16;
        let mut frames = parent.frames.clone();
        frames.extend(seed.frames.clone());
        let upper = self.path_probability(&parent.counts);
        self.current = Some(Archived {
            summary: CandidateSummary {
                id: format!("building-{}-{}", self.generation_rounds, self.preview_steps),
                rows: seed.board.rows,
                spins: seed.spins,
                placed,
                probability: 0.0,
                lower_probability: 0.0,
                upper_probability: upper,
                first_bag_probability: parent.counts.first().map(|count| f64::from(*count) / P7_COUNT as f64),
                first_bag_symmetry: parent.certificates.first().is_some_and(|c| c.first_bag_symmetry_assumed),
                first_bag_symmetry_assumed: parent.certificates.first().is_some_and(|c| c.first_bag_symmetry_assumed),
                complete: seed.spins >= self.goal,
                qualified: false,
                verified: false,
                checked: 0,
                success: 0,
                failed: 0,
                bag_count: parent.counts.len() as u16 + 1,
                budget_pieces: None,
                cdf: vec![],
                ranking_score: seed.score,
                field_quality: seed.quality,
                frame_count: frames.len() as u16,
                frames,
            },
            certificates: parent.certificates.clone(),
            pending_plan: Some(Plan {
                id: 0,
                start: job.start,
                start_spins: job.start_spins,
                start_pieces: job.start_pieces,
                expected: seed.board,
                expected_spins: seed.spins,
                targets: seed.targets.clone(),
                seed_frames: seed.frames.clone(),
                terminal: seed.spins >= self.goal,
                assume_reflection: (job.start_pieces == 0 && (!self.config.ignore_first_bag_probability || self.config.first_bag_min_success_percent > 0.0))
                    || parent.certificates.first().is_some_and(|c| c.first_bag_symmetry_assumed),
                quality: seed.quality,
                quality_escape: job.variant % QUALITY_RESEARCH_INTERVAL == 0,
                future: seed.future.clone(),
                debt: seed.debt,
            }),
        });
    }

    fn update_verifying_preview(&mut self, job: &VerificationJob) {
        self.deepest_spins = self.deepest_spins.max(job.plan.expected_spins);
        let parent = &self.tree[job.parent];
        let id = format!("plan-{}", job.plan.id);
        let mut lower_counts = parent.counts.clone();
        lower_counts.push(job.coverage_success());
        let mut upper_counts = parent.counts.clone();
        upper_counts.push(job.coverage_upper());
        let symmetry_assumed = if job.plan.start_pieces == 0 {
            job.symmetry.is_some() && usize::from(job.coverage_success()) > job.success
        } else { job.plan.assume_reflection };
        let lower = self.path_probability(&lower_counts);
        let upper = self.path_probability(&upper_counts);
        if let Some(current) = &mut self.current {
            if current.summary.id == id && !current.summary.verified {
                current.summary.first_bag_symmetry = symmetry_assumed;
                current.summary.first_bag_symmetry_assumed = symmetry_assumed;
                current.summary.probability = lower;
                current.summary.lower_probability = lower;
                current.summary.upper_probability = upper;
                current.summary.first_bag_probability = lower_counts.first()
                    .map(|count| f64::from(*count) / P7_COUNT as f64);
                current.summary.checked = job.checked as u16;
                current.summary.success = job.success as u16;
                current.summary.failed = (job.checked - job.success) as u16;
                return;
            }
        }
        let mut frames = parent.frames.clone();
        frames.extend(job.plan.seed_frames.clone());
        let quality = job.plan.quality;
        self.current = Some(Archived {
            summary: CandidateSummary {
                id,
                rows: job.plan.expected.rows,
                spins: job.plan.expected_spins,
                placed: job.plan.start_pieces + job.plan.targets.len() as u16,
                probability: lower,
                lower_probability: lower,
                upper_probability: upper,
                first_bag_probability: lower_counts.first().map(|count| f64::from(*count) / P7_COUNT as f64),
                first_bag_symmetry: symmetry_assumed,
                first_bag_symmetry_assumed: symmetry_assumed,
                complete: job.plan.terminal,
                qualified: false,
                verified: false,
                checked: job.checked as u16,
                success: job.success as u16,
                failed: (job.checked - job.success) as u16,
                bag_count: parent.counts.len() as u16 + 1,
                budget_pieces: None,
                cdf: vec![],
                ranking_score: spin_rank(job.plan.expected_spins, quality),
                field_quality: quality,
                frame_count: frames.len() as u16,
                frames,
            },
            certificates: parent.certificates.clone(),
            pending_plan: Some(job.plan.clone()),
        });
    }

    fn finish_verification(&mut self, job: VerificationJob) {
        self.verified_candidates += 1;
        let parent = &self.tree[job.parent];
        let mut counts = parent.counts.clone();
        counts.push(job.coverage_success());
        let total_probability = self.path_probability(&counts);
        let symmetry_costs = job.symmetry.as_ref().map(|_| symmetry_union_costs(&job.costs));
        let symmetry_assumed = job.plan.start_pieces > 0 && job.plan.assume_reflection
            || symmetry_costs.as_ref().is_some_and(|costs| *costs != job.costs);
        let effective_costs = symmetry_costs.as_deref().unwrap_or(&job.costs);
        let mut cdf = vec![];
        let mut budget = None;
        if job.plan.terminal {
            for m in 1..=7 {
                let count = effective_costs.iter().filter(|c| **c > 0 && **c <= m).count() as u16;
                let mut at_budget = parent.counts.clone();
                at_budget.push(count);
                let conditional_start = self.config.ignore_first_bag_probability && parent.pieces == 0;
                let cdf_probability = if conditional_start {
                    f64::from(count) / f64::from(job.coverage_success().max(1))
                } else { self.path_probability(&at_budget) };
                let meets = if conditional_start {
                    job.coverage_success() > 0 && u32::from(count) * 10_000
                        >= u32::from(job.coverage_success())
                            * (self.config.min_success_percent * 100.0).round() as u32
                } else { self.path_qualifies(&at_budget) };
                cdf.push(CdfPoint {
                    pieces: parent.pieces + m as u16,
                    probability: cdf_probability,
                });
                if budget.is_none() && meets {
                    budget = Some(parent.pieces + m as u16);
                }
            }
        }
        let qualified = if job.plan.terminal {
            self.start_qualifies(&counts) && budget.is_some_and(|n| n <= self.config.max_pieces)
        } else {
            self.path_qualifies(&counts)
        };
        let desired = budget.map(|n| n - parent.pieces).unwrap_or(7) as u8;
        let canonical_choice = |rank: usize| job.symmetry.is_none()
            || job.costs[rank] <= job.costs[mirror_rank(rank)];
        let representative = job.costs.iter().enumerate()
            .position(|(rank, c)| *c == desired && canonical_choice(rank))
            .or_else(|| job.costs.iter().enumerate()
                .position(|(rank, c)| *c > 0 && *c <= desired && canonical_choice(rank)));
        let Some(rank) = representative else {
            self.backpropagate(job.parent, 0.0);
            return;
        };
        let (local_frames, inputs) =
            job.graph
                .witness(job.codes[rank], job.costs[rank], parent.pieces);
        let mut frames = parent.frames.clone();
        frames.extend(local_frames);
        let last = frames.last().unwrap();
        // Reuse this boundary's immutable geometry facts locally. No shared
        // cache or certificate policy is introduced by the reuse.
        let facts = BoardFacts::new(&job.plan.expected);
        let quality = boundary_quality_from_facts(&facts, &job.plan.debt, &job.plan.future);
        let obstruction = job.plan.future.obstructed_mass
            .map_or(facts.obstructed_mass, |after| facts.obstructed_mass.min(after));
        let certificate = Rc::new(BagCertificate {
            id: format!("plan-{}", job.plan.id),
            start_rows: job.plan.start.rows,
            end_rows: last.rows,
            start_spins: job.plan.start_spins,
            end_spins: last.spins,
            targets: job.plan.targets.clone(),
            terminal: job.plan.terminal,
            first_bag_symmetry_assumed: symmetry_assumed,
            mirrored_inputs_verified: false,
            symmetry_costs,
            success: job.success as u16,
            effective_success: *counts.last().unwrap(),
            checked: P7_COUNT as u16,
            success_bits: job.bits,
            costs: job.costs,
            order_codes: job.codes,
            representative_queue: permutation(rank),
            representative_inputs: inputs,
        });
        let mut certificates = parent.certificates.clone();
        certificates.push(certificate);
        let summary = CandidateSummary {
            id: format!("plan-{}", job.plan.id),
            rows: last.rows,
            spins: last.spins,
            placed: last.placed,
            probability: total_probability,
            lower_probability: total_probability,
            upper_probability: total_probability,
            first_bag_probability: counts.first().map(|count| f64::from(*count) / P7_COUNT as f64),
            first_bag_symmetry: symmetry_assumed,
            first_bag_symmetry_assumed: symmetry_assumed,
            complete: job.plan.terminal,
            qualified,
            verified: true,
            checked: P7_COUNT as u16,
            success: *counts.last().unwrap(),
            failed: P7_COUNT as u16 - *counts.last().unwrap(),
            bag_count: counts.len() as u16,
            budget_pieces: budget,
            cdf,
            ranking_score: spin_rank(last.spins, quality),
            field_quality: quality,
            frame_count: frames.len() as u16,
            frames: frames.clone(),
        };
        let archived = Archived {
            summary: summary.clone(),
            certificates: certificates.clone(),
            pending_plan: None,
        };
        self.current = Some(archived.clone());
        if qualified {
            self.qualified_bags = self.qualified_bags.max(counts.len() as u16);
            if summary.spins > self.qualified_spins {
                self.last_budget_progress_round = self.generation_rounds;
            }
            self.qualified_spins = self.qualified_spins.max(summary.spins);
            if !job.plan.terminal {
                let best = &mut self.best_spins_by_bag[counts.len()];
                let quality_reference = &mut self.best_quality_by_bag[counts.len()][summary.spins as usize];
                let improved = summary.spins > *best || quality > *quality_reference;
                *best = (*best).max(summary.spins);
                *quality_reference = quality_reference.max(quality);
                if improved {
                    self.benchmark_epoch += 1;
                    self.focus = None;
                }
            }
            if !job.plan.terminal
                && self
                    .best_partial
                    .as_ref()
                    .is_none_or(|prior| better_partial(&summary, &prior.summary))
            {
                self.best_partial = Some(archived.clone());
            }
        }
        if job.plan.terminal {
            if qualified
                && self
                    .best
                    .as_ref()
                    .is_none_or(|b| better_complete(&summary, &b.summary))
            {
                self.best = Some(archived.clone());
            }
            if self
                .other_complete
                .as_ref()
                .is_none_or(|b| summary.probability > b.summary.probability)
            {
                self.other_complete = Some(archived);
            }
        } else if qualified && parent.pieces + 7 < self.config.max_pieces {
            let duplicate = self.tree.iter().any(|n| {
                n.board == job.plan.expected
                    && n.spins == job.plan.expected_spins
                    && n.pieces == parent.pieces + 7
                    && n.certificates.first().is_some_and(|c| c.first_bag_symmetry_assumed) == symmetry_assumed
                    && compare_probability(self.counted_counts(&n.counts), self.counted_counts(&counts))
                        != std::cmp::Ordering::Less
            });
            if !duplicate {
                let pieces = parent.pieces + 7;
                let new_node = TreeNode {
                    parent: Some(job.parent),
                    children: vec![],
                    board: job.plan.expected,
                    spins: job.plan.expected_spins,
                    pieces,
                    counts,
                    certificates,
                    frames,
                    visits: 0,
                    reward: 0.0,
                    future: job.plan.future.clone(),
                    debt: job.plan.debt,
                    attempts: 0,
                    allowance: if self.quality_dominator(pieces, job.plan.expected_spins, quality).is_some() { 1 } else { productive_rounds(
                        job.plan.expected_spins,
                        pieces,
                        job.plan.expected_spins.saturating_sub(parent.spins),
                        job.plan.future.spins,
                        0,
                        self.goal,
                        self.active_budget,
                    ) },
                    lease_epoch: self.benchmark_epoch,
                    last_selected: self.dispatches,
                    quality,
                    last_improved_attempt: 0,
                    best_extension_spins: 0,
                    best_extension_quality: f64::NEG_INFINITY,
                    revisit_after: 0,
                    obstructed_mass: obstruction,
                };
                let gain = new_node.spins - parent.spins;
                let quality = new_node.quality;
                if let Some(child) = self.insert_boundary(job.parent, new_node) {
                    let source = &mut self.tree[job.parent];
                    if gain > source.best_extension_spins
                        || (gain == source.best_extension_spins
                            && quality > source.best_extension_quality)
                    {
                        source.best_extension_spins = gain;
                        source.best_extension_quality = quality;
                        source.last_improved_attempt = source.attempts;
                        source.revisit_after = self.schedule_epoch;
                    }
                    self.focus = Some(child);
                    self.focus_since_dispatch = self.dispatches;
                }
            } else {
                self.duplicates += 1;
            }
        }
        let reward = if job.plan.terminal && qualified {
            2.0
        } else if qualified {
            job.plan.expected_spins as f64 / self.goal as f64
                + (quality / 200.0)
                    .clamp(-0.2, 0.2)
                + 0.2
        } else {
            0.0
        };
        self.backpropagate(job.parent, reward);
    }

    fn insert_boundary(&mut self, parent: usize, node: TreeNode) -> Option<usize> {
        let limit = self.config.beam_width * 8;
        if self.tree.len() < limit {
            let index = self.tree.len();
            self.tree.push(node);
            self.tree[parent].children.push(index);
            return Some(index);
        }
        // Stable slots keep pending jobs and ancestral certificates valid. Only
        // unreferenced leaves are replaced; completed incumbents live separately.
        let mut foundations: Vec<_> = (1..self.tree.len()).filter(|&i| {
            self.tree[i].pieces == 7 && self.tree[i].attempts == 0
                && self.node_selective_reason(i).is_none()
        }).collect();
        foundations.sort_by(|&a, &b| self.tree[b].quality.total_cmp(&self.tree[a].quality));
        foundations.truncate((self.config.beam_width / 4).clamp(2, 8));
        let replaceable = |i: usize| {
            i != parent && self.focus != Some(i) && self.tree[i].children.is_empty()
                && !self.waiting_plans.iter().any(|(pending, _)| *pending == i)
        };
        let victim = (1..self.tree.len())
            .filter(|i| {
                replaceable(*i) && !foundations.contains(i)
            })
            .min_by(|a, b| {
                let score = |i: usize| {
                    let n = &self.tree[i];
                    efficiency(n.spins, n.pieces) + 0.2 * n.quality / (120.0 + n.quality.abs())
                };
                let eligible = |i| self.can_expand(i) && self.node_selective_reason(i).is_none();
                eligible(*a).cmp(&eligible(*b)).then_with(|| score(*a).total_cmp(&score(*b)))
            }).or_else(|| foundations.iter().rev().copied().find(|&i| replaceable(i)));
        let Some(index) = victim else { return None };
        if let Some(old_parent) = self.tree[index].parent {
            self.tree[old_parent].children.retain(|i| *i != index);
        }
        self.tree[index] = node;
        self.tree[parent].children.push(index);
        Some(index)
    }

    pub fn snapshot(&self) -> ProgressSnapshot {
        self.snapshot_with_frames(true)
    }

    pub fn snapshot_compact(&self) -> ProgressSnapshot {
        self.snapshot_with_frames(false)
    }

    fn snapshot_with_frames(&self, include_frames: bool) -> ProgressSnapshot {
        ProgressSnapshot {
            phase: if self.finished {
                "stopped"
            } else if self.verification.is_some() {
                "verifying"
            } else {
                "searching"
            }
            .into(),
            expanded: self.expanded,
            candidates: self.candidates,
            verified_candidates: self.verified_candidates,
            qualified_bags: self.qualified_bags,
            qualified_spins: self.qualified_spins,
            pruned: self.pruned,
            deepest_spins: self.deepest_spins,
            bags: self
                .current
                .as_ref()
                .map(|c| c.summary.bag_count)
                .unwrap_or(0),
            checked: self.last_checked,
            success: self.last_success,
            failed: self.last_checked.saturating_sub(self.last_success),
            frontier: self.tree.len(),
            finished: self.finished,
            current: self
                .current
                .as_ref()
                .map(|a| a.summary.for_snapshot(include_frames)),
            best: self
                .best
                .as_ref()
                .map(|a| a.summary.for_snapshot(include_frames)),
            best_partial: self
                .best_partial
                .as_ref()
                .map(|a| a.summary.for_snapshot(include_frames)),
            duplicates: self.duplicates,
            selective_pruned: self.selective_pruned,
            best_spins_by_bag: self.best_spins_by_bag.clone(),
        }
    }

    pub fn export_json(&self) -> String {
        let archive = self
            .best
            .as_ref()
            .or(self.other_complete.as_ref())
            .or(self.best_partial.as_ref())
            .or(self.current.as_ref());
        self.export_archive(archive)
    }

    pub fn export_selected_json(&self, selection: &str) -> String {
        self.export_archive(match selection {
            "best" => self.best.as_ref(),
            "current" => self.current.as_ref(),
            "partial" => self.best_partial.as_ref(),
            _ => None,
        })
    }

    fn export_archive(&self, archive: Option<&Archived>) -> String {
        let certificates: Vec<_> = archive
            .map(|a| a.certificates.iter().map(|c| c.as_ref()).collect())
            .unwrap_or_default();
        serde_json::json!({
            "format": "neyra", "schemaVersion": 3, "engineVersion": env!("CARGO_PKG_VERSION"),
            "rules": { "width": 10, "height": HEIGHT, "rotation": "SRS+", "spin": "All-mini+", "oSpin": false, "ceiling": "ActualCellDeath24", "movement": "SonicDropOnly", "supply": "UniformP7NormalizedHold", "runtimeReplayVerified": false },
            "config": self.config, "snapshot": self.snapshot(),
            "candidate": archive.map(|a| &a.summary), "bags": certificates,
            "pendingPlan": archive.and_then(|a| a.pending_plan.as_ref()),
            "successBitsEncoding": "hex-u64-lsb0; word=floor(P7rank/64), bit=P7rank%64",
            "interchange": false, "globalOptimalityProven": false,
            "probabilityScope": if self.config.ignore_first_bag_probability { "AfterFirstBag" } else { "AllBags" },
            "firstBagSymmetryPolicy": "AssumeJLAndSZSupplyReflection;chooseLowerCostOnFirstQueue;canonicalOnTie;fixedOrientationAfterStart",
            "firstBagSymmetryAssumed": archive.is_some_and(|a| a.summary.first_bag_symmetry_assumed),
            "mirroredInputsVerified": false,
            "costsMeaning": "0=unknown;1..7=shortestInCanonicalLanguage;255=noPathInCanonicalLanguage;symmetryCosts=assumedReflectionUnion",
            "displayPartialRanking": "spins/max(7,placed);thenSpins;thenFieldQuality;thenFewerLocks;thenProbability",
            "selectiveSearch": {
                "spinDelta": SPIN_DELTA, "residueMarginCells": RESIDUE_MARGIN,
                "paceNearSlack": PACE_NEAR_SLACK, "paceDistanceSlack": PACE_DISTANCE_SLACK,
                "paceIncludesSetup": true,
                "qualityResearchInterval": QUALITY_RESEARCH_INTERVAL, "qualityPruned": self.quality_pruned,
                "activeBudget": self.active_budget,
                "budgetStaleRounds": BUDGET_STALE_ROUNDS,
                "explorationInterval": EXPLORATION_INTERVAL,
                "cutsAreImpossibilityProofs": false
            }
        }).to_string()
    }
}

fn better_partial(a: &CandidateSummary, b: &CandidateSummary) -> bool {
    // The public record includes the setup bag. Only scheduling uses
    // productive_locks; a quality tie must be able to replace an older field.
    let lhs = u32::from(a.spins) * u32::from(b.placed.max(7));
    let rhs = u32::from(b.spins) * u32::from(a.placed.max(7));
    lhs.cmp(&rhs)
        .then_with(|| a.spins.cmp(&b.spins))
        .then_with(|| a.field_quality.partial_cmp(&b.field_quality).unwrap_or(std::cmp::Ordering::Equal))
        .then_with(|| b.placed.cmp(&a.placed))
        .then_with(|| a.probability.partial_cmp(&b.probability).unwrap_or(std::cmp::Ordering::Equal))
        .is_gt()
}

fn better_complete(a: &CandidateSummary, b: &CandidateSummary) -> bool {
    let a_cost = a.budget_pieces.unwrap_or(u16::MAX);
    let b_cost = b.budget_pieces.unwrap_or(u16::MAX);
    let at_budget = |candidate: &CandidateSummary, budget| {
        candidate
            .cdf
            .iter()
            .find(|point| point.pieces == budget)
            .map(|point| point.probability)
            .unwrap_or(candidate.probability)
    };
    a_cost < b_cost
        || (a_cost == b_cost
            && (at_budget(a, a_cost) > at_budget(b, b_cost)
                || (at_budget(a, a_cost) == at_budget(b, b_cost) && a.probability > b.probability)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probability_threshold_is_exact_at_95_percent() {
        assert!(threshold_met(&[4788], 95.0));
        assert!(!threshold_met(&[4787], 95.0));
        assert!(threshold_met(&[5040; 30], 100.0));
        assert!(!threshold_met(&[5039; 30], 100.0));
    }

    #[test]
    fn invalid_config_and_zero_probability_are_rejected() {
        let config = SearchConfig {
            min_success_percent: 0.0,
            ..SearchConfig::default()
        };
        assert!(SearchSession::new(config).is_err());
        let config = SearchConfig {
            min_success_percent: f64::NAN,
            ..SearchConfig::default()
        };
        assert!(SearchSession::new(config).is_err());
    }

    #[test]
    fn logical_rows_preserve_nonuniform_deletions() {
        let deleted = (1 << 1) | (1 << 4);
        assert_eq!(logical_row(0, deleted), Some(0));
        assert_eq!(logical_row(1, deleted), Some(2));
        assert_eq!(logical_row(3, deleted), Some(5));
        assert_eq!(deleted_after(deleted, 1 << 3), Some(deleted | (1 << 5)));
    }

    #[test]
    fn goal_is_not_an_acceptance_ceiling() {
        let config = SearchConfig {
            piece_goal: 112,
            max_pieces: 140,
            ..SearchConfig::default()
        };
        assert!(SearchSession::new(config).is_ok());
        let mut a = CandidateSummary {
            id: "119".into(),
            ranking_score: 40.0,
            field_quality: 0.0,
            frame_count: 0,
            rows: [0; HEIGHT],
            spins: 40,
            placed: 119,
            probability: 1.0,
            lower_probability: 1.0,
            upper_probability: 1.0,
            first_bag_probability: Some(1.0),
            first_bag_symmetry: false,
            first_bag_symmetry_assumed: false,
            complete: true,
            qualified: true,
            verified: true,
            checked: 5040,
            success: 5040,
            failed: 0,
            bag_count: 17,
            budget_pieces: Some(119),
            cdf: vec![],
            frames: vec![],
        };
        let mut b = a.clone();
        b.budget_pieces = Some(126);
        assert!(better_complete(&a, &b));
        a.budget_pieces = Some(120);
        assert!(better_complete(&a, &b));
    }

    #[test]
    fn fresh_search_has_no_fabricated_solution_or_progress() {
        let session = SearchSession::new(SearchConfig::default()).unwrap();
        let snapshot = session.snapshot();
        assert_eq!(snapshot.expanded, 0);
        assert_eq!(snapshot.deepest_spins, 0);
        assert!(snapshot.best.is_none());
        assert!(snapshot.current.is_none());
    }

    #[test]
    fn partial_terminal_verification_preserves_unknown_and_minimum_cost() {
        let mut board = Board::default();
        board.rows[..5].copy_from_slice(&[1023 ^ (1 << 5), 0, (1 << 3) | (1 << 5), 0, 1 << 3]);
        let placement = board
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
        let result = board.apply(&placement).unwrap();
        assert_eq!(result.lines, 1);
        assert!(result.spin);
        let plan = Plan {
            id: 1,
            start: board,
            start_spins: 39,
            start_pieces: 112,
            expected: result.board,
            expected_spins: 40,
            targets: vec![normalize_target(placement.pose, 0).unwrap()],
            seed_frames: vec![],
            terminal: true,
            assume_reflection: false,
            quality: 0.0,
            quality_escape: false,
            future: Continuation::default(),
            debt: [0; HEIGHT],
        };
        let mut job = VerificationJob::new(0, plan);
        let mut cache = PhysicsCache::default();
        assert_eq!(job.costs.iter().filter(|cost| **cost == 0).count(), 5040);
        job.advance(40, &mut cache);
        assert_eq!(job.checked, 0);
        assert!(job.costs.iter().all(|cost| *cost == 0));
        for _ in 0..200 {
            if job.checked == 5040 {
                break;
            }
            job.advance(40, &mut cache);
        }
        assert_eq!(job.checked, 5040);
        // Only S is declared. It must be current or held immediately; the
        // verifier cannot place undeclared pieces merely to reach it.
        assert_eq!(job.success, 1440);
        assert_eq!(job.costs.iter().filter(|cost| **cost == 1).count(), 1440);
        assert_eq!(job.costs.iter().filter(|cost| **cost == 255).count(), 3600);
        let rank = job.costs.iter().position(|cost| *cost == 1).unwrap();
        let (frames, paths) = job.graph.witness(job.codes[rank], 1, 112);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].placed, 113);
        assert_eq!(frames[0].spins, 40);
        assert_eq!(
            board
                .apply(&board.replay(Piece::S, &paths[0]).unwrap())
                .unwrap()
                .board,
            result.board
        );
    }

    #[test]
    fn invalid_single_clear_is_not_an_execution_edge() {
        let mut board = Board::default();
        board.rows[0] = 1023 ^ 15;
        let placement = board
            .replay(
                Piece::I,
                &[Input::Left, Input::Left, Input::Left, Input::HardDrop],
            )
            .unwrap();
        let result = board.apply(&placement).unwrap();
        assert_eq!(result.lines, 1);
        assert!(!result.spin);
        let plan = Plan {
            id: 1,
            start: board,
            start_spins: 39,
            start_pieces: 112,
            expected: result.board,
            expected_spins: 40,
            targets: vec![normalize_target(placement.pose, 0).unwrap()],
            seed_frames: vec![],
            terminal: true,
            assume_reflection: false,
            quality: 0.0,
            quality_escape: false,
            future: Continuation::default(),
            debt: [0; HEIGHT],
        };
        let mut graph = ExecutionGraph::new(&plan);
        graph.expand_one(&plan, 40, &mut PhysicsCache::default());
        assert!(graph.complete());
        assert!(graph.nodes[0].edges.is_empty());
        assert!(!graph.nodes[0].accepting);
    }

    #[test]
    fn exact_probability_comparison_handles_many_bags() {
        assert_eq!(
            compare_probability(&[5040; 31], &[5040]),
            std::cmp::Ordering::Equal
        );
        assert_eq!(
            compare_probability(&[5040, 5039], &[5038, 5040]),
            std::cmp::Ordering::Greater
        );
    }

    #[test]
    fn bitset_export_survives_javascript_json_round_trips() {
        #[derive(Serialize)]
        struct Words(#[serde(serialize_with = "serialize_bit_words")] Vec<u64>);
        let json = serde_json::to_value(Words(vec![u64::MAX, 1_u64 << 63, 1])).unwrap();
        assert_eq!(
            json,
            serde_json::json!([
                "0xffffffffffffffff",
                "0x8000000000000000",
                "0x0000000000000001"
            ])
        );
    }

    #[test]
    fn de_novo_small_goal_search_returns_replayable_fixed_plan() {
        let config = SearchConfig {
            beam_width: 8,
            min_success_percent: 1.0,
            first_bag_min_success_percent: 0.0,
            ..SearchConfig::default()
        };
        let mut session = SearchSession::new(config).unwrap();
        session.goal = 1; // The production constructor always uses forty.
        for _ in 0..3000 {
            session.step(8);
            if session.best.is_some() {
                break;
            }
        }
        let best = session
            .best
            .as_ref()
            .expect("a small-goal plan must be discovered from the empty field");
        assert_eq!(best.summary.spins, 1);
        assert!(best.summary.complete && best.summary.qualified && best.summary.verified);
        let mut board = Board::default();
        let mut spins = 0;
        let mut placed = 0;
        for bag in &best.certificates {
            assert_eq!(board.rows, bag.start_rows);
            assert_eq!(bag.checked, 5040);
            let rank = (0..5040)
                .find(|rank| permutation(*rank) == bag.representative_queue)
                .unwrap();
            let order = decode_order(bag.order_codes[rank], bag.costs[rank]);
            let mut held = bag.representative_queue[0];
            for (i, piece) in order.iter().enumerate() {
                let current = bag.representative_queue.get(i + 1).copied().unwrap_or(7);
                if *piece != current {
                    assert_eq!(*piece, held, "representative may only use current or held");
                    held = current;
                }
                let placement = board
                    .replay(Piece::ALL[*piece as usize], &bag.representative_inputs[i])
                    .unwrap();
                let result = board.apply(&placement).unwrap();
                assert!(result.lines == 0 || (result.lines == 1 && result.spin));
                spins += result.lines;
                placed += 1;
                board = result.board;
                assert_eq!(board.cell_count(), 4 * placed - 10 * spins as u16);
            }
            assert_eq!(board.rows, bag.end_rows);
        }
        assert_eq!(spins, 1);
        assert_eq!(placed, best.summary.placed);
        assert_eq!(board.rows, best.summary.rows);
    }
}
