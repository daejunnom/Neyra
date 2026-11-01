//! Anytime de-novo fixed-bag construction search.
//! Geometry proposals are bounded and heuristic; observed canonical P7 results are exact.
//! Optional first-bag supply reflection is an explicitly exported approximation.
//! Clearra's separation of logical-row operation language and supply language is
//! retained here, with a 24-row SonicDropOnly physics backend.

pub mod labels;

use crate::physics::{Board, Input, Piece, Placement, Pose, HEIGHT};
use crate::structure::{
    map_after_clear, mask_count, placement_bias, remaining_cells, single_goals_for, single_targets,
    BoardFacts, CellMask, SingleGoal,
};
use crate::supply::{
    clearing_only_finish, decode_order, permutation,
    ClearOnlyLanguage, ClearingEdge, ClearingNode, SymmetryOrbits,
    CLEAR_ONLY_HOLD_POLICY, ALL_PIECES, P7_COUNT,
};
use num_bigint::BigUint;
#[cfg(test)]
use crate::supply::LanguageNode;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

const TARGET_SPINS: u8 = 40;
const GRAPH_LIMIT: usize = 4096;
const SPIN_DELTA: u8 = 3;
const RESIDUE_MARGIN: u16 = 12;
const BUDGET_STALE_ROUNDS: u64 = 64;
const EXPLORATION_INTERVAL: u64 = 8;
const LONG_HORIZON_PROBE_INTERVAL: u64 = 32;
const PACE_NEAR_SLACK: u16 = 1;
const PACE_DISTANCE_SLACK: u16 = 1;
// Lower SAO/Sarcelle envelope before bag 10, then the continuous SAO route.
// Sarcelle's loop rebuild and its inherited lag do not relax later progress.
// These are progress references, not physics or supply certificates.
const PACE_REFERENCE: [u8; 18] = [0, 0, 2, 4, 8, 10, 12, 15, 17, 19, 22, 25, 28, 30, 33, 36, 38, 40];

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    #[serde(alias = "minSuccessPercent")]
    pub min_success_percent: f64,
    /// Independent floor for every bag after the separately configured start.
    #[serde(alias = "bagMinSuccessPercent")]
    pub bag_min_success_percent: f64,
    /// Later bags may spend probability only after this many actual Single
    /// finishers. Zero disables this extra floor; the first bag is separate.
    #[serde(alias = "riskMinSpinsPerBag")]
    pub risk_min_spins_per_bag: u8,
    #[serde(alias = "ignoreFirstBagProbability")]
    pub ignore_first_bag_probability: bool,
    /// Independent start filter. Zero removes this floor, never the need for
    /// at least one constructible supply. Later bags also use their own floor.
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
    /// Legacy setting accepted for saved-config compatibility. Search ignores it;
    /// spin-only CNN evaluation belongs to the separate learning workflow.
    #[serde(alias = "cnnEvaluation")]
    pub cnn_evaluation: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            min_success_percent: 100.0,
            bag_min_success_percent: 100.0,
            risk_min_spins_per_bag: 3,
            ignore_first_bag_probability: true,
            first_bag_min_success_percent: 80.0,
            piece_goal: 119,
            max_pieces: 126,
            budget_start: 0,
            seed: 1,
            beam_width: 64,
            search_profile: 0,
            search_lane: 0,
            search_lanes: 1,
            cnn_evaluation: false,
        }
    }
}

impl SearchConfig {
    fn validate(&self) -> Result<(), String> {
        if !self.min_success_percent.is_finite()
            || self.min_success_percent <= 0.0
            || self.min_success_percent > 100.0
        {
            return Err("전체 성공률은 0% 초과 100% 이하여야 합니다.".into());
        }
        if (self.min_success_percent * 100.0 - (self.min_success_percent * 100.0).round()).abs()
            > 1e-7
        {
            return Err("성공률은 소수 둘째 자리까지 입력해 주세요.".into());
        }
        if !self.bag_min_success_percent.is_finite()
            || self.bag_min_success_percent <= 0.0 || self.bag_min_success_percent > 100.0
            || (self.bag_min_success_percent * 100.0
                - (self.bag_min_success_percent * 100.0).round()).abs() > 1e-7
        {
            return Err("가방별 성공률은 0% 초과 100% 이하, 소수 둘째 자리까지 입력해 주세요.".into());
        }
        if !self.first_bag_min_success_percent.is_finite()
            || !(0.0..=100.0).contains(&self.first_bag_min_success_percent)
            || (self.first_bag_min_success_percent * 100.0
                - (self.first_bag_min_success_percent * 100.0).round()).abs() > 1e-7
        {
            return Err("첫 가방 최소 성공률은 0–100%, 소수 둘째 자리까지 입력해 주세요. 0은 제한 없음입니다.".into());
        }
        if self.risk_min_spins_per_bag > 6 {
            return Err("확률 완화 최소 스핀은 0–6의 정수여야 합니다. 0은 제한 없음입니다.".into());
        }
        if self.max_pieces == 0 || self.max_pieces > 700 || self.piece_goal == 0 {
            return Err("탐색 상한은 1–700, 미노 제한은 1 이상이어야 합니다.".into());
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
    /// One retained geometric forecast, never verified progress or supply probability.
    pub forecast_spins: u8,
    pub forecast_spins_by_bag: [u8; CONTINUATION_BAGS],
    pub forecast_placements: u8,
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
            forecast_spins: self.forecast_spins,
            forecast_spins_by_bag: self.forecast_spins_by_bag,
            forecast_placements: self.forecast_placements,
            frame_count: self.frame_count,
            frames: if include_frames {
                self.frames.clone()
            } else {
                vec![]
            },
        }
    }
}

/// Snapshot metadata is derived from retained certificates so compact progress,
/// full detail and exports cannot disagree about the earlier bags' coverage.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CandidateSnapshot {
    #[serde(flatten)]
    pub summary: CandidateSummary,
    pub bag_probabilities: Vec<f64>,
}

impl std::ops::Deref for CandidateSnapshot {
    type Target = CandidateSummary;
    fn deref(&self) -> &Self::Target { &self.summary }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BagCandidatesSnapshot {
    pub bag_count: u16,
    pub best: CandidateSnapshot,
    pub runner_up: Option<CandidateSnapshot>,
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
    pub current: Option<CandidateSnapshot>,
    pub best: Option<CandidateSnapshot>,
    pub best_partial: Option<CandidateSnapshot>,
    /// Two certified whole-bag records per depth, independent of the live tree.
    /// Always compact; fetch one record's frames with candidate:<id> export.
    pub best_by_bag: Vec<BagCandidatesSnapshot>,
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
    /// Newly buried cells created by quiet setup placements. Keep the history
    /// until a cell is filled or deleted; charge only its centrally inaccessible
    /// subset on the current board, so an opened working pocket is not punished.
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
    /// One bounded longer-horizon proposal; ordinary work keeps the preferred pace.
    #[serde(skip)]
    pace_fallback: bool,
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

/// The initial bag builds a foundation. Its incidental clears are recorded,
/// but do not outrank workspace and coherent continuation quality. A genuine
/// goal finish remains a separate higher-priority stage, including small goals.
fn candidate_rank(spins: u8, quality: f64, first_bag: bool, terminal: bool) -> f64 {
    if first_bag && !terminal { spin_rank(0, quality) }
    else { spin_rank(spins, quality) + f64::from(first_bag && terminal) }
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

/// Every ordered first/next pair represents 5! queues. Unknown future
/// targets may overestimate this mask; never remove a choice on a heuristic.
fn first_pair_upper(available: u8, spawn: u8, accepting: bool) -> u16 {
    if accepting {
        return P7_COUNT as u16;
    }
    let ready = (available & spawn & ALL_PIECES).count_ones() as u16;
    let spawnable = (spawn & ALL_PIECES).count_ones() as u16;
    // A ready first piece allows six next labels. Every other spawnable first
    // piece needs one of the ready labels next. This counts the same 42 pairs.
    ready * (6 + spawnable - ready) * 120
}

fn can_be_first(start: &Board, target: &TargetOperation) -> bool {
    projected_poses(target, 0).into_iter().any(|pose| {
        start
            .place_geometry(pose)
            .is_some_and(|result| result.lines == 0 || (result.lines == 1 && start.spin_at(pose)))
    })
}

/// A full zero-clear tiling has no legal hold use. Every ordered first/next
/// pair must therefore admit those two quiet locks in one allowed orientation.
/// Geometry omits input reachability, so this is only a necessary bound. The
/// validated logical union excludes order-dependent clears; unknown tilings
/// remain eligible. Call only for nonterminal plans with no representative clear.
fn quiet_two_prefix_covered(start: &Board, targets: &[TargetOperation], reflected: bool) -> bool {
    if guaranteed_quiet_targets(start, targets.iter(), 0) != Some(ALL_PIECES) {
        return true;
    }
    let initial_spawn = spawn_mask(start);
    let mut second = [0_u8; 7];
    for first in targets {
        if initial_spawn & first.piece.bit() == 0 { continue; }
        let geometry = projected_poses(first, 0).into_iter().find_map(|pose| {
            start.place_geometry(pose).filter(|result| result.lines == 0)
        });
        let Some(geometry) = geometry else { continue; };
        let next_spawn = spawn_mask(&geometry.board);
        for target in targets {
            if target.piece == first.piece || next_spawn & target.piece.bit() == 0 { continue; }
            if projected_poses(target, 0).into_iter().any(|pose| {
                geometry.board.place_geometry(pose).is_some_and(|result| result.lines == 0)
            }) {
                second[first.piece as usize] |= target.piece.bit();
            }
        }
    }
    (0..7_u8).all(|first| (0..7_u8).all(|next| {
        first == next || second[first as usize] & (1 << next) != 0
            || (reflected && second[crate::supply::mirror_piece(first) as usize]
                & (1 << crate::supply::mirror_piece(next)) != 0)
    }))
}

/// Optimistic two-lock supply bound before a complete bag occupies a scarce
/// refinement slot. Geometry and spin shape are necessary; input reachability
/// is deliberately omitted and holding any piece remains an optimistic
/// relaxation. Each active/next/next triple represents 4! queues.
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
    for first in 0..7 {
        for current in 0..7 {
            if first == current || initial_spawn & (1 << first) == 0 {
                continue;
            }
            for next in 0..7 {
                if next == first || next == current {
                    continue;
                }
                let active_first = next_spawn[first] & (1 << current) != 0
                    && (second[first] & (1 << current) != 0
                        || (next_spawn[first] & second[first] & (1 << next) != 0));
                let stored_first = initial_spawn & (1 << current) != 0
                    && next_spawn[current] & (1 << next) != 0
                    && second[current] & ((1 << first) | (1 << next)) != 0;
                prefixes += u16::from(active_first || stored_first);
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

    fn prefix_possible(&self, queue: &[u8; 7], index: usize, cursor: u8,
        held: u8, hold_ready: bool, locks: u8) -> bool {
        let node = &self.nodes[index];
        // An unexpanded node is optimistically accepting. Goal acceptance
        // precedes every next-spawn check, including partial terminal bags.
        if node.accepting || index >= self.cursor || locks >= 2 {
            return true;
        }
        let current = queue[cursor as usize];
        if node.spawn_mask & (1 << current) == 0 {
            return false;
        }
        let locks_possible = node.edges.iter().any(|edge| {
            let next_hold = if edge.piece == current {
                held
            } else if edge.piece == held && hold_ready {
                current
            } else {
                return false;
            };
            self.prefix_possible(queue, edge.child, cursor + 1, next_hold, true, locks + 1)
        });
        // This prefix bound deliberately allows unrestricted holding. The
        // complete policy below applies clear-only deferral and quiet safety.
        locks_possible || (held == crate::supply::NO_PIECE && hold_ready
            && self.prefix_possible(queue, index, cursor + 1, current, false, locks))
    }

    fn two_lock_upper(&self) -> u16 {
        if self.overflow {
            return P7_COUNT as u16;
        }
        (0..210)
            .filter(|prefix| {
                let queue = permutation(prefix * 24);
                self.prefix_possible(&queue, 0, 0, crate::supply::NO_PIECE, true, 0)
            })
            .count() as u16
            * 24
    }

    #[cfg(test)]
    #[allow(dead_code)] // Reference language used by external matched-model checks.
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

    fn clearing_language(&self) -> ClearOnlyLanguage {
        ClearOnlyLanguage::new(self.nodes.iter().map(|node| ClearingNode {
            accepting: node.accepting,
            spawn_mask: node.spawn_mask,
            used: node.key.used,
            edges: node.edges.iter().map(|edge| ClearingEdge {
                piece: edge.piece, child: edge.child, clears: edge.lines == 1,
            }).collect(),
        }).collect(), self.all_targets)
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
    /// Supplies resolved in the selected model, including both members of an orbit.
    checked: u16,
    /// Canonical queues actually evaluated; skipped partners retain unknown costs.
    canonical_checked: u16,
    /// Every model supply has a strategy with no failed quiet lock, even if
    /// that strategy ends at an unavoidable clearing-supply blockage.
    quiet_no_hold_verified: bool,
    quiet_safe_supplies: u16,
    canonical_quiet_safe: u16,
    hold_policy: &'static str,
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
    language: Option<ClearOnlyLanguage>,
    /// Number of supplies resolved in the selected model.
    checked: usize,
    canonical_checked: usize,
    quiet_safe: usize,
    canonical_quiet_safe: usize,
    /// A quiet failure is a whole-plan policy rejection, not a P7 chance loss.
    quiet_rejected: bool,
    /// Known canonical successes; unknown reflected partners are not failures.
    success: usize,
    costs: Vec<u8>,
    codes: Vec<u32>,
    bits: Vec<u64>,
    upper: u16,
    prefix_checked: bool,
    symmetry: Option<SymmetryOrbits>,
    minimum_cost: u8,
}

impl VerificationJob {
    fn new(parent: usize, plan: Plan) -> Self {
        let symmetry = (plan.start_pieces == 0 && plan.start == Board::default() && plan.assume_reflection)
            .then(SymmetryOrbits::default);
        Self {
            graph: ExecutionGraph::new(&plan),
            parent,
            plan,
            language: None,
            checked: 0,
            canonical_checked: 0,
            quiet_safe: 0,
            canonical_quiet_safe: 0,
            quiet_rejected: false,
            success: 0,
            costs: vec![0; P7_COUNT],
            codes: vec![0; P7_COUNT],
            bits: vec![0; 79],
            upper: P7_COUNT as u16,
            prefix_checked: false,
            symmetry,
            minimum_cost: 8,
        }
    }

    fn check_canonical(&mut self, rank: usize) -> (u8, bool) {
        debug_assert_eq!(self.costs[rank], 0);
        let result = clearing_only_finish(self.language.as_ref().unwrap(), &permutation(rank));
        if let Some((cost, code)) = result.finish {
            self.costs[rank] = cost;
            self.codes[rank] = code;
            self.bits[rank / 64] |= 1_u64 << (rank % 64);
            self.success += 1;
        } else {
            self.costs[rank] = 255;
        }
        self.canonical_checked += 1;
        self.canonical_quiet_safe += usize::from(result.quiet_safe);
        (self.costs[rank], result.quiet_safe)
    }

    fn advance(&mut self, goal: u8, cache: &mut PhysicsCache) {
        if self.quiet_rejected { return; }
        if !self.graph.complete() {
            self.graph.expand_one(&self.plan, goal, cache);
            // A zero-clear whole bag has no legal hold use. Missing ordinary
            // first or second locks is a policy rejection, not chance loss.
            let quiet_bag = !self.plan.terminal
                && self.plan.start_spins < goal
                && self.plan.expected_spins == self.plan.start_spins
                && self.graph.all_targets == ALL_PIECES;
            if quiet_bag && !self.graph.overflow && self.graph.cursor == 1 {
                let root = &self.graph.nodes[0];
                let ready = root.edges.iter().filter(|edge| edge.lines == 0)
                    .fold(0, |mask, edge| mask | (1 << edge.piece)) & root.spawn_mask;
                let covered = ready | if self.symmetry.is_some() {
                    crate::supply::mirror_mask(ready)
                } else { 0 };
                if covered != ALL_PIECES { self.quiet_rejected = true; return; }
            }
            self.upper = self.upper.min(self.graph.first_upper());
            if !self.prefix_checked
                && !self.graph.overflow
                && self.graph.cursor > 1
                && self.graph.nodes[0]
                    .edges
                    .iter()
                    .all(|e| e.child < self.graph.cursor)
            {
                if quiet_bag {
                    let possible = |first: u8, second: u8| {
                        let root = &self.graph.nodes[0];
                        root.spawn_mask & (1 << first) != 0 && root.edges.iter().any(|edge| {
                            if edge.piece != first || edge.lines != 0 { return false; }
                            let child = &self.graph.nodes[edge.child];
                            child.spawn_mask & (1 << second) != 0 && child.edges.iter()
                                .any(|edge| edge.piece == second && edge.lines == 0)
                        })
                    };
                    let all_prefixes_safe = (0..7).all(|first| (0..7).all(|second| {
                        first == second || possible(first, second)
                            || (self.symmetry.is_some() && possible(
                                crate::supply::mirror_piece(first),
                                crate::supply::mirror_piece(second)))
                    }));
                    if !all_prefixes_safe { self.quiet_rejected = true; return; }
                }
                self.upper = self.upper.min(self.graph.two_lock_upper());
                self.prefix_checked = true;
            }
            return;
        }
        if self.language.is_none() {
            // Every edge consumes one distinct target. Ignoring the incoming
            // queue can only shorten an accepting path, so this is a safe
            // cost lower bound even for a terminal partial first bag.
            self.minimum_cost = self.graph.nodes.iter()
                .filter(|node| node.accepting)
                .map(|node| node.key.used.count_ones() as u8)
                .min().unwrap_or(8);
            self.language = Some(self.graph.clearing_language());
        }
        if self.symmetry.is_some() {
            let mut queries = 0;
            // Resolve a whole pair atomically. Leave room for its second query
            // so one slice still performs at most 64 canonical evaluations.
            while queries < 63 && self.checked < P7_COUNT {
                let pair = self.symmetry.as_mut().unwrap().next_pair().unwrap();
                let first = self.check_canonical(pair[0]);
                queries += 1;
                let (cost, quiet_safe) = if first.0 == self.minimum_cost {
                    first
                } else {
                    queries += 1;
                    let second = self.check_canonical(pair[1]);
                    (first.0.min(second.0), first.1 || second.1)
                };
                let symmetry = self.symmetry.as_mut().unwrap();
                symmetry.resolve(pair, cost);
                self.checked = usize::from(symmetry.checked());
                if quiet_safe { self.quiet_safe += 2; }
                else { self.quiet_rejected = true; break; }
            }
        } else {
            let end = (self.checked + 64).min(P7_COUNT);
            for rank in self.checked..end {
                let (_, quiet_safe) = self.check_canonical(rank);
                self.checked += 1;
                if quiet_safe { self.quiet_safe += 1; }
                else { self.quiet_rejected = true; break; }
            }
        }
        self.upper = self.upper.min(
            (self.success + P7_COUNT - self.canonical_checked) as u16,
        );
    }

    fn coverage_success(&self) -> u16 {
        self.symmetry.as_ref().map_or(self.success as u16, SymmetryOrbits::success)
    }

    fn coverage_upper(&self) -> u16 {
        self.symmetry.as_ref().map_or(self.upper, |symmetry| {
            symmetry.upper().min(self.upper.saturating_mul(2).min(P7_COUNT as u16))
        })
    }
}

const CONTINUATION_BAGS: usize = 2;
const CONTINUATION_SPINS: u8 = 6;
const CONTINUATION_ALTERNATIVES: usize = 6;
const CONTINUATION_TASKS: usize = 192;
const CONTINUATION_RESERVED: usize = 24;
const CONTINUATION_DEEP_WORK: u16 = 128;
const CONTINUATION_RECIPES: usize = 6;
const CONTINUATION_PAIR_LIMIT: u8 = 3;
const CONTINUATION_READY_CAP: f64 = 15.0;
const CONTINUATION_PREPARED_CAP: f64 = 5.0;
const CONTINUATION_INDEPENDENCE_CAP: f64 = 20.0;

#[derive(Clone, Default)]
struct Continuation {
    placements: Vec<Placement>,
    alternatives: Vec<Vec<Placement>>,
    score: f64,
    /// Best concrete readiness evidence; never a count of alternative slots.
    preparation_score: f64,
    /// Best pair replayed successfully in both orders, not a P7 probability.
    independence_score: f64,
    paid: u16,
    /// Immediate forecast bag only; selective cuts must not borrow bag two.
    spins: u8,
    /// One coherent geometric path, never a sum over the alternatives.
    planned_spins: u8,
    anchor: Option<Rc<SingleGoal>>,
    /// Replayed as the very first operation on the original boundary board.
    immediate_anchor: Option<Rc<SingleGoal>>,
    /// Obstruction after the first forecast bag/prefix, not an exact bound.
    obstructed_mass: Option<u16>,
}

impl Continuation {
    fn displayed_bag_spins(&self) -> [u8; CONTINUATION_BAGS] {
        [self.spins, self.planned_spins.saturating_sub(self.spins)]
    }

    fn ordered_path(&self, variant: u64) -> &[Placement] {
        let index = variant as usize % (self.alternatives.len() + 1);
        if index == 0 { &self.placements } else { &self.alternatives[index - 1] }
    }
}

#[derive(Clone)]
struct FirstBagForecast {
    spins: u8,
    quality: f64,
    paid: u16,
    obstructed_mass: u16,
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
    bag: usize,
    bag_start: Board,
    bag_spawn: u8,
    first_bag: Option<FirstBagForecast>,
    assignments: [[u64; 7]; CONTINUATION_BAGS],
    bridge_used: bool,
}

#[derive(PartialEq)]
struct ProbeKey {
    board: Board,
    inventory: u8,
    debt: CellMask,
    pursuit: Option<Rc<SingleGoal>>,
    anchor: Option<Rc<SingleGoal>>,
    deleted: u32,
    first_mask: u8,
    quiet_since_spin: u8,
    bag: usize,
    assignments: [[u64; 7]; CONTINUATION_BAGS],
    bridge_used: bool,
}

impl From<&ProbeState> for ProbeKey {
    fn from(state: &ProbeState) -> Self {
        Self {
            board: state.board, inventory: state.inventory, debt: state.debt,
            pursuit: state.pursuit.clone(), anchor: state.anchor.clone(), deleted: state.deleted,
            first_mask: state.first_mask, quiet_since_spin: state.quiet_since_spin,
            bag: state.bag, assignments: state.assignments, bridge_used: state.bridge_used,
        }
    }
}

enum ProbeTask {
    Expand(Rc<ProbeState>),
    Finish(Rc<ProbeState>, Pose),
    Support(Rc<ProbeState>, Rc<SingleGoal>, Piece),
    Bank(Rc<ProbeState>, Piece),
    Independence(Box<IndependenceCheck>),
}

struct SpinRecipe {
    /// Quiet targets followed by one finisher, before any earlier line clear.
    targets: Vec<TargetOperation>,
    rotations: Vec<u8>,
    footprint: CellMask,
    used: u8,
    end: Board,
    deleted: u32,
    preparation: u8,
}

struct IndependenceCheck {
    first: Rc<SpinRecipe>,
    second: Rc<SpinRecipe>,
    reverse: bool,
    next: usize,
    board: Board,
    deleted: u32,
    used: u8,
    first_order_end: Option<Board>,
}

struct ForecastLine {
    continuation: Continuation,
    assignments: [[u64; 7]; CONTINUATION_BAGS],
    rank: f64,
}

/// A bounded geometric portfolio, not P7 certification. A second inventory is
/// available only after all seven first-bag labels have actually been placed.
/// Exhaustion or a missing forecast is Unknown, never a proven failed supply.
struct ContinuationProbe {
    tasks: std::collections::VecDeque<ProbeTask>,
    reserved: std::collections::VecDeque<ProbeTask>,
    seen: Vec<ProbeKey>,
    work: u16,
    debt_count: u16,
    base_score: f64,
    best: Continuation,
    lines: Vec<ForecastLine>,
    recipes: Vec<Rc<SpinRecipe>>,
    immediate_anchor: Option<Rc<SingleGoal>>,
    preparation_score: f64,
    independence_score: f64,
    pair_attempts: u8,
    pair_queries: u8,
    pair_recipe_keys: Vec<[[u64; 7]; 2]>,
    minimum_success: u16,
    profile: u8,
    budget: u16,
}

impl ContinuationProbe {
    fn new(board: Board, debt: CellMask, profile: u8, minimum_success: u16) -> Self {
        let facts = BoardFacts::new(&board);
        // Track relief only for initially charged debt. Filling a free working
        // cavity must not earn a second reward for removing an uncharged hole.
        let debt = facts.penalized_debt(&debt);
        Self {
            tasks: std::collections::VecDeque::from([ProbeTask::Expand(Rc::new(ProbeState {
                board, inventory: ALL_PIECES, debt, quiet: 0, quiet_since_spin: 0,
                spins: 0, placements: vec![], pursuit: None, anchor: None,
                deleted: 0, first_mask: ALL_PIECES, bag: 0, bag_start: board,
                bag_spawn: spawn_mask(&board), first_bag: None,
                assignments: [[0; 7]; CONTINUATION_BAGS], bridge_used: false,
            }))]),
            reserved: std::collections::VecDeque::new(),
            seen: vec![], work: 0, debt_count: mask_count(&debt),
            base_score: facts.score(), best: Continuation::default(),
            lines: vec![], recipes: vec![], immediate_anchor: None, preparation_score: 0.0,
            independence_score: 0.0, pair_attempts: 0, pair_queries: 0,
            pair_recipe_keys: vec![], minimum_success, profile,
            budget: if profile == 3 { 96 } else { 64 },
        }
    }

    fn finished(&self) -> bool {
        self.work >= self.budget || (self.tasks.is_empty() && self.reserved.is_empty())
    }

    fn push(&mut self, task: ProbeTask) {
        if self.tasks.len() == CONTINUATION_TASKS { self.tasks.pop_front(); }
        self.tasks.push_back(task);
    }

    /// Interleave finisher kinds and positions without dropping alternate
    /// orientations or support clauses before their actual input queries.
    fn diverse_tasks(tasks: Vec<ProbeTask>, profile: u8) -> Vec<ProbeTask> {
        let mut groups: [Vec<(usize, ProbeTask)>; 7] = std::array::from_fn(|_| vec![]);
        let mut occurrences: [HashMap<u64, usize>; 7] = std::array::from_fn(|_| HashMap::new());
        for task in tasks {
            let (piece, shape) = match &task {
                ProbeTask::Finish(_, pose) => (pose.piece, lock_geometry(*pose, false)),
                ProbeTask::Support(_, goal, _) => (goal.pose.piece, lock_geometry(goal.pose, false)),
                ProbeTask::Bank(_, piece) => (*piece, u64::MAX),
                _ => (Piece::O, u64::MAX - 1),
            };
            let index = piece as usize;
            let order = occurrences[index].entry(shape).or_default();
            groups[index].push((*order, task));
            *order += 1;
        }
        for group in &mut groups {
            group.sort_by_key(|(order, _)| *order);
            group.reverse();
        }
        let mut ordered = vec![];
        loop {
            let before = ordered.len();
            for offset in 0..7 {
                if let Some((_, task)) = groups[(offset + profile as usize) % 7].pop() {
                    ordered.push(task);
                }
            }
            if ordered.len() == before { return ordered; }
        }
    }

    /// The first DFS branch cannot consume every first-finisher/preparation try.
    /// Reserving tasks shares their immutable state, not certificates or caches.
    fn schedule(&mut self, pending: Vec<ProbeTask>, root: bool) {
        if root {
            let mut finish = vec![];
            let mut prepare = vec![];
            for task in pending {
                if matches!(task, ProbeTask::Finish(..)) { finish.push(task); }
                else { prepare.push(task); }
            }
            let mut finish = std::collections::VecDeque::from(Self::diverse_tasks(finish, self.profile));
            let mut prepare = std::collections::VecDeque::from(Self::diverse_tasks(prepare, self.profile));
            if let Some(first) = finish.pop_front().or_else(|| prepare.pop_front()) {
                self.push(first);
            }
            // Keep both ready finishers and distinct preparation goals even
            // when a field offers many rotations of already-open spin slots.
            while self.reserved.len() < CONTINUATION_RESERVED
                && (!finish.is_empty() || !prepare.is_empty()) {
                if let Some(task) = finish.pop_front() { self.reserved.push_back(task); }
                for _ in 0..2 {
                    if self.reserved.len() == CONTINUATION_RESERVED { break; }
                    if let Some(task) = prepare.pop_front() { self.reserved.push_back(task); }
                }
            }
        } else {
            for task in pending.into_iter().rev() { self.push(task); }
        }
    }

    fn step(&mut self, cache: &mut PhysicsCache) {
        if self.finished() { return; }
        let task = if self.tasks.is_empty() || (self.work % 4 == 3 && !self.reserved.is_empty()) {
            self.reserved.pop_front()
        } else { self.tasks.pop_back() };
        let Some(task) = task else { return; };
        self.work += 1;
        match task {
            ProbeTask::Expand(state) => {
                let lock_limit = if state.spins == 0 { 4 } else { CONTINUATION_BAGS * 7 };
                if state.placements.len() >= lock_limit || state.inventory == 0
                    || state.spins >= CONTINUATION_SPINS { return; }
                let mut pending = Vec::new();
                for pose in single_targets(&state.board, state.inventory) {
                    pending.push(ProbeTask::Finish(Rc::clone(&state), pose));
                }
                // A two-spin bag still contains five quiet pieces. Finish its
                // real inventory before borrowing any next-bag finisher.
                let banking = state.bag == 0 && state.spins >= 2;
                let closeout = banking && state.inventory.count_ones() <= 2;
                if closeout {
                    for piece in Piece::ALL {
                        if state.inventory & piece.bit() != 0 {
                            pending.push(ProbeTask::Bank(Rc::clone(&state), piece));
                        }
                    }
                }
                // A quiet mino may first reshape the working cavity before a
                // concrete finisher is selected. Permit one such move per
                // segment; later support remains tied to real remaining pieces.
                if !banking && state.pursuit.is_none() && state.quiet_since_spin == 0
                    && state.inventory.count_ones() > 1 && state.placements.len() + 1 < lock_limit {
                    for piece in Piece::ALL {
                        if state.inventory & piece.bit() != 0 {
                            pending.push(ProbeTask::Bank(Rc::clone(&state), piece));
                        }
                    }
                }
                if state.quiet_since_spin < 3 && state.placements.len() + 1 < lock_limit {
                    let mut goals = Vec::new();
                    if let Some(goal) = &state.pursuit {
                        if state.inventory & goal.pose.piece.bit() != 0
                            && goal.preserved_after(&state.board, None) {
                            goals.push(Rc::clone(goal));
                        }
                    }
                    for goal in single_goals_for(&state.board, state.inventory, 12, self.profile) {
                        if goals.len() == 12 { break; }
                        if !goals.iter().any(|prior| prior.as_ref() == &goal) {
                            goals.push(Rc::new(goal));
                        }
                    }
                    // Interleave goals by piece: a single goal does not occupy
                    // all reserved root tasks before alternatives get a try.
                    for round in 0..goals.len() {
                        for (offset, piece) in Piece::ALL.into_iter().enumerate() {
                            if state.inventory & piece.bit() == 0 { continue; }
                            let goal = &goals[(round + offset) % goals.len()];
                            if piece != goal.pose.piece && !goal.ready(&state.board)
                                && goal.remaining(&state.board) <= 8 {
                                pending.push(ProbeTask::Support(
                                    Rc::clone(&state), Rc::clone(goal), piece,
                                ));
                            }
                        }
                    }
                }
                if banking && !closeout {
                    for piece in Piece::ALL {
                        if state.inventory & piece.bit() != 0 {
                            pending.push(ProbeTask::Bank(Rc::clone(&state), piece));
                        }
                    }
                }
                self.schedule(pending, state.placements.is_empty() || state.quiet_since_spin == 0);
            }
            ProbeTask::Finish(state, pose) => {
                let Some(placement) = cache.target(state.board, pose, true) else { return; };
                let Some(result) = state.board.apply(&placement) else { return; };
                if result.lines != 1 || !result.spin { return; }
                self.accept(&state, placement, result.board, result.cleared_rows, true, None, false);
            }
            ProbeTask::Support(state, goal, piece) => {
                let locks = cache.proposals(state.board, piece, self.profile == 3);
                let mut best = Vec::<(f64, Placement, Board)>::new();
                let mut bridge = None::<(f64, Placement, Board)>;
                for placement in locks.iter() {
                    let direct = goal.accepts_support(&state.board, placement.pose);
                    if !direct && (state.bridge_used || state.quiet_since_spin != 0
                        || !Self::bridge_support(&state.board, &goal, placement.pose)) { continue; }
                    let Some(result) = state.board.apply(placement) else { continue; };
                    if result.lines != 0 || !goal.preserved_after(&result.board, None) { continue; }
                    let score = BoardFacts::new(&result.board).terrain_score(Some(&goal))
                        - goal.remaining(&result.board) as f64 * 8.0;
                    if !direct {
                        if bridge.as_ref().is_none_or(|prior| score > prior.0) {
                            bridge = Some((score, placement.clone(), result.board));
                        }
                        continue;
                    }
                    if best.iter().any(|(_, _, board)| *board == result.board) { continue; }
                    let at = best.iter().position(|b| b.0 < score).unwrap_or(best.len());
                    best.insert(at, (score, placement.clone(), result.board));
                    best.truncate(2);
                }
                // One indirect footing may support a later roof piece (for
                // example an interchangeable pair). It earns no spin itself.
                if let Some((_, placement, board)) = bridge {
                    self.accept(&state, placement, board, 0, false, Some(Rc::clone(&goal)), true);
                }
                for (_, placement, board) in best.into_iter().rev() {
                    self.accept(&state, placement, board, 0, false, Some(Rc::clone(&goal)), false);
                }
            }
            ProbeTask::Bank(state, piece) => {
                let preparation = !(state.bag == 0 && state.spins >= 2);
                // Preparing inside an existing cavity can need a quiet tuck,
                // so this bounded branch uses actual reachable lock proposals.
                let locks = cache.proposals(state.board, piece, preparation);
                let before = BoardFacts::new(&state.board);
                let mut pocket = None::<(f64, Placement, Board)>;
                let mut best = Vec::<(f64, Placement, Board)>::new();
                for placement in locks.iter() {
                    let Some(result) = state.board.apply(placement) else { continue; };
                    if result.lines != 0 { continue; }
                    let score = BoardFacts::new(&result.board).score();
                    if preparation && placement.pose.cells().iter().any(|&(x, y)| {
                        y < before.height as i8 && before.workspace_columns & (1 << x) != 0
                    }) && pocket.as_ref().is_none_or(|prior| score > prior.0) {
                        pocket = Some((score, placement.clone(), result.board));
                    }
                    if best.iter().any(|(_, _, board)| *board == result.board) { continue; }
                    let at = best.iter().position(|b| b.0 < score).unwrap_or(best.len());
                    best.insert(at, (score, placement.clone(), result.board));
                    best.truncate(2);
                }
                if let Some(candidate) = pocket {
                    if !best.iter().any(|(_, _, board)| *board == candidate.2) {
                        if best.len() == 2 { best.pop(); }
                        best.push(candidate);
                    }
                }
                for (_, placement, board) in best.into_iter().rev() {
                    self.accept(&state, placement, board, 0, false, None, false);
                }
            }
            ProbeTask::Independence(check) => self.check_independence(*check, cache),
        }
    }

    fn bridge_support(board: &Board, goal: &SingleGoal, pose: Pose) -> bool {
        if !board.legal(pose) { return false; }
        let mut useful = false;
        for (x, y) in pose.cells() {
            let y = y as usize;
            let bit = 1_u16 << x;
            if goal.required_empty[y] & bit != 0 { return false; }
            if y + 1 < HEIGHT {
                let missing = goal.required_filled[y + 1] & !board.rows[y + 1];
                useful |= missing & bit != 0;
                useful |= goal.support_any.iter().any(|clause| {
                    clause[y + 1] & bit != 0
                        && !(0..HEIGHT).any(|row| clause[row] & board.rows[row] != 0)
                });
            }
        }
        useful
    }

    fn observe(&mut self, state: &ProbeState, facts: Option<&BoardFacts>) {
        if state.spins == 0 { return; }
        let current = facts.map(|facts| FirstBagForecast {
            spins: state.spins,
            quality: -(state.quiet as f64) * 10.0
                + self.debt_count.saturating_sub(mask_count(&state.debt)) as f64 * 6.0
                + (facts.score() - self.base_score) * 0.25,
            paid: self.debt_count.saturating_sub(mask_count(&state.debt)),
            obstructed_mass: facts.obstructed_mass,
        });
        let Some(first) = state.first_bag.as_ref().or(current.as_ref()) else { return; };
        let extra = state.spins.saturating_sub(first.spins);
        // Bag-two singles have a small preparation value only. Immediate
        // future.spins/debt/access remain a snapshot of the first inventory.
        let score = first.spins as f64 * 60.0 + first.quality + extra as f64 * 18.0;
        let rank = first.spins as f64 + extra as f64 * 0.3
            + 0.45 * first.quality / (120.0 + first.quality.abs());
        let line = ForecastLine {
            continuation: Continuation {
                placements: state.placements.clone(), alternatives: vec![], score,
                preparation_score: 0.0, independence_score: 0.0,
                paid: first.paid, spins: first.spins, planned_spins: state.spins,
                anchor: state.anchor.clone(), immediate_anchor: None,
                obstructed_mass: Some(first.obstructed_mass),
            },
            assignments: state.assignments, rank,
        };
        // An extension replaces its own prefix; two incompatible typed tilings
        // remain alternatives even when their occupied union is identical.
        let comparable = |prior: &ForecastLine| {
            prior.continuation.anchor == line.continuation.anchor && (
                prior.assignments.iter().flatten().zip(line.assignments.iter().flatten())
                    .all(|(&old, &new)| old == 0 || old == new)
                || line.assignments.iter().flatten().zip(prior.assignments.iter().flatten())
                    .all(|(&new, &old)| new == 0 || new == old))
        };
        if self.lines.iter().any(|prior| comparable(prior)
            && (prior.continuation.planned_spins > line.continuation.planned_spins
                || (prior.continuation.planned_spins == line.continuation.planned_spins
                    && prior.rank >= line.rank))) { return; }
        self.lines.retain(|prior| !comparable(prior));
        self.lines.push(line);
        self.lines.sort_by(|a, b| b.rank.total_cmp(&a.rank));
        // The best line remains first. Then retain different finisher kinds
        // and different occupied slots before filling with nearby variants.
        let slot = |line: &ForecastLine| line.continuation.anchor.as_ref()
            .map(|goal| (goal.pose.piece as u8, lock_geometry(goal.pose, false), goal.clear_row));
        let mut kept = vec![slot(&self.lines[0])];
        for at in 1..self.lines.len().min(CONTINUATION_ALTERNATIVES) {
            let distinct_piece = (at..self.lines.len()).find(|&i| slot(&self.lines[i]).is_some_and(|key| {
                !kept.iter().flatten().any(|prior| prior.0 == key.0)
            }));
            let distinct_slot = (at..self.lines.len()).find(|&i| !kept.contains(&slot(&self.lines[i])));
            if let Some(index) = distinct_piece.or(distinct_slot) { self.lines.swap(at, index); }
            kept.push(slot(&self.lines[at]));
        }
        self.lines.truncate(CONTINUATION_ALTERNATIVES);
        self.refresh_best();
    }

    fn refresh_best(&mut self) {
        self.best = self.lines.first().map(|line| line.continuation.clone()).unwrap_or_default();
        self.best.immediate_anchor = self.immediate_anchor.clone();
        self.best.alternatives = self.lines.iter().skip(1)
            .map(|line| line.continuation.placements.clone()).collect();
        self.best.preparation_score = self.preparation_score;
        self.best.independence_score = self.independence_score;
        self.best.score += self.preparation_score + self.independence_score;
    }

    fn record_preparation(&mut self, state: &ProbeState) {
        // No earlier clear may be necessary for this readiness preference.
        // The full coherent route has a separate throughput value below.
        if state.bag != 0 || state.spins != 0 { return; }
        let Some(goal) = &state.pursuit else { return; };
        if state.inventory & goal.pose.piece.bit() == 0
            || !goal.preserved_after(&state.board, None) { return; }
        let missing = goal.remaining(&state.board);
        if missing > 4 { return; }
        // This goal still needs a reachable finishing rotation. Its small
        // preparation score cannot masquerade as an executed spin witness.
        let score = CONTINUATION_PREPARED_CAP
            / ((1 + missing) as f64 * (1 + state.quiet) as f64);
        if score > self.preparation_score {
            self.preparation_score = score;
            self.refresh_best();
        }
    }

    fn record_first_spin(&mut self, state: &ProbeState) {
        if state.bag != 0 || state.spins != 1 || state.quiet > 2 { return; }
        if state.quiet == 0 && self.immediate_anchor.is_none() {
            self.immediate_anchor = state.anchor.clone();
            self.refresh_best();
        }
        let readiness = CONTINUATION_READY_CAP / (1 + state.quiet) as f64;
        if readiness > self.preparation_score {
            self.preparation_score = readiness;
            self.refresh_best();
        }
        let mut targets = Vec::new();
        let mut rotations = Vec::new();
        let mut footprint = [0; HEIGHT];
        let mut used = 0;
        for placement in &state.placements {
            let Some(target) = normalize_target(placement.pose, 0) else { return; };
            if used & target.piece.bit() != 0 { return; }
            used |= target.piece.bit();
            for &(x, y) in &target.cells { footprint[y as usize] |= 1 << x; }
            targets.push(target);
            rotations.push(placement.pose.rotation);
        }
        if targets.is_empty() { return; }
        let recipe = Rc::new(SpinRecipe {
            targets, rotations, footprint, used, end: state.board,
            deleted: state.deleted, preparation: state.quiet,
        });
        // Alternative positions of the same finisher are different recipes,
        // not simultaneous spins. Retain them when they change compatibility;
        // only an identical finishing target competes for direct replacement.
        let compatibility = |candidate: &SpinRecipe| self.recipes.iter()
            .filter(|prior| Self::recipes_disjoint(prior, candidate)).count();
        let new_partners = compatibility(&recipe);
        if let Some(index) = self.recipes.iter().position(|prior| {
            prior.targets.last() == recipe.targets.last()
        }) {
            let prior = &self.recipes[index];
            if prior.preparation <= recipe.preparation
                && compatibility(prior) >= new_partners { return; }
            self.recipes.remove(index);
        } else if self.recipes.len() >= CONTINUATION_RECIPES {
            let (index, prior) = self.recipes.iter().enumerate()
                .min_by_key(|(_, prior)| (compatibility(prior), std::cmp::Reverse(prior.preparation)))
                .unwrap();
            if compatibility(prior) >= new_partners
                && prior.preparation <= recipe.preparation { return; }
            self.recipes.remove(index);
        }
        // Reuse each first-spin witness. Only the other recipe on the changed
        // board needs a new exact-target query, once per placement and order.
        let mut partners: Vec<_> = self.recipes.iter()
            .filter(|prior| Self::recipes_disjoint(prior, &recipe)).cloned().collect();
        partners.sort_by_key(|prior| prior.preparation);
        for prior in partners {
            if self.pair_attempts == CONTINUATION_PAIR_LIMIT { break; }
            let mut pair_key = [Self::recipe_key(&prior), Self::recipe_key(&recipe)];
            pair_key.sort_unstable();
            if self.pair_recipe_keys.contains(&pair_key) { continue; }
            self.pair_recipe_keys.push(pair_key);
            self.pair_attempts += 1;
            let check = IndependenceCheck {
                board: prior.end, deleted: prior.deleted, used: prior.used,
                first: prior, second: Rc::clone(&recipe), reverse: false,
                next: 0, first_order_end: None,
            };
            self.reserve_independence(check);
        }
        self.recipes.push(recipe);
    }

    fn recipe_key(recipe: &SpinRecipe) -> [u64; 7] {
        let mut key = [0; 7];
        for (index, target) in recipe.targets.iter().enumerate() {
            key[target.piece as usize] = target_code(target)
                | (u64::from(recipe.rotations[index]) << 40)
                | (u64::from(index + 1 == recipe.targets.len()) << 42);
        }
        key
    }

    fn recipes_disjoint(a: &SpinRecipe, b: &SpinRecipe) -> bool {
        a.used & b.used == 0 && !(0..HEIGHT).any(|y| a.footprint[y] & b.footprint[y] != 0)
    }

    fn reserve_independence(&mut self, check: IndependenceCheck) {
        if self.reserved.len() == CONTINUATION_RESERVED { self.reserved.pop_back(); }
        self.reserved.push_front(ProbeTask::Independence(Box::new(check)));
    }

    fn check_independence(&mut self, mut check: IndependenceCheck, cache: &mut PhysicsCache) {
        let recipe = if check.reverse { &check.first } else { &check.second };
        let Some(target) = recipe.targets.get(check.next) else { return; };
        if check.used & target.piece.bit() != 0 { return; }
        // Retain the original rotation to bound this selective check to one
        // target query. A failed orientation is Unknown, not impossibility.
        let Some(pose) = projected_poses(target, check.deleted).into_iter()
            .find(|pose| pose.rotation == recipe.rotations[check.next]) else { return; };
        let finisher = check.next + 1 == recipe.targets.len();
        self.pair_queries += 1;
        let Some(placement) = cache.target(check.board, pose, finisher) else { return; };
        let Some(result) = check.board.apply(&placement) else { return; };
        if (finisher && (result.lines != 1 || !result.spin))
            || (!finisher && result.lines != 0) { return; }
        let Some(deleted) = deleted_after(check.deleted, result.cleared_rows) else { return; };
        check.board = result.board;
        check.deleted = deleted;
        check.used |= target.piece.bit();
        check.next += 1;
        if check.next < recipe.targets.len() {
            self.reserve_independence(check);
        } else if !check.reverse {
            check.first_order_end = Some(check.board);
            check.reverse = true;
            check.board = check.second.end;
            check.deleted = check.second.deleted;
            check.used = check.second.used;
            check.next = 0;
            self.reserve_independence(check);
        } else if check.first_order_end == Some(check.board) {
            // Pairwise order robustness is distinct from readiness, total
            // future spins, or the success probability of the full fixed bag.
            let preparation = check.first.preparation + check.second.preparation;
            let score = CONTINUATION_INDEPENDENCE_CAP / (1 + preparation) as f64;
            if score > self.independence_score {
                self.independence_score = score;
                self.refresh_best();
            }
        }
    }

    fn accept(
        &mut self, source: &ProbeState, placement: Placement, board: Board,
        cleared_rows: u32, spin: bool, pursuit: Option<Rc<SingleGoal>>, bridge: bool,
    ) {
        let Some(target) = normalize_target(placement.pose, source.deleted) else { return; };
        let mut first_mask = source.first_mask;
        if !can_be_first(&source.bag_start, &target) { first_mask &= !placement.pose.piece.bit(); }
        if first_pair_upper(first_mask, source.bag_spawn, false) < self.minimum_success { return; }
        let Some(deleted) = deleted_after(source.deleted, cleared_rows) else { return; };
        let mut state = source.clone();
        state.first_mask = first_mask;
        state.deleted = deleted;
        state.assignments[state.bag][placement.pose.piece as usize] = target_code(&target);
        // Only an executed first spin owns a terrain region. Preparation and
        // alternative routes cannot claim witnessed-hole exemptions.
        if state.anchor.is_none() && spin {
            let mut empty = [0; HEIGHT];
            for (x, y) in placement.pose.cells() { empty[y as usize] |= 1 << x; }
            let row = cleared_rows.trailing_zeros() as usize;
            let mut filled = [0; HEIGHT];
            filled[row] = crate::physics::FULL_ROW & !empty[row];
            state.anchor = Some(Rc::new(SingleGoal {
                pose: placement.pose, clear_row: row as u8,
                required_filled: filled, required_empty: empty, support_any: vec![],
            }));
        }
        state.inventory &= !placement.pose.piece.bit();
        state.board = board;
        state.debt = remaining_cells(&map_after_clear(&state.debt, cleared_rows), &board);
        state.spins += u8::from(spin);
        state.quiet += u8::from(!spin);
        state.quiet_since_spin = if spin { 0 } else { state.quiet_since_spin + 1 };
        state.placements.push(placement);
        state.bridge_used = !spin && (state.bridge_used || bridge);
        state.pursuit = pursuit;
        if source.bag == 0 && source.spins == 0 {
            if spin { self.record_first_spin(&state); }
            else { self.record_preparation(&state); }
        }
        let base = if self.profile == 3 { 96 } else { 64 };
        if state.spins != 0 {
            if state.bag == 0 {
                let facts = BoardFacts::new(&board);
                self.observe(&state, Some(&facts));
                self.budget = self.budget.max(base + u16::from(state.spins.min(4)) * 32);
                if state.inventory == 0 && state.spins >= 2 {
                    state.first_bag = Some(FirstBagForecast {
                        spins: state.spins,
                        quality: -(state.quiet as f64) * 10.0
                            + self.debt_count.saturating_sub(mask_count(&state.debt)) as f64 * 6.0
                            + (facts.score() - self.base_score) * 0.25,
                        paid: self.debt_count.saturating_sub(mask_count(&state.debt)),
                        obstructed_mass: facts.obstructed_mass,
                    });
                    state.bag = 1;
                    state.inventory = ALL_PIECES;
                    state.bag_start = board;
                    state.bag_spawn = spawn_mask(&board);
                    state.deleted = 0;
                    state.first_mask = ALL_PIECES;
                    state.quiet_since_spin = 0;
                    state.pursuit = None;
                    state.bridge_used = false;
                    self.budget = self.budget.max(base + 128 + CONTINUATION_DEEP_WORK);
                }
            }
        }
        if source.bag == 1 && spin { self.observe(&state, None); }
        let key = ProbeKey::from(&state);
        if !self.seen.contains(&key) {
            self.seen.push(key);
            self.push(ProbeTask::Expand(Rc::new(state)));
        }
    }
}

struct RefinementJob {
    first_bag: bool,
    pending: Vec<SeedState>,
    current: Option<(SeedState, ContinuationProbe)>,
    complete: Vec<SeedState>,
    profile: u8,
    minimum_success: u16,
}

impl RefinementJob {
    fn new(mut pending: Vec<SeedState>, profile: u8, minimum_success: u16, first_bag: bool) -> Self {
        pending.reverse();
        Self {
            first_bag,
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
                seed.score = candidate_rank(seed.spins, seed.quality, self.first_bag, seed.spins >= goal);
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
    start_spawn: u8,
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
    /// A witnessed route only orders alternatives; it never consumes inventory.
    principal: Vec<PrincipalMove>,
    lane: u16,
    lanes: u16,
    root_choice: usize,
    best_bag_spins: u8,
    selective_pruned: u64,
    prune_budget: u16,
    first_bag_symmetry: bool,
    relaxed_setup: bool,
    continuation_minimum_success: u16,
    pace_fallback: bool,
    /// A strict cost cap may remove a prefix only when this bag cannot finish.
    hard_piece_cap: u16,
    /// Qualified complete-bag rank, compared only with an optimistic bag-end rank.
    bag_alpha: f64,
    /// Exact integer equivalent of the current whole-bag pace comparison.
    pace_minimum: u8,
    /// Facts live only while this one parent is expanded over its remaining pieces.
    current_facts: Option<BoardFacts>,
    mino_bound_pruned: u64,
    mino_selective_pruned: u64,
    mino_capacity_pruned: u64,
}

struct PrincipalMove {
    board: Board,
    used: u8,
    placement: Placement,
}

/// Scores for one parent's alternatives of one next piece. Heavy path/tiling
/// copies are made only for the survivors of this sibling batch.
struct PlacementDraft {
    board: Board,
    spins: u8,
    optimistic_spins: u8,
    deleted: u32,
    target: TargetOperation,
    debt: CellMask,
    pursuit: Option<Rc<SingleGoal>>,
    first_mask: u8,
    score: f64,
    quality: f64,
    placed_cells: [(u8, u8); 4],
    lines: u8,
    spin: bool,
}

/// This is selective sibling futility, not dominance between future boards.
/// Never let a weaker supply mask or lower remaining capacity kill an option.
fn sibling_dominates(better: &PlacementDraft, other: &PlacementDraft, goal: u8) -> bool {
    other.optimistic_spins < goal
        && better.spins >= other.spins
        && better.optimistic_spins >= other.optimistic_spins
        && better.first_mask & other.first_mask == other.first_mask
        && better.pursuit == other.pursuit
        && if better.spins > other.spins {
            better.quality > other.quality
        } else {
            better.quality > other.quality + 24.0
        }
}

/// Actual goal finishes use their own complete-plan capacity. A later quiet
/// alternative must not evict a finish through the smaller inventory bucket.
fn insert_sibling_draft(
    batches: &mut [Vec<PlacementDraft>; 2], draft: PlacementDraft,
    goal: u8, continuing_limit: usize, selective: bool,
) -> (u64, u64) {
    let terminal = draft.spins >= goal;
    let siblings = &mut batches[usize::from(terminal)];
    let mut pruned = 0;
    if selective {
        if siblings.iter().any(|other| sibling_dominates(other, &draft, goal)) {
            return (1, 0);
        }
        let before = siblings.len();
        siblings.retain(|other| !sibling_dominates(&draft, other, goal));
        pruned = (before - siblings.len()) as u64;
    }
    let at = siblings.iter().position(|other| other.score < draft.score).unwrap_or(siblings.len());
    siblings.insert(at, draft);
    let limit = if terminal { 12 } else { continuing_limit };
    let capacity = siblings.len() > limit;
    if capacity { siblings.pop(); }
    (pruned, u64::from(capacity))
}

/// Put the witnessed next label first without removing any other inventory.
fn ordered_piece(index: usize, preferred: Option<usize>) -> Piece {
    let index = match preferred {
        Some(preferred) if index == 0 => preferred,
        Some(preferred) if index <= preferred => index - 1,
        _ => index,
    };
    Piece::ALL[index]
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
        let start_spawn = spawn_mask(&node.board);
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
        // Retain every first choice. A prior continuation supplies move-order
        // hints only, including a witnessed tuck absent from the fast list.
        let mut principal = Vec::new();
        if !setup {
            let mut board = node.board;
            let mut used = 0;
            let mut spins = node.spins;
            for placement in node.future.ordered_path(variant).iter().take(7) {
                if spins >= goal || used & placement.pose.piece.bit() != 0 { break; }
                let Some(result) = board.apply(placement) else { break; };
                if result.lines > 1 || (result.lines == 1 && !result.spin) { break; }
                principal.push(PrincipalMove { board, used, placement: placement.clone() });
                board = result.board;
                used |= placement.pose.piece.bit();
                spins += result.lines;
            }
        }
        seed.quality = board_score(&seed.board);
        seed.score = candidate_rank(seed.spins, seed.quality, node.pieces == 0, seed.spins >= goal);
        let depth = seed.targets.len() as u8;
        let terminal = seed.spins >= goal;
        Self {
            parent,
            start: node.board,
            start_spins: node.spins,
            start_pieces: node.pieces,
            start_spawn,
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
            principal,
            lane,
            lanes,
            root_choice: 0,
            best_bag_spins: 0,
            selective_pruned: 0,
            prune_budget: u16::MAX,
            first_bag_symmetry: node.pieces == 0,
            relaxed_setup: false,
            continuation_minimum_success: minimum_success,
            pace_fallback: false,
            hard_piece_cap: u16::MAX,
            bag_alpha: f64::NEG_INFINITY,
            pace_minimum: 0,
            current_facts: None,
            mino_bound_pruned: 0,
            mino_selective_pruned: 0,
            mino_capacity_pruned: 0,
        }
    }

    fn set_prune_budget(&mut self, goal: u8, budget: u16) {
        if self.prune_budget != budget {
            self.prune_budget = budget;
            self.pace_minimum = min_pace_spins(goal, self.start_pieces + 7, budget);
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
        let bounds = MinoBounds {
            start_pieces: self.start_pieces, best_bag_spins: self.best_bag_spins,
            bag_alpha: self.bag_alpha, goal, pace_minimum: self.pace_minimum,
            hard_piece_cap: self.hard_piece_cap,
        };
        let state = &self.frontier[self.state_index];
        // The parent is unchanged while its remaining piece labels are visited.
        // Every child still checks current bounds before entering a frontier.
        let cut = if self.piece_index == 0 {
            mino_cut(&state.board, state.used, state.spins, bounds)
        } else { None };
        if cut.is_some() {
            self.state_index += 1;
            self.piece_index = 0;
            self.current_facts = None;
            if cut == Some(MinoCut::Cost) { self.mino_bound_pruned += 1; self.pruned += 1; }
            else { self.mino_selective_pruned += 1; self.selective_pruned += 1; }
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
            let inventory = if self.start_pieces == 0 {
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
        let hint = self.principal.iter().find(|hint| {
            hint.board == state.board && hint.used == state.used
        }).map(|hint| &hint.placement);
        let preferred = hint.map(|hint| hint.pose.piece as usize);
        let piece = ordered_piece(self.piece_index, preferred);
        self.piece_index += 1;
        if self.piece_index == 7 {
            self.piece_index = 0;
            self.state_index += 1;
        }
        if state.used & (1 << (piece as u8)) != 0 {
            if self.piece_index == 0 { self.current_facts = None; }
            return 0;
        }
        let root_choice = self.start_pieces == 0 && state.used == 0;
        // Every lane partitions the same first-lock geometry list. The rest
        // of each bag is built locally; no worker shares a supply certificate.
        // Quiet setup rounds discard every single. Their narrow proposal set
        // therefore needs no exact spin witness search at all.
        let locks = if self.setup && !self.broad {
            Rc::new(state.board.surface_locks(piece))
        } else { cache.proposals(state.board, piece, self.broad && !root_choice) };
        let hint = hint.filter(|placement| placement.pose.piece == piece);
        if locks.is_empty() && hint.is_none() {
            if self.piece_index == 0 { self.current_facts = None; }
            return 0;
        }
        let hint_key = hint.map(|placement| lock_geometry(placement.pose, placement.spin));
        let per_inventory = self.inventory_limit(state.used | piece.bit());
        let before_facts = self.current_facts.get_or_insert_with(|| BoardFacts::new(&state.board));
        let before_holes = before_facts.holes;
        let mut expanded = 0;
        let mut siblings = [Vec::<PlacementDraft>::new(), Vec::new()];
        // Witnessed spins establish a useful local window before quiet moves.
        // The root list is unchanged (the empty field has no spin single).
        for placement in hint.into_iter()
            .chain(locks.iter().filter(|p| p.spin
                && hint_key.is_none_or(|key| lock_geometry(p.pose, p.spin) != key)))
            .chain(locks.iter().filter(|p| !p.spin
                && hint_key.is_none_or(|key| lock_geometry(p.pose, p.spin) != key)))
        {
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
            let first_upper = first_pair_upper(available, self.start_spawn, self.start_spins >= goal);
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
            // A round reserved for quiet setup can never repair a missing
            // first choice with a later clear. Unknown targets remain allowed.
            // Mixed prefixes may change a target's clear role, so are exempt.
            if !complete && (self.setup || (used == ALL_PIECES && spins == self.start_spins))
                && !quiet_first_covered(first_mask & self.start_spawn, self.first_bag_symmetry)
            {
                self.pruned += 1;
                self.mino_bound_pruned += 1;
                continue;
            }
            let potential = bag_potential(&result.board, used);
            let optimistic_spins = spins.saturating_add(potential.0);
            if let Some(cut) = mino_cut_with_potential(spins, potential, bounds) {
                if cut == MinoCut::Cost { self.mino_bound_pruned += 1; self.pruned += 1; }
                else { self.mino_selective_pruned += 1; self.selective_pruned += 1; }
                continue;
            }
            if !complete && used == ALL_PIECES {
                let quiet = guaranteed_quiet_targets(
                    &self.start, state.targets.iter().chain(std::iter::once(&target)), deleted,
                );
                if quiet.is_some_and(|quiet| {
                    let bad = quiet & !(first_mask & self.start_spawn);
                    if self.first_bag_symmetry {
                        bad & crate::supply::mirror_mask(bad) != 0
                    } else { bad != 0 }
                }) {
                    self.pruned += 1;
                    self.mino_bound_pruned += 1;
                    continue;
                }
            }
            let (bucket, limit) = if used == ALL_PIECES {
                (&self.complete, 12)
            } else { (&self.next_by_inventory[used as usize], per_inventory) };
            if !complete && self.start_pieces != 0 && below_bucket_window(bucket, limit, spins) {
                self.mino_capacity_pruned += 1;
                continue;
            }
            if used == ALL_PIECES && !complete {
                let targets: [TargetOperation; 7] = std::array::from_fn(|index| {
                    state.targets.get(index).unwrap_or(&target).clone()
                });
                if spins == self.start_spins && deleted == 0
                    && !quiet_two_prefix_covered(&self.start, &targets, self.first_bag_symmetry)
                {
                    self.pruned += 1;
                    self.mino_bound_pruned += 1;
                    continue;
                }
                let upper = geometric_two_lock_upper(&self.start, &targets, self.start_spins, goal);
                let upper = if self.first_bag_symmetry {
                    upper.saturating_mul(2).min(P7_COUNT as u16)
                } else { upper };
                if upper < self.minimum_success {
                    self.pruned += 1;
                    continue;
                }
            }
            let pursuit = if result.lines == 0 {
                state.pursuit.as_ref().and_then(|pursuit| {
                    // A speculative footprint is guidance, not an obligation.
                    // Release it if this alternative fills its air cells or
                    // spends the non-setup bag's intended finisher.
                    ((self.start_pieces == 0 || used & pursuit.pose.piece.bit() == 0)
                        && pursuit.preserved_after(&result.board, None))
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
            structural -= mask_count(&facts.penalized_debt(&debt)) as f64 * 4.0;
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
            }
            let quality = if self.start_pieces == 0 {
                // Early floor freedom remains important, but the last four
                // placements must also retain a useful workspace and roof.
                // Every first-bag round uses this foundation order, even after
                // an incidental spin. The intended finisher may be next bag's.
                setup_score(&self.start, state.targets.iter().chain(std::iter::once(&target)))
                    * if self.relaxed_setup { 0.35 } else { 1.0 }
                    + structural * foundation_terrain_weight(state.targets.len() + 1)
            } else {
                structural + setup_score(&self.start, state.targets.iter().chain(std::iter::once(&target))) * 0.25
            };
            // Jitter only orders proposals; it is not evidence of better terrain.
            let score = candidate_rank(spins,
                quality + rng.unit() * 2.0,
                self.start_pieces == 0, complete);
            let (bucket, limit) = if used == ALL_PIECES {
                (&self.complete, 12)
            } else { (&self.next_by_inventory[used as usize], per_inventory) };
            if !complete && bucket.len() >= limit
                && bucket.last().is_some_and(|worst| score <= worst.score)
            {
                self.mino_capacity_pruned += 1;
                continue;
            }
            let draft = PlacementDraft {
                board: result.board, spins, optimistic_spins, deleted, target,
                debt, pursuit, first_mask, score, quality,
                placed_cells: placement.pose.cells().map(|(x, y)| (x as u8, y as u8)),
                lines: result.lines, spin: result.spin,
            };
            let (selective, capacity) = insert_sibling_draft(
                &mut siblings, draft, goal,
                if used == ALL_PIECES { 12 } else { per_inventory }, self.start_pieces != 0,
            );
            self.mino_selective_pruned += selective;
            self.selective_pruned += selective;
            self.mino_capacity_pruned += capacity;
        }
        // All placements of this next piece have competed before their copies
        // or descendants are allocated. Different parents never share a window.
        for draft in siblings.into_iter().flatten() {
            let mut targets = state.targets.clone();
            let mut target_key = state.target_key;
            target_key[piece as usize] = target_code(&draft.target);
            targets.push(draft.target);
            let mut frames = state.frames.clone();
            frames.push(DisplayFrame {
                rows: draft.board.rows,
                piece: piece_name(piece),
                placed_cells: draft.placed_cells,
                spins: draft.spins,
                placed: self.start_pieces + targets.len() as u16,
                lines: draft.lines,
                spin: draft.spin,
            });
            let next = SeedState {
                board: draft.board,
                spins: draft.spins,
                used: state.used | (1 << (piece as u8)),
                deleted: draft.deleted,
                targets,
                target_key,
                frames,
                score: draft.score,
                quality: draft.quality,
                pursuit: draft.pursuit,
                debt: draft.debt,
                future: Continuation::default(),
                first_mask: draft.first_mask,
            };
            if next.spins >= goal || next.used == ALL_PIECES {
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
        if self.piece_index == 0 { self.current_facts = None; }
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
        self.current_facts = None;
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
                    self.start_pieces == 0,
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
        const MASKS: [usize; 8] = [1, 7, 21, 35, 35, 21, 7, 1];
        self.width.div_ceil(MASKS[used.count_ones() as usize]).max(1)
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

/// Preserve independent floor choices early, then give the completed
/// workspace, banks and roof a visible vote before continuation refinement.
fn foundation_terrain_weight(placed: usize) -> f64 {
    0.06 + 0.06 * placed.saturating_sub(3).min(4) as f64
}

/// An intentionally conservative *ordering score*, never an eligibility test.
/// Direct-drop interference and missing base support expose fragile setup bags
/// cheaply. Actual rotation, OR support and hold possibilities remain the exact
/// execution verifier's responsibility.
fn setup_score<'a>(
    base: &Board,
    targets: impl Iterator<Item = &'a TargetOperation> + Clone,
) -> f64 {
    let unsupported = targets
        .clone()
        .filter(|target| {
            !projected_poses(target, 0)
                .into_iter()
                .any(|pose| base.place_geometry(pose).is_some())
        })
        .count();
    let mut incoming = [0_u8; 7];
    for a in targets.clone() {
        for b in targets.clone() {
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
        .clone()
        .flat_map(|target| target.cells.iter())
        .filter(|(x, y)| *y == 0 || base.occupied(*x as i8, *y as i8 - 1))
        .count();
    let remaining_foundation: usize = Piece::ALL
        .iter()
        .filter(|piece| !targets.clone().any(|target| target.piece == **piece))
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

impl Archived {
    fn for_snapshot(&self, include_frames: bool) -> CandidateSnapshot {
        CandidateSnapshot {
            summary: self.summary.for_snapshot(include_frames),
            // A pending bag stays unknown; only the proven prefix is listed.
            bag_probabilities: self.certificates.iter()
                .map(|certificate| f64::from(certificate.effective_success) / P7_COUNT as f64)
                .collect(),
        }
    }
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
    let unexplained = mask_count(&facts.penalized_debt(debt)).saturating_sub(future.paid);
    facts.terrain_score_witnessed(future.immediate_anchor.as_deref()) * 2.0 + future.score
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

/// Optimistic end-of-bag material and single count. Completing k distinct rows
/// needs at least the sum of the k cheapest current row deficits. Any newly
/// introduced empty row costs ten cells, so it cannot improve that lower bound.
/// Geometry, input reachability and ordering can only reduce this capacity.
fn bag_potential(board: &Board, used: u8) -> (u8, u16) {
    let remaining = ALL_PIECES & !used;
    let finishers = (remaining & !Piece::O.bit()).count_ones() as u8;
    let fresh = remaining.count_ones() as u16 * 4;
    let mut deficits = [0_u8; 11];
    let mut cells = 0;
    for row in board.rows {
        let filled = row.count_ones() as u16;
        cells += filled;
        deficits[10 - filled as usize] += 1;
    }
    let mut spent = 0;
    let mut singles = 0;
    'rows: for (missing, &count) in deficits.iter().enumerate() {
        for _ in 0..count {
            if singles == finishers || spent + missing as u16 > fresh { break 'rows; }
            spent += missing as u16;
            singles += 1;
        }
    }
    (singles, cells + fresh - u16::from(singles) * 10)
}

fn quiet_first_covered(available: u8, reflected: bool) -> bool {
    let covered = if reflected { available | crate::supply::mirror_mask(available) } else { available };
    covered == ALL_PIECES
}

/// In a completed fixed bag, a target touching no ultimately full logical row
/// cannot clear in any execution order. Such a piece must be playable first
/// when it leads the supply. This is only a necessary quiet-policy condition;
/// partial/terminal plans must not use it. A malformed union remains unknown.
fn guaranteed_quiet_targets<'a>(
    start: &Board, targets: impl IntoIterator<Item = &'a TargetOperation>, deleted: u32,
) -> Option<u8> {
    let mut rows = [0_u16; 32];
    rows[..HEIGHT].copy_from_slice(&start.rows);
    if rows.iter().any(|row| row & !crate::physics::FULL_ROW != 0) { return None; }
    let mut seen = 0_u8;
    let mut touched = [0_u32; 7];
    for target in targets {
        let piece = target.piece as usize;
        if seen & target.piece.bit() != 0 { return None; }
        seen |= target.piece.bit();
        for &(x, y) in &target.cells {
            if x as usize >= crate::physics::WIDTH || y >= 32 { return None; }
            let cell = 1_u16 << x;
            if rows[y as usize] & cell != 0 { return None; }
            rows[y as usize] |= cell;
            touched[piece] |= 1_u32 << y;
        }
    }
    if seen != ALL_PIECES { return None; }
    let full = rows.iter().enumerate().fold(0_u32, |mask, (y, &row)| {
        mask | if row == crate::physics::FULL_ROW { 1_u32 << y } else { 0 }
    });
    // Comparing the exact deletion mask validates normalization through row31;
    // counting rows alone could silently misclassify a target's clear role.
    if full != deleted { return None; }
    Some(touched.iter().enumerate().fold(0, |quiet, (piece, &rows)| {
        quiet | if rows & full == 0 { 1 << piece } else { 0 }
    }))
}

#[derive(Clone, Copy)]
struct MinoBounds {
    start_pieces: u16,
    best_bag_spins: u8,
    bag_alpha: f64,
    goal: u8,
    pace_minimum: u8,
    hard_piece_cap: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MinoCut { Cost, Selective }

/// Called before expensive field evaluation at every partial placement.
/// A possible same-bag finish is protected: its canonical seven targets are
/// not a lower bound on the number of locks that a supply may actually need.
fn mino_cut(board: &Board, used: u8, spins: u8, bounds: MinoBounds) -> Option<MinoCut> {
    if spins >= bounds.goal { return None; }
    mino_cut_with_potential(spins, bag_potential(board, used), bounds)
}

fn mino_cut_with_potential(
    spins: u8, (more, optimistic_cells): (u8, u16), bounds: MinoBounds,
) -> Option<MinoCut> {
    if spins >= bounds.goal { return None; }
    let upper = spins.saturating_add(more);
    if upper >= bounds.goal { return None; }
    let end = bounds.start_pieces + 7;
    let left = u16::from(bounds.goal - upper);
    let after = left.max((left * 10).saturating_sub(optimistic_cells).div_ceil(4));
    if end.saturating_add(after) > bounds.hard_piece_cap { return Some(MinoCut::Cost); }
    // The initial bag is a foundation even when a peer already cleared there.
    // Keep material bounds; no spin quota, peer delta or alpha applies here.
    if bounds.start_pieces == 0 { return None; }
    if delta_lag(spins, bounds.best_bag_spins, more)
        || upper < bounds.pace_minimum
        || (bounds.start_pieces != 0 && f64::from(upper) + 0.95 <= bounds.bag_alpha)
    { return Some(MinoCut::Selective); }
    None
}

/// Pace is monotone in actual spin count. Cache this exact integer threshold
/// once per bag/horizon, rather than repeat interpolation at every placement.
fn min_pace_spins(goal: u8, pieces: u16, budget: u16) -> u8 {
    let (mut low, mut high) = (0, goal);
    while low < high {
        let middle = low + (high - low) / 2;
        if pace_cut(middle, pieces, goal, budget) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low
}

/// The quality contribution is strictly below 0.95. This shortcut only skips
/// candidates that cannot enter this already full bounded proposal bucket.
fn below_bucket_window(bucket: &[SeedState], limit: usize, spins: u8) -> bool {
    bucket.len() >= limit && bucket.last().is_some_and(|worst| {
        f64::from(spins) + 0.95 <= worst.score
    })
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
    Quality,
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
    // The setup bag and genuine goal finishes need no spin quota. Do not
    // let an incidental first-bag clear erase a better zero-spin foundation.
    if spins >= goal || pieces <= 7 { return None; }
    if delta_lag(spins, best_at_bag, 0) {
        return Some(SelectiveReason::BagDelta);
    }
    // Use all placed locks, with room to prepare while the horizon is far
    // away. The setup exemption above does not permanently shift the curve.
    // Cross multiplication avoids rounding a quota at each bag boundary.
    if pace_cut(spins, pieces, goal, budget) {
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

/// Keep af8's bounded 0.2 quality term in spin units before scaling by
/// productive locks, so its influence cannot grow with bag depth.
fn scheduling_rank(spins: u8, pieces: u16, quality: f64) -> f64 {
    let scale = 7.0 / f64::from(productive_locks(pieces));
    let productive_spins = if pieces <= 7 { 0 } else { spins };
    efficiency(productive_spins, pieces) + 0.2 * quality / (120.0 + quality.abs()) * scale
}

/// Probability is a consumed resource. Shift the bounded signed rank before
/// weighting, so even a zero-spin negative field cannot benefit from more risk.
/// P=1 takes the original path exactly, without extra floating-point rounding.
fn probability_scheduling_rank(spins: u8, pieces: u16, quality: f64, chance: f64) -> f64 {
    if chance == 1.0 { return scheduling_rank(spins, pieces, quality); }
    let shift = 0.2 * 7.0 / f64::from(productive_locks(pieces));
    (scheduling_rank(spins, pieces, quality) + shift) * bounded_probability(chance) - shift
}

fn bounded_probability(chance: f64) -> f64 {
    if chance.is_finite() { chance.clamp(0.0, 1.0) } else { 0.0 }
}

const PROBABILITY_PACE_QUALITY: f64 = 24.0;

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ProbabilityPace {
    enabled: bool,
    counted_bags: usize,
    planned_counted_bags: usize,
    horizon_pieces: u16,
    spent_fraction: f64,
    paced_fraction: f64,
    quality_adjustment: f64,
    expected_prefix_probability: f64,
    projected_final_probability: Option<f64>,
}

/// A scheduling model for already certified probability, not future coverage.
/// Uniform consumption means linear log-budget use across counted bags. Keep
/// the numeric allowance exact elsewhere; this bounded preference cannot admit
/// a route or change a certificate. A safer prefix can never score worse.
fn probability_pace(
    chance: f64, counted_bags: usize, horizon_pieces: u16,
    exclude_first: bool, floor_percent: f64,
) -> ProbabilityPace {
    let planned = usize::from(horizon_pieces.div_ceil(7))
        .saturating_sub(usize::from(exclude_first));
    let floor = floor_percent / 100.0;
    let chance = bounded_probability(chance);
    let enabled = floor > 0.0 && floor < 1.0 && planned != 0 && chance > 0.0;
    let paced = if enabled { (counted_bags as f64 / planned as f64).min(1.0) } else { 0.0 };
    let spent = if enabled { (chance.ln() / floor.ln()).max(0.0) } else { 0.0 };
    ProbabilityPace {
        enabled, counted_bags, planned_counted_bags: planned, horizon_pieces,
        spent_fraction: spent, paced_fraction: paced,
        quality_adjustment: if enabled {
            PROBABILITY_PACE_QUALITY * (paced - spent).clamp(-1.0, 1.0)
        } else { 0.0 },
        expected_prefix_probability: if enabled { floor.powf(paced) } else { 1.0 },
        projected_final_probability: if enabled && counted_bags != 0 {
            Some(chance.powf(planned.max(counted_bags) as f64 / counted_bags as f64))
        } else { None },
    }
}

/// Expected spins as a rational with denominator 40 * horizon. Map a user
/// horizon onto 17 reference bags; interpolate without rounding per-mino quotas.
fn pace_reference_numerator(pieces: u16, goal: u8, budget: u16) -> u64 {
    let horizon = u64::from(budget.max(1));
    let scaled = u64::from(pieces) * 17;
    let bag = (scaled / horizon) as usize;
    let reference = if bag >= 17 {
        u64::from(PACE_REFERENCE[17]) * u64::from(pieces)
    } else {
        let remainder = scaled % horizon;
        u64::from(PACE_REFERENCE[bag]) * (horizon - remainder)
            + u64::from(PACE_REFERENCE[bag + 1]) * remainder
    };
    u64::from(goal) * reference
}

fn pace_cut(spins: u8, pieces: u16, goal: u8, budget: u16) -> bool {
    if pieces <= 7 || spins >= goal { return false; }
    let horizon = u64::from(budget.max(1));
    let slack = u64::from(PACE_NEAR_SLACK) * horizon
        + u64::from(PACE_DISTANCE_SLACK) * u64::from(budget.saturating_sub(pieces));
    pace_reference_numerator(pieces, goal, budget)
        >= u64::from(spins) * 40 * horizon + slack * 40
}

fn pace_lag(spins: u8, pieces: u16, goal: u8, budget: u16) -> f64 {
    if pieces <= 7 { return 0.0; }
    pace_reference_numerator(pieces, goal, budget) as f64
        / (40.0 * f64::from(budget.max(1))) - f64::from(spins)
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
    // Fresh productive boundaries keep substantial work after piece-level cuts.
    // Repeated unsuccessful generations do not retain this minimum lease.
    if stale >= 4 { return 1; }
    if pieces <= 7 { return 2 + u32::from(future_spins != 0 && stale < 2); }
    let rounds = focused_rounds(last_gain, future_spins.min(1), stale).clamp(3, 8);
    if under_pace(spins, pieces, goal, budget) { rounds.min(4) } else { rounds }
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
    best_by_bag: Vec<Vec<Archived>>,
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
    quiet_policy_rejected: u64,
    selective_pruned: u64,
    mino_bound_pruned: u64,
    mino_selective_pruned: u64,
    mino_capacity_pruned: u64,
    last_budget_progress_round: u64,
    benchmark_epoch: u64,
    dispatches: u64,
    focus_since_dispatch: u64,
    last_exploration_dispatch: u64,
    last_pace_probe_dispatch: u64,
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
            best_by_bag: vec![],
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
            quiet_policy_rejected: 0,
            selective_pruned: 0,
            mino_bound_pruned: 0,
            mino_selective_pruned: 0,
            mino_capacity_pruned: 0,
            last_budget_progress_round: 0,
            benchmark_epoch: 0,
            dispatches: 0,
            focus_since_dispatch: 0,
            last_exploration_dispatch: 0,
            last_pace_probe_dispatch: 0,
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
                    self.selective_spin_reference(pieces as usize / 7), self.goal, self.proposal_budget(job.plan.pace_fallback),
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
                self.last_success = job.coverage_success();
                if self.preview_steps % 16 == 0 {
                    self.update_verifying_preview(&job);
                }
                if job.graph.overflow {
                    // Incomplete analysis is not inserted into any failure bitset.
                    self.backpropagate(job.parent, 0.0);
                } else if job.quiet_rejected {
                    // This fixed tiling violates the mandatory no-hold quiet
                    // placement policy, independently of the chance threshold.
                    self.quiet_policy_rejected += 1;
                    self.pruned += 1;
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
                job.best_bag_spins = self.selective_spin_reference((job.start_pieces as usize / 7) + 1);
                job.set_prune_budget(self.goal, self.proposal_budget(job.pace_fallback));
                let prior_pruned = job.pruned;
                let prior_duplicates = job.duplicates;
                let prior_selective = job.selective_pruned;
                let prior_mino = (job.mino_bound_pruned, job.mino_selective_pruned, job.mino_capacity_pruned);
                self.expanded += job.advance(&mut self.rng, self.goal, &mut self.physics_cache);
                self.pruned += job.pruned - prior_pruned;
                self.duplicates += job.duplicates - prior_duplicates;
                self.selective_pruned += job.selective_pruned - prior_selective;
                self.mino_bound_pruned += job.mino_bound_pruned - prior_mino.0;
                self.mino_selective_pruned += job.mino_selective_pruned - prior_mino.1;
                self.mino_capacity_pruned += job.mino_capacity_pruned - prior_mino.2;
                if self.preview_steps % 16 == 0 {
                    self.update_generation_preview(&job);
                }
                if job.next_layer() {
                    let parent = job.parent;
                    let mut completed = job.complete;
                    if job.pace_fallback {
                        // A longer-horizon probe buys at most one ordinary P7
                        // candidate. Genuine finishes always retain verification.
                        let mut ordinary = false;
                        completed.retain(|state| {
                            if state.spins >= self.goal { return true; }
                            if ordinary { return false; }
                            ordinary = true;
                            true
                        });
                    }
                    for state in completed.into_iter().rev() {
                        let pieces = job.start_pieces + state.targets.len() as u16;
                        if state.spins < self.goal
                            && selective_reason(
                                state.board.cell_count(),
                                boundary_obstruction(&state.board, &state.future),
                                pieces,
                                state.spins,
                                self.selective_spin_reference(pieces as usize / 7),
                                self.goal,
                                job.prune_budget,
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
                            assume_reflection: (job.start_pieces == 0 && job.start == Board::default())
                                || self.tree[parent].certificates.first()
                                .is_some_and(|certificate| certificate.first_bag_symmetry_assumed),
                            quality: state.quality,
                            pace_fallback: job.pace_fallback,
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
                    && self.regular_node_eligible(parent);
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
                self.selective_spin_reference(pieces as usize / 7),
                self.goal,
                self.proposal_budget(plan.pace_fallback),
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
        if self.config.min_success_percent < 100.0
            || pieces <= 7 || pieces % 7 != 0 || spins >= self.goal || !quality.is_finite() {
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
        if plan.terminal { return false; }
        let Some(better_spins) = self.quality_dominator(
            plan.start_pieces + plan.targets.len() as u16, plan.expected_spins, plan.quality,
        ) else { return false; };
        let stale = self.tree[parent].attempts
            .saturating_sub(self.tree[parent].last_improved_attempt);
        // Give a coherent, promising preparation its first reduced probes.
        // Repeated unsupported optimism cannot buy unlimited verification.
        !(stale <= 1 && plan.expected_spins.saturating_add(plan.future.spins) > better_spins)
    }

    fn widen_budget(&mut self) {
        let widened = self.active_budget.saturating_add(7).min(self.config.max_pieces);
        if widened != self.active_budget {
            self.active_budget = widened;
            // More available locks do not erase an already qualified spin and
            // quality disadvantage. Longer routes use a separate small probe.
            self.benchmark_epoch += 1;
            self.focus = None;
        }
        self.last_budget_progress_round = self.generation_rounds;
    }

    fn preferred_pace_budget(&self) -> u16 {
        let material_floor = (u16::from(self.goal) * 10).div_ceil(4);
        self.config.piece_goal.max(material_floor).min(self.active_budget).max(1)
    }

    fn proposal_budget(&self, pace_fallback: bool) -> u16 {
        if pace_fallback { self.active_budget } else { self.preferred_pace_budget() }
    }

    fn behind_preferred_pace(&self, index: usize) -> bool {
        let node = &self.tree[index];
        pace_cut(node.spins, node.pieces, self.goal, self.preferred_pace_budget())
    }

    fn regular_node_eligible(&self, index: usize) -> bool {
        self.can_expand(index) && self.node_selective_reason(index).is_none()
            && !self.behind_preferred_pace(index)
    }

    fn node_selective_reason(&self, index: usize) -> Option<SelectiveReason> {
        let node = &self.tree[index];
        if let Some(better_spins) = self.quality_dominator(node.pieces, node.spins, node.quality) {
            // A concrete next-bag witness gets one narrow opportunity only.
            // Widening a horizon or an ageing priority cannot renew it.
            if node.attempts != 0 || node.spins.saturating_add(node.future.spins) <= better_spins {
                return Some(SelectiveReason::Quality);
            }
        }
        selective_reason(
            node.board.cell_count(),
            node.obstructed_mass,
            node.pieces,
            node.spins,
            self.selective_spin_reference(node.pieces as usize / 7),
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

    fn probability_pace_for_counts(&self, counts: &[u16]) -> ProbabilityPace {
        // A widened active search budget must not replenish spent probability.
        // Use the configured completion horizon, with the material lower bound.
        let material_floor = (u16::from(self.goal) * 10).div_ceil(4);
        let horizon = self.config.piece_goal.max(material_floor)
            .min(self.config.max_pieces).max(1);
        probability_pace(self.path_probability(counts), self.counted_counts(counts).len(),
            horizon, self.config.ignore_first_bag_probability, self.config.min_success_percent)
    }

    fn first_bag_minimum(&self) -> u16 {
        // The integer threshold also makes 0 mean any nonzero coverage.
        let basis_points = (self.config.first_bag_min_success_percent * 100.0).round() as u32;
        (basis_points * P7_COUNT as u32).div_ceil(10_000).max(1) as u16
    }

    fn start_qualifies(&self, counts: &[u16]) -> bool {
        counts.first().is_none_or(|&count| count >= self.first_bag_minimum())
    }

    fn bag_minimum(&self) -> u16 {
        let basis_points = (self.config.bag_min_success_percent * 100.0).round() as u32;
        (basis_points * P7_COUNT as u32).div_ceil(10_000).max(1) as u16
    }

    fn bag_minimum_for_spins(&self, actual_bag_spins: u8) -> u16 {
        if actual_bag_spins < self.config.risk_min_spins_per_bag {
            P7_COUNT as u16
        } else { self.bag_minimum() }
    }

    fn next_bag_minimum(&self, counts: &[u16]) -> u16 {
        if counts.is_empty() {
            self.first_bag_minimum().max(if self.config.ignore_first_bag_probability { 1 }
                else { required_success(&[], self.config.min_success_percent) })
        } else {
            self.bag_minimum().max(required_success(self.counted_counts(counts), self.config.min_success_percent))
        }
    }

    /// Generation has no fixed future gain yet, so next_bag_minimum is its
    /// optimistic floor. A completed plan must meet this actual-gain floor.
    /// Forecasts and clearing slots never qualify a low-gain bag for risk.
    fn plan_path_qualifies(&self, counts: &[u16], plan: &Plan) -> bool {
        self.path_qualifies(counts)
            && (plan.start_pieces == 0 || counts.last().is_some_and(|&count| {
                count >= self.bag_minimum_for_spins(
                    plan.expected_spins.saturating_sub(plan.start_spins))
            }))
    }

    fn path_qualifies(&self, counts: &[u16]) -> bool {
        self.start_qualifies(counts)
            && counts.iter().skip(1).all(|&count| count >= self.bag_minimum())
            && threshold_met(self.counted_counts(counts), self.config.min_success_percent)
    }

    /// Raw cross-path maxima carry no probability resource. Keep them as
    /// progress, but do not let a riskier path prune a safer one.
    fn selective_spin_reference(&self, depth: usize) -> u8 {
        if depth > 1 && self.config.min_success_percent == 100.0 { self.best_spins_by_bag[depth] } else { 0 }
    }

    fn node_scheduling_rank(&self, index: usize, quality: f64) -> f64 {
        let node = &self.tree[index];
        let quality = quality + self.probability_pace_for_counts(&node.counts).quality_adjustment;
        probability_scheduling_rank(node.spins, node.pieces, quality, self.path_probability(&node.counts))
    }

    fn node_rounds(&self, index: usize, stale: u32) -> u32 {
        let node = &self.tree[index];
        if self.node_quality_dominated(index) { return 1; }
        productive_rounds(node.spins, node.pieces, self.last_bag_spins(index),
            node.future.spins, if index == 0 { 0 } else { stale }, self.goal,
            self.preferred_pace_budget())
    }

    /// Peers share only verified performance benchmarks. These selective hints
    /// never manufacture local progress, a board, or a supply certificate.
    pub fn observe_benchmarks(&mut self, values: &[u8]) {
        if self.config.min_success_percent < 100.0 { return; }
        let mut changed = false;
        for (depth, &spins) in values.iter().take(self.best_spins_by_bag.len()).enumerate().skip(2) {
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
            .max_by(|(_, (a_parent, a)), (_, (b_parent, b))| {
                let priority = |parent: usize, plan: &Plan| {
                    // The completed boundary score already distinguishes
                    // central working pockets from charged cavity history.
                    // Reuse it instead of reinstating raw-hole penalties or
                    // flooding each waiting board during every comparison.
                    // An unverified pending bag spends no known probability
                    // and earns no extra elapsed-budget credit.
                    let quality = plan.quality
                        + self.probability_pace_for_counts(&self.tree[parent].counts).quality_adjustment;
                    probability_scheduling_rank(plan.expected_spins,
                        plan.start_pieces + plan.targets.len() as u16, quality,
                        self.path_probability(&self.tree[parent].counts))
                };
                a.terminal.cmp(&b.terminal)
                    .then_with(|| priority(*a_parent, a).total_cmp(&priority(*b_parent, b)))
                    .then_with(|| b.id.cmp(&a.id))
            }).map(|(index, _)| index)
    }

    fn explore_pace_probe(&mut self) -> bool {
        if self.active_budget <= self.preferred_pace_budget()
            || self.dispatches.saturating_sub(self.last_pace_probe_dispatch) < LONG_HORIZON_PROBE_INTERVAL
        { return false; }
        let parent = (1..self.tree.len())
            .filter(|&i| self.tree[i].pieces > 7
                && (self.tree[i].attempts > 0 || self.behind_preferred_pace(i))
                && self.dispatches.saturating_sub(self.tree[i].last_selected) >= EXPLORATION_INTERVAL
                && self.can_expand(i) && self.node_selective_reason(i).is_none()
                && !self.node_quality_dominated(i))
            .max_by(|&a, &b| {
                let priority = |i: usize| {
                    let n = &self.tree[i];
                    self.node_scheduling_rank(i, n.quality)
                        + 0.002 * self.dispatches.saturating_sub(n.last_selected).min(128) as f64
                        - 0.05 * f64::from(n.attempts)
                };
                self.behind_preferred_pace(a).cmp(&self.behind_preferred_pace(b))
                    .then_with(|| priority(a).total_cmp(&priority(b)))
            });
        let Some(parent) = parent else { return false; };
        self.last_pace_probe_dispatch = self.dispatches;
        self.last_exploration_dispatch = self.dispatches;
        self.focus = None;
        self.tree[parent].allowance = self.tree[parent].attempts + 1;
        self.tree[parent].lease_epoch = self.benchmark_epoch;
        self.begin_generation_with_pace(parent, true);
        true
    }

    fn explore_once(&mut self) -> bool {
        if self.explore_pace_probe() { return true; }
        if self.dispatches.saturating_sub(self.last_exploration_dispatch) < EXPLORATION_INTERVAL {
            return false;
        }
        self.last_exploration_dispatch = self.dispatches;
        let phase = self.exploration_phase;
        self.exploration_phase = (phase + 1) % 4;
        if phase == 0 {
            if let Some(index) = self.best_waiting_plan(Some(0)) {
                let (parent, plan) = self.waiting_plans.remove(index);
                self.focus = None;
                self.begin_verification(parent, plan);
                return true;
            }
        }
        let eligible = |i: usize| self.regular_node_eligible(i);
        // One bounded exploration phase explicitly advances a deep certified
        // frontier. The old oldest-node fallback could select roots every time.
        let deepest = (phase == 2).then(|| (1..self.tree.len())
            .filter(|&i| self.tree[i].pieces >= 14 && eligible(i)
                && (self.tree[i].attempts.saturating_sub(self.tree[i].last_improved_attempt) < 4
                    || self.dispatches.saturating_sub(self.tree[i].last_selected) >= EXPLORATION_INTERVAL * 4))
            .max_by(|&a, &b| self.tree[a].pieces.cmp(&self.tree[b].pieces)
                .then_with(|| self.tree[b].last_selected.cmp(&self.tree[a].last_selected))
                .then_with(|| self.node_scheduling_rank(a, self.tree[a].quality)
                    .total_cmp(&self.node_scheduling_rank(b, self.tree[b].quality))))).flatten();
        if let Some(parent) = deepest {
            if let Some(index) = self.best_waiting_plan(Some(parent)) {
                let (_, plan) = self.waiting_plans.remove(index);
                self.focus = Some(parent);
                self.focus_since_dispatch = self.dispatches;
                self.begin_verification(parent, plan);
                return true;
            }
        }
        let parent = if phase == 0 && eligible(0) { Some(0) }
        else if phase == 1 {
            (1..self.tree.len()).filter(|&i| eligible(i) && self.tree[i].pieces == 7
                && self.tree[i].attempts == 0)
                .max_by(|&a, &b| {
                    if self.config.min_success_percent == 100.0 {
                        self.tree[a].quality.total_cmp(&self.tree[b].quality)
                    } else {
                        self.node_scheduling_rank(a, self.tree[a].quality)
                            .total_cmp(&self.node_scheduling_rank(b, self.tree[b].quality))
                    }
                })
        } else { deepest };
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
        self.begin_generation_with_pace(parent, false);
    }

    fn begin_generation_with_pace(&mut self, parent: usize, pace_fallback: bool) {
        let attempt = self.tree[parent].attempts;
        let minimum = self.next_bag_minimum(&self.tree[parent].counts);
        let stale = attempt.saturating_sub(self.tree[parent].last_improved_attempt);
        // Failed attempts alone do not earn a wider retry. Fresh productive
        // boundaries keep the full width; stale ones retain only a narrow probe.
        let narrow = pace_fallback || self.node_quality_dominated(parent)
            || (parent != 0 && stale >= 4);
        let productive = self.last_bag_spins(parent) >= 2
            || self.tree[parent].best_extension_spins >= 2;
        let stage = if narrow { 0 }
            else if parent == 0 { (1 + attempt / 4).min(2) }
            else { 1 + u32::from(productive) };
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
            self.selective_spin_reference((self.tree[parent].pieces as usize / 7) + 1);
        generation.bag_alpha = if parent != 0 && self.config.min_success_percent == 100.0 {
            self.best_quality_by_bag[(self.tree[parent].pieces as usize / 7) + 1]
                .iter().enumerate().filter(|(_, quality)| quality.is_finite())
                .map(|(spins, &quality)| spin_rank(spins as u8, quality))
                .fold(f64::NEG_INFINITY, f64::max)
        } else { f64::NEG_INFINITY };
        generation.hard_piece_cap = self.cost_cap();
        generation.set_prune_budget(self.goal, self.proposal_budget(pace_fallback));
        generation.pace_fallback = pace_fallback;
        // Quiet placements must remain supply-safe even when start probability
        // is excluded and its numeric floor is disabled.
        generation.first_bag_symmetry = parent == 0;
        generation.relaxed_setup = parent == 0;
        if parent == 0 {
            // Later unknown bag chances are never assumed to improve the
            // parent's probability. This remains an optimistic necessary floor.
            generation.continuation_minimum_success = self.bag_minimum()
                .max(required_success(&[], self.config.min_success_percent));
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
            .filter(|&i| self.regular_node_eligible(i))
            .map(|i| self.tree[i].revisit_after)
            .min()?;
        self.schedule_epoch = self.schedule_epoch.max(earliest);
        (0..self.tree.len())
            .filter(|&i| {
                self.regular_node_eligible(i)
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
                    // paid tracks only the initially charged cavity ledger;
                    // accessible working pockets cannot earn repayment credit.
                    let speculative = node.future.score.max(0.0)
                        + f64::from(node.future.paid) * 12.0;
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
                    self.node_scheduling_rank(i, quality)
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
        self.plan_path_qualifies(&counts, &job.plan)
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
                forecast_spins: seed.future.planned_spins,
                forecast_spins_by_bag: seed.future.displayed_bag_spins(),
                forecast_placements: seed.future.placements.len() as u8,
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
                assume_reflection: (job.start_pieces == 0 && job.start == Board::default())
                    || parent.certificates.first().is_some_and(|c| c.first_bag_symmetry_assumed),
                quality: seed.quality,
                pace_fallback: job.pace_fallback,
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
            job.symmetry.is_some() && (usize::from(job.coverage_success()) > job.success
                || job.quiet_safe > job.canonical_quiet_safe)
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
                current.summary.success = job.coverage_success();
                current.summary.failed = job.checked as u16 - job.coverage_success();
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
                success: job.coverage_success(),
                failed: job.checked as u16 - job.coverage_success(),
                bag_count: parent.counts.len() as u16 + 1,
                budget_pieces: None,
                cdf: vec![],
                ranking_score: candidate_rank(job.plan.expected_spins, quality,
                    job.plan.start_pieces == 0, job.plan.terminal),
                field_quality: quality,
                forecast_spins: job.plan.future.planned_spins,
                forecast_spins_by_bag: job.plan.future.displayed_bag_spins(),
                forecast_placements: job.plan.future.placements.len() as u8,
                frame_count: frames.len() as u16,
                frames,
            },
            certificates: parent.certificates.clone(),
            pending_plan: Some(job.plan.clone()),
        });
    }

    fn finish_verification(&mut self, job: VerificationJob) {
        debug_assert!(!job.quiet_rejected && job.quiet_safe == P7_COUNT);
        self.verified_candidates += 1;
        let parent = &self.tree[job.parent];
        let mut counts = parent.counts.clone();
        counts.push(job.coverage_success());
        let total_probability = self.path_probability(&counts);
        let symmetry_costs = job.symmetry.as_ref().map(|orbits| orbits.costs().to_vec());
        let symmetry_assumed = job.plan.start_pieces > 0 && job.plan.assume_reflection
            || symmetry_costs.as_ref().is_some_and(|costs| *costs != job.costs)
            || (job.symmetry.is_some() && job.quiet_safe > job.canonical_quiet_safe);
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
                } else { self.plan_path_qualifies(&at_budget, &job.plan) };
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
            self.plan_path_qualifies(&counts, &job.plan)
        };
        let desired = budget.map(|n| n - parent.pieces).unwrap_or(7) as u8;
        // The selected witness must have actually been queried and have
        // the orbit minimum cost. A skipped partner remains zero, not a path.
        let canonical_choice = |rank: usize| job.symmetry.is_none()
            || (job.costs[rank] > 0 && job.costs[rank] == effective_costs[rank]);
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
            canonical_checked: job.canonical_checked as u16,
            quiet_no_hold_verified: true,
            quiet_safe_supplies: job.quiet_safe as u16,
            canonical_quiet_safe: job.canonical_quiet_safe as u16,
            hold_policy: CLEAR_ONLY_HOLD_POLICY,
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
            ranking_score: candidate_rank(last.spins, quality, parent.pieces == 0, job.plan.terminal),
            field_quality: quality,
            forecast_spins: job.plan.future.planned_spins,
            forecast_spins_by_bag: job.plan.future.displayed_bag_spins(),
            forecast_placements: job.plan.future.placements.len() as u8,
            frame_count: frames.len() as u16,
            frames: frames.clone(),
        };
        let archived = Archived {
            summary: summary.clone(),
            certificates: certificates.clone(),
            pending_plan: None,
        };
        self.current = Some(archived.clone());
        Self::retain_bag_candidate(&mut self.best_by_bag, &archived);
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
                if improved && counts.len() > 1 && self.config.min_success_percent == 100.0 {
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
                        self.preferred_pace_budget(),
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
                    // A successful fallback is archived, but cannot start a
                    // large depth-first lease until its actual pace recovers.
                    self.focus = self.regular_node_eligible(child).then_some(child);
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
        foundations.sort_by(|&a, &b| {
            if self.config.min_success_percent == 100.0 {
                self.tree[b].quality.total_cmp(&self.tree[a].quality)
            } else {
                self.node_scheduling_rank(b, self.tree[b].quality)
                    .total_cmp(&self.node_scheduling_rank(a, self.tree[a].quality))
            }
        });
        foundations.truncate((self.config.beam_width / 4).clamp(2, 8));
        let mut untried_deep: Vec<_> = (1..self.tree.len()).filter(|&i| {
            self.tree[i].pieces >= 14 && self.tree[i].attempts == 0
                && self.regular_node_eligible(i)
        }).collect();
        untried_deep.sort_by(|&a, &b| self.tree[b].pieces.cmp(&self.tree[a].pieces)
            .then_with(|| self.node_scheduling_rank(b, self.tree[b].quality)
                .total_cmp(&self.node_scheduling_rank(a, self.tree[a].quality))));
        untried_deep.truncate((self.config.beam_width / 16).clamp(2, 4));
        let replaceable = |i: usize| {
            i != parent && self.focus != Some(i) && self.tree[i].children.is_empty()
                && !self.waiting_plans.iter().any(|(pending, _)| *pending == i)
        };
        let victim = (1..self.tree.len())
            .filter(|i| {
                replaceable(*i) && !foundations.contains(i) && !untried_deep.contains(i)
            })
            .min_by(|a, b| {
                let score = |i: usize| {
                    let n = &self.tree[i];
                    self.node_scheduling_rank(i, n.quality)
                };
                let eligible = |i| self.can_expand(i) && self.node_selective_reason(i).is_none();
                eligible(*a).cmp(&eligible(*b)).then_with(|| score(*a).total_cmp(&score(*b)))
            }).or_else(|| foundations.iter().rev().copied().find(|&i| replaceable(i)))
            .or_else(|| untried_deep.iter().rev().copied().find(|&i| replaceable(i)));
        let Some(index) = victim else { return None };
        if let Some(old_parent) = self.tree[index].parent {
            self.tree[old_parent].children.retain(|i| *i != index);
        }
        self.tree[index] = node;
        self.tree[parent].children.push(index);
        Some(index)
    }

    /// Store display records before live-tree pruning or replacement. Each
    /// entry owns one unchanged certificate chain; alternatives are not merged.
    fn retain_bag_candidate(best_by_bag: &mut Vec<Vec<Archived>>, candidate: &Archived) {
        let summary = &candidate.summary;
        if !summary.verified || !summary.qualified || summary.complete
            || summary.bag_count == 0
            || summary.placed != summary.bag_count * 7
            || candidate.certificates.len() != usize::from(summary.bag_count)
        {
            return;
        }
        let depth = usize::from(summary.bag_count);
        best_by_bag.resize_with(best_by_bag.len().max(depth + 1), Vec::new);
        let records = &mut best_by_bag[depth];
        if let Some(index) = records.iter().position(|prior| prior.summary.id == summary.id) {
            if better_partial(&records[index].summary, summary) { return; }
            records[index] = candidate.clone();
        } else if records.len() < 2 {
            records.push(candidate.clone());
        } else if better_partial(summary, &records[1].summary) {
            records[1] = candidate.clone();
        } else {
            return;
        }
        if records.len() == 2 && better_partial(&records[1].summary, &records[0].summary) {
            records.swap(0, 1);
        }
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
                .map(|a| a.for_snapshot(include_frames)),
            best: self
                .best
                .as_ref()
                .map(|a| a.for_snapshot(include_frames)),
            best_partial: self
                .best_partial
                .as_ref()
                .map(|a| a.for_snapshot(include_frames)),
            best_by_bag: self.best_by_bag.iter().enumerate().filter_map(|(depth, records)| {
                records.first().map(|best| BagCandidatesSnapshot {
                    bag_count: depth as u16,
                    best: best.for_snapshot(false),
                    runner_up: records.get(1).map(|a| a.for_snapshot(false)),
                })
            }).collect(),
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
            _ => selection.strip_prefix("candidate:").and_then(|id| {
                self.best_by_bag.iter().flatten().find(|record| record.summary.id == id)
                    .or_else(|| self.best_partial.as_ref().filter(|record| record.summary.id == id))
                    .or_else(|| self.best.as_ref().filter(|record| record.summary.id == id))
            }),
        })
    }

    fn export_archive(&self, archive: Option<&Archived>) -> String {
        let certificates: Vec<_> = archive
            .map(|a| a.certificates.iter().map(|c| c.as_ref()).collect())
            .unwrap_or_default();
        let spin_preparation = serde_json::json!({
                "preparationEvidence": "BestReachableFirstSpinOrSmallConcretePreparation;NoAlternativeSum",
                "preparationReadyCap": CONTINUATION_READY_CAP,
                "preparationGoalCap": CONTINUATION_PREPARED_CAP,
                "independenceEvidence": "BestDisjointRecipePair;BothOrdersSingleOnly;SameEndBoard;NotP7",
                "independenceCap": CONTINUATION_INDEPENDENCE_CAP,
                "independencePairLimit": CONTINUATION_PAIR_LIMIT,
                "independenceTargetQueryCap": CONTINUATION_PAIR_LIMIT * 6,
        });
        let mut selective_search = serde_json::json!({
                "spinDelta": SPIN_DELTA, "residueMarginCells": RESIDUE_MARGIN,
                "paceNearSlack": PACE_NEAR_SLACK, "paceDistanceSlack": PACE_DISTANCE_SLACK,
                "paceModel": "LowerPreLoopSaoSarcelle;SaoFromBag10;17BagHorizonInterpolation;ReferenceGoal40",
                "paceReferenceCumulativeSpins": PACE_REFERENCE,
                "paceIncludesSetup": true,
                "qualityResearchInterval": 0, "qualityPruned": self.quality_pruned,
                "minoBoundPruned": self.mino_bound_pruned,
                "minoSelectivePruned": self.mino_selective_pruned,
                "minoCapacityPruned": self.mino_capacity_pruned,
                "minoBoundMeaning": "NecessaryWholeBagCostOrQuietFirstAvailabilityOrKnownAllQuietTwoPrefixGeometry;PossibleTerminalPrefixExempt",
                "minoSelectiveMeaning": "optimisticBagEndPaceAndRank;sameParentAndNextPieceSpinQualityCapacitySupplyWindow",
                "minoCapacityMeaning": "cannotEnterCurrentBoundedProposalBucket;notGlobalImpossibility",
                "minoPruning": "rowDeficitUpper;wholeBagPaceUpper;samePieceSiblingWindow;boundedBeamAdmission",
                "normalGenerationRounds": [3, 8],
                "staleGenerationRounds": 1,
                "staleAfterUnimprovedAttempts": 4,
                "generationInventory": "AllSevenFromBagStart;principalContinuationOrdersOnly",
        });
        // Keep each export metadata object below the macro recursion limit.
        let ordering_metadata = serde_json::json!({
                "sourceBaseline": "af8dac63a9a9155fc28e480246fd617bc855bfb1",
                "continuationBags": CONTINUATION_BAGS,
                "continuationSpinTarget": CONTINUATION_SPINS,
                "continuationAlternatives": CONTINUATION_ALTERNATIVES,
                "continuationMeaning": "CoherentGeometricPathsOnly;FirstBagGainAndDebtSeparate;NotP7",
                "continuationTaskLimit": CONTINUATION_TASKS,
                "continuationWorkCap": if self.config.search_profile == 3 { 352 } else { 320 },
                "spinPreparation": spin_preparation,
                "siblingWindow": "SameParentNextPiece;noLessActualOrOptimisticSpins;supplyMaskSuperset;samePursuit",
                "equalSpinSiblingQualityMargin": 24,
                "proposalScope": "FastSurfaceAndWitnessedSingles;periodicBroadSonicLocks;notExhaustive",
                "schedulerQuality": "rank=scale*(productiveSpins+0.2*Q/(120+abs(Q)));productiveSpins=0ForFirstBagElseActualSpins;QIncludesVerifiedLogBudgetPace;risk=(rank+0.2*scale)*VerifiedPrefixProbability-0.2*scale;scale=7/productiveLocks;ExactSelectedSchedulerAtP1",
                "qualityReferencesSurviveBudgetWidening": true,
                "crossPathSpinQualityCutsEnabled": self.config.min_success_percent == 100.0,
                "rawPeerBenchmarksEnabled": self.config.min_success_percent == 100.0,
                "qualityPreparationRounds": 1,
                "preferredPaceBudget": self.preferred_pace_budget(),
                "longHorizonProbeInterval": LONG_HORIZON_PROBE_INTERVAL,
                "activeBudget": self.active_budget,
                "budgetStaleRounds": BUDGET_STALE_ROUNDS,
                "explorationInterval": EXPLORATION_INTERVAL,
                "cutsAreImpossibilityProofs": false
        });
        if let serde_json::Value::Object(extra) = ordering_metadata {
            selective_search.as_object_mut().expect("object literal").extend(extra);
        }
        selective_search["foundationPolicy"] = serde_json::json!({
            "ranking": "AllNonterminalFirstBagPrefixesAndBoundariesUseQualityBeforeActualSpins",
            "terrainWeightsByLock": [0.06, 0.06, 0.06, 0.12, 0.18, 0.24, 0.30],
            "spinPeerCutsFromBag": 2,
            "terminalPriorityPreserved": true,
            "twoPrefixGeometry": "FullKnownZeroClearTilingOnly;BothSpawns;OneOrientationPerOrderedPair;NotInputProof"
        });
        selective_search["continuationDiversity"] = serde_json::json!({
            "order": "InterleaveFinisherTypesAndTargetPositions",
            "quietPreparationLimit": 3,
            "recipeLimit": CONTINUATION_RECIPES,
            "cavityPreparation": "OneGoalFreeStepPerSegment;ActualZeroLinePlacement;NoExtraInventory",
            "deepExploration": "OneOfFourExplorationPhases;EligibleCertifiedDepthAtLeastTwoBags"
        });
        selective_search["fieldModel"] = serde_json::json!(
            "ConnectedCavityResidual;ImmediateWitness;WitnessedOutwardBanks;StrongPreparedAndGoalRelativeRoof;SymmetricGoalWorkspace;StrongCurledWorkspaceRoof;StraightWorkspaceWindowCost;AllAdjacentHeightDependence;SuspendedRoofStepRelief;LowOneCellNotch;OneMinoRoofOrLevelRepair"
        );
        serde_json::json!({
            "format": "neyra", "schemaVersion": 3, "engineVersion": env!("CARGO_PKG_VERSION"),
            "rules": { "width": 10, "height": HEIGHT, "rotation": "SRS+", "spin": "All-mini+", "oSpin": false, "ceiling": "ActualCellDeath24", "movement": "SonicDropOnly", "supply": "UniformP7ClearOnlyHold", "runtimeReplayVerified": false },
            "config": self.config, "snapshot": self.snapshot(),
            "learnedOrdering": { "enabled": false,
                "scope": "Disabled;SpinOnlyModelIsIndependentOfSearch",
                "maxAbsoluteAdjustment": 0.0 },
            "candidate": archive.map(|a| a.for_snapshot(true)), "bags": certificates,
            "pendingPlan": archive.and_then(|a| a.pending_plan.as_ref()),
            "successBitsEncoding": "hex-u64-lsb0; word=floor(P7rank/64), bit=P7rank%64",
            "interchange": false, "globalOptimalityProven": false,
            "probabilityScope": if self.config.ignore_first_bag_probability { "AfterFirstBag" } else { "AllBags" },
            "quietPlacementPolicy": "EverySupplyHasStrategyWithoutQuietBlockageUntilSuccessOrForcedClearBlockage",
            "holdPolicy": CLEAR_ONLY_HOLD_POLICY,
            "quietPolicyRejected": self.quiet_policy_rejected,
            "firstBagSymmetryPolicy": "AssumeJLAndSZSupplyReflection;resolveQueuePairsAtMinimumCost;useVerifiedOrientation;fixedOrientationAfterStart",
            "checkedMeaning": "checked=resolvedModelSupplies;canonicalChecked=actuallyEvaluatedCanonicalSupplies",
            "firstBagSymmetryAssumed": archive.is_some_and(|a| a.summary.first_bag_symmetry_assumed),
            "mirroredInputsVerified": false,
            "costsMeaning": "0=canonicalQueueNotEvaluated;1..7=shortestInCanonicalLanguage;255=noPathInCanonicalLanguage;symmetryCosts=exactMinimumInAssumedReflectionModel",
            "displayPartialRanking": "ProductiveStageOverFoundation;oneBag:FieldQuality,StartProbability,CountedProbability,Spins,FewerLocks;later:spins*CountedProbability/max(7,placed),Probability,MoreBags,Spins,FieldQuality,FewerLocks",
            "bagRecords": "TwoQualifiedVerifiedWholeBagPartialsPerDepth;SamePartialComparator;DistinctPlanIds;IndependentCertificateChains;CandidateIdSelection",
            "bagProbabilityScope": "IndependentFloorAfterFirstBag;BelowActualBagSpinThresholdRequires100Percent;FirstBagUsesItsOwnFloor;DisplayedBagsUseFullCertifiedCoverage",
            "probabilityBudget": archive.filter(|a| a.summary.verified).map(|a| {
                let chance = a.summary.probability;
                let floor = self.config.min_success_percent / 100.0;
                serde_json::json!({
                    "overallFloor": floor,
                    "perBagFloor": self.config.bag_min_success_percent / 100.0,
                    "riskMinSpinsPerBag": self.config.risk_min_spins_per_bag,
                    "countedProbability": chance,
                    "requiredFutureProduct": if chance > 0.0 { Some(floor / chance) } else { None },
                    "spentLogBudgetFraction": if floor < 1.0 && chance > 0.0 { Some(chance.ln() / floor.ln()) }
                        else if chance == 1.0 { Some(0.0) } else { None },
                    "pace": self.probability_pace_for_counts(&a.certificates.iter()
                        .map(|certificate| certificate.effective_success).collect::<Vec<_>>()),
                    "paceMeaning": "UniformLogBudgetToConfiguredHorizon;BoundedSchedulingOnly;ProjectionNotCertificate",
                    "meaning": "AlreadyVerifiedProbabilityResource;FutureProductIsRequirementNotPrediction"
                })
            }),
            "selectiveSearch": selective_search
        }).to_string()
    }
}

fn better_partial(a: &CandidateSummary, b: &CandidateSummary) -> bool {
    // A foundation is selected by its workspace and concrete future evidence.
    // Once a certified follow-up bag exists, display productive-stage records.
    if a.bag_count == 1 && b.bag_count > 1 { return false; }
    if b.bag_count == 1 && a.bag_count > 1 { return true; }
    if a.bag_count == 1 && b.bag_count == 1 {
        let quality = |c: &CandidateSummary| if c.field_quality.is_finite() {
            c.field_quality
        } else { f64::NEG_INFINITY };
        return quality(a).partial_cmp(&quality(b)).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| bounded_probability(a.first_bag_probability.unwrap_or(0.0))
                .partial_cmp(&bounded_probability(b.first_bag_probability.unwrap_or(0.0)))
                .unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| bounded_probability(a.probability).partial_cmp(&bounded_probability(b.probability))
                .unwrap_or(std::cmp::Ordering::Equal))
            .then_with(|| a.spins.cmp(&b.spins))
            .then_with(|| b.placed.cmp(&a.placed)).is_gt();
    }
    // Actual spin efficiency is discounted by already consumed probability.
    // Include the setup bag; equal efficiency preserves the larger chance
    // budget before depth/terrain ties. This is an ordering score, not an
    // expected completion-time or expected partial-spin certificate.
    let a_chance = bounded_probability(a.probability);
    let b_chance = bounded_probability(b.probability);
    let lhs = (f64::from(a.spins) * a_chance) * f64::from(b.placed.max(7));
    let rhs = (f64::from(b.spins) * b_chance) * f64::from(a.placed.max(7));
    lhs.total_cmp(&rhs)
        .then_with(|| a_chance.total_cmp(&b_chance))
        .then_with(|| a.bag_count.cmp(&b.bag_count))
        .then_with(|| a.spins.cmp(&b.spins))
        .then_with(|| {
            if a.bag_count == 1 && b.bag_count == 1 {
                a.first_bag_probability.unwrap_or(0.0).partial_cmp(&b.first_bag_probability.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            } else { std::cmp::Ordering::Equal }
        })
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
            field_quality: 0.0, forecast_spins: 0,
            forecast_spins_by_bag: [0; CONTINUATION_BAGS], forecast_placements: 0,
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
            pace_fallback: false,
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
        // Only S is declared. The first supply is active; an undeclared quiet
        // piece cannot be held merely to peek at S. Other queues are clearing
        // supply failures, not failures of a declared ordinary placement.
        assert!(!job.quiet_rejected);
        assert_eq!(job.quiet_safe, 5040);
        assert_eq!(job.success, 720);
        assert_eq!(job.costs.iter().filter(|cost| **cost == 1).count(), 720);
        assert_eq!(job.costs.iter().filter(|cost| **cost == 255).count(), 4320);
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
            pace_fallback: false,
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
