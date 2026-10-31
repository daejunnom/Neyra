//! Anytime de-novo fixed-bag construction search.
//! Geometry proposals are bounded and heuristic; their P7 certificates are exact.
//! Clearra's separation of logical-row operation language and supply language is
//! retained here, with a 24-row SonicDropOnly physics backend.

use crate::physics::{Board, Input, Piece, Placement, Pose, HEIGHT};
use crate::supply::{
    decode_order, minimum_finish, permutation, LanguageNode, ALL_PIECES, P7_COUNT,
};
use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

const TARGET_SPINS: u8 = 40;
const GRAPH_LIMIT: usize = 4096;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    #[serde(alias = "minSuccessPercent")]
    pub min_success_percent: f64,
    #[serde(alias = "pieceGoal")]
    pub piece_goal: u16,
    #[serde(alias = "maxPieces")]
    pub max_pieces: u16,
    #[serde(alias = "budgetStart")]
    pub budget_start: u16,
    pub seed: u64,
    #[serde(alias = "beamWidth")]
    pub beam_width: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            min_success_percent: 95.0,
            piece_goal: 112,
            max_pieces: 140,
            budget_start: 0,
            seed: 1,
            beam_width: 64,
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
        if self.max_pieces == 0 || self.max_pieces > 700 || self.piece_goal == 0 {
            return Err("미노 상한은 1–700, 목표는 1 이상이어야 합니다.".into());
        }
        if self.budget_start > self.max_pieces || !(4..=256).contains(&self.beam_width) {
            return Err("탐색 시작 예산 또는 탐색 폭을 확인해 주세요.".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DisplayFrame {
    pub rows: [u16; HEIGHT],
    pub piece: String,
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
    pub complete: bool,
    pub qualified: bool,
    pub verified: bool,
    pub checked: u16,
    pub success: u16,
    pub failed: u16,
    pub bag_count: u16,
    pub budget_pieces: Option<u16>,
    pub cdf: Vec<CdfPoint>,
    pub frames: Vec<DisplayFrame>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressSnapshot {
    pub phase: String,
    pub expanded: u64,
    pub candidates: u64,
    pub verified_candidates: u64,
    pub deepest_spins: u8,
    pub bags: u16,
    pub checked: u16,
    pub success: u16,
    pub failed: u16,
    pub frontier: usize,
    pub finished: bool,
    pub current: Option<CandidateSummary>,
    pub best: Option<CandidateSummary>,
}

#[derive(Clone, Debug, Serialize)]
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
    frames: Vec<DisplayFrame>,
    score: f64,
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
    locks: HashMap<(Board, Piece), Rc<Vec<Placement>>>,
    targets: HashMap<(Board, Pose, bool), Option<Placement>>,
}

impl PhysicsCache {
    fn locks(&mut self, board: Board, piece: Piece) -> Rc<Vec<Placement>> {
        if let Some(locks) = self.locks.get(&(board, piece)) {
            return Rc::clone(locks);
        }
        if self.locks.len() >= 512 {
            self.locks.clear();
        }
        let locks = Rc::new(board.search_locks(piece));
        self.locks.insert((board, piece), Rc::clone(&locks));
        locks
    }

    fn target(&mut self, board: Board, pose: Pose, spin: bool) -> Option<Placement> {
        // The search list has only O's orientation-zero representative.
        // Preserve exact-pose queries in other orientations through reverse
        // reachability rather than treating their missing cache entry as failure.
        if pose.piece != Piece::O || pose.rotation == 0 {
            if let Some(locks) = self.locks.get(&(board, pose.piece)) {
                return locks
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
    success: u16,
    checked: u16,
    /// Stable lexicographic P7 rank, I J L O S T Z.
    #[serde(serialize_with = "serialize_bit_words")]
    success_bits: Vec<u64>,
    /// 0=unknown, 1..7=exact shortest accepting length, 255=proven impossible.
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
}

impl VerificationJob {
    fn new(parent: usize, plan: Plan) -> Self {
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
        }
    }

    fn advance(&mut self, goal: u8, cache: &mut PhysicsCache) {
        if !self.graph.complete() {
            self.graph.expand_one(&self.plan, goal, cache);
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
        }
        self.checked = end;
    }
}

struct GenerationJob {
    parent: usize,
    start: Board,
    start_spins: u8,
    start_pieces: u16,
    frontier: Vec<SeedState>,
    next: Vec<SeedState>,
    complete: Vec<SeedState>,
    state_index: usize,
    piece_index: usize,
    width: usize,
    depth: u8,
    setup: bool,
}

impl GenerationJob {
    fn new(parent: usize, node: &TreeNode, width: usize, setup: bool) -> Self {
        Self {
            parent,
            start: node.board,
            start_spins: node.spins,
            start_pieces: node.pieces,
            frontier: vec![SeedState {
                board: node.board,
                spins: node.spins,
                used: 0,
                deleted: 0,
                targets: vec![],
                frames: vec![],
                score: 0.0,
            }],
            next: vec![],
            complete: vec![],
            state_index: 0,
            piece_index: 0,
            width: width.clamp(4, 32),
            depth: 0,
            setup,
        }
    }

    fn advance(&mut self, rng: &mut Random, goal: u8, cache: &mut PhysicsCache) -> u64 {
        if self.state_index >= self.frontier.len() {
            return 0;
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
        let locks = cache.locks(state.board, piece);
        let mut expanded = 0;
        for placement in locks.iter() {
            expanded += 1;
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
            let Some(deleted) = deleted_after(state.deleted, result.cleared_rows) else {
                continue;
            };
            let spins = state.spins + result.lines;
            let mut targets = state.targets.clone();
            targets.push(target);
            let mut frames = state.frames.clone();
            frames.push(DisplayFrame {
                rows: result.board.rows,
                piece: piece_name(piece),
                spins,
                placed: self.start_pieces + targets.len() as u16,
                lines: result.lines,
                spin: result.spin,
            });
            let score = if self.setup {
                setup_score(&self.start, &result.board, &targets) + rng.unit() * 6.0
            } else {
                board_score(&result.board)
                    + (spins - self.start_spins) as f64 * 160.0
                    + setup_score(&self.start, &result.board, &targets) * 0.35
                    + rng.unit() * 3.5
            };
            let next = SeedState {
                board: result.board,
                spins,
                used: state.used | (1 << (piece as u8)),
                deleted,
                targets,
                frames,
                score,
            };
            if spins >= goal || next.used == ALL_PIECES {
                insert_seed(&mut self.complete, next, 8);
            } else {
                insert_seed(&mut self.next, next, self.width);
            }
        }
        expanded
    }

    fn next_layer(&mut self) -> bool {
        if self.state_index < self.frontier.len() {
            return false;
        }
        self.frontier = std::mem::take(&mut self.next);
        self.state_index = 0;
        self.piece_index = 0;
        self.depth += 1;
        self.frontier.is_empty() || self.depth >= 7
    }

    fn display_seed(&self) -> Option<&SeedState> {
        self.complete
            .first()
            .or_else(|| self.next.first())
            .or_else(|| self.frontier.first())
    }
}

fn insert_seed(list: &mut Vec<SeedState>, seed: SeedState, limit: usize) {
    // This is a bounded proposal beam, not an exact transposition proof. Retain
    // inventories and clear histories, and reserve distinct spin-count strata.
    if let Some(index) = list.iter().position(|s| {
        s.board == seed.board
            && s.used == seed.used
            && s.deleted == seed.deleted
            && s.spins == seed.spins
    }) {
        if list[index].score >= seed.score {
            return;
        }
        list.remove(index);
    }
    let at = list
        .iter()
        .position(|s| s.score < seed.score)
        .unwrap_or(list.len());
    list.insert(at, seed);
    if list.len() > limit {
        let quota = if limit >= 16 { 2 } else { 1 };
        let victim = (0..list.len())
            .rev()
            .find(|index| {
                list[..*index]
                    .iter()
                    .filter(|s| s.spins == list[*index].spins)
                    .count()
                    >= quota
            })
            .unwrap_or(list.len() - 1);
        list.remove(victim);
    }
}

/// A proposal heuristic, deliberately not an impossibility test.
fn board_score(board: &Board) -> f64 {
    let mut heights = [0_u8; 10];
    let mut holes = 0;
    let mut near_rows = 0.0;
    let mut overhangs: f64 = 0.0;
    for (x, height) in heights.iter_mut().enumerate() {
        for y in (0..HEIGHT).rev() {
            if board.rows[y] & (1 << x) != 0 {
                *height = (y + 1) as u8;
                break;
            }
        }
        for y in 0..*height as usize {
            if board.rows[y] & (1 << x) == 0 {
                holes += 1;
                let sides = (x == 0 || board.rows[y] & (1 << (x - 1)) != 0) as u8
                    + (x == 9 || board.rows[y] & (1 << (x + 1)) != 0) as u8;
                if sides == 2 && y + 1 < HEIGHT && board.rows[y + 1] & (1 << x) != 0 {
                    overhangs += 1.0;
                }
            }
        }
    }
    for row in board.rows {
        let filled = row.count_ones() as f64;
        if filled >= 6.0 {
            near_rows += (filled - 5.0).powi(2) * 1.8;
        }
    }
    let aggregate = heights.iter().map(|h| *h as f64).sum::<f64>();
    let roughness = heights
        .windows(2)
        .map(|v| (v[0] as f64 - v[1] as f64).abs())
        .sum::<f64>();
    near_rows + overhangs.min(6.0) * 2.5
        - holes as f64 * 1.8
        - aggregate * 0.28
        - roughness * 0.14
        - (board.height() as f64).powi(2) * 0.06
}

/// An intentionally conservative *ordering score*, never an eligibility test.
/// Direct-drop interference and missing base support expose fragile setup bags
/// cheaply. Actual rotation, OR support and hold possibilities remain the exact
/// execution verifier's responsibility.
fn setup_score(base: &Board, board: &Board, targets: &[TargetOperation]) -> f64 {
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
    board_score(board) * 0.12
        - unsupported as f64 * 18.0
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
    finished: bool,
    last_checked: u16,
    last_success: u16,
    generation_rounds: u64,
    active_budget: u16,
    physics_cache: PhysicsCache,
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
        };
        let lower_bound = cost_lower_bound(&Board::default(), 0, TARGET_SPINS);
        let active_budget = if config.budget_start == 0 {
            lower_bound
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
            finished: false,
            last_checked: 0,
            last_success: 0,
            generation_rounds: 0,
            active_budget,
            physics_cache: PhysicsCache::default(),
        })
    }

    /// One unit either expands a geometric piece choice, one execution node,
    /// or at most 64 finite supply products. No step consumes an entire search.
    pub fn step(&mut self, max_work: u32) {
        for _ in 0..max_work.clamp(1, 64) {
            if self.finished {
                break;
            }
            if let Some(mut job) = self.verification.take() {
                job.advance(self.goal, &mut self.physics_cache);
                self.last_checked = job.checked as u16;
                self.last_success = job.success as u16;
                self.update_verifying_preview(&job);
                if job.graph.overflow {
                    // Incomplete analysis is not inserted into any failure bitset.
                    self.backpropagate(job.parent, 0.0);
                } else if job.checked == P7_COUNT {
                    self.finish_verification(job);
                } else if !job.plan.terminal && job.checked > 0 && !self.can_still_qualify(&job) {
                    self.backpropagate(job.parent, 0.0);
                } else {
                    self.verification = Some(job);
                }
                continue;
            }
            if let Some((parent, plan)) = self.waiting_plans.pop() {
                self.verification = Some(VerificationJob::new(parent, plan));
                self.last_checked = 0;
                self.last_success = 0;
                continue;
            }
            if let Some(mut job) = self.generation.take() {
                self.expanded += job.advance(&mut self.rng, self.goal, &mut self.physics_cache);
                self.update_generation_preview(&job);
                if job.next_layer() {
                    let parent = job.parent;
                    for state in job.complete.into_iter().rev() {
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
                        };
                        if self.remember_plan(parent, &plan) {
                            self.candidates += 1;
                            self.waiting_plans.push((parent, plan));
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
            let Some(parent) = self.select_node() else {
                self.finished = true;
                break;
            };
            self.tree[parent].visits += 1;
            let setup = self.generation_rounds % 3 != 2;
            self.generation = Some(GenerationJob::new(
                parent,
                &self.tree[parent],
                self.config.beam_width,
                setup,
            ));
            self.generation_rounds += 1;
            // A heuristic miss at a budget is not an impossibility proof. Broaden
            // the scheduling preference periodically; acceptance never uses it.
            if self.generation_rounds % 12 == 0 {
                self.active_budget = self
                    .active_budget
                    .saturating_add(7)
                    .min(self.config.max_pieces);
            }
        }
    }

    fn remember_plan(&mut self, parent: usize, plan: &Plan) -> bool {
        let mut key = Vec::with_capacity(128);
        for row in plan.start.rows {
            key.extend(row.to_le_bytes());
        }
        key.push(plan.start_spins);
        // The same bag plan can qualify after a stronger probability prefix
        // even when it was rejected after a weaker route to the same board.
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
        // Bounded duplicate cache; clearing affects cost, never certificates.
        if self.seen_plans.len() >= 8192 {
            self.seen_plans.clear();
        }
        self.seen_plans.insert(key)
    }

    fn select_node(&mut self) -> Option<usize> {
        let cap = self
            .best
            .as_ref()
            .and_then(|b| b.summary.budget_pieces)
            .unwrap_or(self.config.max_pieces);
        let can_expand = |node: &TreeNode| {
            node.pieces < self.config.max_pieces
                && node.spins < self.goal
                && node.pieces + cost_lower_bound(&node.board, node.spins, self.goal) <= cap
        };
        if !can_expand(&self.tree[0]) {
            return None;
        }
        let mut index = 0;
        for _ in 0..100 {
            let node = &self.tree[index];
            let widening = (2.0 * ((node.visits + 1) as f64).sqrt()).floor() as usize;
            if node.children.is_empty()
                || (node.children.len() < widening.min(12) && self.rng.unit() < 0.35)
                || self.rng.unit() < 0.08
            {
                return Some(index);
            }
            let log = ((node.visits + 2) as f64).ln();
            let choice = node
                .children
                .iter()
                .copied()
                .filter(|child| can_expand(&self.tree[*child]))
                .max_by(|a, b| {
                    let score = |i: usize| {
                        let c = &self.tree[i];
                        let bound = c.pieces + cost_lower_bound(&c.board, c.spins, self.goal);
                        let preference = self
                            .best
                            .as_ref()
                            .and_then(|b| b.summary.budget_pieces)
                            .map(|n| n.saturating_sub(1))
                            .unwrap_or(self.active_budget);
                        let budget_penalty = bound.saturating_sub(preference) as f64 * 0.015;
                        let target_penalty =
                            bound.saturating_sub(self.config.piece_goal) as f64 * 0.005;
                        c.reward / (c.visits.max(1) as f64)
                            + 0.8 * (log / (c.visits + 1) as f64).sqrt()
                            + c.spins as f64 / self.goal as f64 * 0.25
                            - budget_penalty
                            - target_penalty
                    };
                    score(*a).total_cmp(&score(*b))
                });
            if let Some(child) = choice {
                index = child;
            } else {
                return Some(index);
            }
        }
        Some(index)
    }

    fn backpropagate(&mut self, mut index: usize, reward: f64) {
        loop {
            self.tree[index].visits += 1;
            self.tree[index].reward += reward;
            let Some(parent) = self.tree[index].parent else {
                break;
            };
            index = parent;
        }
    }

    fn can_still_qualify(&self, job: &VerificationJob) -> bool {
        let mut counts = self.tree[job.parent].counts.clone();
        counts.push((job.success + P7_COUNT - job.checked) as u16);
        threshold_met(&counts, self.config.min_success_percent)
    }

    fn update_generation_preview(&mut self, job: &GenerationJob) {
        let Some(seed) = job.display_seed() else {
            return;
        };
        self.deepest_spins = self.deepest_spins.max(seed.spins);
        let parent = &self.tree[job.parent];
        let placed = job.start_pieces + seed.targets.len() as u16;
        let should_update = self.current.as_ref().is_none_or(|c| {
            seed.spins > c.summary.spins
                || (seed.spins == c.summary.spins
                    && placed >= c.summary.placed
                    && !c.summary.verified)
        });
        if !should_update {
            return;
        }
        let mut frames = parent.frames.clone();
        frames.extend(seed.frames.clone());
        let upper = probability(&parent.counts);
        self.current = Some(Archived {
            summary: CandidateSummary {
                id: format!("building-{}-{}", job.parent, self.expanded),
                rows: seed.board.rows,
                spins: seed.spins,
                placed,
                probability: 0.0,
                lower_probability: 0.0,
                upper_probability: upper,
                complete: seed.spins >= self.goal,
                qualified: false,
                verified: false,
                checked: 0,
                success: 0,
                failed: 0,
                bag_count: parent.counts.len() as u16 + 1,
                budget_pieces: None,
                cdf: vec![],
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
            }),
        });
    }

    fn update_verifying_preview(&mut self, job: &VerificationJob) {
        self.deepest_spins = self.deepest_spins.max(job.plan.expected_spins);
        let parent = &self.tree[job.parent];
        let id = format!("plan-{}", job.plan.id);
        if self
            .current
            .as_ref()
            .is_some_and(|c| c.summary.spins > job.plan.expected_spins && c.summary.id != id)
        {
            return;
        }
        let prefix = probability(&parent.counts);
        let lower = prefix * job.success as f64 / P7_COUNT as f64;
        let upper = prefix * (job.success + P7_COUNT - job.checked) as f64 / P7_COUNT as f64;
        let mut frames = parent.frames.clone();
        frames.extend(job.plan.seed_frames.clone());
        self.current = Some(Archived {
            summary: CandidateSummary {
                id,
                rows: job.plan.expected.rows,
                spins: job.plan.expected_spins,
                placed: job.plan.start_pieces + job.plan.targets.len() as u16,
                probability: lower,
                lower_probability: lower,
                upper_probability: upper,
                complete: job.plan.terminal,
                qualified: false,
                verified: false,
                checked: job.checked as u16,
                success: job.success as u16,
                failed: (job.checked - job.success) as u16,
                bag_count: parent.counts.len() as u16 + 1,
                budget_pieces: None,
                cdf: vec![],
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
        counts.push(job.success as u16);
        let total_probability = probability(&counts);
        let mut cdf = vec![];
        let mut budget = None;
        if job.plan.terminal {
            for m in 1..=7 {
                let count = job.costs.iter().filter(|c| **c > 0 && **c <= m).count() as u16;
                let mut at_budget = parent.counts.clone();
                at_budget.push(count);
                cdf.push(CdfPoint {
                    pieces: parent.pieces + m as u16,
                    probability: probability(&at_budget),
                });
                if budget.is_none() && threshold_met(&at_budget, self.config.min_success_percent) {
                    budget = Some(parent.pieces + m as u16);
                }
            }
        }
        let qualified = if job.plan.terminal {
            budget.is_some_and(|n| n <= self.config.max_pieces)
        } else {
            threshold_met(&counts, self.config.min_success_percent)
        };
        let desired = budget.map(|n| n - parent.pieces).unwrap_or(7) as u8;
        let representative = job
            .costs
            .iter()
            .position(|c| *c == desired)
            .or_else(|| job.costs.iter().position(|c| *c > 0 && *c <= desired));
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
        let certificate = Rc::new(BagCertificate {
            id: format!("plan-{}", job.plan.id),
            start_rows: job.plan.start.rows,
            end_rows: last.rows,
            start_spins: job.plan.start_spins,
            end_spins: last.spins,
            targets: job.plan.targets.clone(),
            terminal: job.plan.terminal,
            success: job.success as u16,
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
            complete: job.plan.terminal,
            qualified,
            verified: true,
            checked: P7_COUNT as u16,
            success: job.success as u16,
            failed: (P7_COUNT - job.success) as u16,
            bag_count: counts.len() as u16,
            budget_pieces: budget,
            cdf,
            frames: frames.clone(),
        };
        let archived = Archived {
            summary: summary.clone(),
            certificates: certificates.clone(),
            pending_plan: None,
        };
        if self
            .current
            .as_ref()
            .is_none_or(|c| summary.spins >= c.summary.spins)
        {
            self.current = Some(archived.clone());
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
                    && compare_probability(&n.counts, &counts) != std::cmp::Ordering::Less
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
                };
                self.insert_boundary(job.parent, new_node);
            }
        }
        let reward = if job.plan.terminal && qualified {
            2.0
        } else if qualified {
            job.plan.expected_spins as f64 / self.goal as f64
                + (board_score(&job.plan.expected) / 200.0).clamp(-0.2, 0.2)
                + 0.2
        } else {
            0.0
        };
        self.backpropagate(job.parent, reward);
    }

    fn insert_boundary(&mut self, parent: usize, node: TreeNode) {
        let limit = self.config.beam_width * 8;
        if self.tree.len() < limit {
            let index = self.tree.len();
            self.tree.push(node);
            self.tree[parent].children.push(index);
            return;
        }
        // Stable slots keep pending jobs and ancestral certificates valid. Only
        // unreferenced leaves are replaced; completed incumbents live separately.
        let victim = (1..self.tree.len())
            .filter(|i| {
                *i != parent
                    && self.tree[*i].children.is_empty()
                    && !self.waiting_plans.iter().any(|(pending, _)| pending == i)
            })
            .min_by(|a, b| {
                let score = |i: usize| {
                    let n = &self.tree[i];
                    n.spins as f64 * 8.0 + board_score(&n.board) * 0.05
                        - n.pieces as f64 * 0.1
                        - (n.visits as f64).sqrt() * 0.1
                };
                score(*a).total_cmp(&score(*b))
            });
        let Some(index) = victim else { return };
        if let Some(old_parent) = self.tree[index].parent {
            self.tree[old_parent].children.retain(|i| *i != index);
        }
        self.tree[index] = node;
        self.tree[parent].children.push(index);
    }

    pub fn snapshot(&self) -> ProgressSnapshot {
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
            current: self.current.as_ref().map(|a| a.summary.clone()),
            best: self.best.as_ref().map(|a| a.summary.clone()),
        }
    }

    pub fn export_json(&self) -> String {
        let archive = self
            .best
            .as_ref()
            .or(self.other_complete.as_ref())
            .or(self.current.as_ref());
        let certificates: Vec<_> = archive
            .map(|a| a.certificates.iter().map(|c| c.as_ref()).collect())
            .unwrap_or_default();
        serde_json::json!({
            "format": "neyra", "schemaVersion": 1, "engineVersion": env!("CARGO_PKG_VERSION"),
            "rules": { "width": 10, "height": HEIGHT, "rotation": "SRS+", "spin": "All-mini+", "oSpin": false, "ceiling": "ActualCellDeath24", "movement": "SonicDropOnly", "supply": "UniformP7NormalizedHold", "runtimeReplayVerified": false },
            "config": self.config, "snapshot": self.snapshot(),
            "candidate": archive.map(|a| &a.summary), "bags": certificates,
            "pendingPlan": archive.and_then(|a| a.pending_plan.as_ref()),
            "successBitsEncoding": "hex-u64-lsb0; word=floor(P7rank/64), bit=P7rank%64",
            "interchange": false, "globalOptimalityProven": false
        }).to_string()
    }
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
            rows: [0; HEIGHT],
            spins: 40,
            placed: 119,
            probability: 1.0,
            lower_probability: 1.0,
            upper_probability: 1.0,
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
