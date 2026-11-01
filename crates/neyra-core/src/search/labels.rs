//! Bounded, CPU-only continuation labels from an existing bag boundary.
//!
//! The current field is the model input. Follow-up searches are observations
//! under a recorded budget, never exact values or impossibility certificates.
//! Legacy v2 expands only fully P7-certified parents. Explicit v3 observations
//! use replayed input-reachable parents and a one-bag target, while measuring
//! at least two new inventories under a separate, censored policy.

use super::*;
use std::collections::BTreeMap;

const LABEL_GOAL: u8 = u8::MAX;
const LABEL_POLICY: &str = "neyra-fixed-bag-continuation-v2";
pub const LABEL_TASK: &str = "naga-spin-observations-v2";
pub const NEXT_BAG_TASK: &str = "naga-next-bag-spin-v3";
pub const NEXT_BAG_POLICY: &str = "bounded-next-bag-spin-v3";
const REACHABLE_WITNESS_CAPACITY: usize = 32;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelSource {
    pub kind: String,
    pub reference: String,
    pub family: String,
    /// Number of completed bags before the supplied field. Learning starts at two.
    pub bag_boundary: u16,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
pub struct BoardSample {
    pub id: String,
    /// Physical rows, bottom first; bit zero is the leftmost column.
    pub rows: [u16; HEIGHT],
    pub source: LabelSource,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "camelCase")]
pub struct LabelConfig {
    /// Absent preserves the released two-bag v2 policy and checkpoint meaning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observation_version: Option<u8>,
    pub horizon_bags: u8,
    pub work_budget: u32,
    pub beam_width: usize,
    pub profile: u8,
    pub seed: u64,
    pub max_nodes: usize,
    pub attempts_per_node: u32,
    pub source_revision: String,
}

impl Default for LabelConfig {
    fn default() -> Self {
        Self {
            observation_version: None,
            horizon_bags: 2, work_budget: 131_072, beam_width: 48,
            profile: 0, seed: 1, max_nodes: 96, attempts_per_node: 6,
            source_revision: "unknown".into(),
        }
    }
}

impl LabelConfig {
    fn validate(&self) -> Result<(), String> {
        if !(2..=3).contains(&self.horizon_bags)
            || self.observation_version.is_some_and(|version| version != 2 && version != 3)
            || !(1..=4_194_304).contains(&self.work_budget)
            || !(4..=128).contains(&self.beam_width)
            || self.profile > 3 || !(1..=256).contains(&self.max_nodes)
            || !(1..=16).contains(&self.attempts_per_node)
        {
            return Err("Invalid label budget, horizon, beam, profile or node limits".into());
        }
        if self.observation_version == Some(3) && self.max_nodes < usize::from(self.horizon_bags) {
            return Err("The v3 continuation node capacity must cover its requested horizon".into());
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpinTargets {
    /// Whether this bounded search observed at least one actual future spin.
    /// False is censored absence, not a proof of impossibility.
    pub found_spin: bool,
    pub observed_spins: u8,
}

#[derive(Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DepthObservation {
    pub bags: u8,
    pub geometric_whole_bag_chains: u32,
    pub productive_geometric_whole_bag_chains: u32,
    pub certified_whole_bag_chains: u32,
    pub productive_certified_whole_bag_chains: u32,
    pub max_geometric_whole_bag_spins: Option<u8>,
    pub max_certified_spins: Option<u8>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LabelMeasurements {
    /// Includes input-reachable prefixes in the final explored bag. The
    /// earlier bags, if any, have exact P7 certificates. Not a P7 path count.
    pub max_reachable_observed_spins: u8,
    pub geometric_whole_bag_chains: u32,
    /// Whole-bag chains with at least one actual spin since the input field;
    /// quiet-only alternatives must not inflate the spin-opportunity target.
    pub productive_geometric_whole_bag_chains: u32,
    pub certified_whole_bag_chains: u32,
    pub productive_certified_whole_bag_chains: u32,
    pub max_certified_observed_spins: Option<u8>,
    pub depths: Vec<DepthObservation>,
    pub work_used: u32,
    pub generation_work: u32,
    pub verification_work: u32,
    pub geometric_placements_considered: u64,
    pub canonical_supplies_checked: u64,
    pub quiet_policy_rejections: u32,
    pub supply_rejections: u32,
    pub graph_overflows: u32,
    pub node_capacity_drops: u32,
    pub censored: bool,
    pub stop_reason: String,
    pub evidence_kind: &'static str,
    pub distinctness: &'static str,
    pub work_unit: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lookahead: Option<LookaheadMeasurements>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TwoBagPath {
    pub witness_id: u64,
    pub first_bag_spins: u8,
    pub second_bag_spins: u8,
    pub total_spins: u8,
}

/// Extra depth is an observed, coherent path; it is not another supervised
/// target and does not turn the one-bag prediction into a two-bag prediction.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LookaheadMeasurements {
    pub target_horizon_bags: u8,
    pub requested_bags: u8,
    pub deepest_attempted_bag: u8,
    pub deepest_completed_bag: u8,
    pub deepest_reached_placements: u16,
    pub reachable_whole_bag_chains: u32,
    pub witness_capacity: usize,
    pub capacity_drops: u64,
    pub duplicate_continuations: u64,
    pub best_two_bag_path: Option<TwoBagPath>,
}

/// One physical queue and its replayable inputs. This proves existence only;
/// it has neither a P7 success set nor a construction-success probability.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReachableWholeBagWitness {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub bags: u8,
    pub start_rows: [u16; HEIGHT],
    pub end_rows: [u16; HEIGHT],
    pub start_spins: u8,
    pub end_spins: u8,
    pub deleted_rows: u32,
    pub targets: Vec<TargetOperation>,
    pub representative_queue: [u8; 7],
    pub representative_inputs: Vec<Vec<Input>>,
    pub frames: Vec<DisplayFrame>,
    pub trust: &'static str,
    pub whole_bag: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CertifiedWitness {
    pub id: u64,
    pub parent_id: Option<u64>,
    pub bags: u8,
    pub start_rows: [u16; HEIGHT],
    pub end_rows: [u16; HEIGHT],
    pub start_spins: u8,
    pub end_spins: u8,
    pub targets: Vec<TargetOperation>,
    pub checked: u16,
    pub success: u16,
    pub quiet_safe_supplies: u16,
    pub first_bag_symmetry_assumed: bool,
    pub representative_queue: [u8; 7],
    pub representative_inputs: Vec<Vec<Input>>,
    pub frames: Vec<DisplayFrame>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ReachableWitness {
    pub certified_parent_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reachable_parent_id: Option<u64>,
    pub completed_parent_bags: u8,
    pub additional_placements: u8,
    pub observed_spins: u8,
    pub targets: Vec<TargetOperation>,
    pub frames: Vec<DisplayFrame>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BoardLabel {
    pub schema_version: u8,
    pub task: &'static str,
    pub policy: &'static str,
    pub input: BoardSample,
    pub config: LabelConfig,
    pub targets: SpinTargets,
    pub measured: LabelMeasurements,
    pub certified_witnesses: Vec<CertifiedWitness>,
    pub reachable_witness: Option<ReachableWitness>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_horizon_bags: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_witness: Option<ReachableWitness>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reachable_whole_bag_witnesses: Option<Vec<ReachableWholeBagWitness>>,
}

struct LabelNode {
    node: TreeNode,
    bags: u8,
    certificate_id: Option<u64>,
    chain_key: Vec<u8>,
}

fn node(board: Board, spins: u8, bags: u8, debt: CellMask) -> TreeNode {
    TreeNode {
        parent: None, children: vec![], board, spins,
        // The nonzero offset selects later-bag ordering. Historical setup
        // spins, probabilities and debt are not invented from an input board.
        pieces: 7 * (u16::from(bags) + 1), counts: vec![], certificates: vec![],
        frames: vec![], visits: 0, reward: 0.0, future: Continuation::default(),
        debt, attempts: 0, allowance: 0, lease_epoch: 0, last_selected: 0,
        quality: board_score(&board), last_improved_attempt: 0,
        best_extension_spins: 0, best_extension_quality: f64::NEG_INFINITY,
        revisit_after: 0, obstructed_mass: 0,
    }
}

/// Deliberately stop before the live search's forecast refinement. Its extra
/// inventories would exceed this collector's requested two/three-bag horizon.
fn next_label_layer(job: &mut GenerationJob) -> bool {
    if job.state_index < job.frontier.len() { return false; }
    job.frontier = job.finish_inventory_layer();
    job.current_facts = None;
    job.state_index = 0;
    job.piece_index = 0;
    job.depth += 1;
    job.frontier.is_empty() || job.depth >= 7
}

fn chain_key(parent: &LabelNode, seed: &SeedState) -> Vec<u8> {
    let mut key = parent.chain_key.clone();
    let mut targets = seed.targets.clone();
    targets.sort_by_key(|target| target.piece as u8);
    for target in targets {
        key.push(target.piece as u8);
        key.extend(target_code(&target).to_le_bytes());
    }
    // The fixed typed target assignment alone must not merge incompatible
    // deletion/execution contexts or distinct inherited board histories.
    key.extend(seed.deleted.to_le_bytes());
    key.push(seed.spins);
    for row in seed.board.rows { key.extend(row.to_le_bytes()); }
    key
}

/// Produce a small-sample training observation without changing live search.
/// Every number is tied to this budget and selective proposal policy. A zero
/// observation is not proof that no continuation exists, including when the
/// bounded frontier runs out before the budget does.
pub fn collect_board_label(input: BoardSample, config: LabelConfig) -> Result<BoardLabel, String> {
    if config.observation_version == Some(3) {
        return collect_next_bag_label(input, config);
    }
    collect_legacy_board_label(input, config)
}

fn collect_legacy_board_label(input: BoardSample, config: LabelConfig) -> Result<BoardLabel, String> {
    config.validate()?;
    let board = Board { rows: input.rows };
    if input.id.trim().is_empty() || input.source.bag_boundary < 2
        || input.source.kind.trim().is_empty() || input.source.reference.trim().is_empty()
        || input.source.family.trim().is_empty()
        || !board.is_valid() || board.cell_count() == 0
        || board.rows.iter().any(|row| *row == 1023)
    {
        return Err("Spin labels require a named nonempty 24-row, post-clear field after at least two completed bags, with source provenance".into());
    }
    for forbidden in ["referencePrior", "priorIsMeasuredOutcome", "currentBoardTeacher",
                      "terrainScore", "fieldQuality", "rankingScore", "shapePreference",
                      "observedChoices", "boardEvaluation"] {
        if input.source.extra.contains_key(forbidden) {
            return Err(format!("Legacy evaluation metadata is not a spin-only learning input: {forbidden}"));
        }
    }
    let mut observations = LabelMeasurements {
        max_reachable_observed_spins: 0, geometric_whole_bag_chains: 0,
        productive_geometric_whole_bag_chains: 0, certified_whole_bag_chains: 0,
        productive_certified_whole_bag_chains: 0, max_certified_observed_spins: None,
        depths: (1..=config.horizon_bags).map(|bags| DepthObservation { bags, ..Default::default() }).collect(),
        work_used: 0, generation_work: 0, verification_work: 0,
        geometric_placements_considered: 0, canonical_supplies_checked: 0,
        quiet_policy_rejections: 0, supply_rejections: 0, graph_overflows: 0,
        node_capacity_drops: 0, censored: true, stop_reason: "work_budget".into(),
        evidence_kind: "input-reachable prefixes; exact quiet-safe P7 certificates for every expanded whole bag",
        distinctness: "fixed piece-to-logical-target assignment chains including deleted rows and boundary boards; productive means at least one actual spin since the supplied origin; not input sequences, supply orders or success probability",
        work_unit: "one bounded GenerationJob.advance or VerificationJob.advance; latter expands one execution node or checks at most 64 canonical supplies; forecast refinement disabled",
        lookahead: None,
    };
    let mut nodes = vec![LabelNode { node: node(board, 0, 0, [0; HEIGHT]), bags: 0, certificate_id: None, chain_key: vec![] }];
    let mut cache = PhysicsCache::default();
    let mut random = Random(if config.seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { config.seed });
    let mut generation: Option<GenerationJob> = None;
    let mut verification: Option<(VerificationJob, Vec<u8>)> = None;
    let mut waiting: Vec<(VerificationJob, Vec<u8>)> = vec![];
    let mut seen = HashSet::new();
    let mut witnesses = vec![];
    let mut reachable_witness = None;
    let mut next_id = 1_u64;
    let mut dispatches = 0_u32;
    let mut newest_parent = Some(0_usize);

    while observations.work_used < config.work_budget {
        if let Some((mut job, key)) = verification.take() {
            let before = job.canonical_checked;
            job.advance(LABEL_GOAL, &mut cache);
            observations.work_used += 1;
            observations.verification_work += 1;
            observations.canonical_supplies_checked += (job.canonical_checked - before) as u64;
            if job.quiet_rejected {
                observations.quiet_policy_rejections += 1;
                continue;
            }
            if job.graph.overflow {
                observations.graph_overflows += 1;
                continue;
            }
            if job.coverage_upper() < P7_COUNT as u16 {
                observations.supply_rejections += 1;
                continue;
            }
            if job.checked < P7_COUNT {
                verification = Some((job, key));
                continue;
            }
            if job.coverage_success() != P7_COUNT as u16 || job.quiet_safe != P7_COUNT {
                observations.supply_rejections += 1;
                continue;
            }
            let parent = &nodes[job.parent];
            let bags = parent.bags + 1;
            let rank = job.costs.iter().position(|cost| *cost == 7)
                .ok_or_else(|| "Whole-bag certificate has no seven-lock representative".to_string())?;
            let (mut frames, inputs) = job.graph.witness(job.codes[rank], 7, parent.node.pieces);
            for frame in &mut frames { frame.placed -= 7; }
            let last = frames.last().ok_or_else(|| "Empty whole-bag witness".to_string())?;
            if last.rows != job.plan.expected.rows || last.spins != job.plan.expected_spins {
                return Err("Whole-bag representative disagrees with generated boundary".into());
            }
            observations.certified_whole_bag_chains += 1;
            observations.productive_certified_whole_bag_chains += u32::from(last.spins > 0);
            observations.max_certified_observed_spins = Some(observations.max_certified_observed_spins.unwrap_or(0).max(last.spins));
            let depth = &mut observations.depths[usize::from(bags - 1)];
            depth.certified_whole_bag_chains += 1;
            depth.productive_certified_whole_bag_chains += u32::from(last.spins > 0);
            depth.max_certified_spins = Some(depth.max_certified_spins.unwrap_or(0).max(last.spins));
            witnesses.push(CertifiedWitness {
                id: job.plan.id, parent_id: parent.certificate_id, bags,
                start_rows: job.plan.start.rows, end_rows: last.rows,
                start_spins: job.plan.start_spins, end_spins: last.spins,
                targets: job.plan.targets.clone(), checked: P7_COUNT as u16,
                success: job.coverage_success(), quiet_safe_supplies: job.quiet_safe as u16,
                first_bag_symmetry_assumed: false, representative_queue: permutation(rank),
                representative_inputs: inputs, frames,
            });
            if bags < config.horizon_bags {
                if nodes.len() < config.max_nodes {
                    newest_parent = Some(nodes.len());
                    nodes.push(LabelNode {
                        node: node(job.plan.expected, job.plan.expected_spins, bags, job.plan.debt),
                        bags, certificate_id: Some(job.plan.id), chain_key: key,
                    });
                } else { observations.node_capacity_drops += 1; }
            }
            continue;
        }
        if let Some(mut job) = generation.take() {
            observations.geometric_placements_considered += job.advance(&mut random, LABEL_GOAL, &mut cache);
            observations.work_used += 1;
            observations.generation_work += 1;
            for seed in job.frontier.iter().chain(job.next_by_inventory.iter().flatten()).chain(job.complete.iter()) {
                if seed.spins > observations.max_reachable_observed_spins {
                    observations.max_reachable_observed_spins = seed.spins;
                    let parent = &nodes[job.parent];
                    let mut frames = seed.frames.clone();
                    for frame in &mut frames { frame.placed -= 7; }
                    reachable_witness = Some(ReachableWitness {
                        certified_parent_id: parent.certificate_id,
                        reachable_parent_id: None,
                        completed_parent_bags: parent.bags, additional_placements: seed.targets.len() as u8,
                        observed_spins: seed.spins, targets: seed.targets.clone(), frames,
                    });
                }
            }
            if !next_label_layer(&mut job) {
                generation = Some(job);
                continue;
            }
            let parent = &nodes[job.parent];
            for seed in job.complete {
                if seed.used != ALL_PIECES || seed.targets.len() != 7 { continue; }
                let key = chain_key(parent, &seed);
                if !seen.insert(key.clone()) { continue; }
                let depth = &mut observations.depths[usize::from(parent.bags)];
                depth.geometric_whole_bag_chains += 1;
                depth.productive_geometric_whole_bag_chains += u32::from(seed.spins > 0);
                depth.max_geometric_whole_bag_spins = Some(depth.max_geometric_whole_bag_spins.unwrap_or(0).max(seed.spins));
                observations.geometric_whole_bag_chains += 1;
                observations.productive_geometric_whole_bag_chains += u32::from(seed.spins > 0);
                let plan = Plan {
                    id: next_id, start: job.start, start_spins: job.start_spins,
                    start_pieces: job.start_pieces, expected: seed.board,
                    expected_spins: seed.spins, targets: seed.targets,
                    seed_frames: seed.frames, terminal: false, assume_reflection: false,
                    quality: seed.quality, pace_fallback: false,
                    future: Continuation::default(), debt: seed.debt,
                };
                next_id += 1;
                waiting.push((VerificationJob::new(job.parent, plan), key));
            }
            continue;
        }

        // A newly certified deeper parent receives a generation before its
        // siblings' remaining verifications. Every fourth dispatch returns to
        // the least attempted shallow node so depth cannot starve diversity.
        let eligible = |n: &LabelNode| n.bags < config.horizon_bags && n.node.attempts < config.attempts_per_node;
        let fresh = newest_parent.take().filter(|&index| eligible(&nodes[index]));
        let parent = if dispatches % 4 == 3 {
            nodes.iter().enumerate().filter(|(_, n)| eligible(n))
                .min_by_key(|(index, n)| (n.node.attempts, n.bags, *index)).map(|(index, _)| index)
        } else { fresh };
        if let Some(parent) = parent {
            let n = &mut nodes[parent].node;
            generation = Some(GenerationJob::new(parent, n, config.beam_width, false,
                u64::from(n.attempts), LABEL_GOAL, config.profile, P7_COUNT as u16, 0, 1));
            n.attempts += 1;
            dispatches += 1;
            continue;
        }
        if let Some(index) = waiting.iter().enumerate().max_by_key(|(_, (job, _))| {
            (nodes[job.parent].bags, job.plan.expected_spins, std::cmp::Reverse(job.plan.id))
        }).map(|(index, _)| index) {
            verification = Some(waiting.swap_remove(index));
            dispatches += 1;
            continue;
        }
        if let Some(parent) = nodes.iter().enumerate().filter(|(_, n)| eligible(n))
            .min_by_key(|(index, n)| (n.node.attempts, std::cmp::Reverse(n.bags), *index)).map(|(index, _)| index)
        {
            newest_parent = Some(parent);
            continue;
        }
        observations.stop_reason = "bounded_frontier_exhausted".into();
        break;
    }
    Ok(BoardLabel {
        schema_version: 2, task: LABEL_TASK, policy: LABEL_POLICY, input, config,
        targets: SpinTargets { found_spin: observations.max_reachable_observed_spins > 0,
            observed_spins: observations.max_reachable_observed_spins },
        measured: observations,
        certified_witnesses: witnesses, reachable_witness,
        target_horizon_bags: None, target_witness: None, reachable_whole_bag_witnesses: None,
    })
}

struct ReachableNode {
    node: TreeNode,
    bags: u8,
    witness_id: Option<u64>,
}

fn reachable_prefix(parent: &ReachableNode, seed: &SeedState) -> ReachableWitness {
    let mut frames = seed.frames.clone();
    for frame in &mut frames { frame.placed -= 7; }
    ReachableWitness {
        certified_parent_id: None,
        reachable_parent_id: parent.witness_id,
        completed_parent_bags: parent.bags,
        additional_placements: seed.targets.len() as u8,
        observed_spins: seed.spins,
        targets: seed.targets.clone(), frames,
    }
}

fn replay_reachable_bag(
    parent: &ReachableNode, seed: &SeedState, id: u64, cache: &mut PhysicsCache,
) -> Result<ReachableWholeBagWitness, String> {
    if seed.used != ALL_PIECES || seed.targets.len() != 7 || seed.frames.len() != 7 {
        return Err("An input-reachable boundary requires seven actual locks".into());
    }
    let mut board = parent.node.board;
    let mut deleted = 0;
    let mut spins = parent.node.spins;
    let mut used = 0;
    let mut queue = [0; 7];
    let mut inputs = Vec::with_capacity(7);
    let mut frames = Vec::with_capacity(7);
    for (index, (target, expected)) in seed.targets.iter().zip(&seed.frames).enumerate() {
        if used & target.piece.bit() != 0 {
            return Err("An input-reachable bag reused a piece label".into());
        }
        let mut accepted = None;
        for pose in projected_poses(target, deleted) {
            let Some(placement) = cache.target(board, pose, expected.lines == 1) else { continue; };
            let Some(result) = board.apply(&placement) else { continue; };
            if result.board.rows != expected.rows || result.lines != expected.lines
                || result.lines > 1 || (result.lines == 1 && !result.spin)
                || normalize_target(placement.pose, deleted).as_ref() != Some(target)
            { continue; }
            accepted = Some((placement, result));
            break;
        }
        let (placement, result) = accepted
            .ok_or_else(|| "An observed whole-bag lock has no matching physical input replay".to_string())?;
        deleted = deleted_after(deleted, result.cleared_rows)
            .ok_or_else(|| "Input-reachable whole-bag logical rows exceeded their bound".to_string())?;
        spins += result.lines;
        if spins != expected.spins {
            return Err("Input-reachable whole-bag spin count disagrees with its replay".into());
        }
        queue[index] = target.piece as u8;
        used |= target.piece.bit();
        inputs.push(placement.path);
        frames.push(DisplayFrame {
            rows: result.board.rows, piece: piece_name(target.piece),
            placed_cells: result.cells.map(|(x, y)| (x as u8, y as u8)),
            spins, placed: 7 * u16::from(parent.bags) + index as u16 + 1,
            lines: result.lines, spin: result.spin,
        });
        board = result.board;
    }
    if used != ALL_PIECES || board != seed.board || deleted != seed.deleted || spins != seed.spins {
        return Err("Input-reachable whole-bag boundary disagrees with its replay".into());
    }
    Ok(ReachableWholeBagWitness {
        id, parent_id: parent.witness_id, bags: parent.bags + 1,
        start_rows: parent.node.board.rows, end_rows: board.rows,
        start_spins: parent.node.spins, end_spins: spins, deleted_rows: deleted,
        targets: seed.targets.clone(), representative_queue: queue,
        representative_inputs: inputs, frames,
        trust: "input-reachable", whole_bag: true,
    })
}

/// A distinct policy for the intermediate next-bag model. It uses the ordinary
/// physics/proposal ordering, but a single supplied node owns its entire work
/// budget. No incumbent, completion pace or supply certificate can suppress an
/// input-reachable child. Beam/node/witness limits remain explicit censoring.
fn collect_next_bag_label(input: BoardSample, config: LabelConfig) -> Result<BoardLabel, String> {
    config.validate()?;
    let board = Board { rows: input.rows };
    if input.id.trim().is_empty() || input.source.bag_boundary < 2
        || input.source.kind.trim().is_empty() || input.source.reference.trim().is_empty()
        || input.source.family.trim().is_empty() || !board.is_valid() || board.cell_count() == 0
        || board.rows.iter().any(|row| *row == 1023)
    {
        return Err("Spin labels require a named nonempty 24-row, post-clear field after at least two completed bags, with source provenance".into());
    }
    for forbidden in ["referencePrior", "priorIsMeasuredOutcome", "currentBoardTeacher",
                      "terrainScore", "fieldQuality", "rankingScore", "shapePreference",
                      "observedChoices", "boardEvaluation"] {
        if input.source.extra.contains_key(forbidden) {
            return Err(format!("Legacy evaluation metadata is not a spin-only learning input: {forbidden}"));
        }
    }
    let witness_capacity = config.max_nodes.min(REACHABLE_WITNESS_CAPACITY);
    let mut lookahead = LookaheadMeasurements {
        target_horizon_bags: 1, requested_bags: config.horizon_bags,
        deepest_attempted_bag: 0, deepest_completed_bag: 0, deepest_reached_placements: 0,
        reachable_whole_bag_chains: 0, witness_capacity,
        capacity_drops: 0, duplicate_continuations: 0, best_two_bag_path: None,
    };
    let mut observations = LabelMeasurements {
        max_reachable_observed_spins: 0, geometric_whole_bag_chains: 0,
        productive_geometric_whole_bag_chains: 0, certified_whole_bag_chains: 0,
        productive_certified_whole_bag_chains: 0, max_certified_observed_spins: None,
        depths: (1..=config.horizon_bags).map(|bags| DepthObservation { bags, ..Default::default() }).collect(),
        work_used: 0, generation_work: 0, verification_work: 0,
        geometric_placements_considered: 0, canonical_supplies_checked: 0,
        quiet_policy_rejections: 0, supply_rejections: 0, graph_overflows: 0,
        node_capacity_drops: 0, censored: true, stop_reason: "work_budget".into(),
        evidence_kind: "input-reachable prefixes and replayed whole-bag chains; no P7 or quiet-safe coverage claimed",
        distinctness: "whole-bag continuation states include physical board, depth, actual spin count and inherited debt; one retained physical path per state; not supply orders or success probability",
        work_unit: "one bounded GenerationJob.advance, or one retained lock replay; seven replay units per whole-bag witness; forecast refinement and P7 verification disabled",
        lookahead: None,
    };
    let mut nodes = vec![ReachableNode {
        node: node(board, 0, 0, [0; HEIGHT]), bags: 0, witness_id: None,
    }];
    let mut seen = HashSet::new();
    seen.insert((board, 0_u8, 0_u8, [0_u16; HEIGHT]));
    let mut cache = PhysicsCache::default();
    let mut random = Random(if config.seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { config.seed });
    let mut generation: Option<GenerationJob> = None;
    let mut witnesses = Vec::new();
    let mut target_witness: Option<ReachableWitness> = None;
    let mut reachable_witness: Option<ReachableWitness> = None;
    let mut newest_parent = Some(0_usize);
    let mut next_id = 1_u64;

    'search: while observations.work_used < config.work_budget {
        if witnesses.len() >= witness_capacity {
            observations.stop_reason = "witness_capacity".into();
            break;
        }
        if let Some(mut job) = generation.take() {
            let before_capacity = job.mino_capacity_pruned;
            let before_duplicates = job.duplicates;
            observations.geometric_placements_considered += job.advance(&mut random, LABEL_GOAL, &mut cache);
            observations.work_used += 1;
            observations.generation_work += 1;
            lookahead.capacity_drops += job.mino_capacity_pruned - before_capacity;
            lookahead.duplicate_continuations += job.duplicates - before_duplicates;
            let parent = &nodes[job.parent];
            for seed in job.frontier.iter().chain(job.next_by_inventory.iter().flatten()).chain(job.complete.iter()) {
                if seed.targets.is_empty() { continue; }
                observations.max_reachable_observed_spins = observations.max_reachable_observed_spins.max(seed.spins);
                let placed = 7 * u16::from(parent.bags) + seed.targets.len() as u16;
                lookahead.deepest_reached_placements = lookahead.deepest_reached_placements.max(placed);
                if parent.bags == 0 && target_witness.as_ref().is_none_or(|prior| {
                    (seed.spins, seed.targets.len()) > (prior.observed_spins, usize::from(prior.additional_placements))
                }) {
                    target_witness = Some(reachable_prefix(parent, seed));
                }
                if reachable_witness.as_ref().is_none_or(|prior| {
                    (placed, seed.spins) > (7 * u16::from(prior.completed_parent_bags)
                        + u16::from(prior.additional_placements), prior.observed_spins)
                }) {
                    reachable_witness = Some(reachable_prefix(parent, seed));
                }
            }
            if !next_label_layer(&mut job) {
                generation = Some(job);
                continue;
            }
            let mut fresh = None;
            for seed in job.complete {
                if seed.used != ALL_PIECES || seed.targets.len() != 7 { continue; }
                let parent = &nodes[job.parent];
                let bags = parent.bags + 1;
                let identity = (seed.board, bags, seed.spins, seed.debt);
                if seen.contains(&identity) {
                    lookahead.duplicate_continuations += 1;
                    continue;
                }
                // Keep enough parent-closed proof slots for one complete path
                // to the requested horizon before retaining more siblings.
                // Small maxNodes values must not spend every slot at depth 1.
                let reserved = usize::from(config.horizon_bags - bags);
                if witness_capacity.saturating_sub(witnesses.len()) <= reserved {
                    lookahead.capacity_drops += 1;
                    continue;
                }
                if config.work_budget - observations.work_used < 7 { break 'search; }
                let witness = replay_reachable_bag(parent, &seed, next_id, &mut cache)?;
                observations.work_used += 7;
                observations.verification_work += 7;
                seen.insert(identity);
                next_id += 1;
                lookahead.deepest_completed_bag = lookahead.deepest_completed_bag.max(bags);
                lookahead.reachable_whole_bag_chains += 1;
                observations.geometric_whole_bag_chains += 1;
                observations.productive_geometric_whole_bag_chains += u32::from(seed.spins > 0);
                let depth = &mut observations.depths[usize::from(bags - 1)];
                depth.geometric_whole_bag_chains += 1;
                depth.productive_geometric_whole_bag_chains += u32::from(seed.spins > 0);
                depth.max_geometric_whole_bag_spins = Some(depth.max_geometric_whole_bag_spins.unwrap_or(0).max(seed.spins));
                if bags == 2 && lookahead.best_two_bag_path.as_ref().is_none_or(|prior| {
                    (seed.spins, seed.spins - parent.node.spins) > (prior.total_spins, prior.second_bag_spins)
                }) {
                    lookahead.best_two_bag_path = Some(TwoBagPath {
                        witness_id: witness.id, first_bag_spins: parent.node.spins,
                        second_bag_spins: seed.spins - parent.node.spins, total_spins: seed.spins,
                    });
                }
                if bags < config.horizon_bags {
                    let reserved_nodes = usize::from(config.horizon_bags - bags - 1);
                    if config.max_nodes.saturating_sub(nodes.len()) > reserved_nodes {
                        if fresh.is_none() { fresh = Some(nodes.len()); }
                        nodes.push(ReachableNode {
                            node: node(seed.board, seed.spins, bags, seed.debt),
                            bags, witness_id: Some(witness.id),
                        });
                    } else {
                        observations.node_capacity_drops += 1;
                        lookahead.capacity_drops += 1;
                    }
                }
                witnesses.push(witness);
            }
            newest_parent = fresh;
            continue;
        }

        let eligible = |candidate: &ReachableNode| candidate.bags < config.horizon_bags
            && candidate.node.attempts < config.attempts_per_node;
        // The first actual child is explored before returning to alternatives.
        // Remaining deepest, least-attempted nodes precede another root round.
        let parent = newest_parent.take().filter(|&index| eligible(&nodes[index]))
            .or_else(|| nodes.iter().enumerate().filter(|(_, candidate)| eligible(candidate))
                .max_by_key(|(index, candidate)| (candidate.bags,
                    std::cmp::Reverse(candidate.node.attempts), std::cmp::Reverse(*index)))
                .map(|(index, _)| index));
        let Some(parent) = parent else {
            observations.stop_reason = "bounded_frontier_exhausted".into();
            break;
        };
        lookahead.deepest_attempted_bag = lookahead.deepest_attempted_bag.max(nodes[parent].bags + 1);
        let n = &mut nodes[parent].node;
        generation = Some(GenerationJob::new(parent, n, config.beam_width, false,
            u64::from(n.attempts), LABEL_GOAL, config.profile, 0, 0, 1).for_learning());
        n.attempts += 1;
    }
    let observed_spins = target_witness.as_ref().map_or(0, |witness| witness.observed_spins);
    observations.lookahead = Some(lookahead);
    Ok(BoardLabel {
        schema_version: 3, task: NEXT_BAG_TASK, policy: NEXT_BAG_POLICY,
        input, config, targets: SpinTargets { found_spin: observed_spins > 0, observed_spins },
        measured: observations, certified_witnesses: vec![], reachable_witness,
        target_horizon_bags: Some(1), target_witness,
        reachable_whole_bag_witnesses: Some(witnesses),
    })
}
