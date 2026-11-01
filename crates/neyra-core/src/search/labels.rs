//! Bounded, CPU-only continuation labels from an existing bag boundary.
//!
//! The current field is the model input. Follow-up searches are observations
//! under a recorded budget, never exact values or impossibility certificates.
//! Legacy v2 expands only fully P7-certified parents. Explicit v3 observations
//! use replayed input-reachable parents and a one-bag target, while measuring
//! at least two new inventories under a separate, censored policy.
//! V4 refines a supplied field and a few quality-band sampled first-bag
//! endpoints with equal reserved budgets. Each endpoint's local next-bag
//! target and full-board input proof survive independently of output caps.

use super::*;
use std::collections::BTreeMap;

const LABEL_GOAL: u8 = u8::MAX;
const LABEL_POLICY: &str = "neyra-fixed-bag-continuation-v2";
pub const LABEL_TASK: &str = "naga-spin-observations-v2";
pub const NEXT_BAG_TASK: &str = "naga-next-bag-spin-v3";
pub const NEXT_BAG_POLICY: &str = "bounded-next-bag-spin-v3";
pub const REFINED_BAG_TASK: &str = "naga-next-bag-spin-v4";
pub const REFINED_BAG_POLICY: &str = "concentrated-next-bag-spin-v4";
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
            || self.observation_version.is_some_and(|version| !matches!(version, 2 | 3 | 4))
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
        if self.observation_version == Some(4)
            && (self.horizon_bags != 2 || self.max_nodes < 4 || self.work_budget < 1_024)
        {
            return Err("V4 concentrated observations require horizonBags 2, maxNodes at least 4 and workBudget at least 1024".into());
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dependency_work: Option<u32>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refinement: Option<RefinementMeasurements>,
}

/// A replayed prefix relative to one supplied board and one new inventory.
/// Unlike a supply certificate, this records one actual order and input path.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinedPrefixWitness {
    pub placements: u8,
    pub observed_spins: u8,
    pub deleted_rows: u32,
    pub targets: Vec<TargetOperation>,
    pub representative_queue: Vec<u8>,
    pub representative_inputs: Vec<Vec<Input>>,
    pub frames: Vec<DisplayFrame>,
    pub trust: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinedSearchObservation {
    pub allocated_work: u32,
    pub work_used: u32,
    pub generation_work: u32,
    pub verification_work: u32,
    pub attempts: u32,
    pub targets: SpinTargets,
    pub target_witness: Option<RefinedPrefixWitness>,
    pub deepest_witness: Option<RefinedPrefixWitness>,
    pub deepest_reached_placements: u8,
    pub completed_whole_bag_candidates: u32,
    pub distinct_whole_bag_candidates: u32,
    pub censored: bool,
    pub stop_reason: &'static str,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinedArmObservation {
    pub candidate_witness_id: u64,
    pub selection_stratum: usize,
    pub selection_strata: usize,
    pub selection_pool_size: usize,
    pub selection_index: usize,
    pub observation: RefinedSearchObservation,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComponentOrderObservation {
    pub pieces: Vec<Piece>,
    pub total_orders: u32,
    pub tested_orders: u32,
    pub successful_orders: u32,
    pub interrupted_orders: u32,
    pub work_used: u32,
    pub exhaustive: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComponentDependencyApproximation {
    pub schema_version: u8,
    pub policy: &'static str,
    pub connectivity: &'static str,
    pub scope: &'static str,
    pub max_orders_per_component: usize,
    pub work_used: u32,
    /// These are sums of separate component counts, never global supply odds.
    pub total_orders: u32,
    pub tested_orders: u32,
    pub successful_orders: u32,
    pub components: Vec<ComponentOrderObservation>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinedCandidate {
    pub witness_id: u64,
    /// Proposal/sampling metadata only. It is never a supervised CNN target.
    pub structural_quality: f64,
    /// Only selected first-bag candidates receive this bounded diagnostic.
    pub dependency: Option<ComponentDependencyApproximation>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RefinementMeasurements {
    pub schema_version: u8,
    pub policy: &'static str,
    pub selection_seed: u32,
    pub selection_score_min: f64,
    pub selection_score_max: f64,
    pub selection_band_count: usize,
    pub candidate_pool_capacity: usize,
    pub candidate_pool_size: usize,
    pub candidate_pool_drops: u64,
    pub candidate_duplicates: u64,
    pub arm_limit: usize,
    pub arm_quota: u32,
    pub dependency_work_budget: u32,
    pub dependency_work_used: u32,
    pub proof_capacity: usize,
    pub proof_capacity_drops: u64,
    pub root: RefinedSearchObservation,
    /// Every retained whole-bag witness has one current-board quality entry.
    pub candidates: Vec<RefinedCandidate>,
    pub arms: Vec<RefinedArmObservation>,
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
    if config.observation_version == Some(4) {
        return collect_refined_bag_label(input, config);
    }
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
        work_used: 0, generation_work: 0, verification_work: 0, dependency_work: None,
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
        refinement: None,
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
        work_used: 0, generation_work: 0, verification_work: 0, dependency_work: None,
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
        reachable_whole_bag_witnesses: Some(witnesses), refinement: None,
    })
}

const REFINED_ARM_LIMIT: usize = 3;
const REFINED_DEPENDENCY_ORDERS: usize = 64;

/// Search state is deliberately independent of the final wire proof capacity.
/// The bounded pool contains continuation states, not all-supply certificates.
struct RefinedPhase {
    parent: ReachableNode,
    pool: Vec<SeedState>,
    best: Option<SeedState>,
    best_complete: Option<SeedState>,
    deepest: Option<SeedState>,
    generation_work: u32,
    attempts: u32,
    completed_candidates: u32,
    distinct_candidates: u32,
    candidate_duplicates: u64,
    pool_drops: u64,
    capacity_drops: u64,
    duplicates: u64,
    placements_considered: u64,
    stop_reason: &'static str,
}

fn refined_identity(seed: &SeedState) -> (Board, u8, CellMask) {
    (seed.board, seed.spins, seed.debt)
}

fn record_refined_prefix(phase: &mut RefinedPhase, seed: &SeedState) {
    if seed.targets.is_empty() { return; }
    if phase.best.as_ref().is_none_or(|prior| {
        (seed.spins, seed.targets.len()) > (prior.spins, prior.targets.len())
            || (seed.spins, seed.targets.len()) == (prior.spins, prior.targets.len())
                && seed.quality > prior.quality
    }) { phase.best = Some(seed.clone()); }
    if phase.deepest.as_ref().is_none_or(|prior| {
        (seed.targets.len(), seed.spins) > (prior.targets.len(), prior.spins)
            || (seed.targets.len(), seed.spins) == (prior.targets.len(), prior.spins)
                && seed.quality > prior.quality
    }) { phase.deepest = Some(seed.clone()); }
}

fn record_refined_candidates(
    phase: &mut RefinedPhase, candidates: Vec<SeedState>, capacity: usize,
    seen: &mut HashSet<(Board, u8, CellMask)>, random: &mut Random,
) {
    for candidate in candidates {
        if candidate.used != ALL_PIECES || candidate.targets.len() != 7 { continue; }
        phase.completed_candidates += 1;
        if phase.best_complete.as_ref().is_none_or(|prior| {
            (candidate.spins, candidate.score) > (prior.spins, prior.score)
        }) { phase.best_complete = Some(candidate.clone()); }
        let identity = refined_identity(&candidate);
        let fresh = seen.insert(identity);
        if !fresh { phase.candidate_duplicates += 1; }
        if let Some(index) = phase.pool.iter().position(|prior| refined_identity(prior) == identity) {
            if phase.pool[index].score < candidate.score { phase.pool[index] = candidate; }
        } else if fresh {
            if phase.pool.len() < capacity { phase.pool.push(candidate); }
            else {
                // A uniform reservoir of distinct continuation states retains
                // sparse low/high quality ranges before score-band sampling.
                // The strongest actual target is tracked independently.
                let replacement = (random.unit() * seen.len() as f64) as usize;
                if replacement < capacity { phase.pool[replacement] = candidate; }
                phase.pool_drops += 1;
            }
        }
    }
}

fn run_refined_phase(
    parent: ReachableNode, config: &LabelConfig, generation_budget: u32,
    seed: u64, cache: &mut PhysicsCache,
) -> RefinedPhase {
    let mut phase = RefinedPhase {
        parent, pool: vec![], best: None, best_complete: None, deepest: None, generation_work: 0,
        attempts: 0, completed_candidates: 0, distinct_candidates: 0,
        candidate_duplicates: 0, pool_drops: 0, capacity_drops: 0,
        duplicates: 0, placements_considered: 0, stop_reason: "attempt_limit",
    };
    let mut random = Random(if seed == 0 { 0x9e37_79b9_7f4a_7c15 } else { seed });
    let mut pool_random = Random(seed ^ 0x7265_7365_7276_6f69);
    let mut seen = HashSet::new();
    'attempts: for attempt in 0..config.attempts_per_node {
        if phase.generation_work >= generation_budget {
            phase.stop_reason = "generation_quota";
            break;
        }
        phase.attempts += 1;
        let mut job = GenerationJob::new(0, &phase.parent.node, config.beam_width,
            false, u64::from(attempt), LABEL_GOAL, config.profile, 0, 0, 1).for_learning();
        loop {
            if phase.generation_work >= generation_budget {
                record_refined_candidates(&mut phase, job.complete, config.max_nodes, &mut seen, &mut pool_random);
                phase.stop_reason = "generation_quota";
                break 'attempts;
            }
            let before_capacity = job.mino_capacity_pruned;
            let before_duplicates = job.duplicates;
            phase.placements_considered += job.advance(&mut random, LABEL_GOAL, cache);
            phase.generation_work += 1;
            phase.capacity_drops += job.mino_capacity_pruned - before_capacity;
            phase.duplicates += job.duplicates - before_duplicates;
            for candidate in job.frontier.iter()
                .chain(job.next_by_inventory.iter().flatten()).chain(job.complete.iter())
            {
                record_refined_prefix(&mut phase, candidate);
            }
            if !next_label_layer(&mut job) { continue; }
            record_refined_candidates(&mut phase, job.complete, config.max_nodes, &mut seen, &mut pool_random);
            break;
        }
    }
    phase.distinct_candidates = seen.len() as u32;
    phase
}

struct RefinedArmSelection {
    index: usize,
    stratum: usize,
    strata: usize,
    pool_size: usize,
    selected_index: usize,
}

fn diffuse_refinement_seed(mut value: u64) -> u64 {
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn board_refinement_seed(board: &Board, declared_seed: u64) -> u32 {
    let mut value = declared_seed ^ 0x9e37_79b9_7f4a_7c15;
    for &row in &board.rows {
        value = (value ^ u64::from(row)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    diffuse_refinement_seed(value) as u32
}

/// Equal-width score bands protect sparse quality tails. Select a nonempty
/// band uniformly, then one remaining physical node uniformly within it.
fn refined_arm_selection(
    pool: &[SeedState], seed: u32,
) -> (Vec<RefinedArmSelection>, f64, f64, usize) {
    let scores: Vec<(usize, f64)> = pool.iter().enumerate()
        .filter(|(_, candidate)| candidate.board.cell_count() != 0)
        .map(|(index, candidate)| (index, board_score(&candidate.board))).collect();
    let eligible_count = scores.len();
    let low = scores.iter().map(|&(_, score)| score).reduce(f64::min).unwrap_or(0.0);
    let high = scores.iter().map(|&(_, score)| score).reduce(f64::max).unwrap_or(0.0);
    let count = if high > low { 4 } else { 1 };
    let mut bands = vec![Vec::new(); count];
    for (index, score) in scores {
        let band = if count == 1 { 0 } else {
            (((score - low) / (high - low) * count as f64) as usize).min(count - 1)
        };
        bands[band].push(index);
    }
    let mut random = Random(diffuse_refinement_seed(u64::from(seed) ^ 0x6172_6d73_7472_6174).max(1));
    let mut selected = Vec::new();
    for _ in 0..REFINED_ARM_LIMIT.min(eligible_count) {
        let available: Vec<usize> = bands.iter().enumerate()
            .filter_map(|(index, band)| (!band.is_empty()).then_some(index)).collect();
        let band = available[(random.unit() * available.len() as f64) as usize];
        let pool_size = bands[band].len();
        let selected_index = (random.unit() * pool_size as f64) as usize;
        let index = bands[band].remove(selected_index);
        selected.push(RefinedArmSelection { index, stratum: band, strata: count, pool_size, selected_index });
    }
    (selected, low, high, count)
}

fn replay_refined_prefix(
    parent: &ReachableNode, seed: &SeedState, cache: &mut PhysicsCache,
) -> Result<RefinedPrefixWitness, String> {
    if seed.targets.is_empty() || seed.targets.len() > 7 || seed.frames.len() != seed.targets.len() {
        return Err("A refined observation prefix needs one through seven actual locks".into());
    }
    let mut board = parent.node.board;
    let mut deleted = 0;
    let mut spins = 0;
    let mut used = 0;
    let mut queue = Vec::new();
    let mut inputs = Vec::new();
    let mut frames = Vec::new();
    for (index, (target, expected)) in seed.targets.iter().zip(&seed.frames).enumerate() {
        if used & target.piece.bit() != 0 { return Err("A refined prefix repeated a piece label".into()); }
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
            .ok_or_else(|| "A refined prefix has no matching full-board input replay".to_string())?;
        deleted = deleted_after(deleted, result.cleared_rows)
            .ok_or_else(|| "A refined prefix exceeded logical-row capacity".to_string())?;
        spins += result.lines;
        if parent.node.spins + spins != expected.spins {
            return Err("A refined prefix mixed parent spins into its local target".into());
        }
        used |= target.piece.bit();
        queue.push(target.piece as u8);
        inputs.push(placement.path);
        frames.push(DisplayFrame {
            rows: result.board.rows, piece: piece_name(target.piece),
            placed_cells: result.cells.map(|(x, y)| (x as u8, y as u8)),
            spins, placed: index as u16 + 1, lines: result.lines, spin: result.spin,
        });
        board = result.board;
    }
    if board != seed.board || deleted != seed.deleted || parent.node.spins + spins != seed.spins {
        return Err("A refined prefix disagrees with its actual end field".into());
    }
    Ok(RefinedPrefixWitness {
        placements: seed.targets.len() as u8, observed_spins: spins, deleted_rows: deleted,
        targets: seed.targets.clone(), representative_queue: queue,
        representative_inputs: inputs, frames, trust: "input-reachable",
    })
}

fn finish_refined_observation(
    phase: &RefinedPhase, quota: u32, whole_bag_replay_work: u32,
    cache: &mut PhysicsCache,
) -> Result<RefinedSearchObservation, String> {
    // Both references carry self-contained input evidence. Replaying an
    // identical target/deepest path twice is explicitly charged twice.
    let target_witness = phase.best.as_ref().map(|seed| replay_refined_prefix(&phase.parent, seed, cache)).transpose()?;
    let deepest_witness = phase.deepest.as_ref().map(|seed| replay_refined_prefix(&phase.parent, seed, cache)).transpose()?;
    let verification_work = whole_bag_replay_work
        + target_witness.as_ref().map_or(0, |proof| u32::from(proof.placements))
        + deepest_witness.as_ref().map_or(0, |proof| u32::from(proof.placements));
    let observed_spins = target_witness.as_ref().map_or(0, |proof| proof.observed_spins);
    let work_used = phase.generation_work + verification_work;
    if work_used > quota { return Err("Refined input proofs exceeded their reserved phase budget".into()); }
    Ok(RefinedSearchObservation {
        allocated_work: quota, work_used, generation_work: phase.generation_work,
        verification_work, attempts: phase.attempts,
        targets: SpinTargets { found_spin: observed_spins > 0, observed_spins },
        deepest_reached_placements: deepest_witness.as_ref().map_or(0, |proof| proof.placements),
        target_witness, deepest_witness,
        completed_whole_bag_candidates: phase.completed_candidates,
        distinct_whole_bag_candidates: phase.distinct_candidates,
        censored: true, stop_reason: phase.stop_reason,
    })
}

fn reachable_from_refined(
    proof: &RefinedPrefixWitness, parent_id: Option<u64>, bags: u8, start_spins: u8,
) -> ReachableWitness {
    let mut frames = proof.frames.clone();
    for frame in &mut frames { frame.spins += start_spins; frame.placed += u16::from(bags) * 7; }
    ReachableWitness {
        certified_parent_id: None, reachable_parent_id: parent_id,
        completed_parent_bags: bags, additional_placements: proof.placements,
        observed_spins: start_spins + proof.observed_spins, targets: proof.targets.clone(), frames,
    }
}

fn component_orders(count: usize) -> u32 { (1..=count as u32).product() }

fn component_permutation(indices: &[usize], mut rank: u32) -> Vec<usize> {
    let mut remaining = indices.to_vec();
    let mut result = Vec::with_capacity(indices.len());
    while !remaining.is_empty() {
        let divisor = component_orders(remaining.len() - 1);
        let at = (rank / divisor) as usize;
        rank %= divisor;
        result.push(remaining.remove(at));
    }
    result
}

/// This graph concerns contact among the candidate's typed logical minos.
/// SRS+ kicks, access, earlier lines and spawn space can create dependencies
/// between disconnected components. Component results therefore never prune
/// search, certify a complete order or enter the CNN spin targets.
fn dependency_components(targets: &[TargetOperation]) -> Vec<Vec<usize>> {
    let mut component: Vec<usize> = (0..targets.len()).collect();
    for a in 0..targets.len() {
        for b in a + 1..targets.len() {
            if targets[a].cells.iter().any(|&(ax, ay)| targets[b].cells.iter().any(|&(bx, by)| {
                u16::from(ax.abs_diff(bx)) + u16::from(ay.abs_diff(by)) == 1
            })) {
                let from = component[b];
                let to = component[a];
                for value in &mut component { if *value == from { *value = to; } }
            }
        }
    }
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, group) in component.into_iter().enumerate() { groups.entry(group).or_default().push(index); }
    let mut result: Vec<Vec<usize>> = groups.into_values().collect();
    for group in &mut result { group.sort_by_key(|&index| targets[index].piece as u8); }
    result.sort_by_key(|group| targets[group[0]].piece as u8);
    result
}

fn observe_component_dependencies(
    origin: Board, targets: &[TargetOperation], budget: u32, seed: u32,
    cache: &mut PhysicsCache,
) -> ComponentDependencyApproximation {
    let mut result = ComponentDependencyApproximation {
        schema_version: 1, policy: "touching-components-origin-only-v1",
        connectivity: "orthogonal-logical-target-cells",
        scope: "independent component orders on the origin with other current-bag minos absent; SRS+ can link disconnected components; sums are not P7, quiet-safe coverage or construction probability",
        max_orders_per_component: REFINED_DEPENDENCY_ORDERS,
        work_used: 0, total_orders: 0, tested_orders: 0, successful_orders: 0, components: vec![],
    };
    for (component_index, group) in dependency_components(targets).into_iter().enumerate() {
        let total_orders = component_orders(group.len());
        let sample_count = (total_orders as usize).min(REFINED_DEPENDENCY_ORDERS);
        let mut observation = ComponentOrderObservation {
            pieces: group.iter().map(|&index| targets[index].piece).collect(),
            total_orders, tested_orders: 0, successful_orders: 0,
            interrupted_orders: 0, work_used: 0, exhaustive: false,
        };
        let offset = (seed.wrapping_add((component_index as u32).wrapping_mul(0x9e37_79b9))) % total_orders;
        for sample in 0..sample_count {
            if result.work_used >= budget { break; }
            let rank = if sample_count == total_orders as usize { sample as u32 }
                else { (offset + (sample as u32 * total_orders / sample_count as u32)) % total_orders };
            let order = component_permutation(&group, rank);
            let mut board = origin;
            let mut deleted = 0;
            let mut successful = true;
            let mut interrupted = false;
            for index in order {
                let target = &targets[index];
                let mut accepted = None;
                for pose in projected_poses(target, deleted) {
                    if result.work_used >= budget { interrupted = true; break; }
                    result.work_used += 1;
                    observation.work_used += 1;
                    let Some(mut placement) = cache.target(board, pose, false) else { continue; };
                    let Some(mut placed) = board.apply(&placement) else { continue; };
                    if placed.lines > 1 { continue; }
                    if placed.lines == 1 && !placed.spin {
                        if result.work_used >= budget { interrupted = true; break; }
                        result.work_used += 1;
                        observation.work_used += 1;
                        let Some(spinning) = cache.target(board, pose, true) else { continue; };
                        let Some(spun) = board.apply(&spinning) else { continue; };
                        placement = spinning;
                        placed = spun;
                    }
                    if placed.lines > 1 || (placed.lines == 1 && !placed.spin)
                        || normalize_target(placement.pose, deleted).as_ref() != Some(target)
                    { continue; }
                    let Some(next_deleted) = deleted_after(deleted, placed.cleared_rows) else { continue; };
                    accepted = Some((placed.board, next_deleted));
                    break;
                }
                if interrupted { break; }
                if let Some((next_board, next_deleted)) = accepted {
                    board = next_board; deleted = next_deleted;
                } else { successful = false; break; }
            }
            if interrupted { observation.interrupted_orders += 1; break; }
            observation.tested_orders += 1;
            observation.successful_orders += u32::from(successful);
        }
        observation.exhaustive = observation.tested_orders == total_orders;
        result.total_orders += total_orders;
        result.tested_orders += observation.tested_orders;
        result.successful_orders += observation.successful_orders;
        result.components.push(observation);
    }
    result
}

/// V4 spends a common bounded quota on the supplied field and each of a few
/// sampled actual first-bag endpoints. Their independent next-bag maxima are
/// separately usable targets. The wire proof cap never stops these searches.
fn collect_refined_bag_label(input: BoardSample, config: LabelConfig) -> Result<BoardLabel, String> {
    config.validate()?;
    let board = Board { rows: input.rows };
    if input.id.trim().is_empty() || input.source.bag_boundary < 2
        || input.source.kind.trim().is_empty() || input.source.reference.trim().is_empty()
        || input.source.family.trim().is_empty() || !board.is_valid() || board.cell_count() == 0
        || board.rows.iter().any(|row| *row == 1023)
    {
        return Err("Refined labels require a named nonempty post-clear field after at least two completed bags".into());
    }
    for forbidden in ["referencePrior", "priorIsMeasuredOutcome", "currentBoardTeacher",
        "terrainScore", "fieldQuality", "rankingScore", "shapePreference", "observedChoices", "boardEvaluation"]
    {
        if input.source.extra.contains_key(forbidden) {
            return Err(format!("Evaluation metadata is not a spin-only learning input: {forbidden}"));
        }
    }
    let dependency_budget = 4_096.min(config.work_budget / 32);
    let phase_quota = (config.work_budget - dependency_budget) / (REFINED_ARM_LIMIT as u32 + 1);
    let proof_reserve = 256.min(phase_quota / 4);
    let generation_budget = phase_quota - proof_reserve;
    let proof_capacity = config.max_nodes.min(REACHABLE_WITNESS_CAPACITY);
    // Fixed seed 1 remains reproducible, but cannot choose the same three
    // quality bands for every different board. This explicit recorded draw
    // seed depends only on occupancy and the declared collector seed.
    let selection_seed = board_refinement_seed(&board, config.seed);
    let mut cache = PhysicsCache::default();
    let root_parent = ReachableNode { node: node(board, 0, 0, [0; HEIGHT]), bags: 0, witness_id: None };
    let root = run_refined_phase(root_parent, &config, generation_budget, config.seed, &mut cache);
    let (selected, selection_score_min, selection_score_max, selection_band_count) =
        refined_arm_selection(&root.pool, selection_seed);
    let root_prefix_units = root.best.as_ref().map_or(0, |seed| seed.targets.len() as u32)
        + root.deepest.as_ref().map_or(0, |seed| seed.targets.len() as u32);
    let root_proof_limit = 12.min(proof_capacity.saturating_sub(1))
        .min((phase_quota - root.generation_work - root_prefix_units) as usize / 7);
    if selected.len() > root_proof_limit {
        return Err("The refined arm allocation cannot retain its required parent proofs".into());
    }
    let mut root_seeds = Vec::<SeedState>::new();
    let mut root_identities = HashSet::new();
    for choice in &selected {
        let candidate = &root.pool[choice.index];
        root_identities.insert(refined_identity(candidate));
        root_seeds.push(candidate.clone());
    }
    let mut root_remainder: Vec<&SeedState> = root.pool.iter().collect();
    if let Some(best) = &root.best_complete { root_remainder.push(best); }
    root_remainder.sort_by(|a, b| b.spins.cmp(&a.spins).then_with(|| b.score.total_cmp(&a.score)));
    for candidate in root_remainder {
        if root_seeds.len() >= root_proof_limit { break; }
        if root_identities.insert(refined_identity(candidate)) { root_seeds.push(candidate.clone()); }
    }
    let mut witnesses = Vec::new();
    let mut root_ids = HashMap::new();
    let mut next_id = 1;
    for candidate in &root_seeds {
        let witness = replay_reachable_bag(&root.parent, candidate, next_id, &mut cache)?;
        root_ids.insert(refined_identity(candidate), next_id);
        witnesses.push(witness);
        next_id += 1;
    }
    let root_replay_work = witnesses.len() as u32 * 7;
    let mut arm_phases = Vec::new();
    for choice in &selected {
        let candidate = &root.pool[choice.index];
        let parent_id = root_ids[&refined_identity(candidate)];
        let parent = ReachableNode {
            node: node(candidate.board, candidate.spins, 1, candidate.debt),
            bags: 1, witness_id: Some(parent_id),
        };
        let seed = config.seed ^ parent_id.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ 0x7265_6669_6e65_6172;
        arm_phases.push(run_refined_phase(parent, &config, generation_budget, seed, &mut cache));
    }
    // Output selection retains strong full continuations without changing
    // the independently measured arm targets or their generation allocation.
    let mut first_children = Vec::<(usize, SeedState)>::new();
    let mut other_children = Vec::<(usize, SeedState)>::new();
    for (index, phase) in arm_phases.iter().enumerate() {
        if let Some(best) = &phase.best_complete { first_children.push((index, best.clone())); }
        other_children.extend(phase.pool.iter().cloned().map(|candidate| (index, candidate)));
    }
    let child_order = |a: &(usize, SeedState), b: &(usize, SeedState)| {
        b.1.spins.cmp(&a.1.spins).then_with(|| b.1.score.total_cmp(&a.1.score)).then_with(|| a.0.cmp(&b.0))
    };
    first_children.sort_by(child_order);
    other_children.sort_by(child_order);
    let mut child_seen = HashSet::new();
    let mut arm_replay_work = vec![0_u32; arm_phases.len()];
    for (arm, candidate) in first_children.into_iter().chain(other_children) {
        if witnesses.len() >= proof_capacity { break; }
        if !child_seen.insert((arm, refined_identity(&candidate))) { continue; }
        let phase = &arm_phases[arm];
        let prefix_units = phase.best.as_ref().map_or(0, |seed| seed.targets.len() as u32)
            + phase.deepest.as_ref().map_or(0, |seed| seed.targets.len() as u32);
        if phase.generation_work + arm_replay_work[arm] + prefix_units + 7 > phase_quota { continue; }
        witnesses.push(replay_reachable_bag(&phase.parent, &candidate, next_id, &mut cache)?);
        arm_replay_work[arm] += 7;
        next_id += 1;
    }
    let root_observation = finish_refined_observation(&root, phase_quota, root_replay_work, &mut cache)?;
    let mut arms = Vec::new();
    for ((choice, phase), replay_work) in selected.iter().zip(&arm_phases).zip(arm_replay_work) {
        let observation = finish_refined_observation(phase, phase_quota, replay_work, &mut cache)?;
        if observation.generation_work == 0 { continue; }
        if observation.targets.observed_spins > 6 { return Err("A refined arm exceeded one inventory's spin capacity".into()); }
        arms.push(RefinedArmObservation {
            candidate_witness_id: phase.parent.witness_id.unwrap(),
            selection_stratum: choice.stratum, selection_strata: choice.strata,
            selection_pool_size: choice.pool_size, selection_index: choice.selected_index,
            observation,
        });
    }
    let dependency_per_arm = if arms.is_empty() { 0 } else { dependency_budget / arms.len() as u32 };
    let selected_ids: HashSet<u64> = arms.iter().map(|arm| arm.candidate_witness_id).collect();
    let mut candidates = Vec::new();
    let mut dependency_work = 0;
    for witness in &witnesses {
        let dependency = if selected_ids.contains(&witness.id) {
            let value = observe_component_dependencies(board, &witness.targets, dependency_per_arm,
                selection_seed.wrapping_add(witness.id as u32), &mut cache);
            dependency_work += value.work_used;
            Some(value)
        } else { None };
        candidates.push(RefinedCandidate {
            witness_id: witness.id, structural_quality: board_score(&Board { rows: witness.end_rows }), dependency,
        });
    }
    let target_witness = root_observation.target_witness.as_ref()
        .map(|proof| reachable_from_refined(proof, None, 0, 0));
    let mut reachable_witness = root_observation.deepest_witness.as_ref()
        .map(|proof| reachable_from_refined(proof, None, 0, 0));
    let mut maximum = root_observation.targets.observed_spins;
    for (arm, phase) in arms.iter().zip(&arm_phases) {
        maximum = maximum.max(phase.parent.node.spins + arm.observation.targets.observed_spins);
        if let Some(proof) = &arm.observation.deepest_witness {
            let next = reachable_from_refined(proof, Some(arm.candidate_witness_id), 1, phase.parent.node.spins);
            if reachable_witness.as_ref().is_none_or(|prior| {
                (7 * next.completed_parent_bags + next.additional_placements, next.observed_spins)
                    > (7 * prior.completed_parent_bags + prior.additional_placements, prior.observed_spins)
            }) { reachable_witness = Some(next); }
        }
    }
    let mut depths: Vec<DepthObservation> = (1..=2).map(|bags| DepthObservation { bags, ..Default::default() }).collect();
    let mut best_two_bag_path: Option<TwoBagPath> = None;
    for witness in &witnesses {
        let depth = &mut depths[usize::from(witness.bags - 1)];
        depth.geometric_whole_bag_chains += 1;
        depth.productive_geometric_whole_bag_chains += u32::from(witness.end_spins > 0);
        depth.max_geometric_whole_bag_spins = Some(depth.max_geometric_whole_bag_spins.unwrap_or(0).max(witness.end_spins));
        if witness.bags == 2 && best_two_bag_path.as_ref().is_none_or(|prior| {
            (witness.end_spins, witness.end_spins - witness.start_spins) > (prior.total_spins, prior.second_bag_spins)
        }) {
            best_two_bag_path = Some(TwoBagPath {
                witness_id: witness.id, first_bag_spins: witness.start_spins,
                second_bag_spins: witness.end_spins - witness.start_spins, total_spins: witness.end_spins,
            });
        }
    }
    let generation_work = root_observation.generation_work + arms.iter().map(|arm| arm.observation.generation_work).sum::<u32>();
    let verification_work = root_observation.verification_work + arms.iter().map(|arm| arm.observation.verification_work).sum::<u32>();
    let work_used = generation_work + verification_work + dependency_work;
    if work_used > config.work_budget { return Err("Refined collection exceeded its declared total work budget".into()); }
    let completed_depth = witnesses.iter().map(|witness| witness.bags).max().unwrap_or(0);
    let reached = reachable_witness.as_ref().map_or(0, |proof| 7 * u16::from(proof.completed_parent_bags) + u16::from(proof.additional_placements));
    let distinct_candidates = root.distinct_candidates + arm_phases.iter().map(|phase| phase.distinct_candidates).sum::<u32>();
    let proof_drops = u64::from(distinct_candidates).saturating_sub(witnesses.len() as u64);
    let capacity_drops = root.capacity_drops + root.pool_drops
        + arm_phases.iter().map(|phase| phase.capacity_drops + phase.pool_drops).sum::<u64>() + proof_drops;
    let duplicates = root.duplicates + root.candidate_duplicates
        + arm_phases.iter().map(|phase| phase.duplicates + phase.candidate_duplicates).sum::<u64>();
    let placements_considered = root.placements_considered + arm_phases.iter().map(|phase| phase.placements_considered).sum::<u64>();
    let phase_limited = root.stop_reason == "generation_quota" || arm_phases.iter().any(|phase| phase.stop_reason == "generation_quota");
    let measured = LabelMeasurements {
        max_reachable_observed_spins: maximum,
        geometric_whole_bag_chains: witnesses.len() as u32,
        productive_geometric_whole_bag_chains: witnesses.iter().filter(|witness| witness.end_spins > 0).count() as u32,
        certified_whole_bag_chains: 0, productive_certified_whole_bag_chains: 0,
        max_certified_observed_spins: None, depths, work_used, generation_work, verification_work,
        dependency_work: Some(dependency_work), geometric_placements_considered: placements_considered,
        canonical_supplies_checked: 0, quiet_policy_rejections: 0, supply_rejections: 0,
        graph_overflows: 0, node_capacity_drops: 0, censored: true,
        stop_reason: if phase_limited { "phase_quotas" } else { "bounded_refinement_complete" }.into(),
        evidence_kind: "refined input-reachable one-bag targets per supplied board and selected actual first-bag endpoint; no P7 or quiet-safe coverage claimed",
        distinctness: "physical continuation states by board, actual spins and inherited debt within each origin; per-arm local maxima remain separate; not supply orders or success probability",
        work_unit: "one bounded GenerationJob.advance, one retained lock replay, or one origin-only component target query; replay budgets reserved; P7 verification disabled",
        lookahead: Some(LookaheadMeasurements {
            target_horizon_bags: 1, requested_bags: 2, deepest_attempted_bag: if arms.is_empty() { 1 } else { 2 },
            deepest_completed_bag: completed_depth, deepest_reached_placements: reached,
            reachable_whole_bag_chains: witnesses.len() as u32, witness_capacity: proof_capacity,
            capacity_drops, duplicate_continuations: duplicates, best_two_bag_path,
        }),
    };
    let observed_spins = root_observation.targets.observed_spins;
    if observed_spins > 6 { return Err("A refined root exceeded one inventory's spin capacity".into()); }
    let refinement = RefinementMeasurements {
        schema_version: 1, policy: "uniform-score-band-three-arm-equal-quota-v1", selection_seed,
        selection_score_min, selection_score_max, selection_band_count,
        candidate_pool_capacity: config.max_nodes, candidate_pool_size: root.pool.len(),
        candidate_pool_drops: root.pool_drops, candidate_duplicates: root.candidate_duplicates,
        arm_limit: REFINED_ARM_LIMIT, arm_quota: phase_quota,
        dependency_work_budget: dependency_budget, dependency_work_used: dependency_work,
        proof_capacity, proof_capacity_drops: proof_drops,
        root: root_observation, candidates, arms,
    };
    Ok(BoardLabel {
        schema_version: 4, task: REFINED_BAG_TASK, policy: REFINED_BAG_POLICY,
        input, config, targets: SpinTargets { found_spin: observed_spins > 0, observed_spins },
        measured, certified_witnesses: vec![], reachable_witness,
        target_horizon_bags: Some(1), target_witness,
        reachable_whole_bag_witnesses: Some(witnesses), refinement: Some(refinement),
    })
}
