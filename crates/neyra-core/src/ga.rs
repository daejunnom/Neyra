//! Dataset-free genetic optimization of a small, explicit Sonic Drop policy.
//!
//! Every decision scores every actual reachable lock of every unused bag
//! piece. A selected rollout is a bounded policy, not an exhaustive search.
//! The ordinary search's fixed-plan verifier independently certifies its
//! prefix. Physical progress after the first rejected bag is auxiliary only.

use crate::physics::{Board, MovementMode, Piece, Placement, HEIGHT, WIDTH};
use crate::search::ga_verify::{Proof, Request, Verifier};
use crate::search::{DisplayFrame, SearchConfig, SearchSession};
use crate::supply::{ALL_PIECES, P7_COUNT};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;

mod evolution;
mod checkpoint;
mod policy;
mod physical;
mod repair;

pub const LEGACY_SOURCE_POLICY: &str = "neyra-ga-sonic-full-locks-fixed-plan-v2";
pub const SOURCE_POLICY: &str = "neyra-ga-sonic-full-locks-fixed-plan-v3";
pub const PHYSICAL_SOURCE_POLICY: &str = "neyra-ga-sonic-physical-v4";
pub const FEATURE_SCHEMA: &str = "neyra-ga-features-v1";
pub const FEATURE_COUNT: usize = 24;
pub const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "spinSingle", "maximumHeight", "aggregateHeight", "occupiedCells",
    "columnHoles", "holeDepth", "surfaceBumpiness", "maximumWellDepth",
    "totalWellDepth", "rowTransitions", "columnTransitions", "oneCellRowDeficits",
    "twoCellRowDeficits", "threeCellRowDeficits", "fourCellRowDeficits",
    "roofedAir", "threeCornerEmptyCells", "landingHeight", "blockContacts",
    "floorContacts", "wallContacts", "isT", "isO", "blockedSpawns",
];

/// An explicitly untrained inductive bias. These are neither fitted weights
/// nor a copied engine evaluator, teacher label, opener, or success claim.
pub const INITIAL_WEIGHTS: [f64; FEATURE_COUNT] = [
    8.0, -1.0, -2.0, -0.5, -0.8, -0.5, -0.5, -0.25,
    -0.25, -0.25, -0.25, 0.5, 0.3, 0.2, 0.1, 0.15,
    0.25, -0.5, 0.4, 0.1, 0.1, 0.0, 0.0, -4.0,
];

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase", deny_unknown_fields)]
pub struct GaConfig {
    pub schema_version: String,
    pub population_size: usize,
    pub generations: u32,
    pub seed: u32,
    pub episodes_per_genome: usize,
    pub max_pieces: u16,
    pub piece_goal: u16,
    pub min_success_percent: f64,
    pub bag_min_success_percent: f64,
    pub risk_min_spins_per_bag: u8,
    pub ignore_first_bag_probability: bool,
    pub first_bag_min_success_percent: f64,
    pub mutation_rate: f64,
    pub mutation_scale: f64,
    pub elite_count: usize,
    pub tournament_size: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical: Option<physical::Config>,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            schema_version: "neyra-ga-config-v1".into(), population_size: 8, generations: 3,
            seed: 1, episodes_per_genome: 1, max_pieces: 126, piece_goal: 119,
            min_success_percent: 100.0, bag_min_success_percent: 100.0,
            risk_min_spins_per_bag: 3, ignore_first_bag_probability: true,
            first_bag_min_success_percent: 80.0, mutation_rate: 0.2,
            mutation_scale: 0.3, elite_count: 2, tournament_size: 3, physical: None,
        }
    }
}

impl GaConfig {
    /// New experiments retain actual chance, including the first bag. Legacy
    /// v1 configurations keep their original defaults and exact resume path.
    pub fn frontier_defaults() -> Self {
        Self { schema_version: "neyra-ga-config-v2".into(), min_success_percent: 0.01,
            bag_min_success_percent: 0.01, first_bag_min_success_percent: 0.0,
            risk_min_spins_per_bag: 0, ignore_first_bag_probability: false,
            ..Self::default() }
    }
    pub fn is_frontier(&self) -> bool { self.schema_version == "neyra-ga-config-v2" }
    pub fn is_physical(&self) -> bool { self.schema_version == "neyra-ga-config-v3" }
    pub fn physical_defaults() -> Self {
        Self { schema_version: "neyra-ga-config-v3".into(), episodes_per_genome: 4,
            physical: Some(physical::Config::with_repair_defaults()), ..Self::frontier_defaults() }
    }
    pub fn parameter_count(&self) -> usize {
        if self.physical.as_ref().is_some_and(|p|p.context_features) { policy::CONTEXT_FEATURE_COUNT } else { FEATURE_COUNT }
    }
    fn feature_schema(&self) -> &'static str { policy::feature_schema(self.parameter_count()).unwrap() }
    pub fn source_policy(&self) -> &'static str {
        if self.is_physical() { PHYSICAL_SOURCE_POLICY } else if self.is_frontier() { SOURCE_POLICY } else { LEGACY_SOURCE_POLICY }
    }

    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.schema_version.as_str(), "neyra-ga-config-v1" | "neyra-ga-config-v2" | "neyra-ga-config-v3") {
            return Err("Unknown GA configuration schema".into());
        }
        if self.is_physical() != self.physical.is_some() { return Err("Physical settings require an explicit v4 experiment".into()); }
        if let Some(physical) = &self.physical { physical.validate()?; }
        if self.is_frontier() && self.ignore_first_bag_probability {
            return Err("GA probability-frontier experiments include the first bag".into());
        }
        if !(4..=128).contains(&self.population_size) || !(1..=10_000).contains(&self.generations)
            || !(1..=16).contains(&self.episodes_per_genome) || self.max_pieces == 0 || self.max_pieces > 126
            || self.piece_goal == 0 || self.piece_goal > self.max_pieces
            || self.elite_count == 0 || self.elite_count >= self.population_size
            || !(2..=self.population_size).contains(&self.tournament_size) {
            return Err("Invalid GA population, horizon, generation, or selection limits".into());
        }
        if !self.mutation_rate.is_finite() || !(0.0..=1.0).contains(&self.mutation_rate)
            || !self.mutation_scale.is_finite() || !(0.001..=4.0).contains(&self.mutation_scale) {
            return Err("Invalid GA mutation parameters".into());
        }
        SearchSession::new(self.search_config()).map(|_| ())
    }

    pub fn search_config(&self) -> SearchConfig {
        SearchConfig {
            probability_frontier: self.is_frontier() || self.is_physical(),
            min_success_percent: self.min_success_percent,
            bag_min_success_percent: self.bag_min_success_percent,
            risk_min_spins_per_bag: self.risk_min_spins_per_bag,
            ignore_first_bag_probability: self.ignore_first_bag_probability,
            first_bag_min_success_percent: self.first_bag_min_success_percent,
            piece_goal: self.piece_goal, max_pieces: self.max_pieces,
            seed: u64::from(self.seed), ..SearchConfig::default()
        }
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        if json.len() > 16_384 { return Err("GA configuration is too large".into()); }
        let value: serde_json::Value = serde_json::from_str(json).map_err(|e| format!("Invalid GA settings: {e}"))?;
        let defaults = if value.get("schemaVersion").and_then(|v| v.as_str()) == Some("neyra-ga-config-v1") {
            Self::default()
        } else if value.get("schemaVersion").and_then(|v| v.as_str()) == Some("neyra-ga-config-v2") {
            Self::frontier_defaults()
        } else { Self::physical_defaults() };
        let mut merged = serde_json::to_value(defaults).map_err(|e| e.to_string())?;
        let object = value.as_object().ok_or("GA configuration must be an object")?;
        for (key, value) in object { merged[key] = value.clone(); }
        let config: Self = serde_json::from_value(merged).map_err(|e| format!("Invalid GA settings: {e}"))?;
        config.validate()?;
        Ok(config)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Random { state: u32 }
impl Random {
    fn new(seed: u32) -> Self { Self { state: if seed == 0 { 0x9e37_79b9 } else { seed } } }
    fn next(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.state = x;
        x
    }
    fn unit(&mut self) -> f64 { f64::from(self.next()) / 4_294_967_296.0 }
    fn index(&mut self, n: usize) -> usize { (self.next() as usize) % n }
    fn normal(&mut self) -> f64 { (0..12).map(|_| self.unit()).sum::<f64>() - 6.0 }
}

fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}
fn weights_digest(weights: &[f64]) -> String {
    let bytes: Vec<_> = weights.iter().flat_map(|w| w.to_le_bytes()).collect();
    digest(&bytes)
}
fn validate_weights(weights: &[f64]) -> Result<(), String> {
    policy::validate_weights(weights)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GaModel {
    pub schema_version: String,
    pub source_policy: String,
    pub feature_schema: String,
    pub feature_names: Vec<String>,
    pub model_id: String,
    pub weights_sha256: String,
    pub weights: Vec<f64>,
    pub training_state: String,
    pub trained: bool,
    pub learning_complete: bool,
    pub deployment_ready: bool,
    pub training_config: GaConfig,
    pub completed_generations: u32,
    pub evaluated_episodes: u64,
    pub best: Option<EpisodeResult>,
    pub fitness: Option<Fitness>,
}

impl GaModel {
    fn new(weights: Vec<f64>, config: GaConfig) -> Self {
        let hash = weights_digest(&weights);
        Self { schema_version: "neyra-ga-model-v1".into(), source_policy: config.source_policy().into(),
            feature_schema: policy::feature_schema(weights.len()).unwrap().into(), feature_names: policy::feature_names(weights.len()).unwrap(),
            model_id: format!("ga-{}", &hash[..20]), weights_sha256: hash, weights,
            training_state: "untrained-initializer".into(), trained: false, learning_complete: false,
            deployment_ready: false, training_config: config, completed_generations: 0,
            evaluated_episodes: 0, best: None, fitness: None }
    }
    pub fn initializer(config: GaConfig) -> Result<Self, String> {
        config.validate()?;
        Ok(Self::new(policy::initializer(config.parameter_count() > FEATURE_COUNT), config))
    }
    pub fn from_json(json: &str) -> Result<Self, String> {
        if json.len() > 32 * 1024 * 1024 { return Err("GA model is too large".into()); }
        let model: Self = serde_json::from_str(json).map_err(|e| format!("Invalid GA model: {e}"))?;
        model.validate()?;
        Ok(model)
    }
    pub fn validate(&self) -> Result<(), String> {
        validate_weights(&self.weights)?;
        if self.schema_version != "neyra-ga-model-v1" || self.source_policy != self.training_config.source_policy()
            || self.weights.len() != self.training_config.parameter_count()
            || Some(self.feature_schema.as_str()) != policy::feature_schema(self.weights.len())
            || Some(self.feature_names.clone()) != policy::feature_names(self.weights.len())
            || self.weights_sha256 != weights_digest(&self.weights)
            || self.model_id != format!("ga-{}", &self.weights_sha256[..20])
            || self.learning_complete || self.deployment_ready {
            return Err("GA model schema, policy, feature contract, digest, or readiness is invalid".into());
        }
        self.training_config.validate()
    }
}

/// Measured totals retained for both policies. Legacy v2 reproduction uses
/// this lexicographic ordering; v3 reproduction uses Pareto ranks plus
/// independently preserved raw, probability, strict and foundation niches.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Fitness {
    pub certified_episodes: u32,
    pub qualified_spins: u32,
    pub qualified_bags: u32,
    pub raw_spins: u32,
    pub raw_completed_episodes: u32,
    pub failure_quiet_safe: u32,
    pub failure_success: u32,
    pub raw_pieces: u32,
    pub final_height: u32,
    pub certified_budget_pieces: u32,
}
impl Ord for Fitness {
    fn cmp(&self, other: &Self) -> Ordering {
        self.certified_episodes.cmp(&other.certified_episodes)
            .then_with(|| if self.certified_episodes > 0 {
                other.certified_budget_pieces.cmp(&self.certified_budget_pieces)
            } else { Ordering::Equal })
            .then(self.qualified_spins.cmp(&other.qualified_spins))
            .then(self.qualified_bags.cmp(&other.qualified_bags))
            .then(self.failure_quiet_safe.cmp(&other.failure_quiet_safe))
            .then(self.failure_success.cmp(&other.failure_success))
            .then(self.raw_spins.cmp(&other.raw_spins))
            .then(self.raw_completed_episodes.cmp(&other.raw_completed_episodes))
            .then(self.raw_pieces.cmp(&other.raw_pieces))
            .then(other.final_height.cmp(&self.final_height))
    }
}
impl PartialOrd for Fitness { fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) } }
impl Fitness {
    fn add(&mut self, r: &EpisodeResult) {
        self.certified_episodes += u32::from(r.certified40);
        self.qualified_spins += u32::from(r.qualified_spins);
        self.qualified_bags += u32::from(r.qualified_bags);
        self.raw_spins += u32::from(r.raw_spins);
        self.raw_completed_episodes += u32::from(r.raw_spins >= 40);
        if let Some(p) = r.bag_proofs.iter().find(|p| !p.qualified) {
            self.failure_quiet_safe += u32::from(p.quiet_safe_supplies);
            self.failure_success += u32::from(p.success);
        }
        self.raw_pieces += u32::from(r.raw_pieces);
        self.final_height += u32::from(Board { rows: r.rows }.height());
        if r.certified40 { self.certified_budget_pieces += u32::from(r.budget_pieces.unwrap_or(126)); }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EpisodeResult {
    pub seed: u32,
    pub raw_spins: u8,
    pub raw_pieces: u16,
    pub raw_completed40: bool,
    pub qualified_spins: u8,
    pub qualified_bags: u16,
    pub certified40: bool,
    pub budget_pieces: Option<u16>,
    pub probability: f64,
    pub first_bag_probability: Option<f64>,
    pub first_bag_symmetry_assumed: bool,
    pub rows: [u16; HEIGHT],
    pub qualified_rows: [u16; HEIGHT],
    pub frames: Vec<DisplayFrame>,
    pub qualified_frames: Vec<DisplayFrame>,
    pub placements: Vec<Placement>,
    pub bag_proofs: Vec<Proof>,
    pub unchecked_suffix_bags: u16,
    pub termination: String,
    pub input_replay_verified: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frontier: Option<evolution::FrontierMetrics>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical: Option<physical::EpisodeMetrics>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Counters {
    pub work_units: u64,
    /// Full reachable_locks(piece) calls. Not comparable to legacy search steps.
    pub physics_queries: u64,
    pub candidate_locks: u64,
    pub legal_candidate_locks: u64,
    pub replay_queries: u64,
    /// Canonical verifier increments: graph node or at most 64 supply queries.
    pub verification_work: u64,
    pub supply_queries: u64,
    pub evaluated_episodes: u64,
    pub cache_hits: u64,
    /// Active call wall time; pauses and checkpoint I/O are excluded.
    pub elapsed_ms: f64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Choice { placement: Placement, score: f64 }

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct Episode {
    seed: u32,
    rng: Random,
    board: Board,
    raw_spins: u8,
    raw_pieces: u16,
    used: u8,
    bag_start: Board,
    bag_start_spins: u8,
    bag_start_pieces: u16,
    bag_placements: Vec<Placement>,
    placements: Vec<Placement>,
    frames: Vec<DisplayFrame>,
    qualified_frames: Vec<DisplayFrame>,
    qualified_spins: u8,
    qualified_bags: u16,
    prefix_live: bool,
    counts: Vec<u16>,
    proofs: Vec<Proof>,
    assume_reflection: bool,
    verifier: Option<Verifier>,
    query_piece: usize,
    choice: Option<Choice>,
    done: bool,
    termination: String,
    budget_pieces: Option<u16>,
    #[serde(default, skip_serializing_if = "is_false")]
    frontier_mode: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    physical: Option<physical::EpisodeMetrics>,
}

fn is_false(value: &bool) -> bool { !*value }

impl Episode {
    pub fn new(seed: u32) -> Self {
        Self { seed, rng: Random::new(seed), board: Board::default(), raw_spins: 0, raw_pieces: 0,
            used: 0, bag_start: Board::default(), bag_start_spins: 0, bag_start_pieces: 0,
            bag_placements: vec![], placements: vec![], frames: vec![], qualified_frames: vec![],
            qualified_spins: 0, qualified_bags: 0, prefix_live: true, counts: vec![], proofs: vec![],
            assume_reflection: true, verifier: None, query_piece: 0, choice: None,
            done: false, termination: String::new(), budget_pieces: None, frontier_mode: false, physical: None }
    }
    pub fn for_config(seed: u32, config: &GaConfig) -> Self {
        let mut episode = Self::new(seed); episode.frontier_mode = config.is_frontier();
        if config.is_physical() { episode.physical = Some(physical::EpisodeMetrics::default()); }
        episode
    }
    /// Return to an independently replayed physical prefix. Reusing a prefix
    /// is candidate search, not a fresh from-empty policy evaluation.
    fn from_physical_prefix(seed: u32, config: &GaConfig, prefix: &EpisodeResult) -> Result<Self, String> {
        if !config.is_physical() || prefix.raw_spins >= 40 || prefix.raw_pieces >= config.max_pieces {
            return Err("Invalid physical continuation prefix".into());
        }
        let mut episode = Self::for_config(seed,config);
        for placement in &prefix.placements {
            if episode.used == ALL_PIECES { episode.next_bag_or_finish(config); }
            if episode.used & placement.pose.piece.bit()!=0 { return Err("Continuation repeats a bag piece".into()); }
            episode.commit_placement(placement.clone(), &mut Counters::default())?;
        }
        if episode.board.rows != prefix.rows || episode.raw_spins != prefix.raw_spins {
            return Err("Continuation prefix replay mismatch".into());
        }
        if episode.used == ALL_PIECES { episode.next_bag_or_finish(config); }
        episode.physical.as_mut().unwrap().continuation_prefix_pieces = prefix.raw_pieces;
        Ok(episode)
    }
    pub fn finished(&self) -> bool { self.done }

    fn next_bag_or_finish(&mut self, config: &GaConfig) {
        if self.raw_spins >= 40 {
            self.done = true;
            self.termination = if self.qualified_spins >= 40 { "certified-40" } else { "raw-40-only" }.into();
        } else if self.raw_pieces >= config.max_pieces {
            self.done = true; self.termination = "piece-limit".into();
        } else {
            self.used = 0; self.bag_start = self.board; self.bag_start_spins = self.raw_spins;
            self.bag_start_pieces = self.raw_pieces; self.bag_placements.clear();
            self.query_piece = 0; self.choice = None;
        }
    }

    pub fn advance(&mut self, weights: &[f64], config: &GaConfig, counters: &mut Counters) -> Result<(), String> {
        if self.done { return Ok(()); }
        counters.work_units += 1;
        if let Some(verifier) = &mut self.verifier {
            counters.supply_queries += verifier.advance();
            counters.verification_work += 1;
            if verifier.finished() {
                let proof = verifier.proof(&config.search_config(), &self.counts)?;
                if proof.qualified {
                    self.qualified_spins = proof.end_spins; self.qualified_bags += 1;
                    self.qualified_frames.extend(proof.representative_frames.clone());
                    self.counts.push(proof.success);
                    self.assume_reflection = proof.first_bag_symmetry_assumed;
                    self.budget_pieces = proof.budget_pieces;
                } else { self.prefix_live = false; }
                self.proofs.push(proof); self.verifier = None;
                self.next_bag_or_finish(config);
            }
            return Ok(());
        }
        if self.used == ALL_PIECES || self.raw_spins >= 40 {
            if self.prefix_live && self.physical.is_none() {
                self.verifier = Some(Verifier::new(Request {
                    start: self.bag_start, start_spins: self.bag_start_spins,
                    start_pieces: self.bag_start_pieces, placements: self.bag_placements.clone(),
                    assume_reflection: self.assume_reflection,
                })?);
            } else { self.next_bag_or_finish(config); }
            return Ok(());
        }
        if self.raw_pieces >= config.max_pieces {
            self.done = true; self.termination = "piece-limit-incomplete-bag".into(); return Ok(());
        }
        while self.query_piece < 7 && self.used & Piece::ALL[self.query_piece].bit() != 0 { self.query_piece += 1; }
        if self.query_piece < 7 {
            let piece = Piece::ALL[self.query_piece]; self.query_piece += 1;
            let locks = self.board.reachable_locks(piece);
            counters.physics_queries += 1; counters.candidate_locks += locks.len() as u64;
            for placement in locks {
                let Some(result) = self.board.apply(&placement) else { continue; };
                if result.lines > 1 || (result.lines == 1 && (!result.spin || piece == Piece::O)) { continue; }
                counters.legal_candidate_locks += 1;
                let score = if let Some(physical) = &config.physical {
                    policy::score(weights,&self.board,&result.board,&placement,result.lines,
                        self.used,self.raw_spins,self.raw_pieces,config.max_pieces)
                        + (self.rng.unit() * 2.0 - 1.0) * physical.action_noise
                } else {
                    features(&self.board,&result.board,&placement,result.lines).iter()
                        .zip(weights).map(|(v,w)|v*w).sum::<f64>() + self.rng.unit()*1e-9
                };
                if self.choice.as_ref().is_none_or(|c| score > c.score) {
                    self.choice = Some(Choice { placement, score });
                }
            }
            return Ok(());
        }
        let Some(choice) = self.choice.take() else {
            self.done = true; self.termination = "no-legal-lock".into(); return Ok(());
        };
        self.commit_placement(choice.placement,counters)
    }

    fn commit_placement(&mut self, placement: Placement, counters: &mut Counters) -> Result<(), String> {
        let replay = self.board.replay(placement.pose.piece, &placement.path).ok_or("Selected GA input replay failed")?;
        counters.replay_queries += 1;
        if replay.pose != placement.pose || replay.spin != placement.spin { return Err("GA replay mismatch".into()); }
        let result = self.board.apply(&replay).ok_or("Selected GA placement failed")?;
        if result.lines>1 || (result.lines==1 && (!result.spin || placement.pose.piece==Piece::O)) {
            return Err("Selected GA placement violates Spin Single rules".into());
        }
        self.board = result.board; self.raw_spins += result.lines; self.raw_pieces += 1;
        self.used |= placement.pose.piece.bit(); self.query_piece = 0;
        self.frames.push(DisplayFrame {
            rows: self.board.rows, piece: ["I", "J", "L", "O", "S", "T", "Z"][placement.pose.piece as usize].into(),
            placed_cells: placement.pose.cells().map(|(x, y)| (x as u8, y as u8)),
            spins: self.raw_spins, placed: self.raw_pieces, lines: result.lines, spin: result.spin,
        });
        self.bag_placements.push(placement.clone()); self.placements.push(placement);
        Ok(())
    }

    pub fn result(&self) -> Result<EpisodeResult, String> {
        if !self.done { return Err("GA episode is unfinished".into()); }
        Ok(self.observed_result())
    }

    fn observed_result(&self) -> EpisodeResult {
        let last = self.proofs.iter().rev().find(|p| p.qualified);
        let first = self.proofs.first();
        let mut result = EpisodeResult {
            seed: self.seed, raw_spins: self.raw_spins, raw_pieces: self.raw_pieces,
            raw_completed40: self.raw_spins >= 40, qualified_spins: self.qualified_spins,
            qualified_bags: self.qualified_bags, certified40: self.qualified_spins >= 40,
            budget_pieces: self.budget_pieces, probability: last.map_or(0.0, |p| p.probability),
            first_bag_probability: first.and_then(|p| p.first_bag_probability),
            first_bag_symmetry_assumed: first.is_some_and(|p| p.first_bag_symmetry_assumed),
            rows: self.board.rows,
            qualified_rows: self.qualified_frames.last().map_or([0; HEIGHT], |f| f.rows),
            frames: self.frames.clone(), qualified_frames: self.qualified_frames.clone(),
            placements: self.placements.clone(), bag_proofs: self.proofs.clone(),
            unchecked_suffix_bags: self.raw_pieces.div_ceil(7).saturating_sub(self.proofs.len() as u16),
            termination: if self.done { self.termination.clone() } else { "budget-censored".into() },
            input_replay_verified: true,
            frontier: None, physical: self.physical.clone(),
        };
        if self.frontier_mode { result.frontier = Some(evolution::FrontierMetrics::from_result(&result)); }
        result
    }

    fn validate(&self, config: &GaConfig) -> Result<(), String> {
        if self.frontier_mode != config.is_frontier() || self.physical.is_some()!=config.is_physical()
            || !self.board.is_valid() || !self.bag_start.is_valid() || self.rng.state == 0
            || self.raw_spins > 40 || self.qualified_spins > self.raw_spins || self.raw_pieces > config.max_pieces
            || self.used > ALL_PIECES || self.query_piece > 7 || self.bag_start_pieces % 7 != 0
            || self.bag_placements.len() > 7 || self.placements.len() != usize::from(self.raw_pieces)
            || self.frames.len() != self.placements.len() || self.qualified_frames.len() > self.frames.len()
            || self.choice.as_ref().is_some_and(|c| !c.score.is_finite())
            || self.counts.len() != usize::from(self.qualified_bags)
            || self.counts.iter().any(|&c| c == 0 || usize::from(c) > P7_COUNT) {
            return Err("Invalid GA episode checkpoint".into());
        }
        if let Some(physical)=&self.physical {
            if physical.continuation_prefix_pieces>self.raw_pieces || self.verifier.is_some()
                || (physical.repaired && physical.continuation_prefix_pieces==0)
                || !self.proofs.is_empty() || self.qualified_spins!=0 || self.qualified_bags!=0
                || !self.counts.is_empty() || self.budget_pieces.is_some() {
                return Err("Physical-only checkpoint contains unsupported supply proof".into());
            }
        }
        // Checkpoint import is not an alternate route for manufactured raw
        // evidence: all retained inputs must replay from the empty board.
        let (mut board, mut spins, mut used) = (Board::default(), 0_u8, 0_u8);
        for (i, p) in self.placements.iter().enumerate() {
            if i % 7 == 0 { used = 0; }
            if used & p.pose.piece.bit() != 0 { return Err("Repeated checkpoint bag piece".into()); }
            let replay = board.replay(p.pose.piece, &p.path).ok_or("Invalid checkpoint input replay")?;
            if replay.pose != p.pose || replay.spin != p.spin { return Err("Checkpoint pose mismatch".into()); }
            let lock = board.apply(&replay).ok_or("Invalid checkpoint placement")?;
            if lock.lines > 1 || (lock.lines == 1 && (!lock.spin || p.pose.piece == Piece::O)) {
                return Err("Invalid checkpoint line clear".into());
            }
            used |= p.pose.piece.bit(); spins += lock.lines; board = lock.board;
            if self.frames[i].rows != board.rows || self.frames[i].spins != spins {
                return Err("Checkpoint frame mismatch".into());
            }
        }
        if board != self.board || spins != self.raw_spins { return Err("Checkpoint rollout boundary mismatch".into()); }
        let begin = usize::from(self.bag_start_pieces);
        if begin > self.placements.len() || self.placements.len() - begin != self.bag_placements.len()
            || serde_json::to_value(&self.placements[begin..]).map_err(|e| e.to_string())?
                != serde_json::to_value(&self.bag_placements).map_err(|e| e.to_string())?
            || self.bag_start != if begin == 0 { Board::default() } else { Board { rows: self.frames[begin - 1].rows } }
            || self.bag_start_spins != if begin == 0 { 0 } else { self.frames[begin - 1].spins }
            || self.used != self.bag_placements.iter().fold(0, |mask, p| mask | p.pose.piece.bit())
            || self.prefix_live != self.proofs.iter().all(|p| p.qualified)
            || self.assume_reflection != self.proofs.iter().rev().find(|p| p.qualified)
                .map_or(true, |p| p.first_bag_symmetry_assumed) {
            return Err("Checkpoint bag/qualified-prefix phase mismatch".into());
        }
        if let Some(choice) = &self.choice {
            let p = &choice.placement;
            if self.used & p.pose.piece.bit() != 0 { return Err("Checkpoint choice repeats a bag piece".into()); }
            let replay = self.board.replay(p.pose.piece, &p.path).ok_or("Invalid checkpoint choice input")?;
            let result = self.board.apply(&replay).ok_or("Invalid checkpoint choice lock")?;
            if replay.pose != p.pose || replay.spin != p.spin || result.lines > 1
                || (result.lines == 1 && (!result.spin || p.pose.piece == Piece::O)) {
                return Err("Invalid checkpoint choice physics".into());
            }
        }
        if let Some(verifier) = &self.verifier {
            if !self.prefix_live || (self.used != ALL_PIECES && self.raw_spins < 40)
                || !verifier.matches_request(&Request {
                    start: self.bag_start, start_spins: self.bag_start_spins,
                    start_pieces: self.bag_start_pieces, placements: self.bag_placements.clone(),
                    assume_reflection: self.assume_reflection,
                }) {
                return Err("Checkpoint verifier is detached from its actual rollout".into());
            }
        }
        Ok(())
    }
}

impl EpisodeResult {
    fn audit(&self, config: &GaConfig) -> Result<(), String> {
        if self.frontier.is_some() != config.is_frontier() || self.physical.is_some()!=config.is_physical()
            || self.placements.len() != usize::from(self.raw_pieces) || self.raw_pieces > config.max_pieces
            || self.frames.len() != self.placements.len() || self.raw_spins > 40
            || self.bag_proofs.len() > usize::from(self.raw_pieces.div_ceil(7)) {
            return Err("Invalid retained GA episode bounds".into());
        }
        if let Some(physical)=&self.physical {
            if physical.continuation_prefix_pieces>self.raw_pieces || !self.bag_proofs.is_empty()
                || (physical.repaired && physical.continuation_prefix_pieces==0)
                || self.certified40 || self.qualified_spins!=0 || self.qualified_bags!=0 {
                return Err("A physical candidate cannot contain a supply certificate".into());
            }
        }
        let (mut board, mut spins, mut used) = (Board::default(), 0_u8, 0_u8);
        for (i, p) in self.placements.iter().enumerate() {
            if i % 7 == 0 { used = 0; }
            if used & p.pose.piece.bit() != 0 || spins >= 40 { return Err("Invalid retained bag inventory".into()); }
            let replay = board.replay(p.pose.piece, &p.path).ok_or("Retained GA input replay failed")?;
            if replay.pose != p.pose || replay.spin != p.spin { return Err("Retained GA pose mismatch".into()); }
            let lock = board.apply(&replay).ok_or("Retained GA lock failed")?;
            if lock.lines > 1 || (lock.lines == 1 && (!lock.spin || p.pose.piece == Piece::O)) {
                return Err("Invalid retained GA line clear".into());
            }
            board = lock.board; spins += lock.lines; used |= p.pose.piece.bit();
            let frame = DisplayFrame { rows: board.rows,
                piece: ["I", "J", "L", "O", "S", "T", "Z"][p.pose.piece as usize].into(),
                placed_cells: p.pose.cells().map(|(x, y)| (x as u8, y as u8)),
                spins, placed: i as u16 + 1, lines: lock.lines, spin: lock.spin };
            if frame != self.frames[i] { return Err("Retained GA frame mismatch".into()); }
        }
        if board.rows != self.rows || spins != self.raw_spins || self.raw_completed40 != (spins >= 40)
            || !self.input_replay_verified { return Err("Retained raw GA result mismatch".into()); }
        let (mut counts, mut frames, mut assume_reflection) = (vec![], vec![], true);
        let (mut qualified_spins, mut budget, mut probability) = (0, None, 0.0);
        for (i, proof) in self.bag_proofs.iter().enumerate() {
            let begin = i * 7;
            let end = (begin + 7).min(self.placements.len());
            let start = if begin == 0 { Board::default() } else { Board { rows: self.frames[begin - 1].rows } };
            let start_spins = if begin == 0 { 0 } else { self.frames[begin - 1].spins };
            let mut verifier = Verifier::new(Request { start, start_spins, start_pieces: begin as u16,
                placements: self.placements[begin..end].to_vec(), assume_reflection })?;
            while !verifier.finished() { verifier.advance(); }
            let observed = verifier.proof(&config.search_config(), &counts)?;
            if serde_json::to_value(&observed).map_err(|e| e.to_string())?
                != serde_json::to_value(proof).map_err(|e| e.to_string())? {
                return Err("Retained GA certificate differs from canonical verification".into());
            }
            if proof.qualified {
                counts.push(proof.success); qualified_spins = proof.end_spins;
                frames.extend(proof.representative_frames.clone());
                assume_reflection = proof.first_bag_symmetry_assumed;
                budget = proof.budget_pieces; probability = proof.probability;
            } else if i + 1 != self.bag_proofs.len() {
                return Err("A qualified GA prefix cannot continue after a rejected bag".into());
            }
        }
        if self.qualified_spins != qualified_spins || usize::from(self.qualified_bags) != counts.len()
            || self.certified40 != (qualified_spins >= 40) || self.budget_pieces != budget
            || self.qualified_frames != frames || self.probability != probability
            || self.qualified_rows != frames.last().map_or([0; HEIGHT], |f| f.rows)
            || self.first_bag_probability != self.bag_proofs.first().and_then(|p| p.first_bag_probability)
            || self.first_bag_symmetry_assumed != self.bag_proofs.first().is_some_and(|p| p.first_bag_symmetry_assumed)
            || self.unchecked_suffix_bags != self.raw_pieces.div_ceil(7).saturating_sub(self.bag_proofs.len() as u16) {
            return Err("Retained certified GA result mismatch".into());
        }
        if let Some(frontier) = &self.frontier {
            if *frontier != evolution::FrontierMetrics::from_result(self) {
                return Err("Retained GA frontier/strict100 metrics mismatch".into());
            }
        }
        Ok(())
    }
}

/// Raw geometry/action measurements, normalized by declared board-size
/// constants. No old terrain evaluator, continuation search or label is used.
pub fn score_placement(weights: &[f64], board: &Board, placement: &Placement) -> Option<f64> {
    score_placement_with_mode(weights, board, placement, MovementMode::SonicOnly)
}

/// The same 24 measured features under an explicit movement category. Only
/// input legality changes; feature order, normalization and scoring do not.
pub fn score_placement_with_mode(weights: &[f64], board: &Board, placement: &Placement,
    mode: MovementMode) -> Option<f64> {
    if weights.len() != FEATURE_COUNT || weights.iter().any(|w| !w.is_finite()) { return None; }
    let result = board.apply_with_mode(placement, mode)?;
    if result.lines > 1 || (result.lines == 1 && (!result.spin || placement.pose.piece == Piece::O)) { return None; }
    Some(features(board, &result.board, placement, result.lines).iter()
        .zip(weights).map(|(f, w)| f * w).sum())
}

/// Context-aware proposal score. Legality remains owned by the actual input
/// generator/replay; this value is never a supply probability or proof bound.
pub fn score_placement_context(weights:&[f64],board:&Board,placement:&Placement,used:u8,
    spins:u8,placed:u16,max_pieces:u16)->Option<f64>{
    validate_weights(weights).ok()?;
    let result=board.apply(placement)?;
    if result.lines>1 || (result.lines==1 && (!result.spin || placement.pose.piece==Piece::O)){return None;}
    Some(policy::score(weights,board,&result.board,placement,result.lines,used,spins,placed,max_pieces))
}

fn features(before: &Board, after: &Board, placement: &Placement, lines: u8) -> [f64; FEATURE_COUNT] {
    let mut heights = [0_u16; WIDTH];
    let (mut holes, mut depth, mut roofed) = (0_u32, 0_u32, 0_u32);
    for x in 0..WIDTH {
        for y in (0..HEIGHT).rev() {
            if after.rows[y] & (1 << x) != 0 { heights[x] = y as u16 + 1; break; }
        }
        let mut above = 0_u32;
        for y in (0..usize::from(heights[x])).rev() {
            if after.rows[y] & (1 << x) != 0 { above += 1; }
            else { holes += 1; depth += above; if y + 1 < HEIGHT && after.rows[y + 1] & (1 << x) != 0 { roofed += 1; } }
        }
    }
    let height = heights.iter().copied().max().unwrap_or(0);
    let aggregate: u16 = heights.iter().sum();
    let bump: u16 = heights.windows(2).map(|h| h[0].abs_diff(h[1])).sum();
    let mut wells = [0_u16; WIDTH];
    for x in 0..WIDTH {
        let left = if x == 0 { HEIGHT as u16 } else { heights[x - 1] };
        let right = if x + 1 == WIDTH { HEIGHT as u16 } else { heights[x + 1] };
        wells[x] = left.min(right).saturating_sub(heights[x]);
    }
    let mut row_transitions = 0_u16;
    let mut deficits = [0_u16; 4];
    let mut corners = 0_u16;
    for y in 0..usize::from(height) {
        let mut prev = true;
        for x in 0..WIDTH {
            let occupied = after.rows[y] & (1 << x) != 0;
            row_transitions += u16::from(occupied != prev); prev = occupied;
            if !occupied {
                let count = [(-1, -1), (1, -1), (-1, 1), (1, 1)].iter()
                    .filter(|&&(dx, dy)| after.occupied(x as i8 + dx, y as i8 + dy)).count();
                corners += u16::from(count >= 3);
            }
        }
        row_transitions += u16::from(!prev);
        let missing = WIDTH as u32 - after.rows[y].count_ones();
        if (1..=4).contains(&missing) { deficits[missing as usize - 1] += 1; }
    }
    let mut column_transitions = 0_u16;
    for x in 0..WIDTH {
        let mut prev = true;
        for y in 0..usize::from(height) {
            let occupied = after.rows[y] & (1 << x) != 0;
            column_transitions += u16::from(occupied != prev); prev = occupied;
        }
        column_transitions += u16::from(prev);
    }
    let cells = placement.pose.cells();
    let (mut contacts, mut floor, mut walls, mut landing) = (0_u16, 0_u16, 0_u16, 0_i16);
    for &(x, y) in &cells {
        landing += i16::from(y);
        floor += u16::from(y == 0); walls += u16::from(x == 0 || x == WIDTH as i8 - 1);
        for &(dx, dy) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
            if before.occupied(x + dx, y + dy) { contacts += 1; }
        }
    }
    let blocked = Piece::ALL.iter().filter(|&&p| !after.can_spawn(p)).count();
    [f64::from(lines), f64::from(height) / 24.0, f64::from(aggregate) / 240.0,
        f64::from(after.cell_count()) / 240.0, f64::from(holes) / 240.0, f64::from(depth) / 5760.0,
        f64::from(bump) / 216.0, f64::from(*wells.iter().max().unwrap()) / 24.0,
        f64::from(wells.iter().sum::<u16>()) / 240.0, f64::from(row_transitions) / 264.0,
        f64::from(column_transitions) / 250.0, f64::from(deficits[0]) / 24.0,
        f64::from(deficits[1]) / 24.0, f64::from(deficits[2]) / 24.0, f64::from(deficits[3]) / 24.0,
        f64::from(roofed) / 240.0, f64::from(corners) / 240.0, f64::from(landing) / 96.0,
        f64::from(contacts) / 16.0, f64::from(floor) / 4.0, f64::from(walls) / 4.0,
        f64::from(placement.pose.piece == Piece::T), f64::from(placement.pose.piece == Piece::O),
        blocked as f64 / 7.0]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Genome { weights: Vec<f64>, model_id: String }
impl Genome {
    fn new(weights: Vec<f64>) -> Self { Self { model_id: format!("ga-{}", &weights_digest(&weights)[..20]), weights } }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Evaluation {
    fitness: Fitness, best: Option<EpisodeResult>, seeds: Vec<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    objectives: Option<evolution::Objectives>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    physical: Option<physical::EvaluationMetrics>,
}
impl Evaluation {
    fn empty() -> Self { Self { fitness: Fitness::default(), best: None, seeds: vec![], objectives: None, physical: None } }
    fn add(&mut self, result: EpisodeResult) {
        self.fitness.add(&result); self.seeds.push(result.seed);
        if result.physical.is_some() {
            self.physical.get_or_insert_with(physical::EvaluationMetrics::default).add(&result);
            if self.best.as_ref().is_none_or(|best|physical::compare_result(&result,best).is_gt()) {self.best=Some(result);}
            return;
        }
        if result.frontier.is_some() {
            self.objectives.get_or_insert_with(evolution::Objectives::default).add(&result);
            if self.best.as_ref().is_none_or(|best| evolution::compare_result(&result, best).is_gt()) { self.best = Some(result); }
            return;
        }
        let mut candidate = Fitness::default(); candidate.add(&result);
        let mut old = Fitness::default(); if let Some(best) = &self.best { old.add(best); }
        if self.best.is_none() || candidate > old { self.best = Some(result); }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CachedEvaluation { model_id: String, evaluation: Evaluation }
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Champion { genome: Genome, evaluation: Evaluation }
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Initialization {
    kind: String,
    parent_model_id: String,
    parent_source_policy: String,
    inherited_weights_sha256: String,
    inherited_weights: Vec<f64>,
    parent_completed_generations: u32,
    parent_checkpoint_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    inherited_raw_spins: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent_selected_model_id: Option<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationSummary {
    pub generation: u32, pub best_fitness: Fitness, pub best_qualified_spins: u8,
    pub best_raw_spins: u8, pub certified40_count: usize, pub evaluated_genomes: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_raw_max: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub all_time_raw_max: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict_spins: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frontier_spins: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frontier_probability: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unique_behaviors: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical40_count: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_mean_spins: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_min_spins: Option<u8>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub physical_evaluated_episodes: Option<usize>,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct State {
    config: GaConfig, generation: u32, completed_generations: u32, rng: Random,
    population: Vec<Genome>, evaluations: Vec<Option<Evaluation>>, active_genome_index: usize,
    active_evaluation: Evaluation, active_episode: Option<Episode>, best: Option<Champion>,
    history: Vec<GenerationSummary>, counters: Counters, cache: Vec<CachedEvaluation>,
    total_evaluated_genomes: u64, complete: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frontier: Option<evolution::Archive>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    initialization: Option<Initialization>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    physical: Option<physical::Archive>,
}

pub struct Trainer { state: State, restore_audit_ms: f64 }

#[cfg(target_arch = "wasm32")]
fn now_ms() -> f64 {
    #[wasm_bindgen::prelude::wasm_bindgen]
    extern "C" { #[wasm_bindgen(js_namespace = Date, js_name = now)] fn date_now() -> f64; }
    date_now()
}
#[cfg(not(target_arch = "wasm32"))]
fn now_ms() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64() * 1000.0
}

impl Trainer {
    pub fn new(config: GaConfig) -> Result<Self, String> {
        config.validate()?;
        let mut rng = Random::new(config.seed);
        let initial = policy::initializer(config.parameter_count()>FEATURE_COUNT);
        let mut population = vec![Genome::new(initial.clone())];
        while population.len() < config.population_size {
            let scale = if population.len() % 3 == 0 { 2.0 } else { 0.65 };
            // Expanded models begin with an exactly zero residual. Keeping
            // the same24 base RNG draws also makes capacity ablations start
            // from identical base populations and action policies.
            let weights = initial.iter().enumerate().map(|(i,w)| if i<FEATURE_COUNT {
                (w + rng.normal()*scale).clamp(-16.0,16.0)
            } else {*w}).collect();
            population.push(Genome::new(weights));
        }
        let size = config.population_size;
        let frontier = config.is_frontier().then(evolution::Archive::default);
        let physical = config.is_physical().then(physical::Archive::default);
        Ok(Self { state: State { config, generation: 0, completed_generations: 0, rng,
            population, evaluations: vec![None; size], active_genome_index: 0,
            active_evaluation: Evaluation::empty(), active_episode: None, best: None,
            history: vec![], counters: Counters::default(), cache: vec![], total_evaluated_genomes: 0,
            complete: false, frontier, initialization: None, physical }, restore_audit_ms: 0.0 })
    }

    /// Explicitly start a new v3 experiment from an existing coefficient
    /// vector. No old score, certificate, cache, generation or RNG is copied.
    pub fn with_model(config: GaConfig, model_json: &str) -> Result<Self, String> {
        let model = GaModel::from_json(model_json)?;
        Self::inherit_weights(config, model, None)
    }

    pub fn fork_checkpoint(config: GaConfig, checkpoint_json: &str) -> Result<Self, String> {
        if !config.is_frontier() && !config.is_physical() { return Err("An explicit GA fork requires a new experiment configuration".into()); }
        let previous = Self::from_checkpoint_json(checkpoint_json)?;
        let raw = if config.is_physical() {
            previous.state.frontier.as_ref().and_then(|a|a.raw.as_ref()).map(|r|(r.genome.clone(),r.result.clone()))
                .or_else(||previous.state.physical.as_ref().and_then(|a|a.raw.as_ref()).map(|r|r.inherited()))
        } else {None};
        let selected=previous.champion()?;
        let mut source=selected.clone();
        if let Some((genome,_))=&raw {
            source=GaModel::new(genome.weights.clone(),previous.state.config.clone());
            source.completed_generations=previous.state.completed_generations;
        }
        let mut trainer = Self::inherit_weights(config, source, Some(digest(checkpoint_json.as_bytes())))?;
        if let Some((_,result))=raw {
            let genome=trainer.state.population[0].clone();
            trainer.state.physical.as_mut().unwrap().import(&trainer.state.config,&genome,&result)?;
            let initialization=trainer.state.initialization.as_mut().unwrap();
            initialization.kind="inherited-physical-archive-new-experiment".into();
            initialization.inherited_raw_spins=Some(result.raw_spins);
            let selected_weights=policy::upgrade_weights(&selected.weights,trainer.state.config.parameter_count()>FEATURE_COUNT)?;
            let selected_genome=Genome::new(selected_weights);
            if selected_genome.model_id!=trainer.state.population[0].model_id {
                trainer.state.population[1]=selected_genome;
                initialization.parent_selected_model_id=Some(selected.model_id);
            }
        }
        trainer.restore_audit_ms = previous.restore_audit_ms;
        Ok(trainer)
    }

    fn inherit_weights(config: GaConfig, model: GaModel, checkpoint_hash: Option<String>) -> Result<Self, String> {
        if !config.is_frontier() && !config.is_physical() { return Err("Inherited weights require an explicitly new experiment".into()); }
        let inherited=policy::upgrade_weights(&model.weights,config.parameter_count()>FEATURE_COUNT)?;
        let mut trainer = Self::new(config)?;
        trainer.state.population[0] = Genome::new(inherited.clone());
        // Half the population explores around the inherited policy; the rest
        // keeps the fresh initializer/random population for independent search.
        for i in 1..trainer.state.population.len().div_ceil(2) {
            let scale = if i % 3 == 0 { 1.2 } else { 0.35 };
            let weights = inherited.iter().enumerate().map(|(j,w)| if j<FEATURE_COUNT {
                (w + trainer.state.rng.normal()*scale).clamp(-16.0,16.0)
            } else {*w}).collect();
            trainer.state.population[i] = Genome::new(weights);
        }
        trainer.state.initialization = Some(Initialization {
            kind: "inherited-weights-new-experiment".into(), parent_model_id: model.model_id,
            parent_source_policy: model.source_policy, inherited_weights_sha256: model.weights_sha256,
            inherited_weights: model.weights, parent_completed_generations: model.completed_generations,
            parent_checkpoint_sha256: checkpoint_hash,
            inherited_raw_spins: None,
            parent_selected_model_id: None,
        });
        Ok(trainer)
    }

    pub fn config(&self) -> &GaConfig { &self.state.config }
    pub fn finished(&self) -> bool { self.state.complete }
    pub fn counters(&self) -> &Counters { &self.state.counters }
    pub fn completed_generations(&self) -> u32 { self.state.completed_generations }

    pub fn advance(&mut self, work: u32) -> Result<(), String> {
        let start = now_ms();
        let result = (|| {
            for _ in 0..work.clamp(1, 64) {
                if self.state.complete { break; }
                self.advance_one()?;
            }
            Ok(())
        })();
        self.state.counters.elapsed_ms += (now_ms() - start).max(0.0);
        result
    }

    fn advance_one(&mut self) -> Result<(), String> {
        let state = &mut self.state;
        if let Some(mut repair)=state.physical.as_mut().and_then(|p|p.repair.take()) {
            let result=repair.advance(&state.config,&mut state.counters)?;
            let archive=state.physical.as_mut().unwrap();
            if let Some(result)=result{
                archive.repair_completed+=1;archive.repair40_count+=u64::from(result.raw_completed40);
                archive.observe(state.generation,&repair.genome,&result,&state.config,false);
            }else{archive.repair=Some(repair);}
            return Ok(());
        }
        if let Some(mut continuation)=state.physical.as_mut().and_then(|p|p.continuation.take()) {
            let previous_pieces=continuation.episode.raw_pieces;
            continuation.episode.advance(&continuation.genome.weights,&state.config,&mut state.counters)?;
            let archive=state.physical.as_mut().unwrap();
            if continuation.episode.raw_pieces!=previous_pieces && archive.needs_observation(continuation.episode.raw_spins,continuation.episode.raw_pieces,false) {
                archive.observe(state.generation,&continuation.genome,&continuation.episode.observed_result(),&state.config,false);
            }
            if continuation.episode.finished(){archive.record_completed(&continuation.episode.result()?,true);}
            else{archive.continuation=Some(continuation);}
            return Ok(());
        }
        if state.active_genome_index >= state.population.len() {
            self.finish_generation(); return Ok(());
        }
        let genome = &state.population[state.active_genome_index];
        if state.active_episode.is_none() && state.active_evaluation.seeds.is_empty() {
            if let Some(cached) = state.cache.iter().find(|c| c.model_id == genome.model_id) {
                state.active_evaluation = cached.evaluation.clone(); state.counters.cache_hits += 1;
                if let Some(archive) = &mut state.frontier {
                    archive.generation_raw_max = archive.generation_raw_max.max(cached.evaluation.objectives.as_ref().unwrap().raw_max);
                }
            }
        }
        if state.active_evaluation.seeds.len() >= state.config.episodes_per_genome {
            let evaluation = state.active_evaluation.clone();
            if state.best.as_ref().is_none_or(|b| if state.config.is_physical() {
                physical::compare_evaluation(&evaluation,&b.evaluation).is_gt()
            } else if state.config.is_frontier() {
                evolution::compare_evaluation(&evaluation, &b.evaluation).is_gt()
            } else { evaluation.fitness > b.evaluation.fitness }) {
                state.best = Some(Champion { genome: genome.clone(), evaluation: evaluation.clone() });
            }
            if !state.cache.iter().any(|c| c.model_id == genome.model_id) {
                if state.cache.len() >= 128 { state.cache.remove(0); }
                state.cache.push(CachedEvaluation { model_id: genome.model_id.clone(), evaluation: evaluation.clone() });
            }
            state.evaluations[state.active_genome_index] = Some(evaluation);
            state.active_genome_index += 1; state.total_evaluated_genomes += 1;
            state.active_evaluation = Evaluation::empty();
            return Ok(());
        }
        if state.active_episode.is_none() {
            // Common random numbers across genomes and generations. This seed
            // changes tie/exploration order, not the game's rules or opening.
            let ordinal = state.active_evaluation.seeds.len() as u32;
            let seed = state.config.seed.wrapping_add(ordinal.wrapping_mul(0x9e37_79b9)).wrapping_add(0x6a09_e667);
            state.active_episode = Some(Episode::for_config(seed, &state.config));
        }
        let episode = state.active_episode.as_mut().unwrap();
        let old_proofs = episode.proofs.len();
        let old_pieces = episode.raw_pieces;
        episode.advance(&genome.weights, &state.config, &mut state.counters)?;
        if let Some(archive)=&mut state.physical {
            if episode.raw_pieces!=old_pieces && archive.needs_observation(episode.raw_spins,episode.raw_pieces,true) {
                archive.observe(state.generation,genome,&episode.observed_result(),&state.config,false);
            }
        }
        if let Some(archive) = &mut state.frontier {
            archive.generation_raw_max = archive.generation_raw_max.max(episode.raw_spins);
            let prefix_changed = old_proofs != episode.proofs.len();
            if prefix_changed || archive.raw_should_update(episode.raw_spins, episode.raw_pieces) {
                archive.observe(state.generation, genome, &episode.observed_result(), prefix_changed);
            }
        }
        if episode.finished() {
            let result = episode.result()?;
            if let Some(archive) = &mut state.frontier { archive.completed(&result); }
            if let Some(archive)=&mut state.physical {
                archive.record_completed(&result,false);
                archive.schedule(state.generation,&state.config,genome,&mut state.rng)?;
            }
            state.active_evaluation.add(result); state.counters.evaluated_episodes += 1;
            state.active_episode = None;
        }
        Ok(())
    }

    fn tournament(&mut self, ranked: &[usize]) -> usize {
        let mut best = self.state.rng.index(ranked.len());
        for _ in 1..self.state.config.tournament_size { best = best.min(self.state.rng.index(ranked.len())); }
        ranked[best]
    }

    fn finish_generation(&mut self) {
        if self.state.config.is_physical(){physical::finish_generation(&mut self.state);return;}
        if self.state.config.is_frontier() { evolution::finish_generation(&mut self.state); return; }
        let mut ranked: Vec<_> = (0..self.state.population.len()).collect();
        ranked.sort_by(|&a, &b| self.state.evaluations[b].as_ref().unwrap().fitness
            .cmp(&self.state.evaluations[a].as_ref().unwrap().fitness).then(a.cmp(&b)));
        let top = self.state.evaluations[ranked[0]].as_ref().unwrap();
        self.state.history.push(GenerationSummary {
            generation: self.state.generation, best_fitness: top.fitness.clone(),
            best_qualified_spins: top.best.as_ref().map_or(0, |r| r.qualified_spins),
            best_raw_spins: top.best.as_ref().map_or(0, |r| r.raw_spins),
            certified40_count: self.state.evaluations.iter().filter(|e| e.as_ref().unwrap().fitness.certified_episodes > 0).count(),
            evaluated_genomes: self.state.population.len(),
            generation_raw_max: None, all_time_raw_max: None, strict_spins: None,
            frontier_spins: None, frontier_probability: None, unique_behaviors: None,
            physical40_count:None,physical_mean_spins:None,physical_min_spins:None,physical_evaluated_episodes:None,
        });
        if self.state.history.len() > 256 { self.state.history.remove(0); }
        self.state.completed_generations += 1;
        if self.state.completed_generations >= self.state.config.generations { self.state.complete = true; return; }
        let mut next: Vec<_> = ranked.iter().take(self.state.config.elite_count)
            .map(|&i| self.state.population[i].clone()).collect();
        while next.len() < self.state.config.population_size {
            let a = self.tournament(&ranked); let b = self.tournament(&ranked);
            let mut weights = vec![];
            for j in 0..FEATURE_COUNT {
                let blend = self.state.rng.unit();
                let mut w = blend * self.state.population[a].weights[j] + (1.0 - blend) * self.state.population[b].weights[j];
                if self.state.rng.unit() < self.state.config.mutation_rate {
                    w += self.state.rng.normal() * self.state.config.mutation_scale;
                }
                weights.push(w.clamp(-16.0, 16.0));
            }
            // One broad immigrant per generation retains exploration when
            // the few strict qualified-prefix rewards are sparse.
            if next.len() + 1 == self.state.config.population_size {
                weights = INITIAL_WEIGHTS.iter().map(|w| (w + self.state.rng.normal() * 2.0).clamp(-16.0, 16.0)).collect();
            }
            next.push(Genome::new(weights));
        }
        self.state.population = next; self.state.evaluations = vec![None; self.state.config.population_size];
        self.state.generation += 1; self.state.active_genome_index = 0;
    }

    pub fn champion(&self) -> Result<GaModel, String> {
        let Some(best) = &self.state.best else {
            let mut model = GaModel::new(self.state.population[0].weights.clone(), self.state.config.clone());
            if self.state.initialization.is_some() { model.training_state = "inherited-weight-initializer".into(); }
            return Ok(model);
        };
        let mut model = GaModel::new(best.genome.weights.clone(), self.state.config.clone());
        model.trained = self.state.completed_generations > 0;
        model.training_state = if model.trained { "evolved-candidate" } else { "evaluated-initial-population" }.into();
        model.completed_generations = self.state.completed_generations;
        model.evaluated_episodes = self.state.counters.evaluated_episodes;
        model.best = best.evaluation.best.clone(); model.fitness = Some(best.evaluation.fitness.clone());
        Ok(model)
    }

    pub fn champion_json(&self) -> Result<String, String> { serde_json::to_string(&self.champion()?).map_err(|e| e.to_string()) }

    pub fn snapshot_json(&self) -> Result<String, String> {
        let state = &self.state;
        let best = state.best.as_ref().and_then(|b| b.evaluation.best.as_ref().map(|r| {
            let proofs: Vec<_> = r.bag_proofs.iter().map(|p| serde_json::json!({
                "bagIndex":p.bag_index,"status":p.status,"qualified":p.qualified,"terminal":p.terminal,
                "checked":p.checked,"canonicalChecked":p.canonical_checked,"success":p.success,
                "successUpper":p.success_upper,"quietSafeSupplies":p.quiet_safe_supplies,
                "firstBagSymmetryAssumed":p.first_bag_symmetry_assumed,"mirroredInputsVerified":false,
                "probability":p.probability,"budgetPieces":p.budget_pieces,"startSpins":p.start_spins,
                "endSpins":p.end_spins,"quietNoHoldVerified":p.quiet_no_hold_verified
            })).collect();
            serde_json::json!({"modelId":b.genome.model_id,"fitness":b.evaluation.fitness,"seed":r.seed,
                "rawSpins":r.raw_spins,"rawPieces":r.raw_pieces,"rawCompleted40":r.raw_completed40,
                "qualifiedSpins":r.qualified_spins,"qualifiedBags":r.qualified_bags,"certified40":r.certified40,
                "budgetPieces":r.budget_pieces,"probability":r.probability,"firstBagProbability":r.first_bag_probability,
                "firstBagSymmetryAssumed":r.first_bag_symmetry_assumed,"rows":r.rows,"qualifiedRows":r.qualified_rows,
                "frames":r.frames,"qualifiedFrames":r.qualified_frames,"bagProofs":proofs,
                "uncheckedSuffixBags":r.unchecked_suffix_bags,"termination":r.termination,"inputReplayVerified":true})
        }));
        let current = state.active_episode.as_ref().map(|e| serde_json::json!({
            "rawSpins":e.raw_spins,"rawPieces":e.raw_pieces,"qualifiedSpins":e.qualified_spins,
            "qualifiedBags":e.qualified_bags,"rows":e.board.rows,
            "phase":if e.verifier.is_some(){"verifying-fixed-plan"}else{"enumerating-actual-locks"},
            "checked":e.verifier.as_ref().map_or(0,Verifier::checked),"qualifiedPrefixLive":e.prefix_live
        }));
        let mut snapshot = serde_json::json!({
            "schemaVersion":"neyra-ga-snapshot-v1","sourcePolicy":state.config.source_policy(),"featureSchema":state.config.feature_schema(),
            "status":if state.complete{"complete"}else{"running"},"generation":state.generation,
            "completedGenerations":state.completed_generations,"populationSize":state.config.population_size,
            "evaluatedGenomes":state.active_genome_index,"totalEvaluatedGenomes":state.total_evaluated_genomes,
            "activeGenomeIndex":state.active_genome_index,"best":best,"current":current,
            "history":state.history,"counters":state.counters,"featureNames":policy::feature_names(state.config.parameter_count()).unwrap(),
            "restoreAuditMs":self.restore_audit_ms,
            "learningComplete":false,"deploymentReady":false,
            "workDefinition":"One full reachable-lock piece query, selected-input replay, canonical graph/supply slice, or phase transition. Work units differ from legacy search steps.",
        });
        if let Some(archive) = &state.frontier {
            snapshot["rawArchive"] = archive.raw.as_ref().map(evolution::Route::snapshot).unwrap_or(serde_json::Value::Null);
            snapshot["strictArchive"] = archive.strict.as_ref().map(evolution::Route::snapshot).unwrap_or(serde_json::Value::Null);
            snapshot["probabilityFrontier"] = archive.frontier.iter().map(evolution::Route::snapshot).collect::<Vec<_>>().into();
            let unique: std::collections::BTreeSet<_> = state.evaluations.iter().flatten()
                .flat_map(|e| e.objectives.as_ref().unwrap().behavior_ids.iter()).collect();
            snapshot["diversity"] = archive.snapshot(&state.config, unique.len());
            snapshot["probabilityIncludesFirstBag"] = true.into();
            snapshot["rawArchiveCompleteHistory"] = true.into();
            let generation_best = state.evaluations.iter().enumerate().filter_map(|(i,e)|e.as_ref().map(|e|(i,e)))
                .max_by(|(_,a),(_,b)|evolution::compare_evaluation(a,b));
            snapshot["generationBest"] = generation_best.and_then(|(i,e)| e.best.as_ref().map(|r| {
                serde_json::json!({"modelId":state.population[i].model_id,"fitness":e.fitness,
                    "seed":r.seed,"rawSpins":r.raw_spins,"rawPieces":r.raw_pieces,
                    "qualifiedSpins":r.qualified_spins,"qualifiedBags":r.qualified_bags,
                    "certified40":r.certified40,"budgetPieces":r.budget_pieces,"probability":r.probability})
            })).unwrap_or(serde_json::Value::Null);
            if let Some(best)=state.best.as_ref().and_then(|b|b.evaluation.best.as_ref()) {
                snapshot["best"]["frontier"]=serde_json::to_value(&best.frontier).unwrap();
                snapshot["best"]["strict100"]=best.frontier.as_ref().is_some_and(|f|
                    !f.selected_success_counts.is_empty() && f.selected_success_counts.iter().all(|&n|usize::from(n)==P7_COUNT)).into();
                snapshot["best"]["strictCertified40"]=best.frontier.as_ref().is_some_and(|f|f.strict_certified40).into();
                snapshot["best"]["probabilityIncludesFirstBag"] = true.into();
                snapshot["best"]["fullProbability"] = best.frontier.as_ref().map(|f|f.full_probability).into();
                snapshot["best"]["fullProbabilityBudgetPieces"] = best.frontier.as_ref().and_then(|f|f.full_probability_budget_pieces).into();
            }
        } else {
            snapshot["rawArchiveCompleteHistory"] = false.into();
        }
        if let Some(physical)=&state.physical {
            snapshot["trainingMode"]="physical-only".into();snapshot["probabilityMeasured"]=false.into();
            snapshot["parameterCount"]=state.config.parameter_count().into();
            snapshot["physical"]=physical.snapshot(&state.counters);
            snapshot["rawArchive"]=physical.raw.as_ref().map(physical::Route::snapshot).unwrap_or(serde_json::Value::Null);
            snapshot["bestFromEmptyArchive"]=physical.best_from_empty.as_ref().map(physical::Route::snapshot).unwrap_or(serde_json::Value::Null);
            snapshot["rawArchiveCompleteHistory"]=true.into();
            if let Some(current)=snapshot["current"].as_object_mut(){current.insert("qualifiedPrefixLive".into(),false.into());}
            if let Some(best)=snapshot["best"].as_object_mut(){
                best.insert("probabilityMeasured".into(),false.into());best.insert("probability".into(),serde_json::Value::Null);
                best.insert("trainingMode".into(),"physical-only".into());
            }
            if let Some(c)=&physical.continuation {
                snapshot["current"]=serde_json::json!({"rawSpins":c.episode.raw_spins,"rawPieces":c.episode.raw_pieces,
                    "qualifiedSpins":0,"qualifiedBags":0,"rows":c.episode.board.rows,"phase":"returning-to-physical-prefix",
                    "checked":0,"qualifiedPrefixLive":false,
                    "continuationPrefixPieces":c.episode.physical.as_ref().map_or(0,|p|p.continuation_prefix_pieces)});
            }
            if let Some(repair)=&physical.repair{
                snapshot["current"]=repair.current_snapshot();
            }
        }
        if let Some(initialization) = &state.initialization {
            snapshot["initialization"] = serde_json::to_value(initialization).map_err(|e|e.to_string())?;
        }
        Ok(snapshot.to_string())
    }

    pub fn checkpoint_json(&self) -> Result<String, String> {
        let mut state = serde_json::to_value(&self.state).map_err(|e| e.to_string())?;
        if self.state.config.is_frontier() || self.state.config.is_physical() { checkpoint::compact(&mut state)?; }
        let bytes = serde_json::to_vec(&state).map_err(|e| e.to_string())?;
        Ok(serde_json::json!({"schemaVersion":"neyra-ga-checkpoint-v1","sourcePolicy":self.state.config.source_policy(),
            "featureSchema":self.state.config.feature_schema(),"checksum":digest(&bytes),"state":state}).to_string())
    }

    pub fn from_checkpoint_json(json: &str) -> Result<Self, String> {
        let restore_start = now_ms();
        if json.len() > 32 * 1024 * 1024 { return Err("GA checkpoint is too large".into()); }
        #[derive(Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Envelope { schema_version: String, source_policy: String, feature_schema: String, checksum: String, state: serde_json::Value }
        let e: Envelope = serde_json::from_str(json).map_err(|e| format!("Invalid GA checkpoint: {e}"))?;
        if e.schema_version != "neyra-ga-checkpoint-v1" || !matches!(e.source_policy.as_str(), SOURCE_POLICY | LEGACY_SOURCE_POLICY | PHYSICAL_SOURCE_POLICY)
            || !matches!(e.feature_schema.as_str(),FEATURE_SCHEMA | policy::CONTEXT_FEATURE_SCHEMA)
            || e.checksum != digest(&serde_json::to_vec(&e.state).map_err(|e| e.to_string())?) {
            return Err("GA checkpoint schema, source policy, or checksum mismatch".into());
        }
        let mut encoded_state = e.state;
        if matches!(e.source_policy.as_str(),SOURCE_POLICY | PHYSICAL_SOURCE_POLICY) { checkpoint::expand(&mut encoded_state)?; }
        let state: State = serde_json::from_value(encoded_state).map_err(|e| format!("Invalid GA checkpoint state: {e}"))?;
        state.config.validate()?;
        if e.source_policy != state.config.source_policy() || e.feature_schema!=state.config.feature_schema()
            || state.frontier.is_some()!=state.config.is_frontier() || state.physical.is_some()!=state.config.is_physical() {
            return Err("GA checkpoint policy/config mismatch; fork explicitly instead of silently migrating".into());
        }
        if let Some(physical)=&state.physical {
            if state.counters.supply_queries!=0 || state.counters.verification_work!=0
                || state.counters.evaluated_episodes!=physical.fresh_episodes {
                return Err("Physical-only GA checkpoint contains supply work or mismatched fresh episode totals".into());
            }
        }
        if let Some(initialization) = &state.initialization {
            validate_weights(&initialization.inherited_weights)?;
            if (!state.config.is_frontier() && !state.config.is_physical())
                || !matches!(initialization.kind.as_str(),"inherited-weights-new-experiment"|"inherited-physical-archive-new-experiment")
                || initialization.inherited_weights_sha256 != weights_digest(&initialization.inherited_weights)
                || initialization.parent_model_id != Genome::new(initialization.inherited_weights.clone()).model_id
                || !matches!(initialization.parent_source_policy.as_str(), SOURCE_POLICY | LEGACY_SOURCE_POLICY | PHYSICAL_SOURCE_POLICY)
                || initialization.inherited_raw_spins.is_some_and(|n|n>40)
                || initialization.parent_selected_model_id.as_ref().is_some_and(|id|!id.starts_with("ga-") || id.len()!=23)
                || (initialization.kind=="inherited-physical-archive-new-experiment" && !state.config.is_physical())
                || initialization.parent_checkpoint_sha256.as_ref().is_some_and(|h|h.len()!=64) {
                return Err("Invalid inherited GA initialization provenance".into());
            }
        }
        if state.rng.state == 0 || state.population.len() != state.config.population_size
            || state.evaluations.len() != state.population.len() || state.active_genome_index > state.population.len()
            || state.completed_generations > state.config.generations || state.generation >= state.config.generations
            || state.cache.len() > 128 || state.history.len() > 256
            || state.complete != (state.completed_generations == state.config.generations)
            || !state.counters.elapsed_ms.is_finite() || state.counters.elapsed_ms < 0.0
            || state.active_evaluation.seeds.len() > state.config.episodes_per_genome {
            return Err("Invalid GA checkpoint population or phase".into());
        }
        for genome in state.population.iter().chain(state.best.iter().map(|c| &c.genome)) {
            validate_weights(&genome.weights)?;
            if genome.weights.len()!=state.config.parameter_count() || genome.model_id != Genome::new(genome.weights.clone()).model_id { return Err("GA checkpoint genome digest mismatch".into()); }
        }
        for (i, evaluation) in state.evaluations.iter().enumerate() {
            if (i < state.active_genome_index) != evaluation.is_some() {
                return Err("Invalid GA checkpoint evaluated-population cursor".into());
            }
        }
        if state.generation != if state.complete { state.completed_generations.saturating_sub(1) }
            else { state.completed_generations }
            || (state.active_genome_index == state.population.len()
                && (state.active_episode.is_some() || !state.active_evaluation.seeds.is_empty())) {
            return Err("Invalid GA checkpoint generation phase".into());
        }
        if let Some(episode) = &state.active_episode {
            episode.validate(&state.config)?;
            episode.observed_result().audit(&state.config)?;
        }
        // Every retained physical/certified best is independently replayed.
        // Aggregate optimizer history remains owner-supplied checkpoint data;
        // it is never a deployment/generalization certificate.
        let mut audited = std::collections::HashSet::new();
        let evaluations = state.evaluations.iter().flatten()
            .chain(state.cache.iter().map(|c| &c.evaluation))
            .chain(state.best.iter().map(|c| &c.evaluation))
            .chain(std::iter::once(&state.active_evaluation));
        for evaluation in evaluations {
            let n = evaluation.seeds.len() as u32;
            if n as usize > state.config.episodes_per_genome || evaluation.fitness.certified_episodes > n
                || evaluation.fitness.raw_completed_episodes>n
                || evaluation.fitness.qualified_spins > n * 40 || evaluation.fitness.raw_spins > n * 40
                || evaluation.fitness.raw_pieces > n * u32::from(state.config.max_pieces)
                || evaluation.best.is_none() != (n == 0) {
                return Err("Invalid GA checkpoint evaluation totals".into());
            }
            if evaluation.objectives.is_some() != (state.config.is_frontier() && n>0) {
                return Err("GA checkpoint objective-version mismatch".into());
            }
            if evaluation.physical.is_some()!=(state.config.is_physical() && n>0){return Err("GA physical objective-version mismatch".into());}
            if let Some(p)=&evaluation.physical {
                if p.min_spins.is_none_or(|m|m>40 || m>p.max_spins) || p.max_spins>40 || p.completed_cost>n*126
                    || evaluation.fitness.raw_spins<u32::from(p.min_spins.unwrap_or(0))*n
                    || evaluation.fitness.raw_spins>u32::from(p.max_spins)*n
                    || (evaluation.fitness.raw_completed_episodes>0 && p.max_spins<40)
                    || evaluation.fitness.certified_episodes!=0 || evaluation.fitness.qualified_spins!=0
                    || evaluation.fitness.failure_quiet_safe!=0 || evaluation.fitness.failure_success!=0 {
                    return Err("Invalid physical-only GA evaluation totals".into());
                }
            }
            if let Some(objectives)=&evaluation.objectives {
                if !objectives.probability_sum.is_finite() || objectives.probability_sum<0.0
                    || objectives.probability_sum>f64::from(n) || objectives.strict_spins>n*40
                    || objectives.raw_max>40 || objectives.behavior_ids.len()!=n as usize
                    || objectives.foundation_ids.len()>n as usize {
                    return Err("Invalid GA multiobjective checkpoint totals".into());
                }
            }
            if let Some(best) = &evaluation.best {
                if !evaluation.seeds.contains(&best.seed) || (evaluation.fitness.certified_episodes > 0 && !best.certified40) {
                    return Err("Invalid GA checkpoint selected episode".into());
                }
                let hash = digest(&serde_json::to_vec(best).map_err(|e| e.to_string())?);
                if audited.insert(hash) { best.audit(&state.config)?; }
                if n == 1 {
                    let mut exact = Fitness::default(); exact.add(best);
                    if exact != evaluation.fitness { return Err("GA checkpoint single-episode fitness mismatch".into()); }
                    if let Some(p)=&evaluation.physical{let mut exact=physical::EvaluationMetrics::default();exact.add(best);
                        if exact!=*p{return Err("Physical checkpoint single-episode metrics mismatch".into());}}
                }
            }
        }
        if let Some(archive)=&state.frontier { archive.audit(&state.config,state.generation,&mut audited)?; }
        if let Some(archive)=&state.physical { archive.audit(&state.config,state.generation,&mut audited)?; }
        Ok(Self { state, restore_audit_ms: (now_ms() - restore_start).max(0.0) })
    }
}

/// Frozen-policy evaluation for a native matched-budget comparison. Each
/// completed independent seeded rollout starts from the same empty game.
/// The caller owns its seed roster, work/time cap, and validation/test split.
pub struct FrozenRunner {
    model: GaModel, config: GaConfig, seeds: Vec<u32>, next_seed: usize,
    episode: Option<Episode>, evaluation: Evaluation, counters: Counters,
    maximum_raw_spins: u8, maximum_qualified_spins: u8,
    seed_results: Vec<serde_json::Value>,
}
impl FrozenRunner {
    pub fn new(model: GaModel, config: GaConfig, seeds: Vec<u32>) -> Result<Self, String> {
        model.validate()?; config.validate()?;
        if model.weights.len()!=config.parameter_count(){return Err("Frozen model capacity differs from evaluation configuration".into());}
        if seeds.is_empty() || seeds.len() > 65_536 { return Err("Supply 1..65536 frozen evaluation seeds".into()); }
        let mut unique = std::collections::HashSet::new();
        if !seeds.iter().all(|s| unique.insert(*s)) { return Err("Frozen evaluation seeds must be distinct".into()); }
        Ok(Self { model, config, seeds, next_seed: 0, episode: None, evaluation: Evaluation::empty(), counters: Counters::default(),
            maximum_raw_spins: 0, maximum_qualified_spins: 0, seed_results: vec![] })
    }
    pub fn finished(&self) -> bool { self.next_seed >= self.seeds.len() && self.episode.is_none() }
    pub fn counters(&self) -> &Counters { &self.counters }
    pub fn advance(&mut self, work: u32) -> Result<(), String> {
        let start = now_ms();
        for _ in 0..work.clamp(1, 64) {
            if self.finished() { break; }
            if self.episode.is_none() { self.episode = Some(Episode::for_config(self.seeds[self.next_seed], &self.config)); self.next_seed += 1; }
            let episode = self.episode.as_mut().unwrap();
            episode.advance(&self.model.weights, &self.config, &mut self.counters)?;
            self.maximum_raw_spins = self.maximum_raw_spins.max(episode.raw_spins);
            self.maximum_qualified_spins = self.maximum_qualified_spins.max(episode.qualified_spins);
            if episode.finished() {
                let result=episode.result()?;
                if self.config.is_physical(){
                    let trace:Vec<_>=result.frames.iter().map(|f|(&f.piece,f.rows,f.placed_cells,f.spins)).collect();
                    self.seed_results.push(serde_json::json!({"seed":result.seed,"rawSpins":result.raw_spins,
                        "rawPieces":result.raw_pieces,"rawCompleted40":result.raw_completed40,"termination":result.termination,
                        "behaviorDigest":digest(&serde_json::to_vec(&trace).unwrap())}));
                }
                self.evaluation.add(result); self.counters.evaluated_episodes += 1; self.episode = None;
            }
        }
        self.counters.elapsed_ms += (now_ms() - start).max(0.0);
        Ok(())
    }
    pub fn report_json(&self) -> String {
        let current = self.episode.as_ref().map(Episode::observed_result);
        let mut current_fitness = Fitness::default();
        if let Some(r) = &current { current_fitness.add(r); }
        let mut best_fitness = Fitness::default();
        if let Some(r) = &self.evaluation.best { best_fitness.add(r); }
        let use_current = current.as_ref().is_some_and(|current| self.evaluation.best.as_ref().is_none_or(|best|
            if self.config.is_physical(){physical::compare_result(current,best).is_gt()}
            else if self.config.is_frontier() { evolution::compare_result(current,best).is_gt() }
            else { current_fitness > best_fitness }));
        let best = if use_current { current.as_ref() } else { self.evaluation.best.as_ref() };
        serde_json::json!({"schemaVersion":"neyra-ga-evaluation-v1","sourcePolicy":self.config.source_policy(),"modelSourcePolicy":self.model.source_policy,
            "modelId":self.model.model_id,"weightsSha256":self.model.weights_sha256,
            "config":self.config,"requestedSeeds":self.seeds,"completedSeeds":self.evaluation.seeds,
            "complete":self.finished(),"fitness":self.evaluation.fitness,"best":best,
            "perSeed":if self.config.is_physical(){Some(&self.seed_results)}else{None},
            "maximumObservedRawSpins":self.maximum_raw_spins,"maximumObservedQualifiedSpins":self.maximum_qualified_spins,
            "physical":if self.config.is_physical(){Some(serde_json::json!({"fromEmpty40Count":self.evaluation.fitness.raw_completed_episodes,
                "freshEpisodes":self.evaluation.seeds.len(),"fromEmptyMeanSpins":if self.evaluation.seeds.is_empty(){None}else{Some(f64::from(self.evaluation.fitness.raw_spins)/self.evaluation.seeds.len() as f64)},
                "fromEmptyMinSpins":self.evaluation.physical.as_ref().and_then(|p|p.min_spins),"fromEmptyMaxSpins":self.evaluation.physical.as_ref().map(|p|p.max_spins),
                "probabilityMeasured":false,"prefixReturns":0}))}else{None},
            "bestFromUnfinishedEpisode":use_current,
            "counters":self.counters,"unfinishedEpisode":self.episode.as_ref().map(|e|serde_json::json!({
                "seed":e.seed,"rawSpins":e.raw_spins,"qualifiedSpins":e.qualified_spins,"rawPieces":e.raw_pieces})),
            "deploymentReady":false,"evidence":"Independent fixed-policy rollouts; seeds are exploration order, not different games. Incomplete episodes are excluded from aggregate fitness; an independently certified live prefix can appear in best with bestFromUnfinishedEpisode=true."}).to_string()
    }
}
