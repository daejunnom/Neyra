//! Version-three reporting archives and multiobjective reproduction.
//! These compare measured routes; dominance is never a physical impossibility
//! claim and never removes a legal move from a rollout.

use super::*;
use num_bigint::BigUint;
use std::collections::BTreeSet;

const FRONTIER_CAP: usize = 32;
const FOUNDATION_CAP: usize = 16;
const LEDGER_CAP: usize = 4096;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FrontierMetrics {
    pub probability_includes_first_bag: bool,
    pub success_counts: Vec<u16>,
    pub selected_success_counts: Vec<u16>,
    pub full_probability: f64,
    pub full_probability_budget_pieces: Option<u16>,
    pub strict_spins: u8,
    pub strict_bags: u16,
    pub strict_pieces: u16,
    pub strict_certified40: bool,
    pub strict_budget_pieces: Option<u16>,
    pub behavior_id: String,
    pub foundation_id: Option<String>,
}

impl FrontierMetrics {
    pub(super) fn from_result(result: &EpisodeResult) -> Self {
        let qualified: Vec<_> = result.bag_proofs.iter().take_while(|p| p.qualified).collect();
        let counts: Vec<_> = qualified.iter().map(|p| p.success).collect();
        let mut selected_counts = counts.clone();
        if let Some(proof) = qualified.last().filter(|p| p.terminal) {
            if let Some(budget) = proof.budget_pieces {
                let cost = budget - (proof.bag_index - 1) * 7;
                *selected_counts.last_mut().unwrap() = proof.effective_costs.iter()
                    .filter(|&&c| c > 0 && u16::from(c) <= cost).count() as u16;
            }
        }
        let full_budget = qualified.last().map(|p| if p.terminal {
            (p.bag_index - 1) * 7 + u16::from(p.effective_costs.iter().copied()
                .filter(|c| (1..=7).contains(c)).max().unwrap_or(7))
        } else { p.bag_index * 7 });
        let (mut strict_spins, mut strict_bags, mut strict_pieces, mut strict_budget) = (0, 0, 0, None);
        for proof in &qualified {
            let Some(strict) = &proof.strict100 else { break; };
            if !strict.qualified { break; }
            strict_spins = proof.end_spins; strict_bags += 1;
            strict_pieces = strict.budget_pieces.unwrap_or(proof.bag_index * 7);
            strict_budget = strict.budget_pieces;
        }
        // Logical typed targets and their whole execution context identify a
        // foundation. Occupancy alone cannot identify this fixed tiling.
        let foundation_id = result.bag_proofs.first().map(|p| {
            let mut targets = p.targets.clone(); targets.sort_by_key(|t| t.piece as u8);
            digest(serde_json::json!({"policy":SOURCE_POLICY,"goal":40,
                "startRows":p.start_rows,"endRows":p.end_rows,"startSpins":p.start_spins,
                "endSpins":p.end_spins,"targets":targets,"terminal":p.terminal,
                "symmetry":p.first_bag_symmetry_assumed}).to_string().as_bytes())
        });
        let trace: Vec<_> = result.frames.iter().map(|f| {
            let mut cells = f.placed_cells; cells.sort_unstable();
            (&f.piece, f.rows, cells, f.spins, f.lines, f.spin)
        }).collect();
        let behavior_id = digest(&serde_json::to_vec(&trace).unwrap());
        let full_probability = if counts.is_empty() { 0.0 } else { chance(&counts) };
        Self { probability_includes_first_bag: true, success_counts: counts,
            selected_success_counts: selected_counts, full_probability,
            full_probability_budget_pieces: full_budget, strict_spins, strict_bags,
            strict_pieces, strict_certified40: strict_spins >= 40,
            strict_budget_pieces: strict_budget, behavior_id, foundation_id }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Objectives {
    pub probability_sum: f64,
    pub strict_spins: u32,
    pub raw_max: u8,
    pub behavior_ids: Vec<String>,
    pub foundation_ids: Vec<String>,
}

impl Objectives {
    pub fn add(&mut self, result: &EpisodeResult) {
        let Some(frontier) = &result.frontier else { return; };
        self.probability_sum += result.probability;
        self.strict_spins += u32::from(frontier.strict_spins);
        self.raw_max = self.raw_max.max(result.raw_spins);
        self.behavior_ids.push(frontier.behavior_id.clone());
        if let Some(id) = &frontier.foundation_id { self.foundation_ids.push(id.clone()); }
    }
}

pub(super) fn compare_result(a: &EpisodeResult, b: &EpisodeResult) -> Ordering {
    a.certified40.cmp(&b.certified40)
        .then(a.qualified_spins.cmp(&b.qualified_spins))
        .then(a.probability.total_cmp(&b.probability))
        .then_with(|| if a.certified40 && b.certified40 {
            b.budget_pieces.cmp(&a.budget_pieces)
        } else { Ordering::Equal })
        .then_with(|| {
            let qa = a.bag_proofs.iter().find(|p| !p.qualified);
            let qb = b.bag_proofs.iter().find(|p| !p.qualified);
            qa.map_or(0, |p| p.quiet_safe_supplies).cmp(&qb.map_or(0, |p| p.quiet_safe_supplies))
                .then(qa.map_or(0, |p| p.success).cmp(&qb.map_or(0, |p| p.success)))
        })
        .then(a.raw_spins.cmp(&b.raw_spins))
        .then(b.raw_pieces.cmp(&a.raw_pieces))
}

pub(super) fn compare_evaluation(a: &Evaluation, b: &Evaluation) -> Ordering {
    let ao = a.objectives.as_ref().unwrap(); let bo = b.objectives.as_ref().unwrap();
    a.fitness.certified_episodes.cmp(&b.fitness.certified_episodes)
        .then(a.fitness.qualified_spins.cmp(&b.fitness.qualified_spins))
        .then(ao.probability_sum.total_cmp(&bo.probability_sum))
        .then_with(|| if a.fitness.certified_episodes > 0 {
            b.fitness.certified_budget_pieces.cmp(&a.fitness.certified_budget_pieces)
        } else { Ordering::Equal })
        .then(a.fitness.failure_quiet_safe.cmp(&b.fitness.failure_quiet_safe))
        .then(a.fitness.failure_success.cmp(&b.fitness.failure_success))
        .then(a.fitness.raw_spins.cmp(&b.fitness.raw_spins))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Route {
    pub id: String,
    pub kind: String,
    pub generation: u32,
    pub genome: Genome,
    pub result: EpisodeResult,
    pub spins: u8,
    pub pieces: u16,
    pub probability: Option<f64>,
    pub success_counts: Vec<u16>,
    pub strict100: bool,
}

fn chance(counts: &[u16]) -> f64 {
    counts.iter().fold(1.0, |p, &n| p * f64::from(n) / P7_COUNT as f64)
}

fn compare_chance(a: &[u16], b: &[u16]) -> Ordering {
    let numerator = |counts: &[u16]| counts.iter().fold(BigUint::from(1_u8), |n, &c| n * c);
    (numerator(a) * BigUint::from(P7_COUNT as u32).pow(b.len() as u32))
        .cmp(&(numerator(b) * BigUint::from(P7_COUNT as u32).pow(a.len() as u32)))
}

impl Route {
    fn new(kind: &str, generation: u32, genome: &Genome, result: &EpisodeResult,
        spins: u8, pieces: u16, counts: Vec<u16>) -> Self {
        let probability = (kind != "raw").then(|| chance(&counts));
        let strict100 = kind != "raw" && !counts.is_empty() && counts.iter().all(|&c| usize::from(c) == P7_COUNT);
        // A measured prefix is the same route when its later unqualified raw
        // suffix changes. Identify the fixed typed plans, not that suffix.
        let plans: Vec<_> = result.bag_proofs.iter().take(counts.len()).map(|p| {
            let mut targets = p.targets.clone(); targets.sort_by_key(|target|target.piece as u8);
            serde_json::json!({"startRows":p.start_rows,"startSpins":p.start_spins,
                "targets":targets,"terminal":p.terminal,"symmetry":p.first_bag_symmetry_assumed})
        }).collect();
        let identity = serde_json::json!({"kind":kind,
            "rawBehavior":if kind=="raw"{result.frontier.as_ref().map(|f|&f.behavior_id)}else{None},
            "plans":plans,"pieces":pieces,"counts":counts}).to_string();
        Self { id: format!("{kind}-{}", &digest(identity.as_bytes())[..24]), kind: kind.into(), generation,
            genome: genome.clone(), result: result.clone(), spins, pieces, probability, success_counts: counts, strict100 }
    }

    fn dominates(&self, other: &Self) -> bool {
        let probability = compare_chance(&self.success_counts, &other.success_counts);
        self.spins >= other.spins && self.pieces <= other.pieces && probability != Ordering::Less
            && (self.spins > other.spins || self.pieces < other.pieces || probability == Ordering::Greater)
    }

    fn frames(&self) -> Vec<DisplayFrame> {
        if self.kind == "raw" { return self.result.frames.clone(); }
        if self.kind != "strict" { return self.result.qualified_frames.clone(); }
        let mut frames = vec![];
        for proof in &self.result.bag_proofs {
            let Some(strict) = &proof.strict100 else { break; };
            if !strict.qualified { break; }
            if let Some(witness) = &strict.terminal_witness { frames.extend(witness.frames.clone()); }
            else { frames.extend(proof.representative_frames.clone()); }
        }
        frames
    }

    pub fn snapshot(&self) -> serde_json::Value {
        let frames = self.frames();
        serde_json::json!({"id":self.id,"kind":self.kind,"modelId":self.genome.model_id,
            "generation":self.generation,"spins":self.spins,"pieces":self.pieces,
            "rawSpins":self.result.raw_spins,"rawPieces":self.result.raw_pieces,
            "qualifiedSpins":self.result.qualified_spins,"qualifiedBags":self.result.qualified_bags,
            "probability":self.probability,"probabilityIncludesFirstBag":true,
            "firstBagProbability":self.result.first_bag_probability,
            "firstBagSymmetryAssumed":self.result.first_bag_symmetry_assumed,"mirroredInputsVerified":false,
            "quietSafe":self.kind!="raw","strict100":self.strict100,
            "certified40":self.kind!="raw"&&self.spins>=40,
            "strictCertified40":self.strict100&&self.spins>=40,
            "rawCompleted40":self.result.raw_completed40,
            "rows":frames.last().map_or([0;HEIGHT],|f|f.rows),"frames":frames,
            "representativePieces":frames.last().map_or(0,|f|f.placed),
            "completeHistory":true,"fromGeneration":0,"inputReplayVerified":true})
    }

    fn audit(&self, config: &GaConfig, audited: &mut std::collections::HashSet<String>) -> Result<(), String> {
        validate_weights(&self.genome.weights)?;
        if self.genome.model_id != Genome::new(self.genome.weights.clone()).model_id
            || self.pieces > config.max_pieces || self.success_counts.len() > 18
            || self.success_counts.iter().any(|&c| c == 0 || usize::from(c) > P7_COUNT) {
            return Err("Invalid archived GA genome/counts".into());
        }
        let hash = digest(&serde_json::to_vec(&self.result).map_err(|e|e.to_string())?);
        if audited.insert(hash) { self.result.audit(config)?; }
        let possible = routes(self.generation, &self.genome, &self.result);
        if !possible.iter().any(|r| r.id == self.id && r.kind == self.kind && r.spins == self.spins
            && r.pieces == self.pieces && r.probability == self.probability
            && r.success_counts == self.success_counts && r.strict100 == self.strict100) {
            return Err("Archived GA route differs from its independently verified evidence".into());
        }
        Ok(())
    }
}

fn routes(generation: u32, genome: &Genome, result: &EpisodeResult) -> Vec<Route> {
    let Some(metrics) = &result.frontier else { return vec![]; };
    let mut routes = vec![Route::new("raw", generation, genome, result, result.raw_spins, result.raw_pieces, vec![])];
    if metrics.strict_bags > 0 {
        routes.push(Route::new("strict", generation, genome, result, metrics.strict_spins,
            metrics.strict_pieces, vec![P7_COUNT as u16; usize::from(metrics.strict_bags)]));
    }
    if metrics.success_counts.is_empty() { return routes; }
    let last = result.bag_proofs.iter().take_while(|p| p.qualified).last().unwrap();
    if !last.terminal {
        routes.push(Route::new("frontier", generation, genome, result, result.qualified_spins,
            last.bag_index * 7, metrics.success_counts.clone()));
    } else {
        for cost in 1..=last.targets.len() as u16 {
            let pieces = (last.bag_index - 1) * 7 + cost;
            let count = last.effective_costs.iter().filter(|&&c| c > 0 && c <= cost as u8).count() as u16;
            if count == 0 { continue; }
            let mut counts = metrics.success_counts.clone(); *counts.last_mut().unwrap() = count;
            routes.push(Route::new("frontier", generation, genome, result, result.qualified_spins, pieces, counts));
        }
    }
    routes
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct FoundationNiche {
    foundation_id: String,
    representative: Route,
    last_generation: u32,
    observations: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct Archive {
    pub raw: Option<Route>,
    pub strict: Option<Route>,
    pub frontier: Vec<Route>,
    foundation_niches: Vec<FoundationNiche>,
    pub behavior_ledger: Vec<(String, u32)>,
    pub foundation_ledger: Vec<String>,
    pub duplicate_episodes: u64,
    pub behavior_evictions: u64,
    pub foundation_evictions: u64,
    pub stagnant_generations: u32,
    pub improved_this_generation: bool,
    pub generation_raw_max: u8,
}

impl Default for Archive {
    fn default() -> Self {
        Self { raw: None, strict: None, frontier: vec![], foundation_niches: vec![], behavior_ledger: vec![], foundation_ledger: vec![],
            duplicate_episodes: 0, behavior_evictions: 0, foundation_evictions: 0,
            stagnant_generations: 0, improved_this_generation: false, generation_raw_max: 0 }
    }
}

impl Archive {
    pub fn raw_should_update(&self, spins: u8, pieces: u16) -> bool {
        pieces > 0 && self.raw.as_ref().is_none_or(|old| spins > old.spins || (spins == old.spins && pieces < old.pieces))
    }
    pub fn observe(&mut self, generation: u32, genome: &Genome, result: &EpisodeResult, prefix_changed: bool) {
        self.generation_raw_max = self.generation_raw_max.max(result.raw_spins);
        // Different typed foundations may have different future potential.
        // Keep a separate bounded niche reservoir even when their immediate
        // spin/cost/probability point is globally dominated in the chart.
        if prefix_changed && result.bag_proofs.len() == 1 {
            if let Some(id) = result.frontier.as_ref().and_then(|f|f.foundation_id.as_ref()) {
                if let Some(niche) = self.foundation_niches.iter_mut().find(|n|n.foundation_id == *id) {
                    niche.last_generation = generation; niche.observations += 1;
                } else {
                    if self.foundation_niches.len() == FOUNDATION_CAP {
                        let remove = self.foundation_niches.iter().enumerate()
                            .min_by_key(|(i,n)|(n.last_generation,n.observations,*i)).unwrap().0;
                        self.foundation_niches.remove(remove);
                    }
                    self.foundation_niches.push(FoundationNiche { foundation_id: id.clone(),
                        representative: Route::new("raw",generation,genome,result,result.raw_spins,result.raw_pieces,vec![]),
                        last_generation: generation, observations: 1 });
                }
            }
        }
        for route in routes(generation, genome, result) {
            if route.kind == "raw" {
                if self.raw_should_update(route.spins, route.pieces) {
                    self.improved_this_generation = true; self.raw = Some(route);
                }
            } else if prefix_changed && route.kind == "strict" {
                if self.strict.as_ref().is_none_or(|old| route.spins > old.spins || (route.spins == old.spins && route.pieces < old.pieces)) {
                    self.improved_this_generation = true; self.strict = Some(route);
                }
            } else if prefix_changed && route.kind == "frontier" {
                if self.frontier.iter().any(|r| r.id == route.id || r.dominates(&route)) { continue; }
                let equal_metrics = self.frontier.iter().filter(|r| r.spins == route.spins && r.pieces == route.pieces
                    && compare_chance(&r.success_counts, &route.success_counts) == Ordering::Equal).count();
                if equal_metrics >= 2 { continue; }
                if equal_metrics == 0 { self.improved_this_generation = true; }
                self.frontier.retain(|r| !route.dominates(r)); self.frontier.push(route);
                if self.frontier.len() > FRONTIER_CAP {
                    let vectors: Vec<_> = self.frontier.iter().map(|r| vec![f64::from(r.spins),
                        r.probability.unwrap_or(0.0).max(f64::MIN_POSITIVE).ln(), -f64::from(r.pieces)]).collect();
                    let crowd = crowding(&(0..vectors.len()).collect::<Vec<_>>(), &vectors);
                    let remove = (0..crowd.len()).min_by(|&a,&b| crowd[a].total_cmp(&crowd[b]).then(b.cmp(&a))).unwrap();
                    self.frontier.remove(remove);
                }
            }
        }
    }

    pub fn completed(&mut self, result: &EpisodeResult) {
        let Some(metrics) = &result.frontier else { return; };
        if let Some((_, count)) = self.behavior_ledger.iter_mut().find(|(id,_)| *id == metrics.behavior_id) {
            *count = count.saturating_add(1); self.duplicate_episodes += 1;
        } else {
            if self.behavior_ledger.len() == LEDGER_CAP { self.behavior_ledger.remove(0); self.behavior_evictions += 1; }
            self.behavior_ledger.push((metrics.behavior_id.clone(),1));
        }
        if let Some(id) = &metrics.foundation_id {
            if !self.foundation_ledger.contains(id) {
                if self.foundation_ledger.len() == LEDGER_CAP { self.foundation_ledger.remove(0); self.foundation_evictions += 1; }
                self.foundation_ledger.push(id.clone());
            }
        }
    }

    fn frequency(&self, evaluation: &Evaluation) -> u32 {
        evaluation.objectives.as_ref().map_or(0, |o| o.behavior_ids.iter().map(|id|
            self.behavior_ledger.iter().find(|(known,_)| known == id).map_or(0,|(_,n)|*n)).sum())
    }

    pub fn snapshot(&self, config: &GaConfig, generation_unique: usize) -> serde_json::Value {
        let (immigrants, scale, rate) = reproduction_budget(config, self.stagnant_generations);
        serde_json::json!({"uniqueBehaviors":self.behavior_ledger.len(),"duplicateEpisodes":self.duplicate_episodes,
            "uniqueFoundations":self.foundation_ledger.len(),"generationUniqueBehaviors":generation_unique,
            "stagnantGenerations":self.stagnant_generations,"immigrantCount":immigrants,
            "effectiveMutationScale":scale,"effectiveMutationRate":rate,
            "boundedHistory":true,"ledgerCapacity":LEDGER_CAP,"behaviorEvictions":self.behavior_evictions,
            "foundationEvictions":self.foundation_evictions,"planVerificationCacheEnabled":false})
            .as_object().map(|object| {
                let mut value = serde_json::Value::Object(object.clone());
                value["probabilityFrontierCapacity"] = FRONTIER_CAP.into();
                value["foundationNicheCount"] = self.foundation_niches.len().into();
                value["foundationNicheCapacity"] = FOUNDATION_CAP.into();
                value["foundationParentFraction"] = 0.25.into();
                value
            }).unwrap()
    }

    pub fn audit(&self, config: &GaConfig, generation: u32, audited: &mut std::collections::HashSet<String>) -> Result<(), String> {
        if self.frontier.len() > FRONTIER_CAP || self.behavior_ledger.len() > LEDGER_CAP
            || self.foundation_ledger.len() > LEDGER_CAP || self.generation_raw_max > 40
            || self.foundation_niches.len() > FOUNDATION_CAP
            || self.raw.as_ref().is_some_and(|r|r.kind!="raw")
            || self.strict.as_ref().is_some_and(|r|r.kind!="strict")
            || self.frontier.iter().any(|r|r.kind!="frontier") {
            return Err("Invalid GA archive/ledger capacity".into());
        }
        let mut seen = BTreeSet::new();
        if self.behavior_ledger.iter().any(|(id,n)| id.len()!=64 || *n==0 || !seen.insert(id)) {
            return Err("Invalid GA behavior ledger".into());
        }
        for route in self.raw.iter().chain(self.strict.iter()).chain(self.frontier.iter()) {
            if route.generation > generation { return Err("Archived GA route is from a future generation".into()); }
            route.audit(config, audited)?;
        }
        let mut foundations = BTreeSet::new();
        for niche in &self.foundation_niches {
            if !foundations.insert(&niche.foundation_id) || niche.observations==0 || niche.last_generation>generation
                || niche.representative.generation>generation || niche.representative.kind!="raw"
                || niche.representative.result.bag_proofs.len()!=1
                || niche.representative.result.frontier.as_ref().and_then(|f|f.foundation_id.as_ref())!=Some(&niche.foundation_id) {
                return Err("Invalid independently preserved GA foundation niche".into());
            }
            niche.representative.audit(config, audited)?;
        }
        Ok(())
    }
}

fn axes(evaluation: &Evaluation) -> Vec<f64> {
    let f = &evaluation.fitness; let o = evaluation.objectives.as_ref().unwrap();
    vec![f64::from(f.certified_episodes), f64::from(f.qualified_spins), o.probability_sum,
        f64::from(o.strict_spins), f64::from(f.raw_spins), f64::from(f.failure_quiet_safe), f64::from(f.failure_success),
        if f.certified_episodes>0 {-f64::from(f.certified_budget_pieces)/f64::from(f.certified_episodes)}else{-126.0}]
}

fn dominates(a: &[f64], b: &[f64]) -> bool {
    a.iter().zip(b).all(|(x,y)|x>=y) && a.iter().zip(b).any(|(x,y)|x>y)
}

fn crowding(front: &[usize], values: &[Vec<f64>]) -> Vec<f64> {
    let mut crowd = vec![0.0; values.len()];
    if front.is_empty() { return crowd; }
    for axis in 0..values[0].len() {
        let mut order = front.to_vec(); order.sort_by(|&a,&b|values[a][axis].total_cmp(&values[b][axis]).then(a.cmp(&b)));
        let low=values[order[0]][axis]; let high=values[*order.last().unwrap()][axis];
        if high<=low { continue; }
        crowd[order[0]]=f64::INFINITY;crowd[*order.last().unwrap()]=f64::INFINITY;
        for triple in order.windows(3) { crowd[triple[1]]+=(values[triple[2]][axis]-values[triple[0]][axis])/(high-low); }
    }
    crowd
}

#[derive(Clone, Copy)]
struct Rank { front: usize, crowd: f64, frequency: u32 }
fn rank_order(a: &Rank,b: &Rank) -> Ordering {
    a.front.cmp(&b.front).then(b.crowd.total_cmp(&a.crowd)).then(a.frequency.cmp(&b.frequency))
}

fn ranks(state: &State) -> Vec<Rank> {
    let values: Vec<_>=state.evaluations.iter().map(|e|axes(e.as_ref().unwrap())).collect();
    let mut dominated_by=vec![0_usize;values.len()];let mut dominated=vec![vec![];values.len()];
    for i in 0..values.len(){for j in 0..values.len(){if i!=j&&dominates(&values[i],&values[j]){dominated[i].push(j);dominated_by[j]+=1;}}}
    let mut result:Vec<_>=state.evaluations.iter().map(|e|Rank{front:usize::MAX,crowd:0.0,
        frequency:state.frontier.as_ref().unwrap().frequency(e.as_ref().unwrap())}).collect();
    let mut front:Vec<_>=(0..values.len()).filter(|&i|dominated_by[i]==0).collect();let mut depth=0;
    while !front.is_empty(){let crowd=crowding(&front,&values);let mut next=vec![];
        for &i in &front{result[i].front=depth;result[i].crowd=crowd[i];for &j in &dominated[i]{dominated_by[j]-=1;if dominated_by[j]==0{next.push(j);}}}
        front=next;depth+=1;
    }
    result
}

fn reproduction_budget(config: &GaConfig, stagnant: u32) -> (usize,f64,f64) {
    let percent=if stagnant>=8{25}else{10};
    let immigrants=(config.population_size*percent).div_ceil(100).max(1).min(config.population_size-2);
    let stage=(stagnant/8).min(3);
    (immigrants,(config.mutation_scale*(1.0+f64::from(stage))).min(4.0),
        (config.mutation_rate+f64::from(stage)*0.05).min(1.0))
}

fn select(rng:&mut Random,ranks:&[Rank],size:usize)->usize{
    let mut best=rng.index(ranks.len());for _ in 1..size{let other=rng.index(ranks.len());if rank_order(&ranks[other],&ranks[best]).is_lt(){best=other;}}best
}

pub(super) fn finish_generation(state:&mut State){
    let mut display:Vec<_>=(0..state.population.len()).collect();
    display.sort_by(|&a,&b|compare_evaluation(state.evaluations[b].as_ref().unwrap(),state.evaluations[a].as_ref().unwrap()).then(a.cmp(&b)));
    let top=state.evaluations[display[0]].as_ref().unwrap();
    let archive=state.frontier.as_mut().unwrap();
    let unique:BTreeSet<_>=state.evaluations.iter().flatten().flat_map(|e|e.objectives.as_ref().unwrap().behavior_ids.iter()).collect();
    state.history.push(GenerationSummary{generation:state.generation,best_fitness:top.fitness.clone(),
        best_qualified_spins:top.best.as_ref().map_or(0,|r|r.qualified_spins),best_raw_spins:top.best.as_ref().map_or(0,|r|r.raw_spins),
        certified40_count:state.evaluations.iter().filter(|e|e.as_ref().unwrap().fitness.certified_episodes>0).count(),evaluated_genomes:state.population.len(),
        generation_raw_max:Some(archive.generation_raw_max),all_time_raw_max:Some(archive.raw.as_ref().map_or(0,|r|r.spins)),
        strict_spins:Some(archive.strict.as_ref().map_or(0,|r|r.spins)),
        frontier_spins:Some(archive.frontier.iter().map(|r|r.spins).max().unwrap_or(0)),
        frontier_probability:archive.frontier.iter().max_by(|a,b|a.spins.cmp(&b.spins).then(compare_chance(&a.success_counts,&b.success_counts))).and_then(|r|r.probability),
        unique_behaviors:Some(unique.len())});
    if state.history.len()>256{state.history.remove(0);}
    archive.stagnant_generations=if archive.improved_this_generation{0}else{archive.stagnant_generations+1};
    archive.improved_this_generation=false;archive.generation_raw_max=0;
    state.completed_generations+=1;
    if state.completed_generations>=state.config.generations{state.complete=true;return;}
    let ranks=ranks(state);let mut ranked:Vec<_>=(0..ranks.len()).collect();ranked.sort_by(|&a,&b|rank_order(&ranks[a],&ranks[b]).then(a.cmp(&b)));
    let archive=state.frontier.as_ref().unwrap();
    let (immigrants,scale,rate)=reproduction_budget(&state.config,archive.stagnant_generations);
    let elite_budget=state.config.elite_count.max(3).min(state.population.len()-immigrants-1);
    let mut next=vec![];let mut ids=BTreeSet::new();
    for route in archive.strict.iter().chain(archive.raw.iter()).chain(archive.frontier.iter().rev()){
        if next.len()>=elite_budget{break;}if ids.insert(route.genome.model_id.clone()){next.push(route.genome.clone());}
    }
    for &i in &ranked{if next.len()>=elite_budget{break;}if ids.insert(state.population[i].model_id.clone()){next.push(state.population[i].clone());}}
    while next.len()<state.config.population_size-immigrants{
        let a=select(&mut state.rng,&ranks,state.config.tournament_size);let b=select(&mut state.rng,&ranks,state.config.tournament_size);
        let foundation_parent = if next.len()%4==0 && !archive.foundation_niches.is_empty() {
            Some(&archive.foundation_niches[state.rng.index(archive.foundation_niches.len())].representative.genome)
        } else { None };
        let weights=(0..FEATURE_COUNT).map(|j|{let blend=state.rng.unit();let mut w=blend*foundation_parent.unwrap_or(&state.population[a]).weights[j]+(1.0-blend)*state.population[b].weights[j];
            if state.rng.unit()<rate{w+=state.rng.normal()*scale;}w.clamp(-16.0,16.0)}).collect();next.push(Genome::new(weights));
    }
    while next.len()<state.config.population_size{
        let spread=if archive.stagnant_generations>=8{4.0}else{2.0};
        let weights=INITIAL_WEIGHTS.iter().map(|w|(w+state.rng.normal()*spread).clamp(-16.0,16.0)).collect();next.push(Genome::new(weights));
    }
    state.population=next;state.evaluations=vec![None;state.config.population_size];state.generation+=1;state.active_genome_index=0;
}
