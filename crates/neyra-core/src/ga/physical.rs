//! Probability-free candidate generation. Every stored trajectory replays
//! real Sonic Drop inputs; no route in this module is a supply certificate.
use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default,rename_all="camelCase",deny_unknown_fields)]
pub struct Config {
    pub context_features: bool,
    pub action_noise: f64,
    pub continuation_every: usize,
    pub archive_capacity: usize,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub repair_beam_width: Option<usize>,
}
impl Default for Config {
    fn default()->Self{Self{context_features:false,action_noise:0.025,continuation_every:2,archive_capacity:96,repair_beam_width:None}}
}
impl Config {
    pub(super) fn with_repair_defaults()->Self{Self{repair_beam_width:Some(64),..Self::default()}}
    pub(super) fn validate(&self)->Result<(),String>{
        if !self.action_noise.is_finite() || !(0.0..=1.0).contains(&self.action_noise)
            || self.continuation_every>64 || !(8..=256).contains(&self.archive_capacity)
            || self.repair_beam_width.is_some_and(|width|width>512){
            return Err("Invalid physical GA noise, continuation or archive limits".into());
        } Ok(())
    }
}

#[derive(Clone,Debug,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(rename_all="camelCase",deny_unknown_fields)]
pub struct EpisodeMetrics {
    pub continuation_prefix_pieces:u16,
    #[serde(default,skip_serializing_if="is_false")]
    pub repaired:bool,
}

#[derive(Clone,Debug,Default,PartialEq,Eq,Serialize,Deserialize)]
#[serde(rename_all="camelCase",deny_unknown_fields)]
pub(super) struct EvaluationMetrics { pub min_spins:Option<u8>, pub max_spins:u8, pub completed_cost:u32 }
impl EvaluationMetrics {
    pub fn add(&mut self,r:&EpisodeResult){self.min_spins=Some(self.min_spins.map_or(r.raw_spins,|n|n.min(r.raw_spins)));
        self.max_spins=self.max_spins.max(r.raw_spins);if r.raw_completed40{self.completed_cost+=u32::from(r.raw_pieces);}}
}

pub(super) fn compare_result(a:&EpisodeResult,b:&EpisodeResult)->Ordering{
    a.raw_completed40.cmp(&b.raw_completed40).then(a.raw_spins.cmp(&b.raw_spins))
        .then(b.raw_pieces.cmp(&a.raw_pieces)).then(Board{rows:b.rows}.height().cmp(&Board{rows:a.rows}.height()))
}
pub(super) fn compare_evaluation(a:&Evaluation,b:&Evaluation)->Ordering{
    let ap=a.physical.as_ref().unwrap();let bp=b.physical.as_ref().unwrap();
    a.fitness.raw_completed_episodes.cmp(&b.fitness.raw_completed_episodes)
        .then(a.fitness.raw_spins.cmp(&b.fitness.raw_spins)).then(ap.min_spins.cmp(&bp.min_spins))
        .then(bp.completed_cost.cmp(&ap.completed_cost)).then(b.fitness.final_height.cmp(&a.fitness.final_height))
        .then(b.fitness.raw_pieces.cmp(&a.fitness.raw_pieces))
}

#[derive(Clone,Serialize,Deserialize)]
#[serde(rename_all="camelCase",deny_unknown_fields)]
pub(super) struct Route { id:String, generation:u32, genome:Genome, result:EpisodeResult, imported:bool,
    #[serde(default,skip_serializing_if="Option::is_none")]
    typed_plan_id:Option<String>,
}

fn typed_identity(result:&EpisodeResult)->Option<String>{
    if !result.raw_completed40{return None;}
    let mut bags=vec![];
    for (index,placements) in result.placements.chunks(7).enumerate(){
        let begin=index*7;
        let request=Request{start:if begin==0{Board::default()}else{Board{rows:result.frames[begin-1].rows}},
            start_spins:if begin==0{0}else{result.frames[begin-1].spins},start_pieces:begin as u16,
            placements:placements.to_vec(),assume_reflection:false};
        bags.push(request.typed_identity().ok()?);
    }
    Some(digest(serde_json::json!({"schema":"neyra-physical-typed-plan-v1","bags":bags}).to_string().as_bytes()))
}
impl Route {
    pub(super) fn inherited(&self)->(Genome,EpisodeResult){(self.genome.clone(),self.result.clone())}
    fn new(generation:u32,genome:&Genome,result:&EpisodeResult,imported:bool)->Self{
        let trace:Vec<_>=result.frames.iter().map(|f|(&f.piece,f.rows,f.placed_cells,f.spins)).collect();
        Self{id:format!("physical-{}",&digest(&serde_json::to_vec(&trace).unwrap())[..24]),generation,
            genome:genome.clone(),result:result.clone(),imported,typed_plan_id:typed_identity(result)}
    }
    pub fn snapshot(&self)->serde_json::Value{
        serde_json::json!({"id":self.id,"typedPlanId":self.typed_plan_id,"kind":"raw","modelId":self.genome.model_id,"generation":self.generation,
            "spins":self.result.raw_spins,"pieces":self.result.raw_pieces,"rawSpins":self.result.raw_spins,
            "rawPieces":self.result.raw_pieces,"rawCompleted40":self.result.raw_completed40,
            "qualifiedSpins":0,"qualifiedBags":0,"certified40":false,"strict100":false,"strictCertified40":false,
            "probability":null,"probabilityMeasured":false,"quietSafe":null,"frames":self.result.frames,
            "rows":self.result.rows,"representativePieces":self.result.raw_pieces,"inputReplayVerified":true,
            "completeHistory":true,"imported":self.imported,
            "continuationPrefixPieces":self.result.physical.as_ref().map_or(0,|p|p.continuation_prefix_pieces)})
    }
    fn audit(&self,config:&GaConfig,generation:u32,audited:&mut std::collections::HashSet<String>)->Result<(),String>{
        validate_weights(&self.genome.weights)?;
        if self.genome.weights.len()!=config.parameter_count() || self.generation>generation
            || self.genome.model_id!=Genome::new(self.genome.weights.clone()).model_id {
            return Err("Invalid physical archive genome/provenance".into());}
        // Bounds and physical replay precede every derived identity: an
        // untrusted frame list must never be indexed while canonicalizing.
        let hash=digest(&serde_json::to_vec(&self.result).unwrap());
        if audited.insert(hash){self.result.audit(config)?;}
        if self.id!=Self::new(self.generation,&self.genome,&self.result,self.imported).id{return Err("Physical archive identity mismatch".into());}
        if self.typed_plan_id!=typed_identity(&self.result) || (self.result.raw_completed40 && self.typed_plan_id.is_none()){
            return Err("Physical completion typed-plan identity mismatch".into());}
        Ok(())
    }
}

#[derive(Serialize,Deserialize)]
#[serde(rename_all="camelCase",deny_unknown_fields)]
struct Prefix { key:String, route:Route, visits:u32,
    #[serde(default,skip_serializing_if="is_zero_u32")]
    repair_visits:u32,
}
fn is_zero_u32(n:&u32)->bool{*n==0}
fn is_zero_u64(n:&u64)->bool{*n==0}
#[derive(Serialize,Deserialize)]
#[serde(rename_all="camelCase",deny_unknown_fields)]
pub(super) struct Continuation { pub episode:Episode, pub genome:Genome }

#[derive(Default,Serialize,Deserialize)]
#[serde(rename_all="camelCase",deny_unknown_fields)]
pub(super) struct Archive {
    pub raw:Option<Route>, prefixes:Vec<Prefix>, completions:Vec<Route>,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub best_from_empty:Option<Route>,
    pub continuation:Option<Continuation>,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub repair:Option<repair::Repair>,
    #[serde(default,skip_serializing_if="Option::is_none")]
    pub last_repair_generation:Option<u32>,
    #[serde(default,skip_serializing_if="is_zero_u64")]
    pub repair_attempts:u64,
    #[serde(default,skip_serializing_if="is_zero_u64")]
    pub repair_completed:u64,
    #[serde(default,skip_serializing_if="is_zero_u64")]
    pub repair40_count:u64,
    #[serde(default,skip_serializing_if="Vec::is_empty")]
    repair_attempt_ledger:Vec<String>,
    pub fresh_episodes:u64, pub from_empty40_count:u64, pub from_empty_spin_sum:u64,
    pub from_empty_min_spins:Option<u8>, pub from_empty_max_spins:u8,
    pub continuation_episodes:u64,pub continuation40_count:u64,pub imported_prefixes:u64,
    pub prefix_returns:u64,pub generation_raw_max:u8,pub stagnant_generations:u32,
    pub best_generation_sum:u32, pub repeated_behaviors:u64, behavior_ids:Vec<String>,
}

fn possible_spins(result:&EpisodeResult,max_pieces:u16)->u16{
    let mut used=0_u8;let begin=usize::from(result.raw_pieces/7)*7;
    let Some(suffix)=result.placements.get(begin..)else{return 0;};
    for p in suffix{used|=p.pose.piece.bit();}
    let mut available=u16::from(result.raw_spins);let mut left=max_pieces.saturating_sub(result.raw_pieces);
    while left>0 {let available_labels=7-used.count_ones() as u16;
        let non_o=available_labels-u16::from(used&Piece::O.bit()==0);
        available+=left.min(non_o);if left<=available_labels{break;}left-=available_labels;used=0;
    } available
}

impl Archive {
    pub fn needs_observation(&self,spins:u8,pieces:u16,from_empty:bool)->bool{
        pieces%7==0 || spins>=40 || self.raw.as_ref().is_none_or(|r|spins>r.result.raw_spins
            || (spins==r.result.raw_spins && pieces<r.result.raw_pieces))
            || (from_empty && self.best_from_empty.as_ref().is_none_or(|r|spins>r.result.raw_spins
                || (spins==r.result.raw_spins && pieces<r.result.raw_pieces)))
    }
    pub fn observe(&mut self,generation:u32,genome:&Genome,result:&EpisodeResult,config:&GaConfig,imported:bool){
        if !imported{self.generation_raw_max=self.generation_raw_max.max(result.raw_spins);}
        if result.raw_pieces>0 && self.raw.as_ref().is_none_or(|old|compare_result(result,&old.result).is_gt()){
            self.raw=Some(Route::new(generation,genome,result,imported));
        }
        if !imported && result.physical.as_ref().is_some_and(|p|p.continuation_prefix_pieces==0)
            && result.raw_pieces>0 && self.best_from_empty.as_ref().is_none_or(|old|compare_result(result,&old.result).is_gt()){
            self.best_from_empty=Some(Route::new(generation,genome,result,false));
        }
        if result.raw_completed40 {
            let route=Route::new(generation,genome,result,imported);
            if route.typed_plan_id.is_some() && !self.completions.iter().any(|r|r.typed_plan_id==route.typed_plan_id){self.completions.push(route);
                self.completions.sort_by_key(|r|r.result.raw_pieces);self.completions.truncate(32);}
        }
        // Returning to an irrecoverable horizon cannot produce a 40-spin
        // candidate. This is only the necessary one-clear/non-O lock bound.
        if result.raw_pieces==0 || result.raw_completed40 || possible_spins(result,config.max_pieces)<40{return;}
        if result.raw_pieces%7!=0{return;}
        let key=digest(serde_json::json!({"rows":result.rows,"spins":result.raw_spins,"pieces":result.raw_pieces}).to_string().as_bytes());
        if self.prefixes.iter().any(|p|p.key==key){return;}
        let cap=config.physical.as_ref().unwrap().archive_capacity;
        if self.prefixes.len()>=cap {
            // Preserve depth coverage before preferring productive, low-board
            // prefixes. Cell sharing is physical return-state reuse, never a
            // union of typed plans or supply-success sets.
            let mut counts=[0_usize;19];for p in &self.prefixes{counts[usize::from(p.route.result.raw_pieces/7)]+=1;}
            let new_depth=usize::from(result.raw_pieces/7);
            let remove=self.prefixes.iter().enumerate().filter(|(_,p)|{
                let old_depth=usize::from(p.route.result.raw_pieces/7);
                counts[old_depth]>1 || old_depth==new_depth
            })
                .min_by(|(_,a),(_,b)|{
                    let score=|p:&Prefix|f64::from(p.route.result.raw_spins)-0.10*f64::from(Board{rows:p.route.result.rows}.height())
                        -0.01*f64::from(p.visits.min(100));
                    score(a).total_cmp(&score(b))
                }).map(|(i,_)|i);
            let Some(remove)=remove else{return;};
            if counts[new_depth]>0 {
                let old=&self.prefixes[remove].route.result;
                if result.raw_spins<old.raw_spins && result.raw_pieces>=old.raw_pieces{return;}
            }
            self.prefixes.remove(remove);
        }
        self.prefixes.push(Prefix{key,route:Route::new(generation,genome,result,imported),visits:0,repair_visits:0});
        if imported{self.imported_prefixes+=1;}
    }

    pub fn record_completed(&mut self,result:&EpisodeResult,continuation:bool){
        if continuation{self.continuation_episodes+=1;self.continuation40_count+=u64::from(result.raw_completed40);}
        else{self.fresh_episodes+=1;self.from_empty40_count+=u64::from(result.raw_completed40);
            self.from_empty_spin_sum+=u64::from(result.raw_spins);
            self.from_empty_min_spins=Some(self.from_empty_min_spins.map_or(result.raw_spins,|n|n.min(result.raw_spins)));
            self.from_empty_max_spins=self.from_empty_max_spins.max(result.raw_spins);}
        let trace:Vec<_>=result.frames.iter().map(|f|(&f.piece,f.rows,f.placed_cells)).collect();
        let key=digest(&serde_json::to_vec(&trace).unwrap());
        if self.behavior_ids.contains(&key){self.repeated_behaviors+=1;}else{
            if self.behavior_ids.len()==4096{self.behavior_ids.remove(0);}self.behavior_ids.push(key);}
    }

    pub fn schedule(&mut self,generation:u32,config:&GaConfig,genome:&Genome,rng:&mut Random)->Result<(),String>{
        if let Some(width)=config.physical.as_ref().unwrap().repair_beam_width.filter(|&n|n>0){
            if self.last_repair_generation!=Some(generation) {
                let chosen=self.prefixes.iter().enumerate().filter(|(_,p)|p.repair_visits==0
                    && p.route.result.raw_spins>=30 && config.max_pieces.saturating_sub(p.route.result.raw_pieces)<=14
                    && !self.repair_attempt_ledger.contains(&repair::attempt_key(&p.key,&p.route.genome,width,config.max_pieces)))
                    .min_by(|(_,a),(_,b)|a.repair_visits.cmp(&b.repair_visits)
                        .then(b.route.result.raw_spins.cmp(&a.route.result.raw_spins))
                        .then(a.route.result.raw_pieces.cmp(&b.route.result.raw_pieces))).map(|(i,_)|i);
                if let Some(chosen)=chosen{
                    let prefix=&mut self.prefixes[chosen];prefix.repair_visits+=1;
                    if self.repair_attempt_ledger.len()==4096{self.repair_attempt_ledger.remove(0);}
                    self.repair_attempt_ledger.push(repair::attempt_key(&prefix.key,&prefix.route.genome,width,config.max_pieces));
                    self.repair=Some(repair::Repair::new(prefix.route.result.clone(),prefix.route.genome.clone(),width,rng.next()));
                    self.repair_attempts+=1;self.last_repair_generation=Some(generation);return Ok(());
                }
            }
        }
        let every=config.physical.as_ref().unwrap().continuation_every;
        if every==0 || self.fresh_episodes%every as u64!=0 || self.prefixes.is_empty(){return Ok(());}
        let mut chosen=rng.index(self.prefixes.len());
        // A mixture of broad return states and a small tournament avoids
        // replaying only the latest/highest path with no remaining budget.
        if rng.unit()<0.75 {for _ in 0..3{let other=rng.index(self.prefixes.len());
            let score=|p:&Prefix|f64::from(p.route.result.raw_spins)*7.0/f64::from(p.route.result.raw_pieces.max(7))
                +0.018*f64::from(p.route.result.raw_spins)-0.04*f64::from(p.visits.min(20));
            if score(&self.prefixes[other])>score(&self.prefixes[chosen]){chosen=other;}}}
        let prefix=&mut self.prefixes[chosen];prefix.visits+=1;
        let mut chosen_genome=genome.clone();
        if rng.unit()<0.5{chosen_genome=prefix.route.genome.clone();}
        mutate(&mut chosen_genome.weights,rng,config,0.5);
        chosen_genome=Genome::new(chosen_genome.weights);
        let episode=Episode::from_physical_prefix(rng.next(),config,&prefix.route.result)?;
        self.continuation=Some(Continuation{episode,genome:chosen_genome});self.prefix_returns+=1;Ok(())
    }

    pub fn import(&mut self,config:&GaConfig,genome:&Genome,result:&EpisodeResult)->Result<(),String>{
        let mut episode=Episode::for_config(result.seed,config);
        for placement in &result.placements {
            if episode.raw_pieces>=config.max_pieces || episode.raw_spins>=40{break;}
            if episode.used==ALL_PIECES{episode.next_bag_or_finish(config);}
            episode.commit_placement(placement.clone(),&mut Counters::default())?;
            if episode.used==ALL_PIECES || episode.raw_spins>=40{self.observe(0,genome,&episode.observed_result(),config,true);}
        }
        self.observe(0,genome,&episode.observed_result(),config,true);Ok(())
    }

    pub fn snapshot(&self,counters:&Counters)->serde_json::Value{
        serde_json::json!({"freshEpisodes":self.fresh_episodes,"fromEmpty40Count":self.from_empty40_count,
            "fromEmptyMeanSpins":if self.fresh_episodes>0{Some(self.from_empty_spin_sum as f64/self.fresh_episodes as f64)}else{None},
            "fromEmptyMinSpins":self.from_empty_min_spins,"fromEmptyMaxSpins":self.from_empty_max_spins,
            "fromEmpty40Rate":if self.fresh_episodes>0{Some(self.from_empty40_count as f64/self.fresh_episodes as f64)}else{None},
            "continuationEpisodes":self.continuation_episodes,"continuation40Count":self.continuation40_count,
            "unique40Candidates":self.completions.len(),"retained40Candidates":self.completions.len(),
            "unique40CandidatesScope":"At most32 distinct retained typed fixed plans, including imported records; not an all-history total or independent games.",
            "retained40CandidatesUnit":"typed-fixed-plan",
            "retainedImported40Candidates":self.completions.iter().filter(|r|r.imported).count(),"retainedPrefixCount":self.prefixes.len(),
            "repairAttempts":self.repair_attempts,"repairCompleted":self.repair_completed,"repair40Count":self.repair40_count,
            "repairAttemptLedgerEntries":self.repair_attempt_ledger.len(),"repairAttemptLedgerCapacity":4096,
            "repairProgress":self.repair.as_ref().map(repair::Repair::snapshot),
            "importedPrefixes":self.imported_prefixes,"prefixReturns":self.prefix_returns,
            "repeatedBehaviors":self.repeated_behaviors,"uniqueBehaviors":self.behavior_ids.len(),
            "behaviorLedgerCapacity":4096,"stagnantGenerations":self.stagnant_generations,
            "reusedGenomeEvaluations":counters.cache_hits,"probabilityMeasured":false,
            "completionMeaning":"Replayed physical 40-spin candidate; random-supply construction remains unmeasured."})
    }

    pub fn audit(&self,config:&GaConfig,generation:u32,audited:&mut std::collections::HashSet<String>)->Result<(),String>{
        if self.prefixes.len()>config.physical.as_ref().unwrap().archive_capacity || self.completions.len()>32
            || self.repair_attempt_ledger.len()>4096
            || self.behavior_ids.len()>4096 || self.from_empty40_count>self.fresh_episodes
            || self.continuation40_count>self.continuation_episodes || self.from_empty_spin_sum>self.fresh_episodes.saturating_mul(40)
            || self.from_empty_max_spins>40 || self.generation_raw_max>40 {
            return Err("Invalid physical GA archive bounds".into());}
        let mut attempts=BTreeSet::new();
        if self.repair_attempt_ledger.iter().any(|id|id.len()!=64 || !id.bytes().all(|b|b.is_ascii_hexdigit()) || !attempts.insert(id)){
            return Err("Invalid physical repair attempt ledger".into());}
        if self.prefix_returns!=self.continuation_episodes.saturating_add(u64::from(self.continuation.is_some()))
            || (self.continuation.is_some() && self.repair.is_some())
            || self.repair_attempts!=self.repair_completed.saturating_add(u64::from(self.repair.is_some()))
            || self.repair40_count>self.repair_completed || self.last_repair_generation.is_some_and(|g|g>generation)
            || self.from_empty_min_spins.is_some()!=(self.fresh_episodes>0)
            || self.from_empty_min_spins.is_some_and(|m|m>self.from_empty_max_spins
                || u64::from(m).saturating_mul(self.fresh_episodes)>self.from_empty_spin_sum)
            || self.from_empty_spin_sum>u64::from(self.from_empty_max_spins).saturating_mul(self.fresh_episodes){
            return Err("Invalid physical GA fresh/continuation totals".into());}
        let mut keys=BTreeSet::new();
        for p in &self.prefixes{
            p.route.audit(config,generation,audited)?;
            let r=&p.route.result;
            let expected=digest(serde_json::json!({"rows":r.rows,"spins":r.raw_spins,"pieces":r.raw_pieces}).to_string().as_bytes());
            if p.key!=expected || !keys.insert(&p.key) || r.raw_pieces%7!=0 || r.raw_completed40
                || possible_spins(r,config.max_pieces)<40{return Err("Invalid physical return prefix".into());}
        }
        for route in self.raw.iter().chain(self.completions.iter()).chain(self.best_from_empty.iter()){route.audit(config,generation,audited)?;}
        if self.best_from_empty.as_ref().is_some_and(|r|r.imported || r.result.physical.as_ref().is_none_or(|p|p.continuation_prefix_pieces!=0)){
            return Err("From-empty archive contains a reused or imported trajectory".into());}
        if self.completions.iter().any(|r|!r.result.raw_completed40){return Err("Invalid physical completion archive".into());}
        let mut plans=BTreeSet::new();if self.completions.iter().any(|r|!plans.insert(r.typed_plan_id.as_ref())){
            return Err("Duplicate physical completion typed plan".into());}
        if let Some(c)=&self.continuation{validate_weights(&c.genome.weights)?;
            if c.genome.model_id!=Genome::new(c.genome.weights.clone()).model_id || c.genome.weights.len()!=config.parameter_count()
                || c.episode.physical.as_ref().is_none_or(|p|p.continuation_prefix_pieces==0){return Err("Invalid physical continuation state".into());}
            c.episode.validate(config)?;c.episode.observed_result().audit(config)?;
        }
        if let Some(repair)=&self.repair {repair.audit(config)?;}
        Ok(())
    }
}

fn mutate(weights:&mut [f64],rng:&mut Random,config:&GaConfig,multiplier:f64){
    // Keep a comparable coordinate mutation budget when expanding24→256.
    // Unit-valued local/context terms receive a smaller residual step.
    if config.mutation_rate==0.0{return;}
    let expected=FEATURE_COUNT as f64*config.mutation_rate;
    let count=expected.floor() as usize+usize::from(rng.unit()<expected.fract());
    for _ in 0..count{let i=rng.index(FEATURE_COUNT);
        weights[i]=(weights[i]+rng.normal()*config.mutation_scale*multiplier).clamp(-16.0,16.0);}
    if weights.len()>FEATURE_COUNT{for _ in 0..count{let i=FEATURE_COUNT+rng.index(weights.len()-FEATURE_COUNT);
        weights[i]=(weights[i]+rng.normal()*config.mutation_scale*multiplier*0.20).clamp(-16.0,16.0);}}
}

pub(super) fn finish_generation(state:&mut State){
    let mut ranked:Vec<_>=(0..state.population.len()).collect();
    ranked.sort_by(|&a,&b|compare_evaluation(state.evaluations[b].as_ref().unwrap(),state.evaluations[a].as_ref().unwrap()).then(a.cmp(&b)));
    let top=state.evaluations[ranked[0]].as_ref().unwrap();let archive=state.physical.as_mut().unwrap();
    let episodes=state.evaluations.iter().flatten().map(|e|e.seeds.len()).sum::<usize>();
    let spins=state.evaluations.iter().flatten().map(|e|e.fitness.raw_spins).sum::<u32>();
    let completed=state.evaluations.iter().flatten().map(|e|e.fitness.raw_completed_episodes).sum::<u32>();
    state.history.push(GenerationSummary{generation:state.generation,best_fitness:top.fitness.clone(),best_qualified_spins:0,
        best_raw_spins:top.best.as_ref().map_or(0,|r|r.raw_spins),certified40_count:0,evaluated_genomes:state.population.len(),
        generation_raw_max:Some(archive.generation_raw_max),all_time_raw_max:Some(archive.raw.as_ref().map_or(0,|r|r.result.raw_spins)),
        strict_spins:None,frontier_spins:None,frontier_probability:None,unique_behaviors:None,
        physical40_count:Some(completed),physical_mean_spins:Some(spins as f64/episodes as f64),
        physical_min_spins:state.evaluations.iter().flatten().filter_map(|e|e.physical.as_ref().and_then(|p|p.min_spins)).min(),
        physical_evaluated_episodes:Some(episodes)});
    if state.history.len()>256{state.history.remove(0);}
    if top.fitness.raw_spins>archive.best_generation_sum{archive.best_generation_sum=top.fitness.raw_spins;archive.stagnant_generations=0;}
    else{archive.stagnant_generations+=1;}archive.generation_raw_max=0;
    state.completed_generations+=1;if state.completed_generations>=state.config.generations{state.complete=true;return;}
    let stagnant=archive.stagnant_generations;
    let immigrant_count=(state.population.len()*if stagnant>=8{25}else{10}).div_ceil(100).max(1);
    let mut next:Vec<_>=ranked.iter().take(state.config.elite_count).map(|&i|state.population[i].clone()).collect();
    // Return-search weights earn a fresh from-empty evaluation before their
    // score can influence reproduction. A reused prefix never supplies that
    // fitness. Bounded injection closes the candidate→policy feedback loop.
    if let Some(raw)=&archive.raw {if next.len()<state.population.len()-immigrant_count
        && !next.iter().any(|g|g.model_id==raw.genome.model_id){next.push(raw.genome.clone());}}
    if next.len()<state.population.len()-immigrant_count && !archive.prefixes.is_empty(){
        let candidate=&archive.prefixes[state.rng.index(archive.prefixes.len())].route.genome;
        if !next.iter().any(|g|g.model_id==candidate.model_id){next.push(candidate.clone());}
    }
    while next.len()<state.population.len()-immigrant_count{
        let select=|rng:&mut Random|{let mut pos=rng.index(ranked.len());for _ in 1..state.config.tournament_size{pos=pos.min(rng.index(ranked.len()));}ranked[pos]};
        let a=select(&mut state.rng);let mut weights=state.population[a].weights.clone();
        // Preserve useful parameter interactions in most offspring. Uniform
        // every-coordinate blending flattened independently discovered traits.
        if state.rng.unit()<0.2{let b=select(&mut state.rng);let alpha=state.rng.unit();
            for(i,w)in weights.iter_mut().enumerate(){*w=alpha**w+(1.0-alpha)*state.population[b].weights[i];}}
        let scale=match state.rng.index(4){0=>0.25,1|2=>1.0,_=>if stagnant>=8{4.0}else{2.0}};
        mutate(&mut weights,&mut state.rng,&state.config,scale);next.push(Genome::new(weights));
    }
    while next.len()<state.population.len(){let mut weights=if state.rng.unit()<0.5{state.population[ranked[0]].weights.clone()}
        else{policy::initializer(state.config.parameter_count()>FEATURE_COUNT)};
        for _ in 0..4{mutate(&mut weights,&mut state.rng,&state.config,if stagnant>=8{8.0}else{4.0});}
        next.push(Genome::new(weights));}
    state.population=next;state.evaluations=vec![None;state.config.population_size];state.generation+=1;state.active_genome_index=0;
}
