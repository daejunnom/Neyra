//! A small, resumable physical completion search near a promising endpoint.
//! Every expansion enumerates the canonical reachable locks. Beam capacity
//! and the query cap are selective limits, never probability/impossibility
//! proofs. Exact inputs are replayed before returning a candidate.
use super::*;
use std::collections::{BTreeMap,BTreeSet};

const QUERY_CAP:u64=20_000;

pub(super) fn attempt_key(prefix_key:&str,genome:&Genome,width:usize,max_pieces:u16)->String{
    digest(serde_json::json!({"schema":"neyra-physical-repair-attempt-v1","prefix":prefix_key,
        "modelId":genome.model_id,"width":width,"maxPieces":max_pieces,"queryCap":QUERY_CAP}).to_string().as_bytes())
}

#[derive(Clone,Serialize,Deserialize)]
#[serde(rename_all="camelCase",deny_unknown_fields)]
struct Node { board:Board,used:u8,spins:u8,tail:Vec<Placement>,score:f64 }

#[derive(Serialize,Deserialize)]
#[serde(rename_all="camelCase",deny_unknown_fields)]
pub(super) struct Repair {
    prefix:EpisodeResult,
    pub genome:Genome,
    seed:u32,width:usize,
    beam:Vec<Node>,next:Vec<Node>,best:Node,
    node_index:usize,piece_index:usize,placed:u16,
    queries:u64,expanded:u64,
}

fn upper_spins(spins:u8,used:u8,placed:u16,limit:u16)->u16{
    let(mut mask,mut left,mut possible)=(if placed%7==0{0}else{used},limit.saturating_sub(placed),u16::from(spins));
    while left>0{let labels=7-mask.count_ones() as u16;let non_o=labels-u16::from(mask&Piece::O.bit()==0);
        possible+=left.min(non_o);if left<=labels{break;}left-=labels;mask=0;}
    possible
}

fn key(node:&Node)->([u16;HEIGHT],u8,u8){(node.board.rows,node.used,node.spins)}

fn retain(nodes:&mut Vec<Node>,capacity:usize){
    nodes.sort_by(|a,b|b.score.total_cmp(&a.score).then(a.board.rows.cmp(&b.board.rows)).then(a.used.cmp(&b.used)));
    let mut states=BTreeSet::new();nodes.retain(|node|states.insert(key(node)));
    if nodes.len()<=capacity{return;}
    // Protect each currently represented inventory/spin class before filling
    // by learned ordering. Distinct inventories are not interchangeable.
    let(mut chosen,mut groups,mut selected)=(vec![],BTreeSet::new(),BTreeSet::new());
    for(index,node)in nodes.iter().enumerate(){if chosen.len()==capacity{break;}
        if groups.insert((node.used,node.spins)){chosen.push(node.clone());selected.insert(index);}}
    for(index,node)in nodes.iter().enumerate(){if chosen.len()==capacity{break;}
        if !selected.contains(&index){chosen.push(node.clone());}}
    *nodes=chosen;
}

impl Repair {
    pub fn new(prefix:EpisodeResult,genome:Genome,width:usize,seed:u32)->Self{
        let root=Node{board:Board{rows:prefix.rows},used:0,spins:prefix.raw_spins,tail:vec![],score:0.0};
        let placed=prefix.raw_pieces;
        Self{prefix,genome,seed,width,beam:vec![root.clone()],next:vec![],best:root,
            node_index:0,piece_index:0,placed,queries:0,expanded:0}
    }

    fn result(&self,node:&Node,config:&GaConfig,counters:&mut Counters)->Result<EpisodeResult,String>{
        if usize::from(self.prefix.raw_pieces)+node.tail.len()>usize::from(config.max_pieces){
            return Err("Physical repair exceeds the configured lock cap".into());
        }
        let mut episode=Episode::from_physical_prefix(self.seed,config,&self.prefix)?;
        counters.replay_queries+=u64::from(self.prefix.raw_pieces);
        for placement in &node.tail{
            if episode.used==ALL_PIECES{episode.next_bag_or_finish(config);}
            if episode.used&placement.pose.piece.bit()!=0{return Err("Repair repeats a typed bag piece".into());}
            episode.commit_placement(placement.clone(),counters)?;
        }
        if episode.board!=node.board || episode.raw_spins!=node.spins{return Err("Repair input replay mismatch".into());}
        episode.physical.as_mut().unwrap().repaired=true;
        if episode.raw_spins>=40 || episode.raw_pieces>=config.max_pieces{episode.next_bag_or_finish(config);}
        let mut result=episode.observed_result();
        if !episode.done{result.termination=if self.queries>=QUERY_CAP{"repair-query-cap"}else{"repair-beam-exhausted"}.into();}
        Ok(result)
    }

    /// One full piece reachability query, or one bounded layer transition.
    /// Cursors and both bounded frontiers are in the checkpoint.
    pub fn advance(&mut self,config:&GaConfig,counters:&mut Counters)->Result<Option<EpisodeResult>,String>{
        counters.work_units+=1;
        if self.queries>=QUERY_CAP || self.placed>=config.max_pieces || self.beam.is_empty(){
            return self.result(&self.best,config,counters).map(Some);
        }
        if self.node_index>=self.beam.len(){
            retain(&mut self.next,self.width);
            self.beam=std::mem::take(&mut self.next);self.placed+=1;self.node_index=0;self.piece_index=0;
            return Ok(None);
        }
        let node=&self.beam[self.node_index];
        let used=if self.placed%7==0{0}else{node.used};
        while self.piece_index<7 && used&Piece::ALL[self.piece_index].bit()!=0{self.piece_index+=1;}
        if self.piece_index==7{self.node_index+=1;self.piece_index=0;return Ok(None);}
        let piece=Piece::ALL[self.piece_index];self.piece_index+=1;
        let locks=node.board.reachable_locks(piece);self.queries+=1;counters.physics_queries+=1;
        counters.candidate_locks+=locks.len() as u64;
        let mut positions:BTreeMap<_,_>=self.next.iter().enumerate().map(|(index,node)|(key(node),index)).collect();
        for placement in locks {
            let Some(lock)=node.board.apply(&placement)else{continue;};
            if lock.lines>1 || (lock.lines==1 && (!lock.spin || piece==Piece::O)){continue;}
            counters.legal_candidate_locks+=1;
            let spins=node.spins+lock.lines;let new_used=used|piece.bit();
            if upper_spins(spins,new_used,self.placed+1,config.max_pieces)<40{continue;}
            let score=f64::from(spins)*100.0+policy::score(&self.genome.weights,&node.board,&lock.board,&placement,
                lock.lines,used,node.spins,self.placed,config.max_pieces);
            let state_key=(lock.board.rows,new_used,spins);
            if positions.get(&state_key).is_some_and(|&index|self.next[index].score>=score){continue;}
            let mut tail=node.tail.clone();tail.push(placement);
            let candidate=Node{board:lock.board,used:new_used,spins,tail,score};self.expanded+=1;
            if candidate.spins>self.best.spins || (candidate.spins==self.best.spins
                && (candidate.tail.len()<self.best.tail.len() || (candidate.tail.len()==self.best.tail.len() && candidate.score>self.best.score))){
                self.best=candidate.clone();
            }
            if spins>=40{return self.result(&candidate,config,counters).map(Some);}
            if let Some(&index)=positions.get(&state_key){self.next[index]=candidate;}
            else{positions.insert(state_key,self.next.len());self.next.push(candidate);}
        }
        // Incremental bounded retention yields the same final group winners
        // and top candidates when all groups fit; it is still a finite beam.
        retain(&mut self.next,self.width*8);
        Ok(None)
    }

    pub fn snapshot(&self)->serde_json::Value{
        serde_json::json!({"prefixPieces":self.prefix.raw_pieces,"prefixSpins":self.prefix.raw_spins,
            "placed":self.placed,"bestSpins":self.best.spins,"beamWidth":self.width,
            "physicsQueries":self.queries,"queryCap":QUERY_CAP,"expandedBoards":self.expanded,
            "retainedNodes":self.beam.len(),"pendingNodes":self.next.len(),"probabilityMeasured":false})
    }
    pub fn current_snapshot(&self)->serde_json::Value{
        serde_json::json!({"rawSpins":self.best.spins,"rawPieces":usize::from(self.prefix.raw_pieces)+self.best.tail.len(),
            "qualifiedSpins":0,"qualifiedBags":0,"rows":self.best.board.rows,"phase":"repairing-physical-endgame",
            "checked":0,"qualifiedPrefixLive":false,"continuationPrefixPieces":self.prefix.raw_pieces})
    }

    pub fn audit(&self,config:&GaConfig)->Result<(),String>{
        self.prefix.audit(config)?;validate_weights(&self.genome.weights)?;
        if self.genome.weights.len()!=config.parameter_count() || self.genome.model_id!=Genome::new(self.genome.weights.clone()).model_id
            || self.width==0 || self.width>512 || Some(self.width)!=config.physical.as_ref().and_then(|p|p.repair_beam_width)
            || self.prefix.raw_spins<30 || self.prefix.raw_spins>=40 || self.prefix.raw_pieces%7!=0
            || self.prefix.raw_pieces>=config.max_pieces || config.max_pieces-self.prefix.raw_pieces>14
            || self.placed<self.prefix.raw_pieces || self.placed>config.max_pieces
            || self.beam.len()>self.width || self.next.len()>self.width*8
            || self.node_index>self.beam.len() || self.piece_index>7 || self.queries>QUERY_CAP {
            return Err("Invalid bounded physical repair state".into());
        }
        let validate_node=|node:&Node,expected:Option<usize>|->Result<(),String>{
            if !node.score.is_finite() || node.score.abs()>20_000.0 || node.tail.len()>14
                || usize::from(self.prefix.raw_pieces)+node.tail.len()>usize::from(config.max_pieces)
                || expected.is_some_and(|n|node.tail.len()!=n){return Err("Invalid physical repair node bounds".into());}
            let(mut board,mut used,mut spins)=(Board{rows:self.prefix.rows},0_u8,self.prefix.raw_spins);
            for(i,p)in node.tail.iter().enumerate(){
                if (usize::from(self.prefix.raw_pieces)+i)%7==0{used=0;}
                if used&p.pose.piece.bit()!=0 || spins>=40{return Err("Invalid physical repair inventory".into());}
                let replay=board.replay(p.pose.piece,&p.path).ok_or("Invalid physical repair input")?;
                if replay.pose!=p.pose || replay.spin!=p.spin{return Err("Physical repair pose mismatch".into());}
                let lock=board.apply(&replay).ok_or("Invalid physical repair lock")?;
                if lock.lines>1 || (lock.lines==1 && (!lock.spin || p.pose.piece==Piece::O)){return Err("Invalid physical repair clear".into());}
                used|=p.pose.piece.bit();spins+=lock.lines;board=lock.board;
            }
            if node.board!=board || node.used!=used || node.spins!=spins{return Err("Physical repair node replay mismatch".into());}
            Ok(())
        };
        let depth=usize::from(self.placed-self.prefix.raw_pieces);
        for node in &self.beam{validate_node(node,Some(depth))?;}
        for node in &self.next{validate_node(node,Some(depth+1))?;}
        validate_node(&self.best,None)?;Ok(())
    }
}
