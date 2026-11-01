//! Incremental adapter to the ordinary search's exact fixed-plan verifier.
//! GA has its own proposal policy; it does not have a second supply policy.

use super::*;
use serde::de::Error as _;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct Request {
    pub start: Board,
    pub start_spins: u8,
    pub start_pieces: u16,
    pub placements: Vec<Placement>,
    pub assume_reflection: bool,
}

impl Request {
    fn plan(&self) -> Result<Plan, String> {
        if !self.start.is_valid() || self.start_spins >= TARGET_SPINS
            || self.start_pieces % 7 != 0 || self.placements.is_empty()
            || self.placements.len() > 7 {
            return Err("Invalid fixed-bag replay request".into());
        }
        let (mut board, mut used, mut deleted, mut spins) =
            (self.start, 0_u8, 0_u32, self.start_spins);
        let (mut targets, mut frames) = (vec![], vec![]);
        for placement in &self.placements {
            if used & placement.pose.piece.bit() != 0 || spins >= TARGET_SPINS {
                return Err("A fixed bag cannot repeat a piece or continue past 40".into());
            }
            let replayed = board.replay(placement.pose.piece, &placement.path)
                .ok_or("Fixed-bag input replay failed")?;
            if replayed.pose != placement.pose || replayed.spin != placement.spin {
                return Err("Fixed-bag replay does not match its recorded placement".into());
            }
            let result = board.apply(&replayed).ok_or("Invalid fixed-bag lock")?;
            if result.lines > 1 || (result.lines == 1 && (!result.spin || placement.pose.piece == Piece::O)) {
                return Err("Only non-O Spin Singles and quiet locks are allowed".into());
            }
            targets.push(normalize_target(placement.pose, deleted)
                .ok_or("Logical target row overflow")?);
            deleted = deleted_after(deleted, result.cleared_rows).ok_or("Logical row overflow")?;
            used |= placement.pose.piece.bit();
            board = result.board;
            spins += result.lines;
            frames.push(DisplayFrame {
                rows: board.rows, piece: piece_name(placement.pose.piece),
                placed_cells: placement.pose.cells().map(|(x, y)| (x as u8, y as u8)),
                spins, placed: self.start_pieces + frames.len() as u16 + 1,
                lines: result.lines, spin: result.spin,
            });
        }
        if spins < TARGET_SPINS && used != ALL_PIECES {
            return Err("A nonterminal fixed-bag plan must use all seven pieces".into());
        }
        Ok(Plan {
            id: 0, start: self.start, start_spins: self.start_spins,
            start_pieces: self.start_pieces, expected: board, expected_spins: spins,
            targets, seed_frames: frames, terminal: spins >= TARGET_SPINS,
            assume_reflection: self.assume_reflection, quality: 0.0,
            pace_fallback: false, future: Continuation::default(), debt: [0; HEIGHT],
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Proof {
    pub bag_index: u16,
    pub status: String,
    pub qualified: bool,
    pub terminal: bool,
    pub start_rows: [u16; HEIGHT],
    pub end_rows: [u16; HEIGHT],
    pub start_spins: u8,
    pub end_spins: u8,
    pub targets: Vec<TargetOperation>,
    pub checked: u16,
    pub canonical_checked: u16,
    pub canonical_success: u16,
    pub success: u16,
    pub success_upper: u16,
    pub quiet_safe_supplies: u16,
    pub canonical_quiet_safe: u16,
    pub quiet_no_hold_verified: bool,
    pub first_bag_symmetry_assumed: bool,
    pub mirrored_inputs_verified: bool,
    pub hold_policy: String,
    pub probability: f64,
    pub first_bag_probability: Option<f64>,
    pub budget_pieces: Option<u16>,
    pub cdf: Vec<CdfPoint>,
    pub graph_nodes: usize,
    #[serde(serialize_with = "rle::serialize", deserialize_with = "rle::deserialize")]
    pub canonical_costs: Vec<u8>,
    #[serde(serialize_with = "rle::serialize", deserialize_with = "rle::deserialize")]
    pub effective_costs: Vec<u8>,
    #[serde(serialize_with = "rle::serialize", deserialize_with = "rle::deserialize")]
    pub order_codes: Vec<u32>,
    pub representative_queue: Option<[u8; 7]>,
    pub representative_inputs: Vec<Vec<Input>>,
    pub representative_frames: Vec<DisplayFrame>,
}

/// Lossless storage only. Old plain arrays remain readable, and arrays with
/// too many short runs remain plain. Evaluation always uses the same Vecs.
mod rle {
    use serde::{Deserialize, Serialize};
    use serde::de::Error as _;

    pub fn serialize<T: Copy + Eq + Serialize, S: serde::Serializer>(values: &[T], serializer: S) -> Result<S::Ok, S::Error> {
        let mut runs = Vec::<(u32, T)>::new();
        for &value in values {
            if let Some((count, previous)) = runs.last_mut() {
                if *previous == value { *count += 1; continue; }
            }
            runs.push((1, value));
        }
        #[derive(Serialize)]
        struct Encoded<'a, T> { encoding: &'static str, runs: &'a [(u32, T)] }
        let encoded = Encoded { encoding: "rle-v1", runs: &runs };
        let encoded_size = serde_json::to_vec(&encoded).map_err(serde::ser::Error::custom)?.len();
        let plain_size = serde_json::to_vec(values).map_err(serde::ser::Error::custom)?.len();
        if encoded_size < plain_size { encoded.serialize(serializer) } else { values.serialize(serializer) }
    }

    pub fn deserialize<'de, T: Copy + Deserialize<'de>, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Vec<T>, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Encoded<T> { encoding: String, runs: Vec<(u32, T)> }
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire<T> { Plain(Vec<T>), Encoded(Encoded<T>) }
        let values = match Wire::<T>::deserialize(deserializer)? {
            Wire::Plain(values) => values,
            Wire::Encoded(encoded) => {
                if encoded.encoding != "rle-v1" || encoded.runs.len() > crate::supply::P7_COUNT {
                    return Err(D::Error::custom("Invalid fixed-plan proof encoding"));
                }
                let mut values = Vec::new();
                for (count, value) in encoded.runs {
                    if count == 0 || count as usize > crate::supply::P7_COUNT - values.len() {
                        return Err(D::Error::custom("Invalid fixed-plan proof run length"));
                    }
                    values.extend(std::iter::repeat_n(value, count as usize));
                }
                values
            }
        };
        if values.len() > crate::supply::P7_COUNT { return Err(D::Error::custom("Fixed-plan proof array exceeds 5040")); }
        Ok(values)
    }
}

/// Includes every graph node, resolved orbit and target-query cache entry.
/// Restoring performs no new search and consumes no RNG or work allowance.
pub(crate) struct Verifier {
    request: Request,
    job: VerificationJob,
    cache: PhysicsCache,
}

impl Verifier {
    pub fn new(request: Request) -> Result<Self, String> {
        let plan = request.plan()?;
        Ok(Self { request, job: VerificationJob::new(0, plan), cache: PhysicsCache::default() })
    }

    pub fn advance(&mut self) -> u64 {
        if self.finished() { return 0; }
        let before = self.job.canonical_checked;
        // The public search may reject a known quiet-policy failure early.
        // GA requires a measured gradient, so it completes this exact same
        // execution graph and supply language even after a quiet failure.
        // No failure upper bound is substituted for an observed count.
        let job = &mut self.job;
        if !job.graph.complete() {
            job.graph.expand_one(&job.plan, TARGET_SPINS, &mut self.cache);
            return 0;
        }
        if job.language.is_none() {
            job.minimum_cost = job.graph.nodes.iter().filter(|n| n.accepting)
                .map(|n| n.key.used.count_ones() as u8).min().unwrap_or(8);
            job.language = Some(job.graph.clearing_language());
        }
        if job.symmetry.is_some() {
            let mut queries = 0;
            while queries < 63 && job.checked < P7_COUNT {
                let pair = job.symmetry.as_mut().unwrap().next_pair().unwrap();
                let first = job.check_canonical(pair[0]); queries += 1;
                let (cost, quiet) = if first.0 == job.minimum_cost && first.1 { first } else {
                    let second = job.check_canonical(pair[1]); queries += 1;
                    (first.0.min(second.0), first.1 || second.1)
                };
                let symmetry = job.symmetry.as_mut().unwrap();
                symmetry.resolve(pair, cost); job.checked = usize::from(symmetry.checked());
                if quiet { job.quiet_safe += 2; } else { job.quiet_rejected = true; }
            }
        } else {
            let end = (job.checked + 64).min(P7_COUNT);
            for rank in job.checked..end {
                let (_, quiet) = job.check_canonical(rank); job.checked += 1;
                if quiet { job.quiet_safe += 1; } else { job.quiet_rejected = true; }
            }
        }
        job.upper = (job.success + P7_COUNT - job.canonical_checked) as u16;
        (self.job.canonical_checked - before) as u64
    }

    pub fn finished(&self) -> bool {
        self.job.graph.overflow || self.job.checked == P7_COUNT
    }

    pub fn checked(&self) -> usize { self.job.checked }

    pub fn matches_request(&self, request: &Request) -> bool {
        // Request contains only bounded integer board/input data.
        serde_json::to_value(&self.request).ok() == serde_json::to_value(request).ok()
    }

    pub fn proof(&self, config: &SearchConfig, previous_counts: &[u16]) -> Result<Proof, String> {
        if !self.finished() { return Err("Fixed-plan verification is still pending".into()); }
        // These methods are the public search's canonical thresholds and exact
        // integer-product policy. No GA-specific probability interpretation.
        let policy = SearchSession::new(config.clone())?;
        let job = &self.job;
        let resolved = !job.graph.overflow && job.checked == P7_COUNT;
        let complete = resolved && !job.quiet_rejected;
        let symmetry_costs = job.symmetry.as_ref().map(|orbits| orbits.costs().to_vec());
        let effective = symmetry_costs.as_deref().unwrap_or(&job.costs);
        let symmetry_assumed = job.plan.start_pieces > 0 && job.plan.assume_reflection
            || symmetry_costs.as_ref().is_some_and(|costs| *costs != job.costs)
            || (job.symmetry.is_some() && job.quiet_safe > job.canonical_quiet_safe);
        let mut counts = previous_counts.to_vec();
        counts.push(job.coverage_success());
        let mut cdf = vec![];
        let mut budget = None;
        if complete && job.plan.terminal {
            for m in 1..=7 {
                let count = effective.iter().filter(|c| **c > 0 && **c <= m).count() as u16;
                let mut at_budget = previous_counts.to_vec();
                at_budget.push(count);
                let conditional_start = config.ignore_first_bag_probability && job.plan.start_pieces == 0;
                let p = if conditional_start {
                    f64::from(count) / f64::from(job.coverage_success().max(1))
                } else { policy.path_probability(&at_budget) };
                let meets = if conditional_start {
                    job.coverage_success() > 0 && u32::from(count) * 10_000
                        >= u32::from(job.coverage_success()) * (config.min_success_percent * 100.0).round() as u32
                } else { policy.plan_path_qualifies(&at_budget, &job.plan) };
                cdf.push(CdfPoint { pieces: job.plan.start_pieces + m as u16, probability: p });
                if budget.is_none() && meets { budget = Some(job.plan.start_pieces + m as u16); }
            }
        }
        let qualified = complete && job.quiet_safe == P7_COUNT && if job.plan.terminal {
            policy.start_qualifies(&counts) && budget.is_some_and(|n| n <= config.max_pieces)
        } else { policy.plan_path_qualifies(&counts, &job.plan) };
        let desired = budget.map(|n| n - job.plan.start_pieces).unwrap_or(7) as u8;
        let canonical_choice = |rank: usize| job.symmetry.is_none()
            || (job.costs[rank] > 0 && job.costs[rank] == effective[rank]);
        let representative = complete.then(|| job.costs.iter().enumerate()
            .position(|(rank, c)| *c == desired && canonical_choice(rank))
            .or_else(|| job.costs.iter().enumerate()
                .position(|(rank, c)| *c > 0 && *c <= desired && canonical_choice(rank))))
            .flatten();
        let (frames, inputs) = representative.map(|rank| job.graph.witness(
            job.codes[rank], job.costs[rank], job.plan.start_pieces)).unwrap_or_default();
        Ok(Proof {
            bag_index: job.plan.start_pieces / 7 + 1,
            status: if job.graph.overflow { "graph-capacity" } else if job.quiet_rejected {
                "quiet-policy-rejected"
            } else { "complete" }.into(),
            qualified, terminal: job.plan.terminal,
            start_rows: job.plan.start.rows,
            end_rows: frames.last().map_or(job.plan.expected.rows, |f| f.rows),
            start_spins: job.plan.start_spins,
            end_spins: frames.last().map_or(job.plan.expected_spins, |f| f.spins),
            targets: job.plan.targets.clone(), checked: job.checked as u16,
            canonical_checked: job.canonical_checked as u16, canonical_success: job.success as u16,
            success: job.coverage_success(), success_upper: job.coverage_upper(),
            quiet_safe_supplies: job.quiet_safe as u16, canonical_quiet_safe: job.canonical_quiet_safe as u16,
            quiet_no_hold_verified: complete && job.quiet_safe == P7_COUNT,
            first_bag_symmetry_assumed: symmetry_assumed, mirrored_inputs_verified: false,
            hold_policy: CLEAR_ONLY_HOLD_POLICY.into(),
            probability: if complete { policy.path_probability(&counts) } else { 0.0 },
            first_bag_probability: if resolved || !previous_counts.is_empty() {
                counts.first().map(|&n| f64::from(n) / P7_COUNT as f64)
            } else { None },
            budget_pieces: budget, cdf, graph_nodes: job.graph.nodes.len(),
            canonical_costs: if resolved { job.costs.clone() } else { vec![] },
            effective_costs: if resolved { effective.to_vec() } else { vec![] },
            order_codes: if resolved { job.codes.clone() } else { vec![] },
            representative_queue: representative.map(permutation),
            representative_inputs: inputs, representative_frames: frames,
        })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SavedNode {
    board: Board, used: u8, deleted: u32, spins: u8,
    edges: Vec<SavedEdge>, accepting: bool, spawn_mask: u8,
}
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SavedEdge { piece: u8, child: usize, placement: Placement, lines: u8 }
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SavedTarget { board: Board, pose: Pose, spin: bool, placement: Option<Placement> }
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct SavedVerifier {
    request: Request, nodes: Vec<SavedNode>, cursor: usize, overflow: bool, all_targets: u8,
    has_language: bool, checked: usize, canonical_checked: usize, quiet_safe: usize,
    canonical_quiet_safe: usize, quiet_rejected: bool, success: usize,
    costs: Vec<u8>, codes: Vec<u32>, bits: Vec<u64>, upper: u16,
    prefix_checked: bool, symmetry_costs: Option<Vec<u8>>, minimum_cost: u8,
    targets: Vec<SavedTarget>,
}

impl Serialize for Verifier {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let job = &self.job;
        let mut targets: Vec<_> = self.cache.targets.iter().map(|((board, pose, spin), placement)|
            SavedTarget { board: *board, pose: *pose, spin: *spin, placement: placement.clone() }).collect();
        // HashMap iteration order must not affect checkpoint digests.
        targets.sort_by_key(|t| (t.board.rows, t.pose.piece as u8, t.pose.rotation, t.pose.x, t.pose.y, t.spin));
        SavedVerifier {
            request: self.request.clone(), nodes: job.graph.nodes.iter().map(|node| SavedNode {
                board: node.key.board, used: node.key.used, deleted: node.key.deleted, spins: node.key.spins,
                edges: node.edges.iter().map(|e| SavedEdge { piece: e.piece, child: e.child,
                    placement: e.placement.clone(), lines: e.lines }).collect(),
                accepting: node.accepting, spawn_mask: node.spawn_mask,
            }).collect(), cursor: job.graph.cursor, overflow: job.graph.overflow,
            all_targets: job.graph.all_targets, has_language: job.language.is_some(),
            checked: job.checked, canonical_checked: job.canonical_checked, quiet_safe: job.quiet_safe,
            canonical_quiet_safe: job.canonical_quiet_safe, quiet_rejected: job.quiet_rejected,
            success: job.success, costs: job.costs.clone(), codes: job.codes.clone(), bits: job.bits.clone(),
            upper: job.upper, prefix_checked: job.prefix_checked,
            symmetry_costs: job.symmetry.as_ref().map(|o| o.costs().to_vec()), minimum_cost: job.minimum_cost,
            targets,
        }.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Verifier {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let saved = SavedVerifier::deserialize(deserializer)?;
        if saved.nodes.is_empty() || saved.nodes.len() > GRAPH_LIMIT
            || saved.cursor > saved.nodes.len() || saved.costs.len() != P7_COUNT
            || saved.codes.len() != P7_COUNT || saved.bits.len() != 79 || saved.targets.len() > 8192
            || saved.checked > P7_COUNT || saved.canonical_checked > P7_COUNT {
            return Err(D::Error::custom("Invalid verifier checkpoint bounds"));
        }
        // Checksum detects corruption, not deliberate editing. Reproduce the
        // saved canonical phase on import before trusting any edge, accepting
        // bit, supply cost, cache entry, or quiet-safe claim. This audit does
        // not advance training/RNG; the caller reports restore wall time.
        let mut restored = Verifier::new(saved.request.clone()).map_err(D::Error::custom)?;
        for _ in 0..GRAPH_LIMIT + P7_COUNT {
            if restored.job.graph.cursor == saved.cursor
                && restored.job.checked == saved.checked
                && restored.job.canonical_checked == saved.canonical_checked {
                break;
            }
            if restored.finished() { return Err(D::Error::custom("Unreachable verifier checkpoint phase")); }
            restored.advance();
        }
        let expected = serde_json::to_value(&saved).map_err(D::Error::custom)?;
        let observed = serde_json::to_value(&restored).map_err(D::Error::custom)?;
        if expected != observed {
            return Err(D::Error::custom("Verifier checkpoint differs from canonical replay"));
        }
        Ok(restored)
    }
}
