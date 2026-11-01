//! Exact, independent fixed-plan records. Display dominance is not a reason
//! to delete a different board from the search tree or combine supply sets.
use super::{Archived, CandidateSnapshot, P7_COUNT};
use num_bigint::BigUint;
use serde::Serialize;
use std::{cmp::Ordering, collections::BTreeMap, rc::Rc};

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FrontierPointSnapshot {
    pub pieces: u16,
    pub probability: f64,
    /// Integer product equality, never a rounded floating point prediction.
    pub exact100: bool,
    pub probability_numerator: String,
    pub probability_denominator: String,
    pub candidate: CandidateSnapshot,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GoalFrontierSnapshot {
    pub target_spins: u8,
    pub complete: bool,
    /// Improving breakpoints of the best probability at or below a lock cap.
    /// A missing cap inherits the preceding point, not an unobserved success.
    pub points: Vec<FrontierPointSnapshot>,
}

struct Record {
    archive: Rc<Archived>,
    numerator: BigUint,
    denominator: BigUint,
    probability: f64,
}
impl Record {
    fn cmp_probability(&self, other: &Self) -> Ordering {
        (&self.numerator * &other.denominator).cmp(&(&other.numerator * &self.denominator))
    }
    fn exact100(&self) -> bool { self.numerator == self.denominator }
}

#[derive(Default)]
pub(super) struct Archive {
    records: BTreeMap<(u8, u16), Record>,
    goal: u8,
}

impl Archive {
    pub fn retain(&mut self, candidate: &Archived, goal: u8, cap: u16, exclude_first: bool) {
        let summary = &candidate.summary;
        if !summary.verified || !summary.qualified || summary.spins == 0
            || candidate.certificates.is_empty()
            || candidate.certificates.iter().any(|c| !c.quiet_no_hold_verified
                || usize::from(c.quiet_safe_supplies) != P7_COUNT || c.effective_success == 0) {
            return;
        }
        self.goal = goal;
        let archive = Rc::new(candidate.clone());
        let counts: Vec<u16> = candidate.certificates.iter().map(|c| c.effective_success).collect();
        let costs: Vec<(u16, u16)> = if summary.complete {
            let last = candidate.certificates.last().unwrap();
            let effective = last.symmetry_costs.as_ref().unwrap_or(&last.costs);
            (1..=7).map(|m| ((counts.len() as u16 - 1) * 7 + m,
                effective.iter().filter(|&&cost| cost > 0 && u16::from(cost) <= m).count() as u16))
                .filter(|&(pieces, count)| pieces <= cap && count > 0).collect()
        } else if summary.placed <= cap { vec![(summary.placed, *counts.last().unwrap())] }
        else { vec![] };
        for (pieces, last_count) in costs {
            let mut local = counts.clone();
            *local.last_mut().unwrap() = last_count;
            let (mut numerator, mut denominator) = (BigUint::from(1_u8), BigUint::from(1_u8));
            let mut chance = 1.0;
            if exclude_first && counts.len() == 1 && summary.complete {
                numerator = BigUint::from(last_count);
                denominator = BigUint::from(counts[0]);
                chance = f64::from(last_count) / f64::from(counts[0]);
            } else {
                for &count in local.iter().skip(usize::from(exclude_first)) {
                    numerator *= count;
                    denominator *= P7_COUNT as u16;
                    chance *= f64::from(count) / P7_COUNT as f64;
                }
            }
            for target in 1..=summary.spins.min(goal) {
                let key = (target, pieces);
                let record = Record { archive: Rc::clone(&archive), numerator: numerator.clone(),
                    denominator: denominator.clone(), probability: chance };
                let replace = self.records.get(&key).is_none_or(|prior| {
                    record.cmp_probability(prior).then_with(|| summary.spins.cmp(&prior.archive.summary.spins))
                        .then_with(|| summary.field_quality.total_cmp(&prior.archive.summary.field_quality))
                        == Ordering::Greater
                });
                if replace { self.records.insert(key, record); }
            }
        }
        // Bound retained display data to improving points. The independent
        // search tree and its board diversity are deliberately untouched.
        for target in 1..=goal {
            let mut best: Option<(BigUint, BigUint)> = None;
            let mut remove = vec![];
            for (&key, record) in self.records.range((target, 0)..=(target, u16::MAX)) {
                if best.as_ref().is_some_and(|(n, d)| &record.numerator * d <= n * &record.denominator) {
                    remove.push(key);
                } else { best = Some((record.numerator.clone(), record.denominator.clone())); }
            }
            for key in remove { self.records.remove(&key); }
        }
    }

    pub fn certain_cost(&self, target: u8) -> Option<u16> {
        self.records.range((target, 0)..=(target, u16::MAX))
            .find_map(|(&(_, pieces), r)| r.exact100().then_some(pieces))
    }

    /// An optimistic supply-mass bound for a terminal plan only. Every
    /// unchecked supply is generously counted as success at every cost.
    /// Different unfinished boards must never be discarded by this display
    /// comparison because they can have different future continuations.
    pub fn terminal_bound_dominated(&self, target: u8, start_pieces: u16,
        parent_counts: &[u16], upper_count: u16, cap: u16) -> bool {
        let mut numerator = BigUint::from(upper_count);
        let mut denominator = BigUint::from(P7_COUNT as u16);
        for &count in parent_counts { numerator *= count; denominator *= P7_COUNT as u16; }
        let end = (start_pieces + 7).min(cap);
        start_pieces < end && (start_pieces + 1..=end).all(|cost| {
            self.records.range((target, 0)..=(target, cost)).next_back()
                .is_some_and(|(_, r)| &r.numerator * &denominator >= &numerator * &r.denominator)
        })
    }

    pub fn find(&self, id: &str) -> Option<&Archived> {
        self.records.values().find(|r| r.archive.summary.id == id).map(|r| r.archive.as_ref())
    }

    pub fn snapshot(&self) -> Vec<GoalFrontierSnapshot> {
        (1..=self.goal).filter_map(|target| {
            let points: Vec<_> = self.records.range((target, 0)..=(target, u16::MAX))
                .map(|(&(_, pieces), r)| FrontierPointSnapshot { pieces, probability: r.probability,
                    exact100: r.exact100(), probability_numerator: r.numerator.to_string(),
                    probability_denominator: r.denominator.to_string(), candidate: r.archive.for_snapshot(false) })
                .collect();
            (!points.is_empty()).then_some(GoalFrontierSnapshot { target_spins: target,
                complete: target == self.goal, points })
        }).collect()
    }
}
