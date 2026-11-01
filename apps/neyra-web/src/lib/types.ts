export interface SearchConfig {
  min_success_percent: number;
  ignore_first_bag_probability: boolean;
  /** Independent setup filter. Zero leaves the first-bag chance unrestricted. */
  first_bag_min_success_percent: number;
  piece_goal: number;
  max_pieces: number;
  budget_start: number;
  seed: number;
  beam_width: number;
  /** Browser scheduling only; zero uses all reported logical processors. */
  worker_count: number;
}

export type WorkerSearchConfig = Omit<SearchConfig, 'worker_count'> & {
  search_profile: number;
  search_lane: number;
  search_lanes: number;
};
export type CandidateView = 'current' | 'best' | 'partial';

export interface BoardFrame {
  rows: number[];
  piece: string;
  spins: number;
  placed: number;
  lines: number;
  spin: boolean;
  /** Actual pre-clear physical cells; absent only in older saved records. */
  placedCells?: [number, number][];
}

export interface Candidate {
  id: string;
  rows: number[];
  spins: number;
  placed: number;
  probability: number;
  /** First-bag chance under the selected symmetry policy, even when excluded. */
  firstBagProbability?: number | null;
  firstBagSymmetry?: boolean;
  firstBagSymmetryAssumed?: boolean;
  lowerProbability: number;
  upperProbability: number;
  complete: boolean;
  qualified: boolean;
  verified: boolean;
  checked: number;
  success: number;
  failed: number;
  bagCount: number;
  budgetPieces: number | null;
  cdf: { pieces: number; probability: number }[];
  frames: BoardFrame[];
  frameCount: number;
  rankingScore: number;
  /** Raw forecast/terrain quality before the bounded actual-spin ranking. */
  fieldQuality: number;
  /** One coherent geometric continuation; never added to actual progress. */
  forecastSpins?: number;
  forecastSpinsByBag?: number[];
  forecastPlacements?: number;
}

export interface SearchSnapshot {
  phase: string;
  expanded: number;
  candidates: number;
  verifiedCandidates: number;
  deepestSpins: number;
  qualifiedBags: number;
  qualifiedSpins: number;
  /** Verified same-depth benchmarks for scheduling; never a shared certificate. */
  bestSpinsByBag?: number[];
  pruned: number;
  /** Selective productivity/terrain cuts, separate from proven bounds. */
  selectivePruned?: number;
  duplicates?: number;
  bags: number;
  checked: number;
  success: number;
  failed: number;
  frontier: number;
  finished: boolean;
  current: Candidate | null;
  best: Candidate | null;
  bestPartial: Candidate | null;
  /** Added by the browser coordinator, not combined into any certificate. */
  workers?: number;
}

export type RunStatus = 'idle' | 'loading' | 'running' | 'paused' | 'cancelled' | 'completed' | 'failed';

export type EngineEvent =
  | { type: 'ready'; runId?: string }
  | { type: 'progress'; runId: string; snapshot: SearchSnapshot }
  | { type: 'status'; runId: string; status: RunStatus; message?: string }
  | { type: 'error'; runId: string; message: string }
  | { type: 'warning'; runId: string; message: string }
  | { type: 'incumbent'; runId: string; candidate: Candidate; data: unknown }
  | { type: 'detail'; runId: string; requestId: string; snapshot: SearchSnapshot }
  | { type: 'export'; runId: string; requestId: string; data: unknown; snapshot?: SearchSnapshot }
  | { type: 'export-error'; runId: string; requestId: string; message: string };

export type EngineCommand =
  | { type: 'start'; runId: string; config: WorkerSearchConfig; assetBase: string; assetVersion: string; watchCurrent: boolean }
  | { type: 'pause' | 'resume' | 'cancel'; runId: string }
  | { type: 'benchmarks'; runId: string; bestSpinsByBag: number[] }
  | { type: 'watch'; runId: string; enabled: boolean }
  | { type: 'detail'; runId: string; requestId: string }
  | { type: 'export'; runId: string; requestId: string; view: CandidateView };

export const defaultConfig: SearchConfig = {
  min_success_percent: 100,
  ignore_first_bag_probability: true,
  first_bag_min_success_percent: 80,
  piece_goal: 119,
  max_pieces: 140,
  budget_start: 0,
  seed: 1,
  beam_width: 64,
  worker_count: 0
};
