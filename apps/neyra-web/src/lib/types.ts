export interface SearchConfig {
  min_success_percent: number;
  piece_goal: number;
  max_pieces: number;
  budget_start: number;
  seed: number;
  beam_width: number;
  /** Browser scheduling only; zero selects a conservative automatic count. */
  worker_count: number;
}

export type WorkerSearchConfig = Omit<SearchConfig, 'worker_count'> & { search_profile: number };
export type CandidateView = 'current' | 'best';

export interface BoardFrame {
  rows: number[];
  piece: string;
  spins: number;
  placed: number;
  lines: number;
  spin: boolean;
}

export interface Candidate {
  id: string;
  rows: number[];
  spins: number;
  placed: number;
  probability: number;
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
}

export interface SearchSnapshot {
  phase: string;
  expanded: number;
  candidates: number;
  verifiedCandidates: number;
  deepestSpins: number;
  qualifiedBags: number;
  qualifiedSpins: number;
  pruned: number;
  bags: number;
  checked: number;
  success: number;
  failed: number;
  frontier: number;
  finished: boolean;
  current: Candidate | null;
  best: Candidate | null;
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
  | { type: 'export'; runId: string; requestId: string; data: unknown; snapshot?: SearchSnapshot }
  | { type: 'export-error'; runId: string; requestId: string; message: string };

export type EngineCommand =
  | { type: 'start'; runId: string; config: WorkerSearchConfig; assetBase: string; assetVersion: string }
  | { type: 'pause' | 'resume' | 'cancel'; runId: string }
  | { type: 'export'; runId: string; requestId: string; view: CandidateView };

export const defaultConfig: SearchConfig = {
  min_success_percent: 100,
  piece_goal: 112,
  max_pieces: 140,
  budget_start: 0,
  seed: 1,
  beam_width: 64,
  worker_count: 0
};
