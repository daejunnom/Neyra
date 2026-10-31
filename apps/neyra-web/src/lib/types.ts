export interface SearchConfig {
  min_success_percent: number;
  piece_goal: number;
  max_pieces: number;
  budget_start: number;
  seed: number;
  beam_width: number;
}

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
  bags: number;
  checked: number;
  success: number;
  failed: number;
  frontier: number;
  finished: boolean;
  current: Candidate | null;
  best: Candidate | null;
}

export type RunStatus = 'idle' | 'loading' | 'running' | 'paused' | 'cancelled' | 'completed' | 'failed';

export type EngineEvent =
  | { type: 'ready'; runId?: string }
  | { type: 'progress'; runId: string; snapshot: SearchSnapshot }
  | { type: 'status'; runId: string; status: RunStatus; message?: string }
  | { type: 'error'; runId: string; message: string }
  | { type: 'export'; runId: string; data: unknown };

export type EngineCommand =
  | { type: 'start'; runId: string; config: SearchConfig; assetBase: string }
  | { type: 'pause' | 'resume' | 'cancel' | 'export'; runId: string };

export const defaultConfig: SearchConfig = {
  min_success_percent: 95,
  piece_goal: 112,
  max_pieces: 140,
  budget_start: 0,
  seed: 1,
  beam_width: 64
};
