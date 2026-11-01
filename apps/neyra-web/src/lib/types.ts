export interface SearchConfig {
  min_success_percent: number;
  /** Minimum success of each bag after the independently filtered opening. */
  bag_min_success_percent: number;
  /** Later bags below this actual spin count require 100%; zero disables the restriction. */
  risk_min_spins_per_bag: number;
  ignore_first_bag_probability: boolean;
  /** Independent setup filter. Zero leaves the first-bag chance unrestricted. */
  first_bag_min_success_percent: number;
  piece_goal: number;
  max_pieces: number;
  budget_start: number;
  seed: number;
  beam_width: number;
  /** Compatibility field. Public search always forces the existing evaluator. */
  cnn_evaluation: boolean;
  /** Browser scheduling only; zero uses all reported logical processors. */
  worker_count: number;
}

export type WorkerSearchConfig = Omit<SearchConfig, 'worker_count' | 'cnn_evaluation'> & {
  cnn_evaluation: false;
  search_profile: number;
  search_lane: number;
  search_lanes: number;
};
export type CandidateView = 'current' | 'best' | 'partial' | `candidate:${string}`;

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

/** Actual certified boundary input for the offline label collector. No labels are invented here. */
export interface TrainingField {
  schemaVersion: 1;
  id: string;
  rows: number[];
  source: {
    kind: 'search';
    family: string;
    reference: string;
    runId: string;
    candidateId: string;
    bagBoundary: number;
    probability: number;
    firstBagProbability: number | null;
    ignoreFirstBagProbability: boolean;
    verified: true;
    qualified: true;
    seed: number;
    runSeed: number;
    modelMode: 'heuristic';
    buildVersion: string;
    rules: 'neyra-srsplus-sonicdrop-clearonlyhold-v1';
    searchConfig: {
      minSuccessPercent: number;
      bagMinSuccessPercent: number;
      firstBagMinSuccessPercent: number;
      riskMinSpinsPerBag: number;
      pieceGoal: number;
      maxPieces: number;
      beamWidth: number;
    };
  };
  /** Opt-in observation only; never an input feature or an authoritative training label. */
  trace?: RetainedSearchTrace;
}

export interface SearchRecordCounters {
  expanded: number;
  candidates: number;
  verifiedCandidates: number;
  pruned: number;
}

export interface RetainedSearchTrace {
  schemaVersion: 1;
  kind: 'retained-path';
  prefixOmittedBags: number;
  start: { rows: number[]; spins: number; placed: number };
  steps: Pick<BoardFrame, 'piece' | 'rows' | 'spins' | 'placed' | 'lines' | 'spin'>[];
  evaluation: {
    kind: 'heuristic';
    rawQuality: number | null;
    rankingScore: number | null;
    forecastSpins: number | null;
    forecastSpinsByBag: number[];
    forecastPlacements: number | null;
  };
  search: {
    scope: 'worker-total-at-record';
    budgetStart: number;
    expanded: number | null;
    candidates: number | null;
    verifiedCandidates: number | null;
    pruned: number | null;
  };
  outcome: { spins: number; placed: number; bagCount: number; countedProbability: number };
}

export type TrainingContribution = TrainingField & { trace: RetainedSearchTrace };

/** New bounded observations are generated only while explicit contribution is enabled. */
export interface FocusedInput {
  id: string;
  rows: number[];
  source: {
    kind: 'search';
    family: string;
    reference: string;
    bagBoundary: number;
    runId: string;
    rootCandidateId: string;
    rootBagBoundary: number;
    rootSeed: number;
    buildVersion: string;
    rules: 'neyra-srsplus-sonicdrop-clearonlyhold-v1';
    trust: 'client-reported';
  };
}

export interface FocusedLabelConfig {
  observationVersion?: 2 | 3;
  horizonBags: number;
  workBudget: number;
  beamWidth: number;
  profile: number;
  seed: number;
  maxNodes: number;
  attemptsPerNode: number;
  sourceRevision: string;
}

export interface FocusedWitness {
  id: number;
  parentId: number | null;
  bags: number;
  startRows: number[];
  endRows: number[];
  startSpins: number;
  endSpins: number;
  checked: number;
  success: number;
  quietSafeSupplies: number;
  firstBagSymmetryAssumed: boolean;
  frames: BoardFrame[];
  [key: string]: unknown;
}

export interface FocusedSpinLabel {
  schemaVersion: 2 | 3;
  task: 'naga-spin-observations-v2' | 'naga-next-bag-spin-v3';
  policy: 'neyra-fixed-bag-continuation-v2' | 'bounded-next-bag-spin-v3';
  /** Prediction horizon. Actual follow-up measurement still spans >=2 bags. */
  targetHorizonBags?: 1;
  input: FocusedInput;
  config: FocusedLabelConfig;
  targets: { foundSpin: boolean; observedSpins: number };
  measured: { workUsed: number; censored: boolean; [key: string]: unknown };
  certifiedWitnesses: FocusedWitness[];
  reachableWholeBagWitnesses?: FocusedReachableWitness[];
  targetWitness?: unknown;
  reachableWitness: unknown;
  [key: string]: unknown;
}

/** One exact input-replayed inventory, with no all-supply success claim. */
export interface FocusedReachableWitness {
  id: number;
  parentId: number | null;
  bags: number;
  startRows: number[];
  endRows: number[];
  startSpins: number;
  endSpins: number;
  deletedRows: number;
  targets: { piece: string; cells: [number, number][] }[];
  representativeQueue: number[];
  representativeInputs: string[][];
  frames: BoardFrame[];
  trust: 'input-reachable';
  wholeBag: true;
}

export interface FocusedObservation {
  schemaVersion: 1;
  kind: 'focused-spin-observation';
  id: string;
  runId: string;
  rootCandidateId: string;
  sequence: number;
  parentObservationId: string | null;
  parentWitnessId: number | null;
  label: FocusedSpinLabel;
}

export interface LearningFocusStatus {
  enabled: boolean;
  active: number;
  queued: number;
  awaitingDelivery: number;
  completed: number;
  failed: number;
  deepestOriginBag: number;
  activeBag: number | null;
  latestObservedSpins: number | null;
  rows: number[] | null;
}

export interface Candidate {
  id: string;
  rows: number[];
  spins: number;
  placed: number;
  probability: number;
  /** Certified effective model chance of each bag, including the opening. */
  bagProbabilities?: number[];
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
  /** Existing raw forecast/terrain quality, before actual-spin ranking. */
  fieldQuality: number;
  /** One coherent geometric continuation; never added to actual progress. */
  forecastSpins?: number;
  forecastSpinsByBag?: number[];
  forecastPlacements?: number;
}

export interface BagCandidates {
  bagCount: number;
  best: Candidate;
  runnerUp: Candidate | null;
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
  /** Exact oriented quiet-policy rejection cache; never selective pruning. */
  transpositionHits?: number;
  transpositionStores?: number;
  transpositionImports?: number;
  bags: number;
  checked: number;
  success: number;
  failed: number;
  frontier: number;
  finished: boolean;
  current: Candidate | null;
  best: Candidate | null;
  bestPartial: Candidate | null;
  /** Two distinct fields per depth (reflection-aware display only), each with its own certified route. */
  bestByBag?: BagCandidates[];
  /** Explicit browser selection, pinned while later records improve. */
  selectedCandidate?: Candidate | null;
  /** Added by the browser coordinator, not combined into any certificate. */
  workers?: number;
  /** Conditional deep observation work; distinct from full-build progress. */
  learningFocus?: LearningFocusStatus;
}

export type RunStatus = 'idle' | 'loading' | 'running' | 'paused' | 'cancelled' | 'completed' | 'failed';

export interface QuietTranspositionBatch {
  schemaVersion: 1;
  policy: 'neyra-exact-quiet-tt-v1';
  /** Full serialized verification contexts, with no mirror canonicalization. */
  entries: string[];
}

export type EngineEvent =
  | { type: 'ready'; runId?: string }
  | { type: 'progress'; runId: string; snapshot: SearchSnapshot }
  | { type: 'status'; runId: string; status: RunStatus; message?: string }
  | { type: 'error'; runId: string; message: string }
  | { type: 'warning'; runId: string; message: string }
  | { type: 'incumbent'; runId: string; candidate: Candidate; data: unknown; counters?: SearchRecordCounters }
  | { type: 'bag-record'; runId: string; candidate: Candidate; data: unknown; counters?: SearchRecordCounters }
  | { type: 'transpositions'; runId: string; batch: QuietTranspositionBatch }
  | { type: 'observation'; runId: string; observationId: string; label: FocusedSpinLabel }
  | { type: 'observation-error'; runId: string; observationId: string; message: string }
  | { type: 'detail'; runId: string; requestId: string; snapshot: SearchSnapshot }
  | { type: 'export'; runId: string; requestId: string; data: unknown; snapshot?: SearchSnapshot }
  | { type: 'export-error'; runId: string; requestId: string; message: string };

export type EngineCommand =
  | { type: 'start'; runId: string; config: WorkerSearchConfig; assetBase: string; assetVersion: string; watchCurrent: boolean; learningEnabled?: boolean; transpositionSharing?: boolean }
  | { type: 'learning-mode'; runId: string; enabled: boolean }
  | { type: 'observe'; runId: string; observationId: string; input: FocusedInput }
  | { type: 'observation-idle'; runId: string }
  | { type: 'observation-wait'; runId: string }
  | { type: 'pause' | 'resume' | 'cancel'; runId: string }
  | { type: 'benchmarks'; runId: string; bestSpinsByBag: number[] }
  | { type: 'transpositions'; runId: string; batch: QuietTranspositionBatch }
  | { type: 'watch'; runId: string; enabled: boolean }
  | { type: 'detail'; runId: string; requestId: string }
  | { type: 'export'; runId: string; requestId: string; view: CandidateView };

export const defaultConfig: SearchConfig = {
  min_success_percent: 100,
  bag_min_success_percent: 100,
  risk_min_spins_per_bag: 3,
  ignore_first_bag_probability: true,
  first_bag_min_success_percent: 80,
  piece_goal: 119,
  max_pieces: 126,
  budget_start: 0,
  seed: 1,
  beam_width: 64,
  cnn_evaluation: false,
  worker_count: 0
};
