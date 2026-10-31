import type { BoardFrame, Candidate } from './types';

export const bagPieceOrder = ['I', 'J', 'L', 'O', 'S', 'T', 'Z'] as const;
export type PieceColor = typeof bagPieceOrder[number];
export type BoardCell = PieceColor | '_' | 'X';

/** Clearra SolutionBoardPreview.svelte palette (MIT, Clearra contributors).
 * Source: daejunnom/Clearra@f504c8f02d771da518000f901d2b8704a1d6830b.
 * The complete license is retained in THIRD_PARTY_NOTICES.md. */
export const clearraPalette: Record<string, string> = {
  I: '#60d6db', J: '#6d91e5', L: '#eaa05d', O: '#f2cb52',
  S: '#70c982', T: '#c47bdc', Z: '#ec7771', X: '#78817e', _: '#14231f'
};

export interface BagView {
  bag: number;
  rows: number[];
  /** Bottom-up physical rows, then left-to-right columns. */
  cells: BoardCell[][];
  spins: number;
  spinGain: number;
  placed: number;
  /** Inventory only: this list is never a prescribed placement order. */
  pieces: PieceColor[];
  partial: boolean;
  colored: boolean;
}

export class BuildDocumentError extends Error {
  constructor(message = '배치 기록과 필드가 일치하지 않아 문서를 만들 수 없어요.') {
    super(message);
    this.name = 'BuildDocumentError';
  }
}

const WIDTH = 10;
const HEIGHT = 24;
const FULL_ROW = (1 << WIDTH) - 1;
const pieces = new Set<string>(bagPieceOrder);
const emptyCells = (): BoardCell[][] => Array.from({ length: HEIGHT }, () => Array<BoardCell>(WIDTH).fill('_'));
const rowMask = (row: BoardCell[]): number => row.reduce((mask, cell, x) => mask | (cell === '_' ? 0 : 1 << x), 0);
const cellCount = (rows: number[]): number => rows.reduce((count, row) => {
  for (let value = row; value; value &= value - 1) count++;
  return count;
}, 0);

function validRows(rows: unknown): rows is number[] {
  return Array.isArray(rows) && rows.length === HEIGHT &&
    rows.every(row => Number.isInteger(row) && row >= 0 && row <= FULL_ROW);
}

function grayCells(rows: number[]): BoardCell[][] {
  return rows.map(row => Array.from({ length: WIDTH }, (_, x) => row & (1 << x) ? 'X' : '_'));
}

/** Reconstruct actual placed colors and apply exactly the recorded line clears.
 * Missing legacy coordinates stay gray. Occupancy deltas never guess a mino. */
function applyFrame(cells: BoardCell[][], frame: BoardFrame, index: number, priorSpins: number): BoardCell[][] {
  if (!validRows(frame.rows) || frame.placed !== index + 1 || !pieces.has(frame.piece) ||
      !Number.isInteger(frame.spins) || frame.spins !== priorSpins + frame.lines ||
      (frame.lines !== 0 && frame.lines !== 1) || (frame.lines === 1 && !frame.spin)) {
    throw new BuildDocumentError();
  }
  if (cellCount(frame.rows) !== cellCount(cells.map(rowMask)) + 4 - frame.lines * WIDTH) {
    throw new BuildDocumentError();
  }
  if (frame.placedCells === undefined) return grayCells(frame.rows);
  if (!Array.isArray(frame.placedCells) || frame.placedCells.length !== 4) throw new BuildDocumentError();
  const next = cells.map(row => [...row]);
  const seen = new Set<number>();
  for (const cell of frame.placedCells) {
    if (!Array.isArray(cell) || cell.length !== 2) throw new BuildDocumentError();
    const [x, y] = cell;
    if (!Number.isInteger(x) || !Number.isInteger(y) || x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT ||
        next[y][x] !== '_' || seen.has(y * WIDTH + x)) throw new BuildDocumentError();
    seen.add(y * WIDTH + x);
    next[y][x] = frame.piece as PieceColor;
  }
  const remaining = next.filter(row => rowMask(row) !== FULL_ROW);
  if (HEIGHT - remaining.length !== frame.lines) throw new BuildDocumentError();
  while (remaining.length < HEIGHT) remaining.push(Array<BoardCell>(WIDTH).fill('_'));
  if (!frame.rows.every((row, y) => row === rowMask(remaining[y]))) throw new BuildDocumentError();
  return remaining;
}

/** One real end-state per completed bag, plus an actual terminal partial bag.
 * Compact snapshots without their trace are intentionally not reconstructed. */
export function getBagViews(candidate: Candidate | null): BagView[] {
  if (!candidate || !Array.isArray(candidate.frames) || candidate.frames.length === 0) return [];
  if (candidate.frameCount !== candidate.frames.length || candidate.frames.length > 700 ||
      candidate.placed !== candidate.frames.length || !validRows(candidate.rows)) throw new BuildDocumentError();
  let cells = emptyCells();
  let priorSpins = 0;
  let bagStartSpins = 0;
  let inventory: PieceColor[] = [];
  const bags: BagView[] = [];
  for (let index = 0; index < candidate.frames.length; index++) {
    const frame = candidate.frames[index];
    cells = applyFrame(cells, frame, index, priorSpins);
    priorSpins = frame.spins;
    if (inventory.includes(frame.piece as PieceColor)) throw new BuildDocumentError();
    inventory.push(frame.piece as PieceColor);
    const bagEnd = (index + 1) % 7 === 0;
    const last = index + 1 === candidate.frames.length;
    if (last && !bagEnd && !candidate.complete) throw new BuildDocumentError('미완성 가방은 문서로 저장할 수 없어요.');
    if (bagEnd || last) {
      bags.push({
        bag: bags.length + 1,
        rows: [...frame.rows],
        cells: cells.map(row => [...row]),
        spins: frame.spins,
        spinGain: frame.spins - bagStartSpins,
        placed: frame.placed,
        pieces: bagPieceOrder.filter(piece => inventory.includes(piece)),
        partial: !bagEnd,
        colored: cells.every(row => row.every(cell => cell !== 'X'))
      });
      bagStartSpins = frame.spins;
      inventory = [];
    }
  }
  const last = candidate.frames[candidate.frames.length - 1];
  if (last.spins !== candidate.spins || !last.rows.every((row, y) => row === candidate.rows[y])) {
    throw new BuildDocumentError();
  }
  return bags;
}
