import { CTK3_FILE_MIME_TYPE, encodeCtk3, type Ctk3Document } from 'ctk3';
import { Field, encoder as fumenEncoder, type EncodePages } from 'tetris-fumen';
import { BuildDocumentError, getBagViews, type BagView } from './buildDocument';
import type { Candidate } from './types';

export type BuildDocumentFormat = 'ctk3' | 'fumen';
export interface EncodedBuildDocument { text: string; filename: string; mime: string }

type RecordValue = Record<string, unknown>;
const record = (value: unknown): RecordValue => value !== null && typeof value === 'object' ? value as RecordValue : {};
const chance = (value: unknown): string | null => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
  ? `${Number((value * 100).toFixed(2))}%` : null;

function pageComment(bag: BagView, candidate: Candidate, data: RecordValue): string {
  const bagCertificate = Array.isArray(data.bags) ? record(data.bags[bag.bag - 1]) : {};
  const success = bagCertificate.checked === 5040 && typeof bagCertificate.effectiveSuccess === 'number'
    ? chance(bagCertificate.effectiveSuccess / 5040) : null;
  const config = record(data.config);
  const scope = data.probabilityScope === 'AfterFirstBag' || config.ignore_first_bag_probability === true
    ? '첫 가방 구축 후' : '전체';
  const first = chance(candidate.firstBagProbability);
  const total = candidate.verified ? chance(candidate.probability) : null;
  return [
    `Neyra · ${bag.bag}가방 ${bag.partial ? '목표 도달' : '완료'} 필드`,
    `${bag.pieces.length}미노 · +${bag.spinGain}스핀 · 누적 ${bag.spins}스핀/${bag.placed}미노`,
    `사용 종류: ${bag.pieces.join(' ')}`,
    success ? `이 가방 구축률 ${success}` : null,
    first ? `첫 가방 ${first}` : null,
    total ? `${scope} 성공률 ${total}` : '미검증 후보',
    candidate.firstBagSymmetryAssumed ? '첫 가방 대칭 가정 적용' : null,
    !bag.colored ? '이전 기록의 미확인 색상은 회색' : null,
    '10×24 · SRS+ · Sonic Drop',
    '줄 소거 후의 가방 결과. 배치 순서는 공급에 따라 달라집니다.'
  ].filter(Boolean).join(' | ');
}

/** Encode bag end-state diagrams, never a representative order as universal play.
 * CTK3 uses Clearra's native document API so all 24 physical rows are retained. */
export function encodeBuildDocument(value: unknown, format: BuildDocumentFormat): EncodedBuildDocument {
  const data = record(value);
  const candidate = data.candidate as Candidate | null;
  if (!candidate || !Array.isArray(candidate.frames) || candidate.frames.length === 0) {
    throw new BuildDocumentError('저장할 가방 기록이 아직 없어요. 가방 확인이 끝난 뒤 다시 시도해 주세요.');
  }
  const bags = getBagViews(candidate);
  if (bags.length === 0) throw new BuildDocumentError('저장할 가방 기록이 아직 없어요.');
  const comments = bags.map(bag => pageComment(bag, candidate, data));
  const filename = `neyra-${candidate.spins}spins-${candidate.placed}pieces`;
  if (format === 'ctk3') {
    const document: Ctk3Document = {
      width: 10,
      pages: bags.map((bag, index) => ({
        height: 24,
        cells: bag.cells.flat().map(cell => cell === '_' ? null : cell === 'X' ? 'G' : cell),
        comment: comments[index],
        flags: { lock: false, colorize: true, mirror: false, rise: false, quiz: false }
      }))
    };
    return { text: encodeCtk3(document), filename: `${filename}.ctk3`, mime: CTK3_FILE_MIME_TYPE };
  }
  if (format !== 'fumen') throw new BuildDocumentError('지원하지 않는 문서 형식이에요.');
  if (bags.some(bag => bag.rows[23] !== 0)) {
    throw new BuildDocumentError('24행의 블록은 Fumen에 보존할 수 없어요. CTK3 형식으로 저장하거나 복사해 주세요.');
  }
  const pages: EncodePages = bags.map((bag, index) => ({
    field: Field.create(bag.cells.slice(0, 23).reverse().map(row => row.join('')).join('')),
    comment: comments[index],
    flags: { lock: false, colorize: true, mirror: false, rise: false }
  }));
  return { text: fumenEncoder.encode(pages), filename: `${filename}.fumen`, mime: 'text/plain;charset=utf-8' };
}
