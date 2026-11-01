# 고정 가방 계획을 위한 2차 평가기 초안

2차 평가기는 **이미 실제 Sonic Drop 입력으로 제안한 한 가방의 고정 배치 계획**을 평가한다. 신경망은 비싼 공급 검증을 수행할 후보의 순서를 정한다. 미노 배치의 합법성, 5,040개 공급 결과, 40L 완료 여부, 확률 상·하한은 기존 Rust 물리·검증기가 담당한다.

현재 초안은 작은 비선형 평가기와 실제 학습·추론 경로다. 전체 Go-Explore, AlphaStar, MEME, MCTS 구현이나 검증된 40L 엔진이라고 부르지 않는다. 네트워크를 학습했거나 세대를 마쳤다는 이유로 공개 성능이 개선됐다고 판단하지 않는다.

## 네트워크 계약

| 항목 | 값 |
| --- | --- |
| 모델 스키마 | `neyra-guide-model-v1` |
| 특징 스키마 | `neyra-guide-fixed-bag547-v1` |
| 규칙 프로필 | `neyra-sonic-srsplus-24-clear-hold-first-reflection-v1` |
| 구조 | 547 → 256 → 잔차 블록(256 → 256 → 256) → sigmoid 3개 |
| 파라미터 수 | 272,643 |
| 저장 수치 | 유한한 f32, 각 값의 절댓값 32 이하 |
| 모델 사용 | 완료된 고정 가방 계획의 검증 순서 |

입력은 다음 547개 값이다.

- 시작 보드와 종료 보드: 각각 24×10의 점유 비트, 총 480개.
- I/J/L/O/S/T/Z 순서의 typed target: 각 네 logical cell의 `(x/9, y/31)`, 총 56개. 셀은 `(x,y)` 순서로 정렬한다. 줄 삭제 전 가방별 logical 좌표이며 0..31행을 허용한다.
- target 종류별 존재 비트 7개. 마지막 부분 가방에 없는 종류의 좌표는 0이다.
- 시작 스핀/40, 종료 스핀/40, 시작 미노 수/126, terminal 여부 4개.

고정 계획과 시작 보드가 달라지면 별도 입력이다. 보드만 같다는 이유로 다른 미노-목표 대응을 합치지 않는다. 한 계획에 같은 종류의 미노를 중복 선언하거나 logical cell을 겹쳐 선언할 수 없다. 비종료 계획은 일곱 종류를 모두 사용한다.

세 출력은 **해당 계획에 조건부인 예측치**다.

1. `quietSafeFraction`: 실제 측정한 quiet-safe 공급 수 / 5,040의 예측.
2. `constructionFraction`: 실제 측정한 구축 성공 공급 수 / 5,040의 예측.
3. `bagSpinsFraction`: 대표 계획의 실제 가방 스핀 수 / 6의 예측.

세 번째 출력은 입력의 `(endSpins - startSpins) / 6`만으로 정확히 계산할 수 있는 **보조 재구성 표지**다. 이 출력의 MAE가 작아져도 미래 스핀 예측력을 얻었다는 증거가 되지 않는다. 현재 검증 순서를 보조할 뿐 미래 가방의 가치나 40L 완주 확률로 사용하지 않는다. 후속 가치 head에는 실제로 더 깊이 계속 탐색한 결과와 그 탐색 예산·중단 상태를 별도 표지로 수집해야 한다.

초기 검증 순서 점수는 `2*quiet + construction + 0.25*spins`다. 이는 임시 수 정렬 점수이며 확률 곱이나 기대 완료 성능, admissible bound가 아니다. 모델 출력이 1.0이어도 공급 인증이 생기지 않는다. 모델이 미학습 상태이거나 입력·추론이 실패하면 `priority()`는 힌트를 주지 않아 기존 순서로 계속 진행할 수 있다.

Rust의 CPU 추론과 Python의 NumPy/CUDA 추론은 같은 가중치·정규화·연산 구조를 사용한다. CPU와 GPU의 부동소수 연산 순서 차이는 허용하되 실제 물리와 확률 증거에 영향을 주어서는 안 된다.

### 실제 탐색 연결

`SearchSession::with_guides`는 선택적으로 GA 모델과 이 모델을 받는다. 가이드의 완료 계획 점수는 기존 순서 점수에 작은 힌트로 반영하고 계획별로 캐시한다. 물리·확률 bound나 후보의 합법 여부는 바꾸지 않는다. `enable_guide_collection`으로 최대 4,096개 완료 측정 자료를 모으고 `take_guide_samples`로 꺼낼 수 있다.

네이티브 `neyra-search`는 독립 CPU 스레드의 탐색을 실행하고 미노 수별 확률 frontier를 합친다. 각 스레드의 성공 집합을 합쳐 가짜 확률을 만들지 않는다. 40스핀 종료 계획은 공급별 실제 최소 종료 비용의 CDF를 사용한다. 낮은 스핀 목표의 표시는 해당 목표 이상을 가진 고정 계획 전체를 완성하는 확률이므로, 그 목표의 최초 도달 확률에 대한 검증된 하한이다. 중간 lock에서 목표에 먼저 도달한 뒤 남은 계획이 실패하는 경우까지 별도로 세는 CDF는 아직 구현하지 않았다. 모델 없이도 같은 exact 물리·공급 경로를 사용한다.

```sh
cargo build --locked --release -p neyra-core --bin neyra-search
target/release/neyra-search \
  --guide-model /tmp/neyra-guide-model.json \
  --threads 2 --seconds 10 \
  --output /tmp/neyra-guided-search.json \
  --dataset-output /tmp/neyra-search-measurements.json
```

학습된 GA의 배치 순서도 함께 사용할 때는 `--ga-model /tmp/neyra-ga-model.json`을 추가한다. 모델을 학습했다는 이유로 이 옵션을 기본값으로 자동 승격하지 않는다. 성능 비교에는 같은 시간·스레드 수·물리·확률 조건을 사용한다.

## 모델을 크게 만든 이유와 한계

24개 선형 계수는 수동 특징의 가중합만 표현한다. 이 초안은 전체 보드와 typed targets의 비선형 상호작용을 학습할 공간을 마련했다. 272,643개는 저비용 CPU 경로와 더 큰 표현력을 함께 시험하기 위한 초기 크기이며, 충분한 성능을 보장하는 숫자가 아니다.

추후에는 같은 시간 예산에서 더 작은 모델, 이 모델, 더 넓은 잔차 모델의 **탐색 결과**를 비교한다. 평가 오차가 줄어도 모델 호출 비용 때문에 검증 후보 수가 줄면 전체 탐색 성능은 나빠질 수 있다. 매 미노의 모든 후보에 이 네트워크를 무조건 호출하는 대신 완료된 계획의 비싼 검증 순서를 정하는 데 먼저 사용한다.

## 학습 자료

`learning/guide_train.py extract`는 GA 모델·체크포인트 내부의 `bagProofs`, 또는 동일한 측정 구조의 Search 자료를 읽는다. 다음 조건을 모두 만족하는 레코드만 학습 대상으로 남긴다.

- `checked == 5040`.
- 실행 그래프 용량 초과가 아니고 완료된 결과.
- 구축 성공의 하한과 상한이 같음.
- 기존 clear-only hold 정책과 일치.
- 시작/종료 보드, 실제 스핀, 시작 미노 수, typed targets가 유효함.

v3 체크포인트의 `gaEvidence`는 외곽 checksum과 proof·episode의 content hash를 먼저 확인하고 proof → episode → state 순서로 참조를 해제한다. 누락·중첩 참조와 128 MiB를 넘는 복원은 거부하며, `rle-v1` 공급 배열과 이전 plain 배열을 모두 읽는다. Rust와 Python의 실수 표기 차이로 해시가 바뀌지 않도록 원래 출력의 숫자 표기를 보존한다. 다른 도구로 숫자 표기까지 바꿔 저장한 파일은 원래 네이티브 export로 다시 추출한다. 이전 v2/plain 입력과 모델 입력도 계속 지원한다.

quiet-safe 정책을 충족하지 못했어도 전체 검사가 끝난 경우에는 실제 실패 비율을 학습할 수 있다. 반대로 부분 검증, 시간 중단, overflow는 0점 표지가 아니다. 공급 성공률과 quiet-safe 비율을 하나로 합치지 않는다.

신규 Search exporter가 넘길 수 있는 형식은 다음과 같다. `records`의 각 항목은 위의 완료 조건을 만족해야 한다.

```json
{
  "schemaVersion": "neyra-guide-dataset-v1",
  "rulesProfile": "neyra-sonic-srsplus-24-clear-hold-first-reflection-v1",
  "records": [
    {
      "input": {
        "startRows": [],
        "endRows": [],
        "targets": [],
        "startSpins": 0,
        "endSpins": 0,
        "startPieces": 0,
        "terminal": false
      },
      "checked": 5040,
      "graphOverflow": false,
      "quietSafeSupplies": 5040,
      "success": 5040,
      "bagSpins": 0,
      "holdPolicy": "ActiveFirst;EmptyBagHold;QuietLocksUseCurrent;DeferOnlyProductiveClearCandidates;HeldLocksClear;TerminalUnusedSwapAllowed;EndFlushRequiresAllNextSpawns"
    }
  ]
}
```

이 예시의 빈 배열은 형식 위치를 설명하는 자리 표시자다. 실제 보드에는 24개 행, 비종료 계획에는 일곱 개의 서로 다른 typed targets가 필요하다. 빈 배열 예시는 학습기로 통과하지 않는다. 실제 입력을 만들 때 가짜 성공 개수로 채우지 않는다.

추출기는 정규화한 전체 입력으로 중복을 제거하고, 동일 typed plan의 다른 대표 실행은 같은 group으로 묶는다. group은 시작 보드·typed targets·시작 스핀·미노·terminal 및 규칙/홀드 계약으로 결정된다. 따라서 같은 계획이 학습과 검증에 동시에 들어가지 않는다. 출처 파일의 해시와 제거한 중복 수도 기록한다. 적어도 두 개의 서로 다른 계획 group이 필요하다.

JSON의 해시는 손상 검출과 동일 자료 식별용이다. 소유자가 제공한 완료 검증 기록의 진위를 암호학적으로 인증하지 않는다. 외부 임의 자료의 숫자만으로 새로운 물리 증명이 생기지 않으며, 실제 후보 채택은 계속 정확한 검증을 거친다.

첫 가방의 확률은 기존에 사용자가 승인한 반사 모델에 조건부다. 독립적인 반사 입력 물리 검증을 추가하거나 기존 계약을 변경하지 않는다.

## 실행

필수 Python 의존성은 NumPy다. PyTorch는 CUDA 경로에만 필요하며 없어도 추출·CPU 학습·CPU 추론이 동작한다. 학습 자료와 산출물은 저장소 밖의 개인 경로에 둔다.

```sh
cargo build --locked --release -p neyra-core --bin neyra-guide

target/release/neyra-guide init --seed 1 --output /tmp/neyra-guide-initial.json

python learning/guide_train.py extract \
  --source /tmp/neyra-ga-checkpoint.json \
  --source /tmp/neyra-search-measurements.json \
  --output /tmp/neyra-guide-data.json

python learning/guide_train.py train \
  --data /tmp/neyra-guide-data.json \
  --model /tmp/neyra-guide-initial.json \
  --steps 100 --device auto --cpu-threads 2 \
  --output /tmp/neyra-guide-model.json \
  --checkpoint /tmp/neyra-guide-training.json \
  --report /tmp/neyra-guide-report.json

target/release/neyra-guide evaluate \
  --model /tmp/neyra-guide-model.json \
  --input /tmp/neyra-fixed-plan.json

python learning/guide_train.py predict \
  --model /tmp/neyra-guide-model.json \
  --input /tmp/neyra-fixed-plan.json --device cpu
```

`--device auto`와 `--device cuda`는 CUDA 사용을 시도한 뒤 실패하면 NumPy CPU로 계속한다. `--device cpu`는 GPU를 시도하지 않는다. 요청한 장치와 실제 사용한 장치는 구분해 보고서에 남긴다. 모델에는 `trained`, `learningComplete:false`, `proofModel:false`가 있다. 학습된 모델의 `trained:true`는 gradient update가 수행됐다는 의미다.

재개할 때는 동일한 데이터와 저장된 optimizer 설정을 사용한다.

```sh
python learning/guide_train.py train \
  --data /tmp/neyra-guide-data.json \
  --resume /tmp/neyra-guide-training.json \
  --steps 100 --device cpu \
  --output /tmp/neyra-guide-model.json \
  --checkpoint /tmp/neyra-guide-training.json \
  --report /tmp/neyra-guide-report.json
```

재개의 `--steps`는 추가 update 수다. `--seconds`로 실행 시간을 제한할 수 있지만 한 batch와 마지막 저장은 그 시간 밖까지 이어질 수 있다. 체크포인트에는 가중치, Adam moments, update 수, RNG, 셔플 순서, batch 위치, group split을 함께 저장한다. 같은 데이터라도 checkpoint와 학습 설정을 바꿔 정확한 재개라고 부르지 않는다.

## CPU/GPU 실패 후 계속 실행

CUDA는 실제 forward/backward를 수행한다. 학습의 기준 상태인 가중치와 Adam moments는 CPU에 둔다. GPU 연산은 해당 batch의 임시 계산이며 전체 gradient가 CPU로 돌아와 유한함을 확인한 뒤에만 새 optimizer 상태를 한 번 적용한다.

GPU 초기화 실패, 연산 실패 또는 비정상 gradient가 발생하면 해당 임시 계산을 버리고 **같은 batch를 NumPy에서 다시 계산**한다. 이전 update나 RNG 위치를 되감아 중복 학습하지 않는다. CPU update도 비정상이면 이전 가중치·optimizer를 보존하여 체크포인트를 남기고 명확한 오류로 종료한다. 잘못된 숫자를 조용히 계속 적용하지 않는다.

이 경로는 ‘GPU가 없어도 학습·탐색을 지속’하는 초안이다. 실제 GPU 장치에서 device loss를 일으켜 검증하지 않았다면 그 검증을 완료했다고 보고해서는 안 된다. 모델 추론의 CPU fallback과 탐색의 정확한 물리·증명 경로는 별개이며, 신경망이 없어도 기존 수 정렬로 탐색을 진행할 수 있어야 한다.

체크포인트 파일은 임시 파일을 완성한 후 원자적으로 교체한다. 체크포인트가 학습 재개의 기준이고 모델 JSON은 추론용 파생 산출물이다. 두 파일 교체 사이에 프로세스가 종료되면 체크포인트로 재개하여 모델을 다시 생성할 수 있다.

## 다음 엔진에 연결할 연구 방향

- **Go-Explore:** 검증된 경로를 보관하고 그 접두부로 돌아가 새로운 suffix를 탐색한다. 같은 첫 가방을 반복 생성하는 비용과 이미 찾은 깊은 경로의 상실을 줄이는 방향이다.
- **MAP-Elites·MO-CMA-MAE:** 한 fitness 우승자에 모든 목적을 몰지 않고 높은 확률, 많은 스핀, 적은 미노, 다른 종료 보드의 우수 경로를 나눠 보관한다. exact 100% 경로의 자리를 별도로 보호한다.
- **AlphaStar:** 대규모 대전 league 대신 과거 챔피언과 실패 공급의 회귀 집합, 반례 발견 역할을 참고한다.
- **MEME:** 자료를 재사용할 때 값의 단위와 급격한 정책 변화를 관리한다. 해당 Atari 논문의 큰 학습 계산량을 Neyra의 저비용 성능 근거로 쓰지 않는다.
- **Gumbel planning:** 작은 탐색 예산에서 서로 다른 합법 후보를 중복 없이 비교하는 일정에 적용을 검토한다. NN이 평가한 action value를 증명으로 바꾸지 않는다.
- **SimBa·CCQD:** 더 넓은 비선형 표현과 공유 trunk/다양한 head의 분리를 검토한다. 실제 자료가 늘기 전 신경망 전체를 큰 GA 개체군으로 복제하지 않는다.

공급 확률 100%의 보호는 모델 loss 하나로 해결할 수 없다. 정확한 증명 archive, 최소 탐색 일정, 희귀 성공 표지의 유지, 예측과 무관한 widening 경로가 함께 필요하다. 모델이 낮게 평가한 결과를 exact failure cache에 기록해서는 안 된다.

## 완료 기준

이 평가기의 첫 기술적 완료 기준은 같은 모델·계획의 Rust/Python 유한 출력 일치, 중단·재개의 상태 일치, 잘못된 shape·schema·checksum의 거절, CPU 경로 실제 실행이다. GPU 실측 여부는 따로 기록한다.

성능 채택은 고정한 모델로 같은 시간 예산의 실제 탐색을 비교해 결정한다. 독립 계획 group의 예측 오차는 보조 지표다. 첫 가방 자료만으로 학습한 모델은 깊은 가방이나 40L에 일반화했다고 볼 수 없다. 확률별 최선 결과, exact 100% 진척, 40L 실제 완료, 소요 시간, 검증 후보 수를 함께 본다.

## 1차 참고 자료

- [Go-Explore: First return, then explore (Nature 2021)](https://arxiv.org/abs/2004.12919)
- [AlphaStar (DeepMind, 2019)](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/)
- [MEME: Human-level Atari 200x faster (ICLR 2023)](https://openreview.net/pdf?id=JtC6yOHRoJJ)
- [MO-CMA-MAE (GECCO 2025)](https://arxiv.org/abs/2505.20712)
- [Policy Improvement by Planning with Gumbel (ICLR 2022)](https://openreview.net/pdf?id=bERaNdoegnO)
- [EfficientZero V2 (ICML 2024)](https://proceedings.mlr.press/v235/wang24at.html)
- [CCQD (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/fbdaea4878318e214c0577dae4b8bc43-Abstract-Conference.html)
- [SimBa (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/741aab8b41a2987867acc9939ad50383-Abstract-Conference.html)
- [Invalid Action Masking (FLAIRS 2022)](https://arxiv.org/abs/2006.14171)
- [Valid Action Suppression (RLC 2026, pre-proceedings page)](https://rlj.cs.umass.edu/2026/papers/Paper1.html)
- [Primacy Bias (ICML 2022)](https://proceedings.mlr.press/v162/nikishin22a.html)
- [Dormant Neurons / ReDo (ICML 2023)](https://proceedings.mlr.press/v202/sokar23a.html)
- [Stockfish NNUE 공식 문서](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html)

각 연구의 Neyra 적용은 설계 가설이다. 외부 게임·로봇 벤치마크 수치를 Neyra 개선율로 바꾸지 않는다. Rovent의 PALS도 실제 성능이 검증되지 않은 사용자 프로젝트의 제안·반례·복구 아이디어로만 참고한다.
