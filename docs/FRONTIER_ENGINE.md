# 가방 순서에 대응하는 확률 전선과 학습 탐색

## 목표와 확률의 의미

목표는 가방별 미노 종류와 logical target을 고정하고, 랜덤한 가방 순서에서 그 배치를 만들 수 있는 경로를 찾는 것이다. 제안 단계는 한 미노씩 Sonic Drop의 실제 도달 잠금을 선택한다. 하나의 대표 큐 재생은 경로를 보여 주는 수단이며, 전체 공급 성공의 증명은 별도로 수행한다.

목표 스핀 수 `s`와 미노 상한 `n`에 대해 `F_s(n)`은 **현재까지 찾고 검증한 독립 고정 계획 중, n개 이내에 끝까지 구축하여 s스핀 이상을 얻는 최고 확률**이다. 전역 최적값을 찾았다는 뜻은 아니다. 40스핀 완주와 미완성 목표를 나누며, 같은 목표에서만 확률·비용을 비교한다.

낮은 스핀 목표에서는 그 목표를 달성한 뒤 남은 고정 계획이 실패하는 공급이나 가방 중간의 최초 도달 시점까지 별도로 세지 않는다. 따라서 표시값은 낮은 목표의 **도달 확률에 대한 검증된 하한**이다. 40스핀 종료 계획의 미노별 비용 CDF는 실제로 측정한다. 중간 목표의 최초 도달 CDF는 후속 구현이며 현재 표시와 혼동하지 않는다.

새 `probability_frontier` 모드는 확률 예산 선택을 사용하지 않는다. 첫 가방을 포함하며, 각 가방의 구축 성공이 하나 이상인 경로를 확장한다. 모든 5,040개 모델 공급에 대한 quiet-safe 조건은 유지한다. 성공 확률이 낮다는 것과 조용한 배치가 불법이라는 것은 다른 결과다.

첫 가방의 J/L·S/Z 동시 반사는 기존에 허용된 공급 모델이다. 순서별 공급 검증과 그 반사 가정의 100%를 모든 반사 방향 입력까지 독립 재생한 물리 증명으로 확대하지 않는다. 기존 clear-only hold, 가방 시작의 빈 hold, 다음 가방 spawn 조건, 24행·SRS+·O spin 제외를 바꾸지 않는다. 현재 공급 검증기의 정보·실행 전략 모델을 유지하며 새로운 online 정책의 정보 제한을 증명한 것으로 표현하지 않는다.

## 기록과 가지치기

확률 전선은 정수 분자·분모로 비교한다. 마지막 부분 가방은 실제 비용별 성공 수 CDF를 사용한다. 반올림 표시나 신경망의 1.0은 확률 100%의 근거가 아니다. 같은 목표에서 더 짧은 경로가 정확히 100%라면 그보다 긴 비용 항목을 생략한다. 낮은 확률의 짧은 40스핀 경로만 찾았을 때는 더 긴 고확률 후보를 계속 탐색한다.

미완성 보드가 다르면 후속 가능성도 다르므로 화면에서 지배된 점을 생략하는 것만으로 탐색 노드를 폐기하지 않는다. 별도 CPU lane이나 다른 계획의 성공 공급 집합을 합쳐 한 계획의 확률을 만들지 않는다. 각 표시 점은 그 계획의 전체 증명과 연결되며, 여러 비용의 점이 같은 고정 계획을 참조할 수 있다.

탐색은 기존 재개 가능한 복수 경로 탐색과 material/cost bound를 재사용한다. 확률 전선 모드에서는 확률 자원이 다른 경로에 동일 깊이 raw 스핀 절단을 적용하지 않는다. terminal 계획은 확인된 성공 수와 미검사 공급 전체를 성공으로 보는 낙관적 상한도 모든 비용에서 기존 전선을 이길 수 없을 때 중단할 수 있다. 이 중단은 공급 실패가 아니며 exact failure cache에 넣지 않는다.

Neyra의 기본 제안 트리는 단일 플레이어 계획 탐색이다. 체스 alpha-beta의 상대 MIN 노드를 새로 가정하지 않는다. 가져오는 원리는 실제 상·하한, 수 정렬, 반복 확장, transposition 문맥과 예산 분리다. pace·delta·beam은 계속 선택적 탐색 휴리스틱이며 불가능성 증명이 아니다.

확률 100%의 접두 경로는 별도 서비스 기회를 받는다. 새 전선 모드는 일정 dispatch 간격으로 실제 정수 100% 접두부를 재방문하며 bounded live tree의 일부를 보호한다. 이 경로도 유한한 제안 폭을 사용하므로 모든 100% 해를 발견한다는 완전성 보장은 없다. 보호 경로를 남기는 것은 작은 평균 점수 차이로 희귀한 안전 경로를 잃는 문제를 줄이기 위한 장치다.

## 학습한 GA와 추가 탐색

`SearchSession::with_ga_model`은 명시적으로 선택한 학습 GA의 24개 또는 256개 계수를 복수 경로 탐색의 배치 순서에 사용한다. 문맥 정책에는 현재 가방의 사용 종류·실제 스핀·미노 수·비용 상한을 전달한다. 실제 `reachable_locks`를 열거하며, 선택적 beam·다음 가방 탐색과 정확한 P7 검증은 별도의 탐색 계층이다. 이 탐색 계층을 GA의 훈련 환경으로 되돌려 사용하지 않는다.

2026-11-02의 새 GA v4는 확률을 제외하고 물리 40L 후보를 우선 생성한다. 이 변경은 여기서 정의한 검색기의 quiet-safe·확률 증명을 없애지 않는다. 물리 후보의 `rawCompleted40`은 공개 탐색의 완료 인증이 아니며, 저장된 입력·typed plan을 다시 검증해야 한다. 새 GA의 가방 접두 복귀와 이 검색기의 인증 접두 재방문은 서로 다른 기록이다.

GA 점수는 탐색의 field-quality 단위에 맞춰 40배로 변환한다. 이 수는 실제 스핀이나 확률을 추가하지 않는다. 완료된 배치 평가를 구조 평가로 조용히 덮어쓰지 않는다. 잘못된 모델이나 미학습 초기값은 기존 구조 평가로 전환하고 원인을 반환한다. 모델의 학습 여부는 파일의 상태이며 공개 승격·40L 완료 판정이 아니다.

모델에 남은 첫 가방 경로는 시작 제안으로 재사용할 수 있다. 입력을 다시 재생하고 실제 P7 검증을 통과한 뒤에만 탐색 트리에 들어간다. 모델에 저장된 성공 숫자를 신뢰해 증명을 만들지 않는다. 현재 초안은 이 첫 가방 제안과 탐색 중 유지되는 접두 보드 재방문을 구현한다. 임의 길이의 과거 archive 전체를 영구 저장·복귀하는 Go-Explore 완전 재현은 후속 범위다.

공개 페이지는 모델을 로컬 파일로 읽어 Worker 메모리에서만 사용한다. 모델과 학습 자료를 서버에 전송하지 않는다. 선택한 파일은 새 실행에만 적용한다. 기존 공개 기본 평가기는 계속 사용할 수 있다.

## 후속 엔진과 실제 표지

후속 guide는 보드 하나에서 무조건부 성공 확률을 예측하지 않는다. 시작·종료 보드, 미노별 logical targets와 진행을 함께 입력받아 **해당 고정 계획의** quiet-safe 비율, 구축 성공 비율, 스핀을 예측한다. 첫 초안은 547입력·256폭 잔차 신경망과 3개 출력으로 구성한다. 자세한 크기·학습·CPU/GPU 계약은 [GUIDE_ENGINE.md](GUIDE_ENGINE.md)에 둔다.

예측치는 완성된 계획의 검증 순서를 개선하는 보조값이다. 실제 CPU 검사·5,040개 공급 증거가 정답이며, 중단·overflow·미검사는 실패 표지로 쓰지 않는다. 학습 자료가 작은 초기에는 정확한 보드/목표 문맥을 기준으로 중복을 제거하고, 동일 계획을 train과 validation으로 나누지 않는다. 단순히 계수 hash가 다르다는 이유로 같은 행동을 새 자료로 세지 않는다.

학습 완료 기준은 세대 수나 loss 감소가 아니다. 먼저 허용 126미노 안의 실제 검증된 40스핀 경로를 확보하고, 저장·복원 후 재검증하며, 고정 예산 비교에서 기존 평가기보다 유리한지 확인한다. 그 후 119미노와 확률·비용 전선을 개선한다. 40스핀 표지가 없는 상태에서 완주 확률 head를 임의로 지도하지 않는다.

## 네이티브 실행

```sh
cargo build --locked --release -p neyra-core --bin neyra-search
neyra-search --seconds 30 --threads 4 --output /tmp/neyra-frontier.json
neyra-search --ga-model /tmp/ga-champion.json --seconds 30 --threads 4 --output /tmp/neyra-ga-search.json
neyra-search --ga-model /tmp/ga-champion.json --guide-model /tmp/guide-model.json --seconds 30 --threads 4 --dataset-output /tmp/guide-data.json --output /tmp/neyra-guided-search.json
```

각 CPU thread는 독립적인 물리·증명 상태를 소유한다. 전체 벽시계 deadline을 공유하고 첫 배치 제안을 lane별로 나눈다. 결과 병합은 같은 목표·비용에서 독립 계획의 최대 확률을 고른다. 전체 시간에는 모델 준비와 최종 증명 내보내기 비용도 기록되며, 요청된 탐색 시간과 같다고 가정하지 않는다. 물리 worker가 하나 실패하면 완료된 다른 worker의 결과는 유지하고 오류를 함께 보고한다.

브라우저는 독립 Web Worker를 유지하므로 WASM shared-memory thread 지원을 필수로 요구하지 않는다. GPU는 정확한 물리·공급 검증을 소유하지 않는다. 현 Rust 검색 추론은 CPU이며, 선택적 GPU 학습은 같은 모델을 내보낸다.

## 검토한 연구와 참고 소스

다음은 아이디어의 근거이며 Neyra 성능 향상 수치가 아니다. 원저자 코드·가중치를 복사하지 않았다.

| 자료 | 채택하는 원리와 적용 범위 |
|---|---|
| [Go-Explore, Nature 2021](https://adrien.ecoffet.com/files/go-explore-nature.pdf) | 유망한 상태와 도달 경로 보관, 복귀와 새로운 탐색의 분리. 현재는 첫 가방 제안 재검증과 유지 중인 접두 상태 재방문. |
| [AlphaStar, Nature 2019](https://deepmind.google/blog/alphastar-grandmaster-level-in-starcraft-ii-using-multi-agent-reinforcement-learning/) | 과거 전략·약점 집합을 유지해 망각 감시. 대규모 league·모방 데이터 훈련 전체를 요구하지 않음. |
| [MEME, ICLR 2023](https://arxiv.org/abs/2209.07550) | 경험 재사용·정규화·정책 변화 안정화. 논문의 200배 경험 효율을 Neyra CPU 속도로 환산하지 않음. |
| [MO-CMA-MAE, GECCO 2025](https://arxiv.org/abs/2505.20712) | 다양한 행동 영역 안에서 다목적 우수해 보관. 현재 GA는 이 알고리즘의 완전 재현이 아님. |
| [SimBa, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/741aab8b41a2987867acc9939ad50383-Abstract-Conference.html) | 비선형 표현·잔차 경로·규모 확장 검토. 같은 모델/정규화 구조를 재현했다는 뜻은 아님. |
| [EfficientZero V2, ICML 2024](https://proceedings.mlr.press/v235/wang24at.html) | 적은 경험에서 탐색과 재사용. 정확한 전이가 이미 있으므로 world model 학습은 우선하지 않음. |
| [Primacy bias, ICML 2022](https://proceedings.mlr.press/v162/nikishin22a.html) | 초기 자료 편향 감시와 부분 재초기화 실험 후보. 현재 GA 회귀를 이 신경망 현상으로 단정하지 않음. |
| [Valid action suppression, RLC 2026 pre-proceedings](https://rlj.cs.umass.edu/2026/papers/Paper1.html) | 불법수 패널티 학습의 부작용. Neyra는 실제 합법 잠금만 평가. |
| [BRPG-MAP-Elites, 2026](https://spiral.imperial.ac.uk/entities/publication/0e2a4a4d-e8d3-471d-a9e4-c2e96d936021) | 행동 문맥 누락이 잘못된 학습 신호를 만들 수 있다는 주의. 후속 입력에 typed plan/가방 상태 유지. |
| [Rovent PALS develop](https://github.com/daejunnom/Rovent/tree/609819ded605795c9ff0653adad391e875b1c4b2) | 제안·반례·수정과 근거 소유권. 문서와 실제 인터페이스를 참고했으며 기력·성능은 검증된 것으로 취급하지 않음. |
| [Clearra](https://github.com/daejunnom/Clearra/tree/87f2e81182d3cd92bc4f3353bef5982e90e2cc0b) | 실제 입력 도달·typed operation·공급/hold 언어·문맥 캐시 경계. Neyra Sonic Drop 계약 유지. |
| [solution-finder](https://github.com/knewjade/solution-finder/tree/0e7c935a5399159a3d9c42fb8721e3c6842ae17d) | 구축 가능성 cover와 도달성 모듈 분리. `LockedReachable`의 soft-drop식 역방향 이동을 Sonic Drop 물리로 복사하지 않음. |

## 후속 작업의 판단 순서

다양성 유지 → 실제 확률 전선 개선 → 같은 비용에서 GA+검색 및 guide 제거 대조 → 깊은 보드의 표지 확보 → 모델 폭·탐색 폭 선택 순서로 진행한다. 데이터가 없다는 이유만으로 매우 큰 신경망이나 대규모 self-play를 시작하지 않는다. 100% 경로는 독립 보호 기록과 실패 공급 반례 집합으로 계속 평가하며 평균 loss만으로 모델을 승격하지 않는다.
