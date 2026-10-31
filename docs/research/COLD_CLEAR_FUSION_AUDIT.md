# Cold Clear / 공개 MochBot 관련 코드 탐색 감사

작성: 2026-10-31. 이 문서는 소스를 읽고 적용 방향을 도출한 기록이다. 업스트림 엔진 실행·성능 벤치마크나 현재 운영 중 MochBot의 내부 구현 검증을 수행했다는 뜻은 아니다. 아래 적용안의 실제 구현 상태는 README와 v2 수용 기준을 함께 확인한다.

## 1. 식별과 확인 범위

| 대상 | 확인 커밋 | 실제 확인 범위 |
| --- | --- | --- |
| [MinusKelvin/cold-clear](https://github.com/MinusKelvin/cold-clear) | 279edd7c3177ff8077f6a930193397814b281f27 | 원본 Rust DAG, 후보 생성, 평가, 이동 탐색 |
| [MinusKelvin/cold-clear-2](https://github.com/MinusKelvin/cold-clear-2) | ed8b19327b6bd1410ddd873d8611485bd45d8fae | 별도 재작성 엔진. rank 기반 확률적 확장, DAG, 충돌 맵 |
| [MosaicTETR/fusion](https://github.com/MosaicTETR/fusion) | 1eb879f34a77b868ef3ef0025714e0094bf06eca | 기존 MochBot/fusion에서 이동한 공개 리플레이 코칭 엔진 |
| [halp1/triangle](https://github.com/halp1/triangle) | 71c0f9cf1cd724961bc049f4d077ca32564b5812 | README로 공개 TETR.IO 클라이언트·시뮬레이션 라이브러리임을 확인 |

MochBot 제작자 Joshua Liu의 [프로젝트 소개](https://summer.hackclub.com/projects/3518)는 본체가 비공개이며 공개한 부분은 클라이언트라고 명시하고 halp1/triangle을 연결한다. 따라서 fusion의 beam search를 운영 중 MochBot 본체의 탐색 알고리즘이라고 단정할 근거가 없다. 해당 소개의 falcon-4w 통합 언급도 본체의 전체 탐색 구조를 증명하지 않는다.

fusion의 고정 커밋 README는 TETR.IO replay coaching engine이라고 설명한다. 아래 fusion 분석은 그 공개 코드의 분석이다. Cold Clear와 CC2는 서로 다른 버전으로 구분한다.

라이선스 확인 범위: 원본 Cold Clear README는 MPLv2, CC2 README는 MIT 또는 Apache-2.0을 명시한다. fusion의 검사한 전체 트리에는 LICENSE/COPYING 파일이 없고 Cargo.toml에 license 항목도 없다. 이번 제안은 탐색 원리를 Neyra 계약에 맞춰 재구현하는 것이며 fusion 코드를 제품에 복사하는 변경안이 아니다.

## 2. 원본 Cold Clear: 표준 UCT와 다른 확률적 DAG 탐색

### 실제 제어 흐름

[bot/src/dag.rs](https://github.com/MinusKelvin/cold-clear/blob/279edd7c3177ff8077f6a930193397814b281f27/bot/src/dag.rs)의 find_and_mark_leaf는 다음 순서로 확장할 미전개 노드를 찾는다.

1. forced_analysis_lines에 있는 이전 유망 수순을 우선 따라간다.
2. 자식들을 평가 순서로 정렬한다.
3. Standard 평가의 weight로 자식을 확률적으로 고른다.
4. 아직 다른 작업자가 예약하지 않은 leaf를 marked로 예약한다.
5. [bot/src/modes/normal.rs](https://github.com/MinusKelvin/cold-clear/blob/279edd7c3177ff8077f6a930193397814b281f27/bot/src/modes/normal.rs)의 Thinker가 가능한 다음·홀드 미노의 이동과 자식 평가를 생성한다.
6. DAG에 넣고 영향을 받는 부모들만 갱신한다.

Standard의 weight는 다음 형태다. rank는 0부터 시작한다.

    weight = (value - minimum_value + 10)^2 / (rank^2 + 1)

실제 [standard.rs의 weight/improve](https://github.com/MinusKelvin/cold-clear/blob/279edd7c3177ff8077f6a930193397814b281f27/bot/src/evaluation/standard.rs#L735-L751)와 Node 필드에는 일반 UCT의 방문 횟수·평균 rollout 보상·log(parent_visits)/child_visits 항이 없다. 주어진 버전을 단순 UCT라고 설명하면 부정확하다. 각 확장에서 모든 자식을 만든 뒤 beam 폭으로 잘라내는 구조도 아니다. 명시적 IDA* 비용 반복 심화는 없다.

### 중복 상태와 메모리

DAG는 배치 깊이별 generation을 둔다. 같은 generation에서 SimplifiedBoard를 키로 중복 보드를 합치며, 키에는 실제 field rows, combo, 남은 bag, reserve 미노, B2B, reserve가 hold인지 여부가 들어간다. 높이와 표면만 같다고 합치지 않는다. 서로 다른 부모는 동일 자식을 공유한다.

자식 배열과 부모 링크는 generation별 bump arena에 배치한다. 실제 수를 선택하면 루트를 자식으로 옮기고 이미 지난 generation을 제거한다. 새 next가 공개되면 speculated generation을 known으로 바꾸며 이후 구조를 재사용한다. 비동기 작업 결과는 generation 번호로 오래된 결과를 구분한다.

Neyra에서는 실시간 플레이 루트를 전진시키는 대신 다음을 재사용한다.

- 동일한 가방 경계의 후속 후보 스트림과 인증 결과.
- 동일한 exact board/piece/profile의 물리 질의.
- 동일한 고정 계획의 실행 DAG와 P7 bitsets.
- 중단·재개와 비용 contour 변경 사이에 유지되는 parent arena와 후보 cursor.

프로필·규칙·가방 계획이 다르면 키도 달라야 한다. 보드만으로 고정 계획의 서로 다른 의미를 합치지 않는다.

### 미지 큐 처리

원본의 speculated node는 남은 bag의 미노 종류별 자식 집합을 가진다. 탐색 시 해당 종류를 균등하게 고르며, 역전파는 각 종류에서 선택 가능한 좋은 결과들의 평균을 사용하고 죽는 경우에 평가 페널티를 준다.

이는 실제 플레이의 기대 평가 최적화다. Neyra의 고정 타일링에 대한 정확한 P7 성공률을 대체하지 못한다. Neyra에는 C-/C+ 집합, 고정 geometry, 실제 홀드 공급 언어를 그대로 유지해야 한다.

### 평가가 보여 주는 유용한 원리

[standard.rs](https://github.com/MinusKelvin/cold-clear/blob/279edd7c3177ff8077f6a930193397814b281f27/bot/src/evaluation/standard.rs#L242-L365)는 가능한 T 자원을 보고 알려진 T-slot을 가상 소비한 뒤 남은 보드의 높이·표면·cavity·overhang 등을 평가한다. 즉 생산적인 포켓을 일반적인 나쁜 구멍과 똑같이 감점하지 않는다.

Neyra에 적용할 것은 all-spin 목표별 포켓 계약과 소비 후 구조의 평가다. 원본 TSD/TST/PC/공격량/이동시간 보상과 가상 T-slot 소비 결과를 그대로 복사하지 않는다. 가상 소비는 휴리스틱이며 물리 가능성이나 미래 스핀 수의 증명된 상한이 아니다.

Value::improve는 score와 spike의 성분별 최댓값을 별개로 유지한다. Neyra의 최소 비용과 구축률은 이런 식으로 합치면 안 된다. 예컨대 110개·90%와 120개·99%에서 110개·99%라는 실현되지 않은 결과를 만들어서는 안 된다. 비용·확률·witness를 함께 가진 label 또는 실제 결과의 Pareto 집합을 유지해야 한다.

## 3. Cold Clear 2: 충돌 맵과 작업 공유, 가져오지 않을 가정

[CC2 README](https://github.com/MinusKelvin/cold-clear-2/blob/ed8b19327b6bd1410ddd873d8611485bd45d8fae/README.md)는 MCTS-inspired expansion과 transposition-aware tree를 명시한다.

[src/dag/known.rs](https://github.com/MinusKelvin/cold-clear-2/blob/ed8b19327b6bd1410ddd873d8611485bd45d8fae/src/dag/known.rs#L51-L75)의 선택은 정렬된 자식 인덱스를 난수 s로부터 floor((-ln(s)/exploration) mod child_count)로 선택한다. 방문 횟수 기반 UCT가 아니다. [speculated.rs](https://github.com/MinusKelvin/cold-clear-2/blob/ed8b19327b6bd1410ddd873d8611485bd45d8fae/src/dag/speculated.rs)는 남은 bag 종류를 먼저 균등하게 고른다.

[src/bot/freestyle.rs](https://github.com/MinusKelvin/cold-clear-2/blob/ed8b19327b6bd1410ddd873d8611485bd45d8fae/src/bot/freestyle.rs#L43-L98)는 동일 상태에서 필요한 종류의 movegen을 한 번씩 수행하고, 여러 next 시나리오에서 같은 reserve 미노의 후보를 공유한다. Neyra의 5,040개 공급 순열에서도 동일한 물리 상태·미노의 검증을 한 번만 수행하도록 하는 원리와 맞는다.

[src/movegen.rs](https://github.com/MinusKelvin/cold-clear-2/blob/ed8b19327b6bd1410ddd873d8611485bd45d8fae/src/movegen.rs#L326-L369)의 CollisionMaps는 board/piece/rotation별 충돌 좌표를 bitboard로 만들고 각 질의에서 한 비트를 읽는다. Neyra의 역방향 질의에서도 반복되는 cell 충돌 검사에 적용할 수 있다. 단 SRS+, 원점, 24행 사망 경계, first-success kick은 Neyra의 canonical rules로 새로 계산한다.

가져오지 않을 가정:

- 낮은 스택일 때 모든 열·회전을 자유롭게 진입 가능한 출발점으로 두는 fast path. Neyra에서는 현재 보드의 실제 no-drop 진입 증명이 필요하다.
- 원본의 T 중심 스핀 규칙과 64비트 높이 모델.
- [src/map.rs](https://github.com/MinusKelvin/cold-clear-2/blob/ed8b19327b6bd1410ddd873d8611485bd45d8fae/src/map.rs)의 u64 hash만 저장하는 상태 동일성 판정. Neyra의 불가능 증명/성공률 인증에는 exact key equality를 추가한다.
- [src/dag.rs](https://github.com/MinusKelvin/cold-clear-2/blob/ed8b19327b6bd1410ddd873d8611485bd45d8fae/src/dag.rs#L108-L116)의 add_piece에는 despeculated 평가 역전파 TODO가 남아 있다. 이 버전의 모든 캐시 무효화 동작을 완성된 정답으로 간주하지 않는다.

원본 Cold Clear [libtetris/src/moves.rs](https://github.com/MinusKelvin/cold-clear/blob/279edd7c3177ff8077f6a930193397814b281f27/libtetris/src/moves.rs#L57-L112)에도 낮은 스택용 출발점과 ZeroG/ZeroGComplete의 완전성 차이가 명시되어 있다. Neyra에서는 이런 속도 제한을 쓰더라도 프로필 제외 또는 heuristic mode로 구분하고 실패 증명으로 쓰지 않는다.

## 4. 공개 fusion: beam, 점진적 폭 확대, 완료 스냅샷

### 기본 탐색

[src/search.rs](https://github.com/MosaicTETR/fusion/blob/1eb879f34a77b868ef3ef0025714e0094bf06eca/src/search.rs)는 폭 우선 확장 후 각 깊이에서 다음 처리를 한다.

- 생존/의무 등급의 최상위 tier 유지.
- 최고 점수에서 FUTILITY_DELTA 이상 떨어진 후보 제거.
- 정렬 후 beam_width만 유지.
- 강제로 보존할 root move의 기존 대표가 있으면 truncation/futility에서 재삽입.

[src/search_config.rs](https://github.com/MosaicTETR/fusion/blob/1eb879f34a77b868ef3ef0025714e0094bf06eca/src/search_config.rs)의 기본값은 beam 800, depth 14, quiescence 최대 3, extension beam 비율 0.15, policy expansion cap 32다. 이것들은 해당 코칭 엔진의 설정이며 Neyra에 적절한 수치라는 증거가 아니다.

time_budget_ms가 없으면 한 번 설정 폭으로 탐색한다. 있으면 폭 min(200,max_width)에서 시작해 2배씩 늘린다. 따라서 여기의 반복은 깊이 반복 심화가 아니라 iterative widening이다. 각 iteration은 root부터 다시 실행되며 이 경로에서 이전 iteration의 검색 DAG를 재사용하지 않는다.

### 완료된 작업의 보존

일반 깊이 확장이나 NN batch 도중 deadline을 만나면 미완성 레벨을 새 beam으로 교체하지 않고 이전 완료 beam을 유지한다. quiescence에서도 완료 snapshot을 보존한다.

Neyra에는 이 원리를 activePreview와 certified incumbent 분리, candidate revision, 취소된 작업이 인증 집합을 덮어쓰지 않는 commit 계약으로 적용한다. UI는 처리 중 board를 계속 표시할 수 있지만 완료 결과와 섞지 않는다.

중요한 WASM 차이: 해당 코드의 실제 시간 deadline은 native cfg에 묶여 있다. wasm32에서는 time_budget_ms를 iteration count 한도로 변환한다. 이를 실제 밀리초 취소 보장으로 가져오면 안 된다. Neyra는 자체적인 짧은 work quantum과 worker 메시지 양보를 구현한다.

### Quiescence를 바꾸어 적용

SearchNode::is_loud는 combo>0 또는 B2B>0 또는 과거 clear event가 존재하면 참이다. Neyra에 그대로 쓰면 스핀 빌드의 대부분 상태가 계속 loud로 남을 수 있다.

Neyra에서는 현재 계획이 만든 포켓의 마감 의무, 즉시 실행해야 할 kick 진입, 다음 공통 경계까지 확인해야 할 지지 조건을 extension trigger로 사용한다. 단순히 '스핀을 했다'는 이유로 확장하지 않는다. 제한된 extension에서 찾지 못한 것은 실패 증명이 아니다.

### 후보 생성과 expensive scoring

[src/search_expand.rs](https://github.com/MosaicTETR/fusion/blob/1eb879f34a77b868ef3ef0025714e0094bf06eca/src/search_expand.rs#L801-L820)는 NN policy가 있는 경로에서 min(policy cap, current beam width, action count)개를 먼저 남긴 뒤 자식 평가를 수행한다. level batch는 부모별 action을 준비한 후 모델 입력을 묶는다.

Neyra에는 학습 모델 없이도 다음처럼 적용할 수 있다.

1. 국소 목표/coverage/필수 의존성으로 싼 priority를 계산한다.
2. 이번 quantum에서 인증할 상위 후보만 꺼낸다.
3. 나머지는 삭제하지 않고 deferred cursor에 보관한다.
4. 물리 검증/P7 결과를 받으면 다음 후보를 요청한다.

후보 선별은 정렬·작업 할당이며, 확률적 score margin을 불가능 증명으로 저장하지 않는다.

### 큐 예측과 TT를 과장하지 않는다

[src/bag.rs](https://github.com/MosaicTETR/fusion/blob/1eb879f34a77b868ef3ef0025714e0094bf06eca/src/bag.rs#L82-L112)는 남은 종류가 두 개 이하이면 정해진 종류 순서로 queue에 붙인다. 두 종류의 순서는 아직 불확실하다. 현재 search_impl에서 그 반환 vector는 max_depth 계산에 사용되지만 실제 root expansion에는 원래 state.queue가 전달된다. 이를 정확한 7-bag chance tree 또는 P7 성공률 계산으로 설명할 수 없다. Neyra의 정확한 공급 언어에는 사용하지 않는다.

search_config의 board_score 필드에 'cached in TT' 주석이 있지만, 검사한 search.rs/search_expand.rs의 실제 호출 경로에는 검색 노드 TT probe/insert가 없다. search_expand의 HashSet은 profiling을 위한 action key 중복 계수다. 이를 transposition 기반 search dedup으로 소개하지 않는다.

현재 movegen production dispatch는 [src/movegen.rs](https://github.com/MosaicTETR/fusion/blob/1eb879f34a77b868ef3ef0025714e0094bf06eca/src/movegen.rs#L856-L896)의 smear_core다. 파일 앞의 scalar engine 코드는 oracle/비교용 경로도 포함하므로 파일 전체를 동일 production 경로로 읽지 않는다. 이 분석에서는 smear kernel 자체의 정확성·성능을 실행 검증하지 않았다.

## 5. Neyra 변경 우선순위

| 순위 | 모듈 | 구체 변경 | 참고한 원리 |
| --- | --- | --- | --- |
| P0 | execution / p7 / rules | exact board와 logical rows, SRS+ fingerprint, C-/C+, 최종 CDF 보존 | Neyra의 증명 계약을 우선 유지 |
| P1 | solver / telemetry | root부터 재시작하지 않는 resumable frontier, candidate cursor, runId/revision, 완료 결과 원자 반영 | CC subtree/arena 재사용, fusion completed snapshot |
| 조건부 | execution / reachability | worker 내부 exact board/piece/profile 질의의 실제 재사용을 확인한 뒤 제한된 공유 선택 | CC marked leaf, CC2 종류별 movegen 공유 |
| P1 | structure / goals | productive pocket은 소비 후 successor를 평가하고 미지정 holes와 분리 | CC의 자원 한도 T-slot cutout 평가를 all-spin 계약으로 일반화 |
| P2 | solver | 가방 macro 후보 quota를 점진 확대하되 미처리 후보는 Deferred로 보존 | fusion iterative widening을 complete 후보 stream 위에 적용 |
| 조건부 | reachability | 질의 밀도가 사전 계산 비용을 상쇄할 때 충돌 anchor mask 사용 | CC2 collision maps |
| 기본 해제 | solver / bounds | 가방 경계·worker 간 공유는 중복률과 유지 비용을 확인한 뒤 검토; 상태 equality와 proof scope 유지 | CC DAG, CC2 hash-only 동일성은 배제 |
| P3 | goals / solver | 마감이 있는 pocket obligation만 제한적 extension | fusion quiescence의 목적을 Neyra에 맞게 수정 |
| 제외 | learned evaluation | NNUE·학습 의존 평가기 대신 명시적인 정적 특징과 구조 계약 사용 | 대전 평가의 목적을 그대로 이식하지 않음 |

### 실제 추천 scheduler

주 제어기는 v2의 가방 경계 branch-and-bound를 유지한다. 각 경계 상태는 '새 후보를 조금 더 생성', '기존 후보의 모호한 P7 edge 검증', '인증된 다음 가방으로 내려가기'의 재개 가능한 작업을 가진다.

- 첫 순서는 현재 incumbent의 유효 접두 수순, 싼 구조 평가, 기대 검증 비용으로 정한다.
- 검증은 같은 후보 내부의 실행 상태를 우선 공유한다. 추가 메모이제이션은 조회·복사·할당·메모리 비용까지 보고 선택한다.
- 좋지 않은 순위는 Deferred이며 완전성 증명으로 처리하지 않는다.
- 예산 contour가 바뀌어도 기하 후보·물리 witness·P7 certificate는 fingerprint가 같으면 재사용한다.
- 각 work quantum은 작은 개수의 목표/간선을 처리한 뒤 UI와 취소 메시지에 양보한다.

많은 target 질의가 같은 보드·미노에 모이면 단일 reverse query를 반복하는 것보다 strict24 정방향 reachable table 한 번이 더 싼 교차점이 있을 수 있다. 이는 계측 후 선택할 내부 실행 방식이다. 아직 타깃이 없는 모든 보드에서 무조건 전체 movegen을 호출하는 전단으로 돌아가지는 않는다.

독립 worker가 서로 다른 경계 상태를 담당한다면 동일 보드가 충분히 재등장하지 않을 수 있다. 위 공유 원리는 도입을 보장하는 성능 근거가 아니며, work stealing이나 전역 동기화 캐시를 기본으로 계획하지 않는다. 불필요한 회전·동일 기하 후보를 먼저 줄이는 최적화와 캐시 추가는 구분한다.

## 6. 측정과 인증 경계

이 감사는 구현 방향을 제공하며 속도 배율을 주장하지 않는다. 성능 비교는 동일 rules fingerprint, seed, 시간/메모리 한도에서 다음을 측정한다.

- 최초 검증된 40-spin 완주까지 시간과 그 N_p.
- 반복된 board/piece 물리 질의 비율과 cache hit.
- 후보 생성 대비 exact P7 확장 비율.
- 폭 확대·재개 때 다시 실행한 작업량.
- expensive 질의 하나가 해소한 P7 unknown 순열 수.
- activePreview 지연, cancel/pause 응답 시간.
- heuristic Deferred와 ProvenImpossible의 분리.

Cold Clear의 대전 승률이나 fusion의 depth=14를 'Neyra 14가방 탐색 가능'이라는 성능 근거로 사용하지 않는다. 이 코드에서 ply는 보통 미노 하나이며, Neyra의 외부 edge는 완전히 인증한 고정 가방이다.
