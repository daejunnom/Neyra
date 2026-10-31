# Neyra

**더 적은 미노로 완주하는 새로운 Naga Eyes 빌드를 찾는 브라우저 탐색기.**

[Neyra 열기 →](https://daejunnom.github.io/Neyra/)

Rust/Wasm 탐색기와 Svelte 화면으로 구성됩니다. 빈 필드에서 40번의 스핀 싱글을 목표로 탐색하며, 계산은 브라우저의 Web Worker에서 실행합니다.

## MVP 범위

- **10×24 필드**, 중력 대기와 한 칸 낙하 없이 이동·회전·Sonic Drop으로 도달하는 배치.
- 가방별 고정 배치를 새로 만드는 **single-tiling 탐색**. UCT 선택과 제한된 배치 탐색으로 후보를 제안하며, 기존 빌드나 반복 구조를 시작점으로 사용하지 않습니다.
- 제안한 배치의 가능한 실행 순서와 홀드를 검사하는 **정확한 P7 검증**. 가방의 5,040개 순열을 확인하고, 검증 결과를 JSON에 보존합니다.
- 마지막 가방의 일부 미노만 쓰는 완주를 허용하며, 사용 미노 수에 따른 성공률을 평가합니다.
- 현재 후보·미검증 결과·기준을 만족한 완주 결과를 구분하고, 진행 상황과 최선 결과를 중간 표시합니다. 일시정지·재개·중지·배치 재생·JSON 저장을 지원합니다.

최소 성공률, 목표 미노 수, 탐색 상한과 시작 예산을 조정할 수 있습니다. **112개는 개선 목표이며 합격 조건이 아닙니다.** 목표보다 긴 완주도 성공률 기준을 만족하면 보관합니다.

MVP의 후보 생성은 시간과 메모리를 제한하는 휴리스틱입니다. 완주 발견이나 전역 최적성을 보장하지 않으며, 탐색하지 못한 상태를 불가능으로 판정하지 않습니다. 같은 모양의 다른 타일링을 합치는 교차 배치, 미러 결합, 재시작 비용 가중치는 아직 포함하지 않습니다. P7 검증의 정확성은 구현한 24행 물리·홀드 모델과 해당 고정 배치에 대한 것입니다.

## 개발

Node.js 24, pnpm 11.25.0, Rust 1.90.0과 `wasm-bindgen-cli` 0.2.104를 사용합니다.

```sh
npm install -g pnpm@11.25.0
rustup toolchain install 1.90.0 --profile minimal --target wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.104 --locked
pnpm install --frozen-lockfile
cargo test --locked
pnpm check
pnpm build:wasm
pnpm dev
```

개발 서버는 `http://localhost:4173`에서 열립니다. 정적 배포 빌드는 다음과 같습니다.

```sh
NEYRA_BASE_PATH=/Neyra pnpm build
```

출력은 `apps/neyra-web/build`입니다. `main`에 반영하면 GitHub Actions가 테스트·빌드를 거쳐 Pages에 배포합니다. 저장소 Pages 설정의 배포 원본은 **GitHub Actions**를 사용합니다.

## 라이선스

[MIT](LICENSE). Clearra의 킥 테이블과 GUI, Spilink에 포함된 Triangle.js의 규칙 자료를 활용했습니다. 원본 출처·적용 범위·라이선스 전문은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)에 있습니다.
