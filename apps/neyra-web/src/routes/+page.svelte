<script lang="ts">
  import { onMount } from 'svelte';
  import BoardPreview from '$lib/BoardPreview.svelte';
  import ProgressStrip from '$lib/ProgressStrip.svelte';
  import Icon from '$lib/Icon.svelte';
  import { SearchClient, workerLimit, MAX_WORKERS } from '$lib/engine';
  import { defaultConfig, type Candidate, type EngineEvent, type RunStatus, type SearchConfig, type SearchSnapshot } from '$lib/types';
  import './styles.css';

  let client: SearchClient;
  let status: RunStatus = 'idle';
  let config = { ...defaultConfig };
  let draft = { ...defaultConfig };
  let runConfig = { ...defaultConfig };
  let snapshot: SearchSnapshot | null = null;
  let error = '';
  let settings: HTMLDialogElement;
  let about: HTMLDialogElement;
  let settingsError = '';
  let elapsedMs = 0;
  let previousTick = 0;
  let view: 'current' | 'best' = 'current';
  let frameIndex = -1;
  let playing = false;
  let visibleCandidateId = '';
  let savedMessage = '';
  let notice = '';
  let availableWorkers = 1;

  // Clearra SolutionBoardPreview.svelte palette (MIT, pinned in THIRD_PARTY_NOTICES).
  const pieceColors: Record<string, string> = {
    I: '#60d6db', O: '#f2cb52', T: '#c47bdc', S: '#70c982',
    Z: '#ec7771', J: '#6d91e5', L: '#eaa05d'
  };
  const number = new Intl.NumberFormat('ko-KR');
  const compactNumber = new Intl.NumberFormat('ko-KR', { notation: 'compact', maximumFractionDigits: 1 });
  $: active = status === 'running' || status === 'loading';
  $: hasSession = status !== 'idle' && status !== 'failed';
  $: bestRoute = snapshot?.best ?? snapshot?.bestPartial ?? null;
  $: candidate = view === 'best' ? bestRoute : snapshot?.current ?? null;
  $: frames = candidate?.frames ?? [];
  $: selectedIndex = frameIndex < 0 ? Math.max(0, frames.length - 1) : Math.min(frameIndex, Math.max(0, frames.length - 1));
  $: frame = frames[selectedIndex];
  $: boardRows = frame?.rows ?? candidate?.rows ?? [];
  $: boardSpins = frame?.spins ?? candidate?.spins ?? 0;
  $: boardPlaced = frame?.placed ?? candidate?.placed ?? 0;
  $: verifying = status === 'running' && snapshot?.phase === 'verifying';
  $: statusLabel = ({
    idle: '탐색 준비',
    loading: '탐색기 준비 중',
    running: '새로운 빌드 탐색 중',
    paused: '일시정지',
    cancelled: '탐색 중지',
    completed: '이번 탐색 완료',
    failed: '탐색을 멈췄어요'
  } as Record<RunStatus, string>)[status];
  $: if (candidate?.id !== visibleCandidateId) {
    visibleCandidateId = candidate?.id ?? '';
    frameIndex = -1;
    playing = false;
  }

  function percent(value: number) {
    return (Math.max(0, Math.min(1, value)) * 100).toLocaleString('ko-KR', { maximumFractionDigits: 2 }) + '%';
  }
  function time(ms: number) {
    const total = Math.floor(ms / 1000);
    return String(Math.floor(total / 60)).padStart(2, '0') + ':' + String(total % 60).padStart(2, '0');
  }
  function coverageLabel(item: Candidate | null | undefined) {
    if (!item) return '—';
    if (item.verified) return percent(item.probability);
    if (item.lowerProbability === item.upperProbability) return percent(item.lowerProbability);
    return percent(item.lowerProbability) + '–' + percent(item.upperProbability);
  }
  function budgetCoverageLabel(item: Candidate) {
    const point = item.cdf.find((entry) => entry.pieces === item.budgetPieces);
    return point ? percent(point.probability) : coverageLabel(item);
  }
  function receive(event: EngineEvent) {
    if (event.type === 'progress') {
      snapshot = event.snapshot;
      if (status === 'loading') status = 'running';
      if (snapshot.finished && status === 'running') status = 'completed';
    } else if (event.type === 'status') {
      status = event.status;
      if (event.status === 'running') previousTick = performance.now();
    } else if (event.type === 'error') {
      status = 'failed';
      error = event.message;
    } else if (event.type === 'warning') {
      notice = event.message;
    } else if (event.type === 'export') {
      download(event.data, 'neyra-result.json');
    }
  }
  onMount(() => {
    availableWorkers = workerLimit();
    client = new SearchClient(receive);
    try {
      const stored = localStorage.getItem('neyra-config-v1');
      if (stored) {
        const values = JSON.parse(stored);
        const merged = { ...defaultConfig, ...values };
        // A manual count intentionally overrides the browser's processor hint.
        // Preserve it when that hint changes, subject to the application limit.
        if (Number.isInteger(merged.worker_count) && merged.worker_count > 0) {
          merged.worker_count = Math.min(merged.worker_count, MAX_WORKERS);
        }
        if (validateConfig(merged) === '') config = merged;
      }
    } catch { /* A fresh configuration remains usable when storage is unavailable. */ }
    previousTick = performance.now();
    const clock = window.setInterval(() => {
      const now = performance.now();
      if (status === 'running') elapsedMs += now - previousTick;
      previousTick = now;
      if (playing && frames.length > 0) {
        if (selectedIndex >= frames.length - 1) playing = false;
        else frameIndex = selectedIndex + 1;
      }
    }, 250);
    return () => { window.clearInterval(clock); client.dispose(); };
  });
  function start() {
    if (!client || active) return;
    status = 'loading';
    error = '';
    savedMessage = '';
    notice = '';
    elapsedMs = 0;
    snapshot = null;
    view = 'current';
    frameIndex = -1;
    playing = false;
    runConfig = { ...config };
    previousTick = performance.now();
    client.start(runConfig);
  }
  function pause() { client.pause(); }
  function resume() {
    view = 'current';
    frameIndex = -1;
    playing = false;
    previousTick = performance.now();
    client.resume();
  }
  function stop() { playing = false; client.cancel(); }
  function validateConfig(value: SearchConfig) {
    if (!Number.isFinite(value.min_success_percent) || value.min_success_percent <= 0 || value.min_success_percent > 100) return '성공률은 0% 초과, 100% 이하로 입력해 주세요.';
    if (!Number.isInteger(value.piece_goal) || value.piece_goal < 1 || value.piece_goal > 1000) return '목표 미노 수는 1–1,000 사이의 정수로 입력해 주세요.';
    if (!Number.isInteger(value.max_pieces) || value.max_pieces < 1 || value.max_pieces > 700) return '탐색 상한은 1–700 사이의 정수로 입력해 주세요.';
    if (!Number.isInteger(value.budget_start) || value.budget_start < 0 || value.budget_start > value.max_pieces) return '시작 예산은 자동(0) 또는 탐색 상한 이하로 입력해 주세요.';
    if (!Number.isSafeInteger(value.seed) || value.seed < 1 || value.seed > 4294967295) return '시드는 1–4,294,967,295 사이의 정수로 입력해 주세요.';
    if (![16, 32, 64, 128].includes(value.beam_width)) return '탐색 폭 설정을 다시 선택해 주세요.';
    if (!Number.isInteger(value.worker_count) || value.worker_count < 0 || value.worker_count > MAX_WORKERS) return `동시 탐색 수는 자동(0) 또는 1–${MAX_WORKERS} 사이의 정수로 입력해 주세요.`;
    return '';
  }
  function openSettings() { draft = { ...config }; settingsError = ''; settings.showModal(); }
  function saveSettings(event: SubmitEvent) {
    event.preventDefault();
    settingsError = validateConfig(draft);
    if (settingsError) return;
    config = { ...draft };
    try { localStorage.setItem('neyra-config-v1', JSON.stringify(config)); } catch { /* Settings still apply to this tab. */ }
    settings.close();
  }
  function newSeed() {
    draft.seed = (crypto.getRandomValues(new Uint32Array(1))[0] % 4294967295) + 1;
  }
  function download(data: unknown, filename: string) {
    const url = URL.createObjectURL(new Blob([typeof data === 'string' ? data : JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const anchor = document.createElement('a');
    anchor.href = url; anchor.download = filename; anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    savedMessage = '결과를 저장했어요.';
  }
  function exportResult() { client.export(view); }
  function togglePlayback() {
    if (frames.length < 2) return;
    if (selectedIndex === frames.length - 1) frameIndex = 0;
    playing = !playing;
  }
  function previousFrame() { playing = false; frameIndex = Math.max(0, selectedIndex - 1); }
  function nextFrame() { playing = false; frameIndex = Math.min(frames.length - 1, selectedIndex + 1); }
</script>

<svelte:head>
  <title>Neyra — Naga Eyes 빌드 탐색</title>
  <meta name="description" content="더 적은 미노로 완주하는 Naga Eyes 빌드를 브라우저에서 탐색하세요." />
</svelte:head>

<main class="app-shell">
  <header class="app-header">
    <a class="brand" href="./" aria-label="Neyra 홈">
      <span class="brand-mark" aria-hidden="true"><i></i><i></i><i></i><i></i></span>
      <span>Neyra<span class="brand-dot">.</span></span>
      <span class="version">MVP</span>
    </a>
    <div class="header-actions">
      <button class="icon-button help-button" onclick={() => about.showModal()} aria-label="도움말"><Icon name="info" size={17} /></button>
      <a href="https://github.com/daejunnom/Neyra" class="github-link" target="_blank" rel="noreferrer">GitHub <Icon name="external" size={13} /></a>
    </div>
  </header>

  <section class="page-heading">
    <div><p class="eyebrow">NAGA EYES / BUILD SEARCH</p><h1>더 적은 미노, 새로운 빌드.</h1></div>
    <button class="settings-button" onclick={openSettings} aria-label="탐색 설정"><Icon name="settings" size={17} /><span>탐색 설정</span></button>
  </section>

  <section class="workspace" aria-label="빌드 탐색 작업 공간">
    <section class="board-panel" aria-label="후보 필드">
      <div class="panel-top">
        <div class="view-tabs" role="tablist" aria-label="필드 보기">
          <button role="tab" aria-selected={view === 'current'} class:selected={view === 'current'} onclick={() => view = 'current'}>탐색 중</button>
          <button role="tab" aria-selected={view === 'best'} class:selected={view === 'best'} disabled={!bestRoute} onclick={() => view = 'best'}>최선 경로 {#if bestRoute}<span class="tab-dot"></span>{/if}</button>
        </div>
        <span class="board-dimensions">10 × 24</span>
      </div>

      <div class="board-stage">
        <div class="field-wrap"><BoardPreview rows={boardRows} ariaLabel={view === 'best' ? '최선 결과 필드' : '탐색 중인 후보 필드'} empty={!candidate} /></div>
        <div class="field-summary">
          <div class="field-stat primary"><span>스핀 싱글</span><strong>{boardSpins}<small>/ 40</small></strong><div class="goal-track" aria-hidden="true"><i style={'height:' + boardSpins / 40 * 100 + '%'}></i></div></div>
          <div class="field-stat"><span>배치한 미노</span><strong>{boardPlaced}<small>개</small></strong></div>
          <div class="field-stat"><span>현재 가방</span><strong>{candidate ? Math.max(1, Math.ceil(boardPlaced / 7)) : '—'}</strong></div>
          <div class="candidate-label" class:complete={candidate?.complete}>{candidate?.complete ? '완주 후보' : candidate?.verified && candidate.qualified ? '검증 통과' : candidate ? '탐색 후보' : '빈 필드'}</div>
          {#if frame?.piece}<div class="last-piece" style={'--piece-color:' + (pieceColors[frame.piece] || '#96c7b5')}><span>{frame.piece}</span><small>{frame.lines > 0 && frame.spin ? 'SPIN SINGLE' : 'LOCK'}</small></div>{/if}
        </div>
      </div>

      <div class="timeline">
        <button class="icon-button" aria-label="이전 배치" disabled={frames.length < 2 || selectedIndex === 0} onclick={previousFrame}><Icon name="back" size={15} /></button>
        <button class="icon-button playback-button" aria-label={playing ? '재생 일시정지' : '배치 재생'} disabled={frames.length < 2} onclick={togglePlayback}><Icon name={playing ? 'pause' : 'play'} size={14} /></button>
        <input type="range" aria-label="배치 순서" min="0" max={Math.max(0, frames.length - 1)} value={selectedIndex} disabled={frames.length < 2} oninput={(event) => { frameIndex = Number(event.currentTarget.value); playing = false; }} />
        <button class="icon-button" aria-label="다음 배치" disabled={frames.length < 2 || selectedIndex === frames.length - 1} onclick={nextFrame}><Icon name="arrow" size={15} /></button>
        <span class="frame-count">{frames.length ? selectedIndex + 1 : 0}<small> / {frames.length}</small></span>
      </div>
    </section>

    <aside class="control-panel">
      <section class="run-section">
        <div class="section-heading"><span class="eyebrow">SEARCH SESSION</span><span class="session-time"><Icon name="clock" size={13} />{time(elapsedMs)}</span></div>
        <h2>{status === 'idle' ? '가능성을 탐색하세요.' : status === 'paused' ? '잠시 쉬어가는 중.' : status === 'cancelled' ? '여기까지 탐색했어요.' : status === 'completed' ? '이번 탐색을 마쳤어요.' : status === 'failed' ? '다시 시작할 수 있어요.' : '다음 빌드를 찾는 중.'}</h2>
        <p class="run-description">{status === 'idle' ? '빈 필드에서 40번의 스핀 싱글을 완성할 새로운 구축을 찾습니다.' : status === 'paused' ? '지금까지의 결과는 그대로 유지됩니다.' : status === 'cancelled' || status === 'completed' ? '발견한 후보를 확인하거나 새 탐색을 시작하세요.' : status === 'failed' ? '아래 내용을 확인한 뒤 탐색을 다시 시작해 주세요.' : '발견한 구조와 검증 상황이 실시간으로 갱신됩니다.'}</p>
        <div class="goal-summary">
          <div><span>성공률 기준</span><strong>{(hasSession ? runConfig : config).min_success_percent}%<small> 이상</small></strong></div>
          <div><span>개선 목표</span><strong>{(hasSession ? runConfig : config).piece_goal}<small> 미노</small></strong></div>
        </div>
        <div class="run-actions">
          {#if status === 'running' || status === 'loading'}
            <button class="button primary" onclick={pause} disabled={status === 'loading'}><Icon name="pause" size={16} />일시정지</button>
            <button class="button quiet stop-button" onclick={stop} aria-label="탐색 중지"><Icon name="stop" size={13} /></button>
          {:else if status === 'paused'}
            <button class="button primary" onclick={resume}><Icon name="play" size={15} />계속 탐색</button>
            <button class="button quiet stop-button" onclick={stop} aria-label="탐색 중지"><Icon name="stop" size={13} /></button>
          {:else}
            <button class="button primary" onclick={start}><Icon name={status === 'idle' ? 'play' : 'restart'} size={16} />{status === 'idle' ? '탐색 시작' : '새 탐색 시작'}</button>
          {/if}
        </div>
        {#if error}<div class="error-message" role="alert">{error}</div>{/if}
        {#if notice}<p class="session-notice" role="status">{notice}</p>{/if}
      </section>

      <section class="progress-section" aria-label="작업 진척도">
        <ProgressStrip active={active} {verifying} checked={snapshot?.checked ?? 0} label={statusLabel} />
        <dl class="metrics">
          <div><dt>살펴본 배치</dt><dd>{compactNumber.format(snapshot?.expanded ?? 0)}</dd></div>
          <div><dt>가방 후보</dt><dd>{number.format(snapshot?.candidates ?? 0)}</dd></div>
          <div><dt>확인한 가방</dt><dd>{number.format(snapshot?.verifiedCandidates ?? 0)}</dd></div>
          <div><dt>검증 통과 깊이</dt><dd>{snapshot?.qualifiedBags ?? 0}<small> 가방</small></dd></div>
        </dl>
        <p class="verified-progress">검증 통과 최대 {snapshot?.qualifiedSpins ?? 0}스핀<span>{snapshot?.workers ?? 0}개 동시 탐색</span></p>
      </section>

      <section class="best-section" aria-label="현재 최선 결과">
        <div class="section-heading"><h3><Icon name="spark" size={16} />현재 최선</h3><button class="text-button" disabled={!candidate} onclick={exportResult} aria-label="선택한 후보 저장"><Icon name="download" size={14} />저장</button></div>
        {#if bestRoute}
          <button class="best-card" onclick={() => view = 'best'}>
            {#if bestRoute.complete}
              <div><strong>{bestRoute.budgetPieces ?? bestRoute.placed}<span> 미노</span></strong><span class="verified-tag"><Icon name="check" size={12} />완주</span></div>
              <p>{budgetCoverageLabel(bestRoute)} 성공률 <Icon name="arrow" size={13} /></p>
            {:else}
              <div><strong>{bestRoute.spins}<span> 스핀</span></strong><span class="verified-tag"><Icon name="check" size={12} />{bestRoute.bagCount}가방</span></div>
              <p>{bestRoute.placed}미노 · {coverageLabel(bestRoute)} 성공률 <Icon name="arrow" size={13} /></p>
            {/if}
          </button>
        {:else}
          <div class="best-empty"><span class="empty-line"></span><strong>첫 가방을 확인하고 있어요.</strong><p>검증을 통과한 최선 경로를 여기에 표시합니다.</p></div>
        {/if}
        {#if candidate?.complete && !candidate.qualified}
          <div class="secondary-result"><span>완주 후보 · {candidate.verified ? '기준 미달' : '확인 중'}</span><strong>{coverageLabel(candidate)}</strong></div>
        {/if}
        <p class="verification-note">{candidate ? candidate.verified ? `선택한 후보 구축률 · ${coverageLabel(candidate)}` : candidate.checked > 0 ? `구축률 확인 중 · ${coverageLabel(candidate)}` : '선택한 후보 · 구축률 미검증' : '발견한 후보의 구축률을 여기에 표시합니다.'}</p>
        <span class="save-status" role="status">{savedMessage}</span>
      </section>
    </aside>
  </section>

  <footer class="app-footer"><span>모든 탐색은 이 브라우저에서 실행됩니다.</span><span>Built with <a href="https://github.com/daejunnom/Clearra" target="_blank" rel="noreferrer">Clearra</a></span></footer>
</main>

<dialog class="settings-dialog" bind:this={settings}>
  <form onsubmit={saveSettings}>
    <div class="dialog-heading"><div><p class="eyebrow">SEARCH OPTIONS</p><h2>탐색 설정</h2></div><button class="icon-button" type="button" onclick={() => settings.close()} aria-label="설정 닫기"><Icon name="close" /></button></div>
    <div class="settings-grid">
      <label><span>최소 성공률 <small>%</small></span><input type="number" min="0.01" max="100" step="0.01" required bind:value={draft.min_success_percent} /><small>전체 빌드의 성공률 기준</small></label>
      <label><span>개선 목표 <small>미노</small></span><input type="number" min="1" max="1000" step="1" required bind:value={draft.piece_goal} /><small>목표보다 긴 결과도 보관해요.</small></label>
      <label class="full-row"><span>탐색 상한 <small>미노</small></span><input type="number" min="1" max="700" step="1" required bind:value={draft.max_pieces} /></label>
    </div>
    <details class="advanced-options"><summary>추가 설정 <span>동시 탐색 · 시드 · 범위</span></summary><div class="settings-grid">
      <label><span>시작 예산 <small>0 = 자동</small></span><input type="number" min="0" max={draft.max_pieces} step="1" required bind:value={draft.budget_start} /></label>
      <label><span>탐색 폭</span><select bind:value={draft.beam_width}><option value={16}>작게 · 16</option><option value={32}>보통 · 32</option><option value={64}>넓게 · 64</option><option value={128}>아주 넓게 · 128</option></select></label>
      <label class="full-row"><span>동시 탐색 수 <small>0 = 전체 자동</small></span><input type="number" min="0" max={MAX_WORKERS} step="1" required bind:value={draft.worker_count} /><small>자동 {availableWorkers}개 · 직접 지정하면 최대 {MAX_WORKERS}개</small></label>
      <label class="full-row"><span>시드</span><div class="seed-input"><input type="number" min="1" max="4294967295" step="1" required bind:value={draft.seed} /><button type="button" class="icon-button" onclick={newSeed} aria-label="새 시드"><Icon name="restart" size={17} /></button></div></label>
    </div></details>
    {#if active || status === 'paused'}<p class="settings-note">변경한 설정은 다음 탐색부터 적용됩니다.</p>{/if}
    {#if settingsError}<p class="error-message" role="alert">{settingsError}</p>{/if}
    <div class="dialog-actions"><button class="button quiet" type="button" onclick={() => settings.close()}>닫기</button><button class="button primary" type="submit">설정 저장</button></div>
  </form>
</dialog>

<dialog class="help-dialog" bind:this={about}>
  <div class="dialog-heading"><div><p class="eyebrow">NEYRA</p><h2>새 빌드를 찾는 작은 작업실</h2></div><button class="icon-button" onclick={() => about.close()} aria-label="도움말 닫기"><Icon name="close" /></button></div>
  <p>Neyra는 24행 안에서, 중력 대기 없이 40번의 스핀 싱글을 만드는 빌드를 탐색합니다.</p>
  <p>왼쪽은 가방 단위의 후보, 오른쪽은 작업 진척도입니다. <strong>최선 경로</strong>에서 검증을 통과한 배치를 확인하고 저장할 수 있어요. 완주 결과가 나오면 사용 미노 수로 비교합니다.</p>
  <p>112개는 개선 목표예요. 그보다 긴 빌드도 성공률 기준을 만족하면 결과에 남습니다.</p>
  <div class="dialog-actions"><button class="button primary" onclick={() => about.close()}>알겠어요</button></div>
</dialog>
