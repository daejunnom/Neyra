<script lang="ts">
  import { onMount } from 'svelte';
  import BoardPreview from '$lib/BoardPreview.svelte';
  import ProgressStrip from '$lib/ProgressStrip.svelte';
  import Icon from '$lib/Icon.svelte';
  import { SearchClient, spinsPerBag, probabilityAdjustedSpinsPerBag, workerLimit, MAX_WORKERS } from '$lib/engine';
  import { encodeBuildDocument } from '$lib/exportFormats';
  import { createPiecePlayback } from '$lib/buildDocument';
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
  let view: 'current' | 'best' = 'best';
  let frameIndex = -1;
  let playing = false;
  let visibleCandidateKey = '';
  let savedMessage = '';
  let notice = '';
  let availableWorkers = 1;

  type ExportFormat = 'ctk3' | 'fumen';
  type BuildDocument = ReturnType<typeof encodeBuildDocument>;
  let exportFormat: ExportFormat = 'ctk3';
  let exportBusy = false;
  let exportEpoch = 0;
  let copyDialog: HTMLDialogElement;
  let copyText: HTMLTextAreaElement;
  let preparedDocument: BuildDocument | null = null;
  let clipboardAvailable = false;
  let copyNotice = '';
  let pendingDocument: {
    requestId: string; format: ExportFormat;
    resolve: (document: BuildDocument) => void; reject: (reason: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  } | null = null;

  const number = new Intl.NumberFormat('ko-KR');
  const compactNumber = new Intl.NumberFormat('ko-KR', { notation: 'compact', maximumFractionDigits: 1 });
  const evaluationNumber = new Intl.NumberFormat('ko-KR', {
    minimumFractionDigits: 1, maximumFractionDigits: 1, signDisplay: 'exceptZero'
  });
  const piecePlayback = createPiecePlayback();
  $: active = status === 'running' || status === 'loading';
  $: hasSession = status !== 'idle' && status !== 'failed';
  $: displayedConfig = snapshot !== null || hasSession ? runConfig : config;
  $: conditionalProbability = displayedConfig.ignore_first_bag_probability;
  $: bestRoute = snapshot?.best ?? snapshot?.bestPartial ?? null;
  $: bagProbabilities = bestRoute?.bagProbabilities ?? [];
  $: candidate = view === 'best' ? bestRoute : snapshot?.current ?? null;
  $: if (client) client.observeCurrent(view === 'current');
  $: frames = candidate?.frames ?? [];
  $: selectedIndex = view === 'current' || frameIndex < 0 ? Math.max(0, frames.length - 1) : Math.min(frameIndex, Math.max(0, frames.length - 1));
  $: frame = frames[selectedIndex];
  $: boardRows = view === 'current' ? candidate?.rows ?? [] : frame?.rows ?? candidate?.rows ?? [];
  $: boardColors = piecePlayback(frames, selectedIndex);
  $: boardSpins = view === 'current' ? candidate?.spins ?? 0 : frame?.spins ?? candidate?.spins ?? 0;
  $: boardPlaced = view === 'current' ? candidate?.placed ?? 0 : frame?.placed ?? candidate?.placed ?? 0;
  $: if (view === 'current') playing = false;
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
  $: candidateKey = candidate ? JSON.stringify([candidate.id, candidate.verified, candidate.spins,
    candidate.placed, candidate.bagCount, candidate.fieldQuality,
    candidate.verified ? candidate.firstBagProbability : null,
    candidate.verified ? candidate.probability : null, candidate.frameCount, candidate.rows]) : '';
  $: if (candidateKey !== visibleCandidateKey) {
    visibleCandidateKey = candidateKey;
    frameIndex = -1;
    playing = false;
  }

  function percent(value: number) {
    if (!Number.isFinite(value)) return '—';
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
  function efficiencyLabel(item: Candidate) {
    return spinsPerBag(item).toLocaleString('ko-KR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function adjustedEfficiencyLabel(item: Candidate) {
    return probabilityAdjustedSpinsPerBag(item).toLocaleString('ko-KR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }
  function evaluationLabel(item: Candidate | null) {
    if (!item || !Number.isFinite(item.fieldQuality)) return '—';
    return evaluationNumber.format(Math.abs(item.fieldQuality) < 0.05 ? 0 : item.fieldQuality);
  }
  function verificationLabel(item: Candidate | null) {
    if (!item) return '발견한 후보의 구축률을 여기에 표시합니다.';
    if (!item.verified && item.checked === 0) return '선택한 후보 · 구축률 미검증';
    const scope = conditionalProbability ? '2가방부터' : item.verified ? '선택한 후보 구축률' : '구축률 확인 중';
    const first = conditionalProbability && item.firstBagProbability != null
      ? ` · 첫 가방 ${percent(item.firstBagProbability)}` : '';
    const symmetry = item.firstBagSymmetryAssumed ? ' · 대칭 가정' : '';
    return `${scope} · ${coverageLabel(item)}${first}${symmetry}`;
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
    } else if (event.type === 'export' && pendingDocument?.requestId === event.requestId) {
      const pending = pendingDocument;
      pendingDocument = null;
      clearTimeout(pending.timer);
      try { pending.resolve(encodeBuildDocument(event.data, pending.format)); }
      catch (reason) { pending.reject(reason instanceof Error ? reason : new Error(String(reason))); }
    } else if (event.type === 'export-error' && pendingDocument?.requestId === event.requestId) {
      rejectDocument(event.message);
    }
  }
  onMount(() => {
    availableWorkers = workerLimit();
    clipboardAvailable = typeof navigator.clipboard?.writeText === 'function';
    client = new SearchClient(receive);
    try {
      const stored = localStorage.getItem('neyra-config-v1');
      if (stored) {
        const values = JSON.parse(stored);
        const merged = { ...defaultConfig, ...values };
        // The old cumulative floor already implied this floor for every later
        // bag. Preserve that acceptance range when migrating saved settings.
        if (values && !Object.prototype.hasOwnProperty.call(values, 'bag_min_success_percent') &&
            Number.isFinite(values.min_success_percent) && values.min_success_percent > 0 && values.min_success_percent <= 100) {
          merged.bag_min_success_percent = values.min_success_percent;
        }
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
      if (view === 'best' && playing && frames.length > 0) {
        if (selectedIndex >= frames.length - 1) playing = false;
        else frameIndex = selectedIndex + 1;
      }
    }, 250);
    return () => { exportEpoch += 1; rejectDocument('화면을 닫아 문서 준비를 멈췄어요.'); window.clearInterval(clock); client.dispose(); };
  });
  function start() {
    if (!client || active) return;
    exportEpoch += 1;
    rejectDocument('새 탐색을 시작해 문서 준비를 멈췄어요.');
    exportBusy = false;
    preparedDocument = null;
    copyDialog?.close();
    status = 'loading';
    error = '';
    savedMessage = '';
    notice = '';
    elapsedMs = 0;
    snapshot = null;
    view = 'best';
    frameIndex = -1;
    playing = false;
    runConfig = { ...config };
    previousTick = performance.now();
    client.observeCurrent(false);
    client.start(runConfig);
  }
  function pause() { client.pause(); }
  function resume() {
    view = 'best';
    frameIndex = -1;
    playing = false;
    previousTick = performance.now();
    client.observeCurrent(false);
    client.resume();
  }
  function stop() { playing = false; client.cancel(); }
  function validateConfig(value: SearchConfig) {
    if (!Number.isFinite(value.min_success_percent) || value.min_success_percent <= 0 || value.min_success_percent > 100) return '전체 성공률은 0% 초과, 100% 이하로 입력해 주세요.';
    if (Math.abs(value.min_success_percent * 100 - Math.round(value.min_success_percent * 100)) > 1e-7) return '전체 성공률은 소수 둘째 자리까지 입력해 주세요.';
    if (!Number.isFinite(value.bag_min_success_percent) || value.bag_min_success_percent <= 0 || value.bag_min_success_percent > 100) return '가방별 성공률은 0% 초과, 100% 이하로 입력해 주세요.';
    if (Math.abs(value.bag_min_success_percent * 100 - Math.round(value.bag_min_success_percent * 100)) > 1e-7) return '가방별 성공률은 소수 둘째 자리까지 입력해 주세요.';
    if (typeof value.ignore_first_bag_probability !== 'boolean') return '첫 가방 성공률 설정을 다시 선택해 주세요.';
    if (!Number.isFinite(value.first_bag_min_success_percent) || value.first_bag_min_success_percent < 0 || value.first_bag_min_success_percent > 100) return '첫 가방 최소 성공률은 0–100%로 입력해 주세요. 0%는 제한 없음입니다.';
    if (Math.abs(value.first_bag_min_success_percent * 100 - Math.round(value.first_bag_min_success_percent * 100)) > 1e-7) return '첫 가방 성공률은 소수 둘째 자리까지 입력해 주세요.';
    if (!Number.isInteger(value.piece_goal) || value.piece_goal < 1 || value.piece_goal > 1000) return '미노 제한은 1–1,000 사이의 정수로 입력해 주세요.';
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
  function rejectDocument(message: string) {
    if (!pendingDocument) return;
    const pending = pendingDocument;
    pendingDocument = null;
    clearTimeout(pending.timer);
    pending.reject(new Error(message));
  }
  function requestDocument(format: ExportFormat): Promise<BuildDocument> {
    return new Promise((resolve, reject) => {
      const requestId = crypto.randomUUID();
      pendingDocument = { requestId, format, resolve, reject,
        timer: setTimeout(() => rejectDocument('문서를 받지 못했어요. 잠시 후 다시 시도해 주세요.'), 15000) };
      try {
        if (!client.export(view, requestId)) rejectDocument('저장할 후보를 먼저 선택해 주세요.');
      } catch (reason) {
        rejectDocument(reason instanceof Error ? reason.message : '문서를 준비하지 못했어요.');
      }
    });
  }
  function download(document: BuildDocument) {
    const url = URL.createObjectURL(new Blob([document.text], { type: document.mime }));
    const anchor = window.document.createElement('a');
    anchor.href = url; anchor.download = document.filename; anchor.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    savedMessage = '문서를 저장했어요.';
  }
  async function exportResult(action: 'download' | 'copy') {
    if (!client || !candidate || exportBusy) return;
    const epoch = exportEpoch;
    exportBusy = true;
    savedMessage = '문서를 준비하고 있어요.';
    const document = requestDocument(exportFormat);
    // Register a handler immediately even if the clipboard refuses the write
    // before the worker has returned its requested certificate.
    document.catch(() => {});
    try {
      if (action === 'download') {
        const result = await document;
        if (epoch === exportEpoch) download(result);
      } else {
        try {
          if (navigator.clipboard?.write && typeof ClipboardItem !== 'undefined') {
            // Invoke write during the user's click; the promised content may
            // arrive later from the worker without losing that activation.
            const text = document.then(result => new Blob([result.text], { type: 'text/plain' }));
            text.catch(() => {});
            await navigator.clipboard.write([new ClipboardItem({ 'text/plain': text })]);
          } else {
            const result = await document;
            if (epoch !== exportEpoch) return;
            if (!navigator.clipboard?.writeText) throw new Error('클립보드 복사를 지원하지 않는 브라우저입니다.');
            await navigator.clipboard.writeText(result.text);
          }
          if (epoch === exportEpoch) savedMessage = '문서를 클립보드에 복사했어요.';
        } catch {
          const result = await document;
          if (epoch !== exportEpoch) return;
          preparedDocument = result;
          copyNotice = '자동으로 복사하지 못했어요. 아래 문서를 선택해 복사할 수 있습니다.';
          copyDialog.showModal();
          queueMicrotask(() => copyText?.select());
          savedMessage = '복사할 문서를 준비했어요.';
        }
      }
    } catch (reason) {
      if (epoch === exportEpoch) savedMessage = reason instanceof Error ? reason.message : '문서를 준비하지 못했어요.';
    } finally {
      if (epoch === exportEpoch) exportBusy = false;
    }
  }
  function selectCopyText() {
    copyText.focus();
    copyText.select();
    copyNotice = '텍스트를 선택했어요. 브라우저의 복사 기능을 사용해 주세요.';
  }
  async function copyPrepared() {
    if (!preparedDocument || !navigator.clipboard?.writeText) { selectCopyText(); return; }
    try {
      await navigator.clipboard.writeText(preparedDocument.text);
      copyNotice = '문서를 클립보드에 복사했어요.';
      savedMessage = copyNotice;
    } catch { selectCopyText(); }
  }
  function togglePlayback() {
    if (view !== 'best' || frames.length < 2) return;
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
          <button role="tab" aria-selected={view === 'best'} class:selected={view === 'best'} onclick={() => view = 'best'}>최선 경로 {#if bestRoute}<span class="tab-dot"></span>{/if}</button>
          <button role="tab" aria-selected={view === 'current'} class:selected={view === 'current'} onclick={() => view = 'current'}>탐색 중</button>
        </div>
        <span class="board-dimensions">10 × 24</span>
      </div>

      <div class="board-stage">
        <div class="field-wrap"><BoardPreview rows={boardRows} colors={boardColors} ariaLabel={view === 'best' ? '최선 경로 필드' : '탐색 중인 후보의 최종 필드'} empty={!candidate} /></div>
        <div class="field-summary">
          <div class="field-stat primary"><span>스핀 싱글</span><strong>{boardSpins}<small>/ 40</small></strong><div class="goal-track" aria-hidden="true"><i style={'height:' + boardSpins / 40 * 100 + '%'}></i></div></div>
          <div class="field-stat"><span>배치한 미노</span><strong>{boardPlaced}<small>개</small></strong></div>
          <div class="field-stat"><span>현재 가방</span><strong>{candidate ? Math.max(1, Math.ceil(boardPlaced / 7)) : '—'}</strong></div>
          <div class="candidate-label" class:complete={candidate?.complete}>{candidate?.complete ? '완주 후보' : candidate?.verified && candidate.qualified ? '검증 통과' : candidate ? '탐색 후보' : '빈 필드'}</div>
        </div>
      </div>

      {#if view === 'best'}<div class="timeline">
        <button class="icon-button" aria-label="이전 미노" disabled={frames.length < 2 || selectedIndex === 0} onclick={previousFrame}><Icon name="back" size={15} /></button>
        <button class="icon-button playback-button" aria-label={playing ? '재생 일시정지' : '미노 재생'} disabled={frames.length < 2} onclick={togglePlayback}><Icon name={playing ? 'pause' : 'play'} size={14} /></button>
        <input type="range" aria-label="미노 배치 선택" min="0" max={Math.max(0, frames.length - 1)} value={selectedIndex} disabled={frames.length < 2} oninput={(event) => { frameIndex = Number(event.currentTarget.value); playing = false; }} />
        <button class="icon-button" aria-label="다음 미노" disabled={frames.length < 2 || selectedIndex === frames.length - 1} onclick={nextFrame}><Icon name="arrow" size={15} /></button>
        <span class="frame-count">{frame?.placed ?? candidate?.placed ?? 0}<small> / {candidate?.placed ?? 0} 미노</small></span>
      </div>{/if}
    </section>

    <aside class="control-panel">
      <section class="run-section">
        <div class="section-heading"><span class="eyebrow">SEARCH SESSION</span><span class="session-time"><Icon name="clock" size={13} />{time(elapsedMs)}</span></div>
        <h2>{status === 'idle' ? '가능성을 탐색하세요.' : status === 'paused' ? '잠시 쉬어가는 중.' : status === 'cancelled' ? '여기까지 탐색했어요.' : status === 'completed' ? '이번 탐색을 마쳤어요.' : status === 'failed' ? '다시 시작할 수 있어요.' : '다음 빌드를 찾는 중.'}</h2>
        <p class="run-description">{status === 'idle' ? '빈 필드에서 40번의 스핀 싱글을 완성할 새로운 구축을 찾습니다.' : status === 'paused' ? '지금까지의 결과는 그대로 유지됩니다.' : status === 'cancelled' || status === 'completed' ? '발견한 후보를 확인하거나 새 탐색을 시작하세요.' : status === 'failed' ? '아래 내용을 확인한 뒤 탐색을 다시 시작해 주세요.' : '발견한 구조와 검증 상황이 실시간으로 갱신됩니다.'}</p>
        <div class="goal-summary">
          <div><span>전체 성공률{conditionalProbability ? ' · 2가방부터' : ''}</span><strong>{displayedConfig.min_success_percent}%{#if displayedConfig.min_success_percent < 100}<small> 이상</small>{/if}</strong></div>
          <div><span>미노 제한</span><strong>{displayedConfig.piece_goal}<small> 미노</small></strong></div>
        </div>
        <p class="bag-floor-summary">가방별 {displayedConfig.bag_min_success_percent}%{displayedConfig.bag_min_success_percent < 100 ? ' 이상' : ''} · 첫 가방 {displayedConfig.first_bag_min_success_percent > 0 ? `${displayedConfig.first_bag_min_success_percent}% 이상` : '제한 없음'}</p>
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
        <div class="section-heading"><h3><Icon name="spark" size={16} />{bestRoute?.complete ? '최선 완주' : '최선 경로'}</h3><div class="result-actions">
          <select bind:value={exportFormat} disabled={exportBusy} aria-label="내보내기 형식"><option value="ctk3">CTK3</option><option value="fumen">Fumen</option></select>
          <button class="icon-button" disabled={!candidate || exportBusy} onclick={() => exportResult('download')} aria-label="선택한 후보 저장" title="문서 다운로드"><Icon name="download" size={15} /></button>
          <button class="icon-button" disabled={!candidate || exportBusy} onclick={() => exportResult('copy')} aria-label="선택한 후보 복사" title="클립보드 복사"><Icon name="copy" size={15} /></button>
        </div></div>
        {#if bestRoute}
          <button class="best-card" onclick={() => view = 'best'}>
            {#if !bestRoute.complete}<span class="efficiency-caption">성공률 반영 효율{#if bestRoute.probability < 1}<small>실제 {efficiencyLabel(bestRoute)}스핀/가방</small>{/if}</span>{/if}
            <div>
              {#if bestRoute.complete}
                <strong>{bestRoute.budgetPieces ?? bestRoute.placed}<span> 미노</span></strong>
              {:else}
                <strong title={`실제 ${efficiencyLabel(bestRoute)}스핀/가방에 ${conditionalProbability ? '2가방부터의 ' : ''}전체 성공률을 반영한 값`}>{adjustedEfficiencyLabel(bestRoute)}<span> 스핀/가방</span></strong>
              {/if}
              <span class="best-card-meta">
                <span class="verified-tag"><Icon name="check" size={12} />{bestRoute.complete ? '완주' : `${bestRoute.bagCount}가방`}</span>
                <span class="field-evaluation" title="최선 경로의 마지막 필드와 후속 스핀 계획 평가">필드 평가 <b>{evaluationLabel(bestRoute)}</b></span>
              </span>
            </div>
            {#if bestRoute.complete}
              <p>{budgetCoverageLabel(bestRoute)} {conditionalProbability ? '2가방부터 성공률' : '성공률'} <Icon name="arrow" size={13} /></p>
            {:else}
              <p>{bestRoute.spins}스핀 · {bestRoute.placed}미노 · {coverageLabel(bestRoute)}{conditionalProbability ? ' (2가방부터)' : ''} <Icon name="arrow" size={13} /></p>
            {/if}
          </button>
          {#if bagProbabilities.length > 0}
            <section class="bag-chances" aria-label="최선 경로의 가방별 구축률">
              <h4>가방별 구축률</h4>
              <ol>
                {#each bagProbabilities as probability, bag}
                  <li><span>{bag + 1}가방{#if bag === 0 && conditionalProbability}<small>전체 제외</small>{/if}</span><strong>{percent(probability)}</strong></li>
                {/each}
              </ol>
            </section>
          {/if}
        {:else}
          <div class="best-empty"><span class="empty-line"></span><div class="best-empty-heading"><strong>첫 가방을 확인하고 있어요.</strong><span class="field-evaluation">필드 평가 <b>—</b></span></div><p>전체 성공률을 반영한 스핀 효율로 경로를 비교합니다.</p></div>
        {/if}
        {#if candidate?.complete && !candidate.qualified}
          <div class="secondary-result"><span>완주 후보 · {candidate.verified ? '기준 미달' : '확인 중'}</span><strong>{coverageLabel(candidate)}</strong></div>
        {/if}
        <p class="verification-note">{verificationLabel(candidate)}</p>
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
      <label><span>전체 성공률 <small>%</small></span><input type="number" min="0.01" max="100" step="0.01" required bind:value={draft.min_success_percent} /><small>{draft.ignore_first_bag_probability ? '2가방부터 누적 성공률의 하한' : '첫 가방부터 누적 성공률의 하한'}</small></label>
      <label><span>가방별 성공률 <small>%</small></span><input type="number" min="0.01" max="100" step="0.01" required bind:value={draft.bag_min_success_percent} /><small>2가방부터 각 가방의 하한 · 첫 가방은 별도 설정</small></label>
      <label><span>미노 제한 <small>미노</small></span><input type="number" min="1" max="1000" step="1" required bind:value={draft.piece_goal} /><small>탐색 상한 안의 더 긴 결과도 보관해요.</small></label>
      <label><span>탐색 상한 <small>미노</small></span><input type="number" min="1" max="700" step="1" required bind:value={draft.max_pieces} /></label>
    </div>
    <details class="advanced-options"><summary>추가 설정 <span>첫 가방 · 동시 탐색 · 범위</span></summary><div class="settings-grid">
      <label class="full-row checkbox-setting" title="첫 가방은 별도 최소값으로 확인하고, 전체 성공률은 2가방부터 계산합니다."><span><input type="checkbox" bind:checked={draft.ignore_first_bag_probability} aria-describedby="first-bag-probability-note" />첫 가방 성공률 제외</span><small id="first-bag-probability-note">전체 성공률은 2가방부터 계산해요.</small></label>
      <label><span>첫 가방 최소 성공률 <small>%</small></span><input type="number" min="0" max="100" step="0.01" required bind:value={draft.first_bag_min_success_percent} /><small>0%는 제한 없음</small></label>
      <label><span>시작 예산 <small>0 = 자동</small></span><input type="number" min="0" max={draft.max_pieces} step="1" required bind:value={draft.budget_start} /></label>
      <label><span>탐색 폭</span><select bind:value={draft.beam_width}><option value={16}>작게 · 16</option><option value={32}>보통 · 32</option><option value={64}>넓게 · 64</option><option value={128}>아주 넓게 · 128</option></select></label>
      <label><span>동시 탐색 수 <small>0 = 전체 자동</small></span><input type="number" min="0" max={MAX_WORKERS} step="1" required bind:value={draft.worker_count} /><small>자동 {availableWorkers}개 · 직접 지정하면 최대 {MAX_WORKERS}개</small></label>
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
  <p>일반 배치는 홀드 없이 가능한지 확인하며, 홀드와 성공률 기준은 줄을 지우는 전개에 적용합니다.</p>
  <p><strong>최선 경로</strong>에서는 미노별 배치를 재생할 수 있고, <strong>탐색 중</strong>에서는 후보의 마지막 필드만 보여 줍니다. 오른쪽에서 작업 진척도와 가방별 구축률을 확인할 수 있습니다.</p>
  <p>전체 성공률과 가방별 성공률을 함께 만족하는 경로를 찾습니다. 첫 가방의 하한과 전체 성공률에서 제외할지는 별도로 정할 수 있습니다.</p>
  <p>중간 최선 경로는 첫 가방을 포함한 스핀 효율에 전체 성공률을 곱해 비교합니다. 확률을 많이 소모한 경로는 낮게 평가하며, 완주 결과는 조건을 만족하는 사용 미노 수를 우선합니다.</p>
  <p>미노 제한을 우선해 탐색하며, 탐색 상한 안의 더 긴 빌드도 성공률 기준을 만족하면 결과에 남습니다.</p>
  <div class="dialog-actions"><button class="button primary" onclick={() => about.close()}>알겠어요</button></div>
</dialog>


<dialog class="copy-dialog" bind:this={copyDialog}>
  <div class="dialog-heading"><div><p class="eyebrow">BUILD DOCUMENT</p><h2>문서 복사</h2></div><button class="icon-button" onclick={() => copyDialog.close()} aria-label="복사 창 닫기"><Icon name="close" /></button></div>
  <p class="copy-notice" role="status">{copyNotice}</p>
  <textarea bind:this={copyText} readonly value={preparedDocument?.text ?? ''} rows="6" aria-label="복사할 빌드 문서" spellcheck="false"></textarea>
  <div class="dialog-actions"><button class="button quiet" onclick={selectCopyText}>텍스트 선택</button>{#if clipboardAvailable}<button class="button primary" onclick={copyPrepared}><Icon name="copy" size={15} />복사</button>{/if}</div>
</dialog>
