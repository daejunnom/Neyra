<!-- Adapted from Clearra WorkspaceProgressStatus.svelte, MIT, 5c98563332706a718330bc7ae8e403d2583afe9d. -->
<script lang="ts">
  export let active = false;
  export let verifying = false;
  export let checked = 0;
  export let label = '준비됨';
  $: percent = Math.max(0, Math.min(100, checked / 5040 * 100));
</script>

<div class="progress-status">
  <div class="overall"><div><span class:active class="status-dot"></span><strong>{label}</strong></div><span>{verifying ? checked.toLocaleString('ko-KR') + ' / 5,040' : active ? '계속 탐색 중' : ''}</span></div>
  {#if verifying}
    <div class="track" role="progressbar" aria-label="현재 가방의 순열 검증" aria-valuemin="0" aria-valuemax="5040" aria-valuenow={checked}><span style={'width:' + percent + '%'}></span></div>
  {:else}
    <div class="track" class:indeterminate={active} aria-hidden="true"><span></span></div>
  {/if}
</div>

<style>
  .progress-status { display:grid; gap:12px; }
  .overall { display:flex; align-items:center; justify-content:space-between; gap:10px; }
  .overall>div { display:flex; align-items:center; gap:8px; min-width:0; }
  .overall strong { font-size:12px; font-weight:650; color:#d4e5dc; }
  .overall>span { font-size:10px; color:#8a9d92; font-variant-numeric:tabular-nums; white-space:nowrap; }
  .status-dot { width:6px; height:6px; border-radius:50%; background:#587065; flex-shrink:0; }
  .status-dot.active { background:#b5eb89; box-shadow:0 0 9px #b5eb8938; }
  .track { height:3px; overflow:hidden; background:#2a3d32; border-radius:4px; position:relative; }
  .track>span { background:#b5eb89; height:100%; display:block; width:0; transition:width 120ms linear; border-radius:inherit; }
  .track.indeterminate>span { animation:progress 2s ease-in-out infinite; width:35%; position:absolute; left:0; }
  @keyframes progress { from{transform:translateX(-120%)}to{transform:translateX(390%)} }
  @media(prefers-reduced-motion:reduce){.track.indeterminate>span{animation:none;left:32%}}
</style>
