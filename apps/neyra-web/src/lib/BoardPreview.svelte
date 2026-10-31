<!--
  Adapted from Clearra SolutionBoardPreview.svelte (MIT).
  Copyright (c) 2026 Clearra contributors.
  Upstream: 5c98563332706a718330bc7ae8e403d2583afe9d.
  Grid layout and cell semantics retained; 24-row mask view added.
-->
<script lang="ts">
  export let rows: number[] = [];
  export let ariaLabel = '탐색 필드';
  export let empty = false;
  $: cells = Array.from({ length: 240 }, (_, index) => {
    const y = 23 - Math.floor(index / 10);
    const x = index % 10;
    return Boolean((rows[y] || 0) & (1 << x));
  });
</script>

<div class="preview">
  <div class="row-labels" aria-hidden="true"><span>24</span><span>20</span><span>10</span><span>01</span></div>
  <div class="solution-board" role="img" aria-label={ariaLabel}>
    {#each cells as filled, index}
      <span class="cell" class:empty={!filled} class:filled class:buffer={index < 40} class:boundary={index >= 40 && index < 50} aria-hidden="true"></span>
    {/each}
    {#if empty}
      <div class="board-placeholder"><span class="placeholder-mino"><i></i><i></i><i></i><i></i></span><span>새로운 빌드의 시작</span></div>
    {/if}
  </div>
</div>

<style>
  .preview { position:relative; width:100%; padding-left:22px; }
  .solution-board { aspect-ratio:10/24; background:#111e1c; border:1px solid #43534d; border-radius:5px; display:grid; grid-template-columns:repeat(10,minmax(0,1fr)); grid-template-rows:repeat(24,minmax(0,1fr)); overflow:hidden; width:100%; position:relative; }
  .cell { min-height:0; min-width:0; box-shadow:inset 0 0 0 .5px #ffffff06; }
  .empty { background:#14231f; }
  .filled { background:#96c7b5; box-shadow:inset 0 0 0 .8px #152b23, inset 0 2px 0 #ffffff29; }
  .buffer.empty { background:#0f1b18; }
  .boundary { border-top:1px dashed #557468; }
  .row-labels { position:absolute; left:0; top:0; bottom:0; width:16px; color:#61776d; font-size:9px; font-variant-numeric:tabular-nums; }
  .row-labels span { position:absolute; right:0; } .row-labels span:nth-child(1){top:0}.row-labels span:nth-child(2){top:16.666%}.row-labels span:nth-child(3){top:58.333%}.row-labels span:nth-child(4){bottom:0}
  .board-placeholder { position:absolute; inset:0; display:flex; align-items:center; justify-content:center; flex-direction:column; gap:16px; color:#718c7e; font-size:11px; background:linear-gradient(transparent,#10221c22); pointer-events:none; }
  .placeholder-mino { display:grid; width:53px; grid-template-columns:repeat(3,1fr); grid-template-rows:repeat(2,16px); gap:2px; opacity:.6; }
  .placeholder-mino i { background:#527364; border:1px solid #7ca99066; border-radius:2px; }
  .placeholder-mino i:first-child { grid-column:2; } .placeholder-mino i:nth-child(2){grid-row:2}
</style>
