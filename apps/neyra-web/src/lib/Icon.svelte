<script lang="ts">
  export let name = 'play';
  export let size = 18;
  const paths: Record<string, string> = {
    play: 'M8 5v14l11-7z',
    pause: 'M8 5v14M16 5v14',
    stop: 'M6 6h12v12H6z',
    settings: 'M4 7h16M4 17h16M8 4v6M16 14v6',
    download: 'M12 3v12m-5-5 5 5 5-5M5 16v4h14v-4',
    arrow: 'm9 5 7 7-7 7',
    back: 'm15 5-7 7 7 7',
    close: 'm6 6 12 12M18 6 6 18',
    check: 'm5 12 4 4L19 6',
    external: 'M14 3h7v7m0-7L10 14M10 5H4v15h15v-6',
    restart: 'M4 10a8 8 0 1 1 1 8M4 4v6h6',
    info: 'M12 11v6M12 7h.01',
    clock: 'M12 7v5l3 2',
    spark: 'm12 3 2.7 6.3L21 12l-6.3 2.7L12 21l-2.7-6.3L3 12l6.3-2.7z'
  };
</script>

<svg width={size} height={size} viewBox="0 0 24 24" fill={name === 'play' || name === 'stop' ? 'currentColor' : 'none'} stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
  {#if name === 'info' || name === 'clock'}<circle cx="12" cy="12" r="9" />{/if}
  <path d={paths[name] || paths.play} />
</svg>
