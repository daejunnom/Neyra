// Adapted from Clearra apps/clearra-web/svelte.config.js (MIT).
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

const base = process.env.NEYRA_BASE_PATH || '';
if (base && (!base.startsWith('/') || base.endsWith('/') || base.includes('..'))) {
  throw new Error('NEYRA_BASE_PATH must be empty or an absolute path without a trailing slash');
}

export default {
  preprocess: vitePreprocess(),
  kit: {
    adapter: adapter({ pages: 'build', assets: 'build', strict: true }),
    paths: { base, relative: false }
  }
};
