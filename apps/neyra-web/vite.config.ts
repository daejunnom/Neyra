import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [sveltekit()],
  worker: { format: 'es' },
  server: { port: 4173, strictPort: true },
  preview: { port: 4173, strictPort: true },
  build: { target: 'es2022' }
});
