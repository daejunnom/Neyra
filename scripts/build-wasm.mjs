import { execFileSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const out = resolve(root, 'apps/neyra-web/static/wasm');
const bindgenVersion = execFileSync('wasm-bindgen', ['--version'], { encoding: 'utf8' }).trim();
if (bindgenVersion !== 'wasm-bindgen 0.2.104') {
  throw new Error('Expected wasm-bindgen 0.2.104; install the version matching Cargo.lock.');
}
execFileSync('cargo', ['build', '--release', '--locked', '--target', 'wasm32-unknown-unknown', '-p', 'neyra-core'], {
  cwd: root, stdio: 'inherit'
});
mkdirSync(out, { recursive: true });
execFileSync('wasm-bindgen', [
  resolve(root, 'target/wasm32-unknown-unknown/release/neyra_core.wasm'),
  '--target', 'web', '--out-dir', out
], { cwd: root, stdio: 'inherit' });
