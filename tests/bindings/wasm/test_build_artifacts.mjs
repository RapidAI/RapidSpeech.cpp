// WASM build-product sanity test (no instantiation).
//
// Just checks that build-wasm.sh produced the three artifacts and that
// the .js loader exports a Module / createRapidSpeechModule symbol that
// loaders downstream rely on.

import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..', '..');
const wasmDir = path.join(repoRoot, 'wasm-examples');

const required = ['rapidspeech-wasm.js', 'rapidspeech-wasm.wasm'];
let ok = true;
for (const f of required) {
  const p = path.join(wasmDir, f);
  if (!existsSync(p)) {
    console.error(`FAIL: missing ${p}`);
    ok = false;
  } else {
    console.log(`  ok  ${f}`);
  }
}

const js = readFileSync(path.join(wasmDir, 'rapidspeech-wasm.js'), 'utf8');
// Emscripten outputs a factory; existence of either symbol is fine.
const factoryFound =
  /createRapidSpeechModule|var Module|export default \w+/.test(js);
if (!factoryFound) {
  console.error('FAIL: rapidspeech-wasm.js has no recognizable Emscripten entry point');
  ok = false;
} else {
  console.log('  ok  rapidspeech-wasm.js exports an Emscripten factory');
}

if (!ok) process.exit(1);
console.log('\nWASM build artifacts look sane.');
