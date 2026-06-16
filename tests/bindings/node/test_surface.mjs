// Node binding smoke test (no models needed).
//
// Verifies:
//   - the bridge module loads from a Node context
//   - all three classes are exported with the documented method names
//   - RS_TASK enum has the documented members
//   - the WASM .js loader path exists next to the bridge
//
// Run:  node tests/bindings/node/test_surface.mjs

import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..', '..');

const fails = [];
function check(name, cond, hint = '') {
  if (cond) {
    console.log(`  ok  ${name}`);
  } else {
    fails.push(`${name}${hint ? ' — ' + hint : ''}`);
    console.log(`  FAIL ${name}${hint ? ' — ' + hint : ''}`);
  }
}

// ── load bridge ──────────────────────────────────────────────────────
const bridgePath = path.join(repoRoot, 'wasm-examples', 'rapidspeech-bridge.js');
check('bridge file exists', existsSync(bridgePath), bridgePath);

const wasmJsPath = path.join(repoRoot, 'wasm-examples', 'rapidspeech-wasm.js');
const wasmBin = path.join(repoRoot, 'wasm-examples', 'rapidspeech-wasm.wasm');
check('rapidspeech-wasm.js present', existsSync(wasmJsPath));
check('rapidspeech-wasm.wasm present', existsSync(wasmBin));

const mod = require(bridgePath);
check('module.exports has RapidSpeechWASM', typeof mod.RapidSpeechWASM === 'function');
check('module.exports has RapidSpeechVAD', typeof mod.RapidSpeechVAD === 'function');
check('module.exports has RapidSpeechKWS', typeof mod.RapidSpeechKWS === 'function');
check('module.exports has RS_TASK', typeof mod.RS_TASK === 'object' && mod.RS_TASK !== null);

// ── RS_TASK enum ────────────────────────────────────────────────────
const taskKeys = ['ASR_OFFLINE', 'ASR_ONLINE', 'TTS_OFFLINE', 'TTS_ONLINE'];
for (const k of taskKeys) {
  check(`RS_TASK.${k} defined`, typeof mod.RS_TASK?.[k] === 'number');
}

// ── instance methods (without invoking WASM) ────────────────────────
function requireMethods(Cls, names) {
  const proto = Cls.prototype;
  for (const n of names) {
    check(`${Cls.name}.prototype.${n}`, typeof proto?.[n] === 'function');
  }
}

requireMethods(mod.RapidSpeechWASM, [
  'initAsr', 'initTts', 'free',
  'pushAudio', 'pushText',
  'pushReferenceAudio', 'pushReferenceText',
  'reset',
]);
requireMethods(mod.RapidSpeechVAD, [
  'free', 'reset', 'setThreshold', 'pushAudio',
  'drainSegments', 'drainFrames',
]);
requireMethods(mod.RapidSpeechKWS, [
  'free', 'reset', 'pushAudio',
]);

// ── summary ─────────────────────────────────────────────────────────
if (fails.length) {
  console.error(`\nFAIL — ${fails.length} check(s) failed:`);
  for (const f of fails) console.error(`  - ${f}`);
  process.exit(1);
}
console.log('\nAll surface checks passed.');
