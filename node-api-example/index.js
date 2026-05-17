/**
 * RapidSpeech Node.js API Example
 *
 * Transcribes a WAV file using a GGUF ASR model loaded via WebAssembly.
 * All inference runs locally — no network calls after model load.
 *
 * Usage:
 *   node index.js --model /path/to/model.gguf --wav /path/to/audio.wav
 *   node index.js -m model.gguf -w audio.wav --threads 4 --runs 5
 *
 * Prerequisites:
 *   Build the WASM module first:
 *     cd rapidspeech/wasm && ./build-wasm.sh
 *   Then copy the build outputs, or run from the project root so that
 *   require('../wasm-examples/rapidspeech-wasm.js') resolves correctly.
 */

'use strict';

const fs = require('fs');
const path = require('path');

// ── CLI Argument Parsing ────────────────────────────────────
function parseArgs() {
  const args = {
    model: null,
    wav: null,
    threads: 2,
    runs: 1,
    help: false,
  };

  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case '-m':
      case '--model':   args.model = argv[++i]; break;
      case '-w':
      case '--wav':     args.wav = argv[++i]; break;
      case '-t':
      case '--threads': args.threads = parseInt(argv[++i], 10); break;
      case '-r':
      case '--runs':    args.runs = parseInt(argv[++i], 10); break;
      case '-h':
      case '--help':    args.help = true; break;
    }
  }
  return args;
}

function showHelp() {
  console.log(`
RapidSpeech Node.js API Example

Usage:
  node index.js -m <model.gguf> -w <audio.wav> [options]

Required:
  -m, --model <path>    Path to GGUF ASR model
  -w, --wav <path>      Path to 16kHz mono WAV file

Options:
  -t, --threads <n>     Number of CPU threads (default: 2)
  -r, --runs <n>        Benchmark runs (default: 1)
  -h, --help            Show this help

Examples:
  node index.js -m sense-voice-small.gguf -w test.wav
  node index.js -m model.gguf -w audio.wav --threads 4 --runs 10
`);
}

// ── WAV Reader (PCM, 8/16/24/32-bit) ────────────────────────
function readWav(filePath) {
  const buf = fs.readFileSync(filePath);
  if (buf.length < 44) throw new Error('File too small to be WAV');

  if (buf.toString('ascii', 0, 4) !== 'RIFF') throw new Error('Not a valid RIFF file');
  if (buf.toString('ascii', 8, 12) !== 'WAVE') throw new Error('Not a valid WAV file');

  const audioFormat = buf.readUInt16LE(20);
  if (audioFormat !== 1) throw new Error('Only PCM WAV files are supported');

  const numChannels = buf.readUInt16LE(22);
  const sampleRate = buf.readUInt32LE(24);
  const bitsPerSample = buf.readUInt16LE(34);

  // Find data chunk
  let offset = 36;
  while (offset + 8 < buf.length) {
    const chunkId = buf.toString('ascii', offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);
    if (chunkId === 'data') {
      const dataOffset = offset + 8;
      const bytesPerSample = bitsPerSample / 8;
      const totalSamples = Math.floor(chunkSize / bytesPerSample);
      const samplesPerChannel = Math.floor(totalSamples / numChannels);

      const pcm = new Float32Array(samplesPerChannel);

      for (let i = 0; i < samplesPerChannel; i++) {
        let sample = 0;
        const byteOff = dataOffset + i * numChannels * bytesPerSample;

        if (bitsPerSample === 8) {
          sample = buf.readUInt8(byteOff) - 128;
          pcm[i] = sample / 128.0;
        } else if (bitsPerSample === 16) {
          sample = buf.readInt16LE(byteOff);
          pcm[i] = sample / 32768.0;
        } else if (bitsPerSample === 24) {
          // 24-bit signed in little-endian
          const lo = buf.readUInt8(byteOff);
          const mi = buf.readUInt8(byteOff + 1);
          const hi = buf.readInt8(byteOff + 2);
          sample = (hi << 16) | (mi << 8) | lo;
          pcm[i] = sample / 8388608.0;
        } else if (bitsPerSample === 32) {
          sample = buf.readInt32LE(byteOff);
          pcm[i] = sample / 2147483648.0;
        } else {
          throw new Error(`Unsupported bit depth: ${bitsPerSample}`);
        }
      }

      return { pcm, sampleRate, numChannels, bitsPerSample };
    }
    offset += 8 + chunkSize;
  }

  throw new Error('No data chunk found in WAV file');
}

// ── Load WASM module ────────────────────────────────────────
async function loadWasmModule() {
  const searchPaths = [
    path.join(__dirname, '..', 'wasm-examples', 'rapidspeech-wasm.js'),
    path.join(__dirname, '..', 'rapidspeech', 'wasm', 'build', 'rapidspeech-wasm.js'),
  ];

  let wasmJsPath = null;
  for (const p of searchPaths) {
    if (fs.existsSync(p)) { wasmJsPath = p; break; }
  }

  if (!wasmJsPath) {
    console.error('WASM module not found. Build it first:');
    console.error('  cd rapidspeech/wasm && ./build-wasm.sh');
    console.error('Searched:');
    searchPaths.forEach(p => console.error(`  ${p}`));
    process.exit(1);
  }

  console.log(`WASM:   ${wasmJsPath}`);
  const RapidSpeechModule = require(wasmJsPath);
  console.log('Initializing WASM runtime...');
  const module = await RapidSpeechModule();
  const version = module.ccall('rs_wasm_get_version', 'string', [], []);
  console.log(`Version: ${version}`);
  return module;
}

// ── Main ────────────────────────────────────────────────────
async function main() {
  const args = parseArgs();
  if (args.help || !args.model || !args.wav) {
    if (!args.help) console.error('Error: --model and --wav are required.\n');
    showHelp();
    process.exit(args.help ? 0 : 1);
  }

  const modelPath = path.resolve(args.model);
  const wavPath = path.resolve(args.wav);

  if (!fs.existsSync(modelPath)) { console.error(`Model not found: ${modelPath}`); process.exit(1); }
  if (!fs.existsSync(wavPath)) { console.error(`WAV not found: ${wavPath}`); process.exit(1); }

  console.log('=== RapidSpeech Node.js API Example ===\n');

  // ── 1. Read WAV ───────────────────────────────────────────
  console.log(`WAV:    ${wavPath}`);
  let wav;
  try { wav = readWav(wavPath); }
  catch (err) { console.error(`Failed to read WAV: ${err.message}`); process.exit(1); }

  const audioDuration = wav.pcm.length / wav.sampleRate;
  console.log(`        ${wav.sampleRate} Hz, ${wav.numChannels} ch, ${wav.bitsPerSample}-bit`);
  console.log(`        ${wav.pcm.length} samples, ${audioDuration.toFixed(2)} s\n`);

  // ── 2. Load WASM ──────────────────────────────────────────
  const module = await loadWasmModule();

  // ── 3. Load model ─────────────────────────────────────────
  console.log(`\nModel:  ${modelPath}`);
  console.log('Loading model into WASM...');

  const modelData = fs.readFileSync(modelPath);
  const modelName = '/model.gguf';
  module.FS.writeFile(modelName, modelData);

  const initRet = module.ccall('rs_wasm_init', 'number', ['string', 'number'],
                                [modelName, args.threads]);
  if (initRet !== 0) {
    console.error('Failed to initialize model.');
    module.ccall('rs_wasm_free', null, [], []);
    process.exit(1);
  }

  const archName = module.ccall('rs_wasm_get_arch_name', 'string', [], []);
  const sampleRate = module.ccall('rs_wasm_get_sample_rate', 'number', [], []);
  console.log(`        Architecture: ${archName}, Sample rate: ${sampleRate} Hz\n`);

  // ── 4. Run ASR ────────────────────────────────────────────
  console.log(`Running ASR (${args.runs} run(s))...`);

  // Prepare audio on WASM heap once (reused for benchmarks)
  const nSamples = wav.pcm.length;
  const ptr = module._malloc(nSamples * 4);
  module.HEAPF32.set(wav.pcm, ptr / 4);

  const times = [];

  for (let r = 0; r < args.runs; r++) {
    module.ccall('rs_wasm_reset', null, [], []);
    module.ccall('rs_wasm_push_audio', 'number', ['number', 'number'], [ptr, nSamples]);

    const t0 = performance.now();
    const status = module.ccall('rs_wasm_process', 'number', [], []);
    const elapsed = (performance.now() - t0) / 1000;
    times.push(elapsed);

    if (status > 0) {
      const text = module.ccall('rs_wasm_get_text', 'string', [], []);
      if (args.runs === 1) {
        console.log(`\n──────────────────────────────────────────────────────────`);
        console.log(` ${text || '(no speech recognized)'}`);
        console.log(`──────────────────────────────────────────────────────────\n`);
      } else {
        const rtf = elapsed / audioDuration;
        console.log(`  Run ${r+1}: ${elapsed.toFixed(3)}s  RTF=${rtf.toFixed(3)}  ${text || '(no speech)'}`);
      }
    } else {
      console.log(`  Run ${r+1}: no output`);
    }
  }

  module._free(ptr);

  if (args.runs > 1) {
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const min = Math.min(...times);
    const max = Math.max(...times);
    console.log(`\n  Avg: ${avg.toFixed(3)}s  Min: ${min.toFixed(3)}s  Max: ${max.toFixed(3)}s  RTF=${(avg/audioDuration).toFixed(3)}`);
  }

  console.log(`Elapsed: ${times.reduce((a,b)=>a+b,0).toFixed(2)}s | Audio: ${audioDuration.toFixed(2)}s`);

  // ── 5. Cleanup ────────────────────────────────────────────
  module.ccall('rs_wasm_free', null, [], []);
  console.log('Done.');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
