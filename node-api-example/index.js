/**
 * RapidSpeech Node.js API Example
 *
 * Demonstrates offline ASR (with optional VAD pre-segmentation) and offline
 * TTS on the WASM build of RapidSpeech. Inference runs locally — no network
 * calls after model load.
 *
 * Usage:
 *   # ASR — whole-clip transcription
 *   node index.js asr -m <asr.gguf> -w <audio.wav> [--two-pass] [--runs N]
 *
 *   # ASR — VAD-segmented (transcribe each detected speech region)
 *   node index.js asr -m <asr.gguf> -w <audio.wav> \
 *       --vad <silero-vad.gguf> [--vad-threshold 0.5] [--vad-min-seg 0.3] \
 *       [--two-pass]
 *
 *   # TTS
 *   node index.js tts -m <tts.gguf> -t "Hello world" -o out.wav
 *       [--instruct "female"] [--lang Chinese] [--seed 42] [--n-steps 16]
 *
 * Prerequisites:
 *   Build the WASM module first:
 *     cd rapidspeech/wasm && ./build-wasm.sh
 */

'use strict';

const fs = require('fs');
const path = require('path');

const { RapidSpeechWASM, RapidSpeechVAD, RS_TASK } = require('../wasm-examples/rapidspeech-bridge.js');

const ASR_VAD_SR = 16000; // VAD always runs at 16 kHz

// ── CLI Argument Parsing ────────────────────────────────────
function parseArgs(argv) {
  const args = {
    mode: null,         // 'asr' | 'tts'
    model: null,
    wav: null,
    text: null,
    output: 'out.wav',
    threads: 2,
    runs: 1,
    twoPass: false,
    ctcPrecheck: false,
    noLlm: false,
    vad: null,
    vadThreshold: 0.5,
    vadMinSeg: 0.3,
    instruct: 'male',
    lang: 'English',
    seed: 42,
    nSteps: 32,
    refWav: null,
    refText: null,
    help: false,
  };

  if (argv[0] && !argv[0].startsWith('-')) {
    args.mode = argv.shift().toLowerCase();
  }

  for (let i = 0; i < argv.length; i++) {
    switch (argv[i]) {
      case '-m': case '--model':    args.model = argv[++i]; break;
      case '-w': case '--wav':      args.wav = argv[++i]; break;
      case '-t': case '--text':     args.text = argv[++i]; break;
      case '-o': case '--output':   args.output = argv[++i]; break;
      case '--threads':             args.threads = parseInt(argv[++i], 10); break;
      case '-r': case '--runs':     args.runs = parseInt(argv[++i], 10); break;
      case '--two-pass':            args.twoPass = true; break;
      case '--no-llm':              args.noLlm = true; break;
      case '--ctc-precheck':        args.ctcPrecheck = true; break;
      case '--vad':                 args.vad = argv[++i]; break;
      case '--vad-threshold':       args.vadThreshold = parseFloat(argv[++i]); break;
      case '--vad-min-seg':         args.vadMinSeg = parseFloat(argv[++i]); break;
      case '--instruct':            args.instruct = argv[++i]; break;
      case '--lang':                args.lang = argv[++i]; break;
      case '--seed':                args.seed = parseInt(argv[++i], 10); break;
      case '--n-steps':             args.nSteps = parseInt(argv[++i], 10); break;
      case '--ref':                 args.refWav = argv[++i]; break;
      case '--ref-text':            args.refText = argv[++i]; break;
      case '-h': case '--help':     args.help = true; break;
      default:
        console.error(`Unknown argument: ${argv[i]}`); args.help = true; break;
    }
  }
  return args;
}

function showHelp() {
  console.log(`
RapidSpeech Node.js API Example (WASM-based, ASR + TTS)

Usage:
  node index.js asr -m <model.gguf> -w <audio.wav> [options]
  node index.js tts -m <model.gguf> -t "text"    [options]

ASR options:
  -w, --wav <path>          Input WAV file (any sample rate; downstream model decides)
  --two-pass                Run CTC greedy then LLM rescore (FunASRNano)
  --no-llm                  Disable the LLM rescoring pass entirely (CTC only)
  --ctc-precheck            Skip LLM on silence using a quick CTC precheck
  -r, --runs <n>            Inference runs for benchmarking (default: 1)

ASR + VAD pre-segmentation:
  --vad <path>              GGUF VAD model (silero-vad / firered-vad)
  --vad-threshold <f>       Speech threshold (default: 0.5)
  --vad-min-seg <s>         Drop segments shorter than this (default: 0.3)

TTS options:
  -t, --text <text>         Text to synthesize
  -o, --output <path>       Output WAV file (default: out.wav)
  --instruct <text>         Voice description (default: "male")
  --lang <lang>             Target language     (default: English)
  --seed <n>                Random seed         (default: 42)
  --n-steps <n>             Diffusion steps     (default: 32)
  --ref <path>              Reference WAV for voice cloning
  --ref-text <text>         Transcript of the reference audio

Common:
  -m, --model <path>        GGUF model path
  --threads <n>             CPU threads (default: 2)
  -h, --help                Show this help
`);
}

// ── WAV reader (8/16/24/32-bit PCM) ─────────────────────────
function readWav(filePath) {
  const buf = fs.readFileSync(filePath);
  if (buf.length < 44) throw new Error('File too small to be WAV');
  if (buf.toString('ascii', 0, 4) !== 'RIFF') throw new Error('Not a valid RIFF file');
  if (buf.toString('ascii', 8, 12) !== 'WAVE') throw new Error('Not a valid WAV file');
  if (buf.readUInt16LE(20) !== 1) throw new Error('Only PCM WAV is supported');

  const numChannels  = buf.readUInt16LE(22);
  const sampleRate   = buf.readUInt32LE(24);
  const bitsPerSample= buf.readUInt16LE(34);

  // Find data chunk
  let offset = 36;
  while (offset + 8 < buf.length) {
    const chunkId   = buf.toString('ascii', offset, offset + 4);
    const chunkSize = buf.readUInt32LE(offset + 4);
    if (chunkId === 'data') {
      const dataOff = offset + 8;
      const bps = bitsPerSample / 8;
      const totalSamples       = Math.floor(chunkSize / bps);
      const samplesPerChannel  = Math.floor(totalSamples / numChannels);
      const pcm = new Float32Array(samplesPerChannel);

      for (let i = 0; i < samplesPerChannel; i++) {
        let s = 0;
        const byteOff = dataOff + i * numChannels * bps;
        switch (bitsPerSample) {
          case 8:  s = buf.readUInt8(byteOff) - 128;  pcm[i] = s / 128.0; break;
          case 16: s = buf.readInt16LE(byteOff);      pcm[i] = s / 32768.0; break;
          case 24: {
            const lo = buf.readUInt8(byteOff);
            const mi = buf.readUInt8(byteOff + 1);
            const hi = buf.readInt8 (byteOff + 2);
            s = (hi << 16) | (mi << 8) | lo;
            pcm[i] = s / 8388608.0;
            break;
          }
          case 32: s = buf.readInt32LE(byteOff);      pcm[i] = s / 2147483648.0; break;
          default: throw new Error(`Unsupported bit depth: ${bitsPerSample}`);
        }
      }
      return { pcm, sampleRate, numChannels, bitsPerSample };
    }
    offset += 8 + chunkSize;
  }
  throw new Error('No data chunk found in WAV file');
}

// Linear-interpolation resampler — good enough for VAD's 16 kHz feed.
function resampleLinear(pcm, srcSr, dstSr) {
  if (srcSr === dstSr) return pcm;
  const ratio = srcSr / dstSr;
  const nOut = Math.ceil(pcm.length / ratio);
  const out = new Float32Array(nOut);
  for (let i = 0; i < nOut; ++i) {
    const x = i * ratio;
    const i0 = Math.floor(x);
    const i1 = Math.min(pcm.length - 1, i0 + 1);
    const t = x - i0;
    out[i] = pcm[i0] * (1 - t) + pcm[i1] * t;
  }
  return out;
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = s - 60 * m;
  return `${String(m).padStart(2, '0')}:${sec.toFixed(2).padStart(5, '0')}`;
}

// ── WAV writer (16-bit mono PCM) ────────────────────────────
function writeWavMono16(filePath, pcm, sampleRate) {
  const numSamples = pcm.length;
  const byteRate   = sampleRate * 2;
  const dataSize   = numSamples * 2;
  const buf = Buffer.alloc(44 + dataSize);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1,  20);    // PCM
  buf.writeUInt16LE(1,  22);    // mono
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(byteRate,   28);
  buf.writeUInt16LE(2,  32);    // block align
  buf.writeUInt16LE(16, 34);    // bits/sample
  buf.write('data', 36);
  buf.writeUInt32LE(dataSize, 40);
  for (let i = 0; i < numSamples; i++) {
    const v = Math.max(-1, Math.min(1, pcm[i]));
    buf.writeInt16LE((v * 32767) | 0, 44 + i * 2);
  }
  fs.writeFileSync(filePath, buf);
}

// ── Load WASM module ────────────────────────────────────────
async function loadWasmModule() {
  const searchPaths = [
    path.join(__dirname, '..', 'wasm-examples', 'rapidspeech-wasm.js'),
    path.join(__dirname, '..', 'rapidspeech', 'wasm', 'build', 'rapidspeech-wasm.js'),
  ];
  const wasmJsPath = searchPaths.find((p) => fs.existsSync(p));
  if (!wasmJsPath) {
    console.error('WASM module not found. Build it first:');
    console.error('  cd rapidspeech/wasm && ./build-wasm.sh');
    console.error('Searched:'); searchPaths.forEach((p) => console.error('  ' + p));
    process.exit(1);
  }
  console.log(`WASM:    ${wasmJsPath}`);
  const Module = require(wasmJsPath);
  return await Module();
}

// ── Transcribe a single clip (with optional 2-pass) ─────────
async function transcribeClip(rs, pcm, { twoPass, noLlm }) {
  rs.reset();
  rs.pushAudio(pcm);

  if (twoPass) {
    rs.setUseLlm(false);
    const t0 = performance.now();
    const ctc = await rs.process();
    const tCtc = (performance.now() - t0) / 1000;

    rs.setUseLlm(true);
    const t1 = performance.now();
    const llm = await rs.redecode();
    const tLlm = (performance.now() - t1) / 1000;

    return { ctc: ctc.text, llm: llm.text, tCtc, tLlm };
  }

  if (noLlm) rs.setUseLlm(false);
  const t0 = performance.now();
  const r = await rs.process();
  const t = (performance.now() - t0) / 1000;
  return { ctc: null, llm: null, single: r.text, tSingle: t };
}

// ── ASR runner ──────────────────────────────────────────────
async function runAsr(args) {
  const wav = readWav(args.wav);
  const duration = wav.pcm.length / wav.sampleRate;
  console.log(`WAV:     ${args.wav} (${wav.sampleRate} Hz, ${wav.numChannels} ch, ${wav.bitsPerSample}-bit, ${duration.toFixed(2)}s)`);

  const mod = await loadWasmModule();
  const rs  = new RapidSpeechWASM(mod);

  const modelData = fs.readFileSync(args.model);
  await rs.initAsr(modelData, args.threads);
  console.log(`Arch:    ${rs.archName}  SR=${rs.sampleRate} Hz  version=${rs.version}`);

  if (args.ctcPrecheck) rs.setCtcPrecheck(true);
  if (args.noLlm)       rs.setUseLlm(false);

  // Resample to the ASR model's native rate.
  const targetSr = rs.sampleRate;
  const asrPcm = wav.sampleRate === targetSr
    ? wav.pcm
    : resampleLinear(wav.pcm, wav.sampleRate, targetSr);
  if (wav.sampleRate !== targetSr) {
    console.log(`Resampled: ${wav.sampleRate} → ${targetSr} Hz`);
  }

  // ── VAD-driven path: detect segments, transcribe each ──────
  if (args.vad) {
    const vad = new RapidSpeechVAD(mod);
    const vadData = fs.readFileSync(args.vad);
    await vad.load(vadData, Math.max(2, (args.threads / 2) | 0));
    vad.setThreshold(args.vadThreshold);
    console.log(`VAD:     ${vad.arch}  threshold=${args.vadThreshold}`);

    // VAD always runs at 16 kHz. Build a parallel 16 k feed if needed.
    const vadPcm = targetSr === ASR_VAD_SR
      ? asrPcm
      : resampleLinear(wav.pcm, wav.sampleRate, ASR_VAD_SR);
    vad.reset();
    vad.pushAudio(vadPcm);
    const segments = vad.drainSegments(1024)
      .filter((s) => s.end_s - s.start_s >= args.vadMinSeg);
    console.log(`VAD:     ${segments.length} segment(s) ≥ ${args.vadMinSeg}s`);

    if (segments.length === 0) {
      console.log('  (no speech detected by VAD)');
      vad.free();
      rs.free();
      return;
    }

    let totalT = 0;
    for (const seg of segments) {
      const s0 = Math.max(0, Math.floor(seg.start_s * targetSr));
      const s1 = Math.min(asrPcm.length, Math.floor(seg.end_s * targetSr));
      if (s1 <= s0) continue;
      const segPcm = asrPcm.subarray(s0, s1);
      const segDur = (s1 - s0) / targetSr;
      const tag = `[${fmtTime(seg.start_s)} → ${fmtTime(seg.end_s)} | ${segDur.toFixed(2)}s]`;
      const r = await transcribeClip(rs, segPcm, { twoPass: args.twoPass, noLlm: args.noLlm });
      if (args.twoPass) {
        totalT += r.tCtc + r.tLlm;
        console.log(`${tag} CTC: ${r.ctc || '(no speech)'}`);
        console.log(`${' '.repeat(tag.length)} LLM: ${r.llm || '(no speech)'}`);
      } else {
        totalT += r.tSingle;
        console.log(`${tag} ${r.single || '(no speech)'}`);
      }
    }
    console.log(`\nTotal inference: ${totalT.toFixed(2)}s, audio: ${duration.toFixed(2)}s, RTF=${(totalT/duration).toFixed(3)}`);
    vad.free();
    rs.free();
    return;
  }

  // ── Whole-clip path ─────────────────────────────────────────
  const times1 = [];
  const times2 = [];
  for (let r = 0; r < args.runs; r++) {
    const res = await transcribeClip(rs, asrPcm, { twoPass: args.twoPass, noLlm: args.noLlm });

    if (args.twoPass) {
      times1.push(res.tCtc);
      times2.push(res.tLlm);
    } else {
      times1.push(res.tSingle);
    }

    if (args.runs === 1) {
      if (args.twoPass) {
        console.log(`\n  CTC :  ${res.ctc || '(no speech)'}`);
        console.log(`  LLM :  ${res.llm || '(no speech)'}\n`);
      } else {
        console.log(`\n  Result: ${res.single || '(no speech)'}\n`);
      }
    } else {
      const t = args.twoPass ? res.tCtc : res.tSingle;
      const rtf = t / duration;
      const out = args.twoPass ? (res.llm || res.ctc || '(no speech)') : (res.single || '(no speech)');
      console.log(`  Run ${r+1}: ${t.toFixed(3)}s RTF=${rtf.toFixed(3)} ${out}`);
    }
  }

  if (args.runs > 1) {
    const avg1 = times1.reduce((a,b)=>a+b,0) / times1.length;
    console.log(`\n  ${args.twoPass ? 'CTC' : 'Single'} avg: ${avg1.toFixed(3)}s  RTF=${(avg1/duration).toFixed(3)}`);
    if (times2.length) {
      const avg2 = times2.reduce((a,b)=>a+b,0) / times2.length;
      console.log(`  LLM avg:    ${avg2.toFixed(3)}s  RTF=${(avg2/duration).toFixed(3)}`);
    }
  }

  rs.free();
}

// ── TTS runner ──────────────────────────────────────────────
async function runTts(args) {
  const mod = await loadWasmModule();
  const rs  = new RapidSpeechWASM(mod);

  const modelData = fs.readFileSync(args.model);
  await rs.initTts(modelData, args.threads);
  console.log(`Arch:    ${rs.archName}  SR=${rs.sampleRate} Hz  version=${rs.version}`);

  rs.setTtsParams({ instruct: args.instruct, language: args.lang, seed: args.seed });
  rs.setDiffusionSteps(args.nSteps);
  console.log(`Params:  instruct="${args.instruct}" lang=${args.lang} seed=${args.seed} steps=${args.nSteps}`);

  if (args.refWav) {
    if (!args.refText) { console.error('--ref requires --ref-text'); process.exit(1); }
    const ref = readWav(args.refWav);
    rs.pushReferenceAudio(ref.pcm, ref.sampleRate);
    rs.pushReferenceText(args.refText);
    console.log(`Cloning: ${args.refWav} (${ref.sampleRate} Hz, ${(ref.pcm.length/ref.sampleRate).toFixed(2)}s)`);
  }

  console.log(`Text:    ${JSON.stringify(args.text)}`);
  const t0 = performance.now();
  const pcm = await rs.synthesize(args.text);
  const elapsed = (performance.now() - t0) / 1000;
  const duration = pcm.length / rs.sampleRate;
  const rtf = elapsed / Math.max(duration, 1e-9);
  console.log(`Output:  ${pcm.length} samples @ ${rs.sampleRate} Hz (${duration.toFixed(2)}s, elapsed=${elapsed.toFixed(2)}s, RTF=${rtf.toFixed(3)})`);

  writeWavMono16(args.output, pcm, rs.sampleRate);
  console.log(`Wrote    ${args.output}`);

  rs.free();
}

// ── Main ───────────────────────────────────────────────────
async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help || !args.mode || !args.model
      || (args.mode === 'asr' && !args.wav)
      || (args.mode === 'tts' && !args.text)) {
    if (!args.help) console.error('Error: missing required arguments.\n');
    showHelp();
    process.exit(args.help ? 0 : 1);
  }

  if (args.twoPass && args.noLlm) {
    console.error('--two-pass and --no-llm are mutually exclusive'); process.exit(1);
  }

  if (!fs.existsSync(args.model)) {
    console.error(`Model not found: ${args.model}`); process.exit(1);
  }
  if (args.vad && !fs.existsSync(args.vad)) {
    console.error(`VAD model not found: ${args.vad}`); process.exit(1);
  }

  console.log('=== RapidSpeech Node.js API Example ===\n');
  if (args.mode === 'asr')      await runAsr(args);
  else if (args.mode === 'tts') await runTts(args);
  else { console.error(`Unknown mode: ${args.mode}`); showHelp(); process.exit(1); }

  console.log('Done.');
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
