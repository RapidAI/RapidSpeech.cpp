/**
 * RapidSpeech WASM JavaScript bridge.
 *
 * Wraps the Emscripten-compiled rapidspeech-wasm module with a high-level JS
 * API. Supports both ASR (push PCM → text) and TTS (push text → PCM chunks).
 *
 * Usage (browser):
 *   const mod = await RapidSpeechModule();
 *   const asr = new RapidSpeechWASM(mod);
 *   await asr.initAsr(modelUrl);
 *   asr.pushAudio(float32Pcm);
 *   const { text } = asr.process();
 *
 *   const tts = new RapidSpeechWASM(mod);
 *   await tts.initTts(modelUrl);
 *   tts.setTtsParams({ instruct: 'female', language: 'English', seed: 42 });
 *   const pcm = tts.synthesize('Hello world');  // Float32Array @ tts.sampleRate
 */

// Mirror of rs_task_type_t in rapidspeech.h
const RS_TASK = Object.freeze({
  ASR_OFFLINE: 0,
  ASR_ONLINE:  1,
  TTS_OFFLINE: 2,
  TTS_ONLINE:  3,
  E2E_SPEECH_LLM: 4,
});

class RapidSpeechWASM {
  constructor(module) {
    this._mod = module;
    this._ready = false;

    // ── Bind C functions (cwrap = lighter than ccall on hot paths) ──
    // Functions that may suspend through WebGPU's async device/buffer ops are
    // bound with {async: true} so cwrap returns a Promise instead of dropping
    // the result when ASYNCIFY unwinds. Always await these.
    this._initEx        = module.cwrap('rs_wasm_init_ex', 'number', ['string', 'number', 'number'], { async: true });
    this._init          = module.cwrap('rs_wasm_init',    'number', ['string', 'number'],           { async: true });
    this._free          = module.cwrap('rs_wasm_free',    null,     []);

    this._pushAudio     = module.cwrap('rs_wasm_push_audio',          'number', ['number', 'number']);
    this._pushText      = module.cwrap('rs_wasm_push_text',           'number', ['string']);
    this._pushRefAudio  = module.cwrap('rs_wasm_push_reference_audio','number', ['number', 'number', 'number']);
    this._pushRefText   = module.cwrap('rs_wasm_push_reference_text', 'number', ['string']);

    this._process       = module.cwrap('rs_wasm_process',      'number', [], { async: true });
    this._redecode      = module.cwrap('rs_wasm_redecode',     'number', [], { async: true });
    this._getText       = module.cwrap('rs_wasm_get_text',     'string', []);
    this._getAudioPtr   = module.cwrap('rs_wasm_get_audio_ptr','number', []);
    this._getAudioLen   = module.cwrap('rs_wasm_get_audio_len','number', []);
    this._reset         = module.cwrap('rs_wasm_reset',        'number', []);

    this._setPrompt     = module.cwrap('rs_wasm_set_user_input_prompt',  'number', ['string']);
    this._setUseLlm     = module.cwrap('rs_wasm_set_use_llm',            'number', ['number']);
    this._setCtcPrecheck= module.cwrap('rs_wasm_set_ctc_precheck',       'number', ['number']);
    this._setTtsParams  = module.cwrap('rs_wasm_set_tts_params',         'number', ['string','string','number']);
    this._setTtsSteps   = module.cwrap('rs_wasm_set_tts_diffusion_steps','number', ['number']);

    this._getSampleRate = module.cwrap('rs_wasm_get_sample_rate', 'number', []);
    this._getArchName   = module.cwrap('rs_wasm_get_arch_name',   'string', []);
    this._isReady       = module.cwrap('rs_wasm_is_ready',        'number', []);
    this._getVersion    = module.cwrap('rs_wasm_get_version',     'string', []);

    this._sampleRate = 16000;
    this._taskType   = RS_TASK.ASR_OFFLINE;
  }

  // ── Model loading ──────────────────────────────────────────
  /**
   * Generic init: fetch model from URL, write to /model.gguf, init context.
   * @param {string}   modelUrl
   * @param {number}   taskType   one of RS_TASK.*
   * @param {number}   nThreads
   * @param {function} onProgress(fraction, {loaded, total})
   */
  async init(modelUrl, taskType = RS_TASK.ASR_OFFLINE, nThreads = 2, onProgress = null) {
    const fileName = '/model.gguf';

    let buffer;
    if (modelUrl instanceof Uint8Array || modelUrl instanceof ArrayBuffer) {
      buffer = modelUrl instanceof ArrayBuffer ? new Uint8Array(modelUrl) : modelUrl;
    } else {
      const response = await fetch(modelUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
      }
      const total = parseInt(response.headers.get('Content-Length') || '0', 10);
      const reader = response.body.getReader();
      const chunks = [];
      let loaded = 0;
      if (onProgress) onProgress(0, { loaded: 0, total });
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        if (onProgress) {
          const frac = total > 0 ? Math.min(loaded / total, 1.0) : 0;
          onProgress(frac, { loaded, total });
        }
      }
      buffer = new Uint8Array(loaded);
      let off = 0;
      for (const c of chunks) { buffer.set(c, off); off += c.length; }
      if (onProgress) onProgress(1, { loaded, total: total || loaded });
    }

    this._mod.FS.writeFile(fileName, buffer);

    const ret = await this._initEx(fileName, taskType, nThreads);
    if (ret !== 0) throw new Error('rs_wasm_init_ex failed with code ' + ret);

    this._sampleRate = this._getSampleRate();
    this._taskType   = taskType;
    this._ready = true;
  }

  initAsr(modelUrl, nThreads = 2, onProgress = null) {
    return this.init(modelUrl, RS_TASK.ASR_OFFLINE, nThreads, onProgress);
  }

  initTts(modelUrl, nThreads = 2, onProgress = null) {
    return this.init(modelUrl, RS_TASK.TTS_ONLINE, nThreads, onProgress);
  }

  free() {
    if (this._ready) {
      this._free();
      this._ready = false;
    }
  }

  // ── ASR ───────────────────────────────────────────────────
  /** Push float32 PCM samples to the audio buffer. */
  pushAudio(pcm) {
    if (!this._ready) return;
    const n = pcm.length;
    const ptr = this._mod._malloc(n * 4);
    this._mod.HEAPF32.set(pcm, ptr / 4);
    this._pushAudio(ptr, n);
    this._mod._free(ptr);
  }

  /** Set the LLM decoder prompt (FunASRNano). */
  setUserInputPrompt(prompt) { this._setPrompt(String(prompt)); }
  setUseLlm(enable)          { this._setUseLlm(enable ? 1 : 0); }
  setCtcPrecheck(enable)     { this._setCtcPrecheck(enable ? 1 : 0); }
  async redecode() {
    const status = await this._redecode();
    return { status, text: status > 0 ? this._getText() : '' };
  }

  // ── TTS ───────────────────────────────────────────────────
  /** Set TTS generation params (OmniVoice). */
  setTtsParams({ instruct = 'male', language = 'English', seed = 42 } = {}) {
    this._setTtsParams(instruct, language, seed | 0);
  }

  setDiffusionSteps(nSteps) { this._setTtsSteps(nSteps | 0); }

  /**
   * Push a reference audio buffer for voice cloning.
   * @param {Float32Array} pcm
   * @param {number} sampleRate
   */
  pushReferenceAudio(pcm, sampleRate = 16000) {
    if (!this._ready) return;
    const n = pcm.length;
    const ptr = this._mod._malloc(n * 4);
    this._mod.HEAPF32.set(pcm, ptr / 4);
    this._pushRefAudio(ptr, n, sampleRate | 0);
    this._mod._free(ptr);
  }

  pushReferenceText(text) { this._pushRefText(String(text)); }

  /** Push text for TTS to consume on the next process() call. */
  pushText(text) {
    if (!this._ready) return -1;
    return this._pushText(String(text));
  }

  /**
   * Synthesize text and return the assembled Float32Array (all chunks
   * concatenated). Use processStreaming() if you need chunked playback.
   */
  async synthesize(text) {
    if (!this._ready) throw new Error('Not initialized');
    this.reset();
    if (this._pushText(String(text)) !== 0) {
      throw new Error('rs_wasm_push_text failed');
    }
    const chunks = [];
    let total = 0;
    let ret;
    do {
      ret = await this._process();
      if (ret < 0) throw new Error('TTS inference error');
      const n = this._getAudioLen();
      if (n > 0) {
        const ptr = this._getAudioPtr();
        const view = this._mod.HEAPF32.subarray(ptr / 4, ptr / 4 + n);
        const chunk = new Float32Array(n);
        chunk.set(view); // copy out of WASM heap before the next process()
        chunks.push(chunk);
        total += n;
      }
    } while (ret > 0);

    const out = new Float32Array(total);
    let off = 0;
    for (const c of chunks) { out.set(c, off); off += c.length; }
    return out;
  }

  /**
   * Async generator yielding TTS chunks as they arrive.
   *   for await (const chunk of tts.streamSynthesize('hi')) { play(chunk); }
   */
  async *streamSynthesize(text) {
    if (!this._ready) throw new Error('Not initialized');
    this.reset();
    if (this._pushText(String(text)) !== 0) {
      throw new Error('rs_wasm_push_text failed');
    }
    let ret;
    do {
      ret = await this._process();
      if (ret < 0) throw new Error('TTS inference error');
      const n = this._getAudioLen();
      if (n > 0) {
        const ptr = this._getAudioPtr();
        const view = this._mod.HEAPF32.subarray(ptr / 4, ptr / 4 + n);
        const chunk = new Float32Array(n);
        chunk.set(view);
        yield chunk;
      }
      // Yield to the event loop so playback / UI can keep up.
      await new Promise((r) => setTimeout(r, 0));
    } while (ret > 0);
  }

  // ── Generic ───────────────────────────────────────────────
  /**
   * Run one inference step. Returns { status, text } for ASR; for TTS use
   * synthesize() / streamSynthesize() which handle the loop.
   */
  async process() {
    if (!this._ready) return { status: -1, text: '' };
    const status = await this._process();
    const text = status > 0 ? this._getText() : '';
    return { status, text };
  }

  reset() {
    if (this._ready) this._reset();
  }

  get sampleRate() { return this._sampleRate; }
  get isReady()    { return this._ready; }
  get archName()   { return this._ready ? this._getArchName() : 'unknown'; }
  get version()    { return this._getVersion(); }
  get taskType()   { return this._taskType; }
}

// ─────────────────────────────────────────────────────────────
// RapidSpeechVAD — unified wrapper around silero-vad / firered-vad
// ─────────────────────────────────────────────────────────────
//
// Same WASM module the speech context uses. Auto-detects the model
// architecture from GGUF. Push 16 kHz mono Float32 PCM, drain segments
// or per-frame events on demand.

const VAD_SEGMENT_BYTES = 8;   // {float start_s; float end_s}
const VAD_FRAME_BYTES   = 24;  // {i32 idx; f32 raw; f32 smooth; i32 is_sp; i32 is_st; i32 is_ed}

class RapidSpeechVAD {
  constructor(module) {
    this._mod = module;
    this._ready = false;

    this._init        = module.cwrap('rs_wasm_vad_init',         'number', ['string', 'number'], { async: true });
    this._free        = module.cwrap('rs_wasm_vad_free',         null,     []);
    this._reset       = module.cwrap('rs_wasm_vad_reset',        'number', []);
    this._setThresh   = module.cwrap('rs_wasm_vad_set_threshold','number', ['number']);
    this._push        = module.cwrap('rs_wasm_vad_push_audio',   'number', ['number', 'number']);
    this._isSpeech    = module.cwrap('rs_wasm_vad_is_speech',    'number', []);
    this._getProb     = module.cwrap('rs_wasm_vad_get_probability','number', []);
    this._getArch     = module.cwrap('rs_wasm_vad_get_arch',     'string', []);
    this._drainSegs   = module.cwrap('rs_wasm_vad_drain_segments','number', ['number', 'number']);
    this._drainFrames = module.cwrap('rs_wasm_vad_drain_frames', 'number', ['number', 'number']);

    // Reusable IO buffers in the WASM heap. Grown on demand.
    this._segBuf = { ptr: 0, capacity: 0 };
    this._frmBuf = { ptr: 0, capacity: 0 };
  }

  /**
   * Load a VAD model. URL points to a GGUF file whose general.architecture
   * is "silero-vad" or "firered-vad".
   */
  async load(modelUrl, nThreads = 2, onProgress = null) {
    const fileName = '/vad.gguf';

    let buffer;
    if (modelUrl instanceof Uint8Array || modelUrl instanceof ArrayBuffer) {
      buffer = modelUrl instanceof ArrayBuffer ? new Uint8Array(modelUrl) : modelUrl;
    } else {
      const response = await fetch(modelUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch VAD model: ${response.status} ${response.statusText}`);
      }
      const total = parseInt(response.headers.get('Content-Length') || '0', 10);
      const reader = response.body.getReader();
      const chunks = [];
      let loaded = 0;
      if (onProgress) onProgress(0, { loaded: 0, total });
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        if (onProgress) {
          const frac = total > 0 ? Math.min(loaded / total, 1.0) : 0;
          onProgress(frac, { loaded, total });
        }
      }
      buffer = new Uint8Array(loaded);
      let off = 0;
      for (const c of chunks) { buffer.set(c, off); off += c.length; }
      if (onProgress) onProgress(1, { loaded, total: total || loaded });
    }

    this._mod.FS.writeFile(fileName, buffer);
    const ret = await this._init(fileName, nThreads);
    if (ret !== 0) throw new Error('rs_wasm_vad_init failed with code ' + ret);
    this._ready = true;
  }

  free() {
    if (this._segBuf.ptr) { this._mod._free(this._segBuf.ptr); this._segBuf = { ptr: 0, capacity: 0 }; }
    if (this._frmBuf.ptr) { this._mod._free(this._frmBuf.ptr); this._frmBuf = { ptr: 0, capacity: 0 }; }
    if (this._ready) { this._free(); this._ready = false; }
  }

  reset()                  { if (this._ready) this._reset(); }
  setThreshold(threshold)  { if (this._ready) this._setThresh(threshold); }
  get isSpeech()           { return this._ready ? !!this._isSpeech() : false; }
  get probability()        { return this._ready ? this._getProb() : 0; }
  get arch()               { return this._ready ? this._getArch() : ''; }
  get isReady()            { return this._ready; }

  /** Push 16 kHz mono Float32Array PCM. Updates internal queues. */
  pushAudio(pcm) {
    if (!this._ready) return;
    const n = pcm.length;
    const ptr = this._mod._malloc(n * 4);
    this._mod.HEAPF32.set(pcm, ptr / 4);
    this._push(ptr, n);
    this._mod._free(ptr);
  }

  _ensureBuf(slot, capacity, elemBytes) {
    if (slot.capacity >= capacity) return;
    if (slot.ptr) this._mod._free(slot.ptr);
    slot.ptr = this._mod._malloc(capacity * elemBytes);
    slot.capacity = capacity;
  }

  /** Drain queued segments as [{start_s, end_s}, ...]. */
  drainSegments(maxCount = 64) {
    if (!this._ready) return [];
    this._ensureBuf(this._segBuf, maxCount, VAD_SEGMENT_BYTES);
    const n = this._drainSegs(this._segBuf.ptr, maxCount);
    const out = new Array(n);
    const base = this._segBuf.ptr / 4;
    for (let i = 0; i < n; ++i) {
      const off = base + i * 2;
      out[i] = { start_s: this._mod.HEAPF32[off], end_s: this._mod.HEAPF32[off + 1] };
    }
    return out;
  }

  /** Drain queued per-frame events. */
  drainFrames(maxCount = 256) {
    if (!this._ready) return [];
    this._ensureBuf(this._frmBuf, maxCount, VAD_FRAME_BYTES);
    const n = this._drainFrames(this._frmBuf.ptr, maxCount);
    const out = new Array(n);
    const i32 = this._mod.HEAP32;
    const f32 = this._mod.HEAPF32;
    const base32 = this._frmBuf.ptr / 4; // 6 i32-slots per frame
    for (let i = 0; i < n; ++i) {
      const off = base32 + i * 6;
      out[i] = {
        frame_idx:        i32[off],
        raw_prob:         f32[off + 1],
        smoothed_prob:    f32[off + 2],
        is_speech:        !!i32[off + 3],
        is_speech_start:  !!i32[off + 4],
        is_speech_end:    !!i32[off + 5],
      };
    }
    return out;
  }
}

// ─────────────────────────────────────────────────────────────
// RapidSpeechKWS — open-vocabulary streaming wake-word spotter
// ─────────────────────────────────────────────────────────────
//
// Backed by SenseVoice + ContextGraph + CTC Aho-Corasick decoder. Independent
// of the main RapidSpeechWASM context — KWS holds its own model so the
// asr/tts tabs can coexist on the same MODULARIZE'd module.
//
// keywords.txt format (one per line, sherpa-onnx compatible):
//   <tok-id> <tok-id> ...  [:boost] [#threshold] [@human-phrase]
// Tokens are SenseVoice token ids; use scripts/sensevoice_tokenize.py to
// produce them from natural-language phrases.

const KWS_RECORD_BYTES = 24; // {double time_s; float avg_prob; u32 off; u32 len; u32 pad}

class RapidSpeechKWS {
  constructor(module) {
    this._mod = module;
    this._ready = false;

    this._init       = module.cwrap('rs_wasm_kws_init',         'number',
        ['string', 'string', 'number', 'number', 'number', 'number', 'number'],
        { async: true });
    this._free       = module.cwrap('rs_wasm_kws_free',         null,     []);
    this._reset      = module.cwrap('rs_wasm_kws_reset',        'number', []);
    this._push       = module.cwrap('rs_wasm_kws_push_audio',   'number', ['number', 'number']);
    this._poll       = module.cwrap('rs_wasm_kws_poll',         'number', []);
    this._hitsPtr    = module.cwrap('rs_wasm_kws_get_hits_ptr', 'number', []);
    this._hitsCount  = module.cwrap('rs_wasm_kws_get_hits_count','number', []);
    this._poolPtr    = module.cwrap('rs_wasm_kws_get_phrase_pool_ptr','number', []);
    this._poolLen    = module.cwrap('rs_wasm_kws_get_phrase_pool_len','number', []);
    this._sampleRate = module.cwrap('rs_wasm_kws_get_sample_rate','number', []);
    this._archName   = module.cwrap('rs_wasm_kws_get_arch_name','string', []);

    this._decoder = (typeof TextDecoder !== 'undefined') ? new TextDecoder('utf-8') : null;
  }

  /**
   * Load model + register keywords.
   * @param {string} modelUrl   GGUF URL (SenseVoice)
   * @param {string} keywordsText  newline-separated keywords.txt content
   * @param {object} opts  { nThreads, windowMs, hopMs, debounceMs, defaultThreshold }
   * @param {function} onProgress(frac, {loaded,total})
   */
  async load(modelUrl, keywordsText, opts = {}, onProgress = null) {
    const {
      nThreads = 2, windowMs = 1600, hopMs = 200,
      debounceMs = 1500, defaultThreshold = 0.0,
    } = opts;

    const modelPath = '/kws_model.gguf';
    const kwPath    = '/keywords.txt';

    let buffer;
    if (modelUrl instanceof Uint8Array || modelUrl instanceof ArrayBuffer) {
      buffer = modelUrl instanceof ArrayBuffer ? new Uint8Array(modelUrl) : modelUrl;
    } else {
      const response = await fetch(modelUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch KWS model: ${response.status} ${response.statusText}`);
      }
      const total = parseInt(response.headers.get('Content-Length') || '0', 10);
      const reader = response.body.getReader();
      const chunks = [];
      let loaded = 0;
      if (onProgress) onProgress(0, { loaded: 0, total });
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.length;
        if (onProgress) {
          const frac = total > 0 ? Math.min(loaded / total, 1.0) : 0;
          onProgress(frac, { loaded, total });
        }
      }
      buffer = new Uint8Array(loaded);
      let off = 0;
      for (const c of chunks) { buffer.set(c, off); off += c.length; }
      if (onProgress) onProgress(1, { loaded, total: total || loaded });
    }

    this._mod.FS.writeFile(modelPath, buffer);
    const enc = new TextEncoder();
    this._mod.FS.writeFile(kwPath, enc.encode(String(keywordsText || '')));

    const ret = await this._init(
      modelPath, kwPath, nThreads | 0,
      windowMs | 0, hopMs | 0, debounceMs | 0,
      Number(defaultThreshold) || 0.0,
    );
    if (ret !== 0) throw new Error('rs_wasm_kws_init failed with code ' + ret);
    this._ready = true;
  }

  free() {
    if (this._ready) { this._free(); this._ready = false; }
  }

  reset() { if (this._ready) this._reset(); }

  /** Push 16 kHz mono Float32Array PCM. */
  pushAudio(pcm) {
    if (!this._ready) return;
    const n = pcm.length;
    const ptr = this._mod._malloc(n * 4);
    this._mod.HEAPF32.set(pcm, ptr / 4);
    this._push(ptr, n);
    this._mod._free(ptr);
  }

  /**
   * Run as many KWS analysis windows as the buffered audio supports.
   * Returns an array of hits: [{ phrase, avg_prob, time_s }, ...]
   */
  poll() {
    if (!this._ready) return [];
    const n = this._poll();
    if (n <= 0) return [];

    const recPtr  = this._hitsPtr();
    const poolPtr = this._poolPtr();
    const poolLen = this._poolLen();
    if (!recPtr || !poolLen) return [];

    const pool = this._mod.HEAPU8.subarray(poolPtr, poolPtr + poolLen);
    const decode = (off, len) => {
      const slice = pool.subarray(off, off + len);
      if (this._decoder) return this._decoder.decode(slice);
      // Fallback for very old environments.
      let s = '';
      for (let i = 0; i < slice.length; ++i) s += String.fromCharCode(slice[i]);
      return decodeURIComponent(escape(s));
    };

    const out = new Array(n);
    const getValue = this._mod.getValue;
    for (let i = 0; i < n; ++i) {
      const base = recPtr + i * KWS_RECORD_BYTES;
      const time_s   = getValue(base + 0,  'double');
      const avg_prob = getValue(base + 8,  'float');
      const off      = this._mod.HEAPU32[(base + 12) >> 2];
      const len      = this._mod.HEAPU32[(base + 16) >> 2];
      out[i] = { time_s, avg_prob, phrase: decode(off, len) };
    }
    return out;
  }

  get sampleRate() { return this._ready ? this._sampleRate() : 16000; }
  get archName()   { return this._ready ? this._archName()   : 'unknown'; }
  get isReady()    { return this._ready; }
}

// Export for both browser and Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { RapidSpeechWASM, RapidSpeechVAD, RapidSpeechKWS, RS_TASK };
} else if (typeof globalThis !== 'undefined') {
  globalThis.RapidSpeechWASM = RapidSpeechWASM;
  globalThis.RapidSpeechVAD  = RapidSpeechVAD;
  globalThis.RapidSpeechKWS  = RapidSpeechKWS;
  globalThis.RS_TASK = RS_TASK;
}
