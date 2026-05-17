/**
 * RapidSpeech WASM JavaScript bridge.
 *
 * Provides a clean JS API on top of the Emscripten-compiled WASM module.
 * Handles memory management, string marshaling, and model fetching.
 */
class RapidSpeechWASM {
  constructor(module) {
    this._mod = module;
    this._ready = false;

    // Bind C functions
    this._init = module.cwrap('rs_wasm_init', 'number', ['string', 'number']);
    this._free = module.cwrap('rs_wasm_free', null, []);
    this._pushAudio = module.cwrap('rs_wasm_push_audio', 'number', ['number', 'number']);
    this._process = module.cwrap('rs_wasm_process', 'number', []);
    this._getText = module.cwrap('rs_wasm_get_text', 'string', []);
    this._reset = module.cwrap('rs_wasm_reset', 'number', []);
    this._getSampleRate = module.cwrap('rs_wasm_get_sample_rate', 'number', []);
    this._getArchName = module.cwrap('rs_wasm_get_arch_name', 'string', []);
    this._isReady = module.cwrap('rs_wasm_is_ready', 'number', []);
    this._getVersion = module.cwrap('rs_wasm_get_version', 'string', []);

    this._sampleRate = 16000;
  }

  /**
   * Fetch a GGUF model from a URL and initialize the recognizer.
   * @param {string} modelUrl - URL of the .gguf file
   * @param {number} nThreads - number of CPU threads (default: 2)
   * @param {function} onProgress - progress callback (0.0 - 1.0)
   * @returns {Promise<void>}
   */
  async init(modelUrl, nThreads = 2, onProgress = null) {
    const fileName = '/model.gguf';

    // Fetch model data
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }

    const contentLength = response.headers.get('Content-Length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;

    // Read into ArrayBuffer with progress tracking
    let loaded = 0;
    const reader = response.body.getReader();
    const chunks = [];
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (onProgress && total > 0) {
        onProgress(Math.min(loaded / total, 1.0));
      }
    }

    // Concatenate chunks
    const buffer = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
      buffer.set(chunk, offset);
      offset += chunk.length;
    }

    // Write to Emscripten virtual filesystem
    this._mod.FS.writeFile(fileName, buffer);

    // Initialize the model
    const ret = this._init(fileName, nThreads);
    if (ret !== 0) {
      throw new Error('rs_wasm_init failed with code ' + ret);
    }

    this._sampleRate = this._getSampleRate();
    this._ready = true;
  }

  /**
   * Push PCM audio samples (float32, -1.0 to 1.0, 16kHz mono).
   * @param {Float32Array} pcm
   */
  pushAudio(pcm) {
    if (!this._ready) return;
    const nSamples = pcm.length;
    const ptr = this._mod._malloc(nSamples * 4); // 4 bytes per float
    this._mod.HEAPF32.set(pcm, ptr / 4);
    this._pushAudio(ptr, nSamples);
    this._mod._free(ptr);
  }

  /**
   * Run one inference step.
   * @returns {{status: number, text: string}}
   */
  process() {
    if (!this._ready) return { status: -1, text: '' };
    const status = this._process();
    const text = status > 0 ? this._getText() : '';
    return { status, text };
  }

  /**
   * Reset the recognizer for a new utterance.
   */
  reset() {
    if (!this._ready) return;
    this._reset();
  }

  /**
   * Release resources.
   */
  free() {
    if (this._ready) {
      this._free();
      this._ready = false;
    }
  }

  get sampleRate() { return this._sampleRate; }
  get isReady() { return this._ready; }
  get archName() { return this._ready ? this._getArchName() : 'unknown'; }
  get version() { return this._getVersion(); }
}

// Export for both browser and Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { RapidSpeechWASM };
}
