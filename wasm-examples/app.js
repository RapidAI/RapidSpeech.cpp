/**
 * RapidSpeech Web Demo — Application Logic
 *
 * Microphone capture → VAD → ASR pipeline running entirely in the browser.
 * Requires: RapidSpeechModule (from rapidspeech-wasm.js) and RapidSpeechWASM.
 */

// ── DOM Elements ────────────────────────────────────────────
const elModelUrl = document.getElementById('model-url');
const elLoadBtn = document.getElementById('load-btn');
const elStartBtn = document.getElementById('start-btn');
const elStopBtn = document.getElementById('stop-btn');
const elClearBtn = document.getElementById('clear-btn');
const elStatus = document.getElementById('status');
const elProgress = document.getElementById('progress-bar');
const elProgressContainer = document.getElementById('progress-container');
const elVadBar = document.getElementById('vad-bar');
const elVadLabel = document.getElementById('vad-label');
const elTranscript = document.getElementById('transcript');
const elPartial = document.getElementById('partial');
const elInfo = document.getElementById('info');

// ── State ───────────────────────────────────────────────────
let rs = null;
let audioCtx = null;
let scriptNode = null;
let mediaStream = null;
let running = false;
let partialText = '';
let segmentCount = 0;

// ── SIMPLE VAD ──────────────────────────────────────────────
// Energy-based VAD: computes RMS of audio chunk and compares to
// a threshold. Simple but effective for browser demos where we
// don't have the full Silero VAD compiled to WASM.
const ENERGY_THRESHOLD = 0.01;
const SILENCE_FRAMES_MAX = 15; // ~480ms at 32ms frame
let silenceFrames = 0;
let inSpeech = false;
let speechBuffer = [];
let speechStartTime = 0;
let audioTime = 0;
const SAMPLE_RATE = 16000;

function rmsEnergy(pcm) {
  let sum = 0;
  for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
  return Math.sqrt(sum / pcm.length);
}

// ── Initialize ──────────────────────────────────────────────
async function loadModel() {
  const url = elModelUrl.value.trim();
  if (!url) {
    setStatus('Please enter a model URL.', 'error');
    return;
  }

  setStatus('Downloading model...', 'loading');
  elLoadBtn.disabled = true;
  elProgressContainer.style.display = 'block';

  try {
    // MODULARIZE=1: RapidSpeechModule() returns a Promise<instance>
    const wasmInstance = await RapidSpeechModule();
    rs = new RapidSpeechWASM(wasmInstance);
    await rs.init(url, 2, (p) => {
      elProgress.style.width = (p * 100).toFixed(1) + '%';
    });

    elProgressContainer.style.display = 'none';
    elStartBtn.disabled = false;
    setStatus(`Ready — ${rs.archName} @ ${rs.sampleRate}Hz`, 'ready');
    elInfo.textContent = `RapidSpeech v${rs.version} | ${rs.archName} | ${rs.sampleRate} Hz`;
  } catch (err) {
    setStatus('Failed: ' + err.message, 'error');
    elProgressContainer.style.display = 'none';
  } finally {
    elLoadBtn.disabled = false;
  }
}

// ── Microphone ──────────────────────────────────────────────
async function startMic() {
  if (running) return;

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        sampleRate: { ideal: SAMPLE_RATE },
        echoCancellation: true,
        noiseSuppression: true,
      }
    });
  } catch (err) {
    setStatus('Microphone access denied.', 'error');
    return;
  }

  audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });

  // Use AudioWorklet if available, fallback to ScriptProcessorNode
  if (audioCtx.audioWorklet) {
    await startWithAudioWorklet();
  } else {
    startWithScriptProcessor();
  }

  running = true;
  elStartBtn.disabled = true;
  elStopBtn.disabled = false;
  setStatus('Listening...', 'listening');
}

function startWithScriptProcessor() {
  const source = audioCtx.createMediaStreamSource(mediaStream);
  scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);

  scriptNode.onaudioprocess = (event) => {
    if (!running) return;
    const input = event.inputBuffer.getChannelData(0);
    processAudioChunk(input);
  };

  source.connect(scriptNode);
  scriptNode.connect(audioCtx.destination);
}

// AudioWorklet path — lower latency, better performance
async function startWithAudioWorklet() {
  // Inline worklet processor
  const workletCode = `
    class RSProcessor extends AudioWorkletProcessor {
      process(inputs) {
        const input = inputs[0];
        if (input && input.length > 0) {
          this.port.postMessage(input[0]);
        }
        return true;
      }
    }
    registerProcessor('rs-processor', RSProcessor);
  `;

  const blob = new Blob([workletCode], { type: 'application/javascript' });
  const url = URL.createObjectURL(blob);
  await audioCtx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);

  const workletNode = new AudioWorkletNode(audioCtx, 'rs-processor');
  workletNode.port.onmessage = (event) => {
    if (!running) return;
    processAudioChunk(event.data);
  };

  const source = audioCtx.createMediaStreamSource(mediaStream);
  source.connect(workletNode);
  workletNode.connect(audioCtx.destination);
}

// ── Audio Processing ────────────────────────────────────────
function processAudioChunk(buffer) {
  const chunkSize = 512; // 32ms at 16kHz
  const data = new Float32Array(buffer);

  for (let offset = 0; offset + chunkSize <= data.length; offset += chunkSize) {
    const chunk = data.subarray(offset, offset + chunkSize);
    const energy = rmsEnergy(chunk);

    // Update VAD display
    const prob = Math.min(energy / 0.05, 1.0);
    updateVadDisplay(prob, energy >= ENERGY_THRESHOLD);

    audioTime += chunkSize / SAMPLE_RATE;

    if (energy >= ENERGY_THRESHOLD) {
      silenceFrames = 0;
      if (!inSpeech) {
        inSpeech = true;
        speechStartTime = audioTime - chunkSize / SAMPLE_RATE;
        speechBuffer = [];
      }
      speechBuffer.push(...chunk);
    } else if (inSpeech) {
      speechBuffer.push(...chunk);
      silenceFrames++;
      if (silenceFrames >= SILENCE_FRAMES_MAX) {
        // End of speech segment — run ASR
        finalizeSegment();
      }
    }
  }
}

function finalizeSegment() {
  if (speechBuffer.length === 0) return;

  const pcm = new Float32Array(speechBuffer);
  rs.pushAudio(pcm);
  const result = rs.process();

  segmentCount++;
  const startStr = formatTime(speechStartTime);
  const endStr = formatTime(audioTime);

  if (result.status > 0 && result.text) {
    appendTranscript(startStr, endStr, result.text);
  }

  rs.reset();
  inSpeech = false;
  speechBuffer = [];
  silenceFrames = 0;

  // Update partial display
  partialText = '';
  elPartial.textContent = '';
}

function appendTranscript(start, end, text) {
  const line = document.createElement('div');
  line.className = 'transcript-line';
  line.innerHTML = `<span class="timestamp">[${start} → ${end}]</span> ${escapeHtml(text)}`;
  elTranscript.appendChild(line);
  elTranscript.scrollTop = elTranscript.scrollHeight;
}

// ── Stop / Clear ────────────────────────────────────────────
function stopMic() {
  running = false;
  elStartBtn.disabled = false;
  elStopBtn.disabled = true;

  // Flush remaining speech
  if (inSpeech && speechBuffer.length > 0) {
    finalizeSegment();
  }

  if (scriptNode) {
    scriptNode.disconnect();
    scriptNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  if (audioCtx && audioCtx.state !== 'closed') {
    audioCtx.close().catch(() => {});
    audioCtx = null;
  }

  setStatus('Stopped.', 'idle');
  updateVadDisplay(0, false);
}

function clearTranscript() {
  elTranscript.innerHTML = '';
  elPartial.textContent = '';
  partialText = '';
  segmentCount = 0;
  audioTime = 0;
}

// ── UI Helpers ──────────────────────────────────────────────
function setStatus(msg, cls) {
  elStatus.textContent = msg;
  elStatus.className = 'status ' + cls;
}

function updateVadDisplay(prob, isSpeech) {
  elVadBar.style.width = (prob * 100).toFixed(1) + '%';
  elVadBar.className = 'vad-fill' + (isSpeech ? ' speech' : '');
  elVadLabel.textContent = isSpeech ? 'SPEECH' : 'silence';
  elVadLabel.className = 'vad-label' + (isSpeech ? ' speech' : '');
}

function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(1);
  return String(m).padStart(2, '0') + ':' + String(s).padStart(4, '0');
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ── Event Bindings ──────────────────────────────────────────
elLoadBtn.addEventListener('click', loadModel);
elStartBtn.addEventListener('click', startMic);
elStopBtn.addEventListener('click', stopMic);
elClearBtn.addEventListener('click', clearTranscript);

// ── Initial State ───────────────────────────────────────────
setStatus('Enter a model URL and click Load.', 'idle');
elStopBtn.disabled = true;
elStartBtn.disabled = true;
