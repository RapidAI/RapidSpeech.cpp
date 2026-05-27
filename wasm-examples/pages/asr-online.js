/**
 * ASR Online — microphone capture → VAD (neural or energy fallback) → ASR.
 *
 * When a VAD model (silero-vad or firered-vad) is loaded, audio is fed through
 * the neural VAD and segments are dispatched to ASR on `is_speech_end` events.
 * Otherwise the page falls back to an RMS-energy gate with `silence-frames`
 * hysteresis.
 *
 * Exposes window.RSPageAsrOnline.init({root, createInstance}).
 */
(function () {
  const { formatTime, escapeHtml, statusSetter, makeDownloadProgress } = window.RSUtils;
  const SAMPLE_RATE = 16000;

  function $(root, id) { return root.querySelector(`[data-id="${id}"]`); }

  async function init({ root, createInstance }) {
    const els = {
      modelUrl: $(root, 'model-url'),
      nThreads: $(root, 'n-threads'),
      useLlm:   $(root, 'use-llm'),
      twoPass:  $(root, 'two-pass'),
      loadBtn:  $(root, 'load-btn'),
      status:   $(root, 'status'),
      progress: $(root, 'progress-container'),
      progressBar: $(root, 'progress-bar'),
      progressInfo: $(root, 'progress-info'),

      vadUrl:        $(root, 'vad-url'),
      vadLoadBtn:    $(root, 'vad-load-btn'),
      vadStatus:     $(root, 'vad-status'),
      vadProgress:   $(root, 'vad-progress-container'),
      vadProgressBar:$(root, 'vad-progress-bar'),
      vadProgressInfo:$(root, 'vad-progress-info'),

      threshold:    $(root, 'vad-threshold'),
      thresholdVal: $(root, 'vad-threshold-val'),
      silenceFrames:    $(root, 'silence-frames'),
      silenceFramesVal: $(root, 'silence-frames-val'),

      startBtn: $(root, 'start-btn'),
      stopBtn:  $(root, 'stop-btn'),
      clearBtn: $(root, 'clear-btn'),
      vadBar:   $(root, 'vad-bar'),
      vadLabel: $(root, 'vad-label'),
      info:     $(root, 'info'),
      transcript: $(root, 'transcript'),
      partial:    $(root, 'partial'),
      rtfWarn:   $(root, 'rtf-warn'),
    };

    const setStatus    = statusSetter(els.status);
    const setVadStatus = statusSetter(els.vadStatus);

    // Mutable runtime state (rebuilt each Start).
    let rs = null;
    let vad = null;
    let audioCtx = null;
    let mediaStream = null;
    let running = false;
    let carry = new Float32Array(0);
    let silenceFrames = 0;
    let inSpeech = false;
    let speechBuffer = [];
    let speechStartTime = 0;
    let audioTime = 0;
    let workletNode = null;
    let scriptNode = null;
    // VAD-mode (neural) speech-buffer state.
    let vadSpeechBuf = [];
    let vadSpeechStart = 0;
    // Serialize ASR dispatches — rs is a single shared context and calling
    // rs.reset/pushAudio/process concurrently from overlapping VAD segments
    // corrupts state and deadlocks the WebGPU queue.
    let asrQueue = Promise.resolve();

    function vadParams() {
      return {
        threshold: parseFloat(els.threshold.value),
        silenceMax: parseInt(els.silenceFrames.value, 10),
      };
    }

    els.threshold.addEventListener('input', () => {
      const v = parseFloat(els.threshold.value);
      els.thresholdVal.textContent = v.toFixed(3);
      if (vad && vad.isReady) vad.setThreshold(v);
    });
    els.silenceFrames.addEventListener('input', () => {
      els.silenceFramesVal.textContent = els.silenceFrames.value;
    });
    els.thresholdVal.textContent    = parseFloat(els.threshold.value).toFixed(3);
    els.silenceFramesVal.textContent = els.silenceFrames.value;

    els.loadBtn.addEventListener('click', loadModel);
    els.vadLoadBtn.addEventListener('click', loadVad);
    els.startBtn.addEventListener('click', startMic);
    els.stopBtn.addEventListener('click', stopMic);
    els.clearBtn.addEventListener('click', clearTranscript);

    window.addEventListener('beforeunload', stopMic);

    async function loadModel() {
      const url = els.modelUrl.value.trim();
      if (!url) { setStatus('Please enter a model URL.', 'error'); return; }

      setStatus('Downloading model...', 'loading');
      els.loadBtn.disabled = true;
      els.progress.style.display = 'block';

      try {
        rs = await createInstance();
        const nThreads = parseInt(els.nThreads.value, 10) || 4;
        const onProg = makeDownloadProgress(els.progressBar, els.progressInfo);
        await rs.initAsr(url, nThreads, onProg);
        rs.setUseLlm(els.useLlm.checked);
        els.useLlm.addEventListener('change', () => rs.setUseLlm(els.useLlm.checked));

        els.progress.style.display = 'none';
        els.startBtn.disabled = false;
        setStatus(`Ready — ${rs.archName} @ ${rs.sampleRate}Hz`, 'ready');
        els.info.textContent = `${rs.archName} | ${rs.sampleRate} Hz`;
      } catch (err) {
        setStatus('Failed: ' + err.message, 'error');
        els.progress.style.display = 'none';
      } finally {
        els.loadBtn.disabled = false;
      }
    }

    async function loadVad() {
      const url = els.vadUrl.value.trim();
      if (!url) { setVadStatus('Enter a VAD model URL first.', 'error'); return; }
      setVadStatus('Downloading VAD...', 'loading');
      els.vadLoadBtn.disabled = true;
      els.vadProgress.style.display = 'block';
      try {
        if (!rs) { rs = await createInstance(); }
        vad = new window.RapidSpeechVAD(rs._mod);
        const onProg = makeDownloadProgress(els.vadProgressBar, els.vadProgressInfo);
        await vad.load(url, 2, onProg);
        vad.setThreshold(parseFloat(els.threshold.value));
        els.vadProgress.style.display = 'none';
        setVadStatus(`Neural VAD — ${vad.arch}`, 'ready');
      } catch (err) {
        setVadStatus('Failed: ' + err.message, 'error');
        els.vadProgress.style.display = 'none';
        vad = null;
      } finally {
        els.vadLoadBtn.disabled = false;
      }
    }

    async function startMic() {
      if (running) return;
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: { ideal: SAMPLE_RATE },
                   echoCancellation: true, noiseSuppression: true }
        });
      } catch {
        setStatus('Microphone access denied.', 'error');
        return;
      }

      audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
      if (audioCtx.audioWorklet) {
        await startWithAudioWorklet();
      } else {
        startWithScriptProcessor();
      }

      // Reset segmentation state.
      carry = new Float32Array(0);
      inSpeech = false;
      silenceFrames = 0;
      speechBuffer = [];
      vadSpeechBuf = [];
      vadSpeechStart = 0;
      asrQueue = Promise.resolve();
      rtfHistory.length = 0;
      rtfWarned = false;
      if (els.rtfWarn) els.rtfWarn.style.display = 'none';
      if (vad && vad.isReady) vad.reset();

      running = true;
      els.startBtn.disabled = true;
      els.stopBtn.disabled  = false;
      setStatus(vad && vad.isReady ? 'Listening (neural VAD)...' : 'Listening (energy gate)...',
                'listening');
    }

    function startWithScriptProcessor() {
      const source = audioCtx.createMediaStreamSource(mediaStream);
      scriptNode = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptNode.onaudioprocess = (event) => {
        if (!running) return;
        processAudioChunk(event.inputBuffer.getChannelData(0));
      };
      source.connect(scriptNode);
      scriptNode.connect(audioCtx.destination);
    }

    async function startWithAudioWorklet() {
      const workletCode = `
        class RSProcessor extends AudioWorkletProcessor {
          process(inputs) {
            const input = inputs[0];
            if (input && input.length > 0) this.port.postMessage(input[0]);
            return true;
          }
        }
        registerProcessor('rs-processor', RSProcessor);
      `;
      const blob = new Blob([workletCode], { type: 'application/javascript' });
      const url = URL.createObjectURL(blob);
      await audioCtx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);

      workletNode = new AudioWorkletNode(audioCtx, 'rs-processor');
      workletNode.port.onmessage = (event) => {
        if (!running) return;
        processAudioChunk(event.data);
      };
      const source = audioCtx.createMediaStreamSource(mediaStream);
      source.connect(workletNode);
      workletNode.connect(audioCtx.destination);
    }

    function processAudioChunk(buffer) {
      // If the audio device gave us a non-16k stream, resample on the fly.
      // (most browsers honor the AudioContext's 16 k sampleRate hint.)
      let pcm = buffer;
      if (audioCtx && audioCtx.sampleRate !== SAMPLE_RATE) {
        // Cheap linear resample for the live path — segments going to ASR
        // already pass through the high-quality OfflineAudioContext resampler
        // in decodeFileToMono16k; here it just feeds VAD/energy gating.
        const ratio = audioCtx.sampleRate / SAMPLE_RATE;
        const outLen = Math.floor(buffer.length / ratio);
        const tmp = new Float32Array(outLen);
        for (let i = 0; i < outLen; ++i) tmp[i] = buffer[Math.floor(i * ratio)] || 0;
        pcm = tmp;
      }

      if (vad && vad.isReady) {
        processWithNeuralVad(pcm);
      } else {
        processWithEnergyGate(pcm);
      }
    }

    function processWithNeuralVad(pcm) {
      audioTime += pcm.length / SAMPLE_RATE;
      vad.pushAudio(pcm);
      const prob = vad.probability;
      updateVadDisplay(Math.min(prob, 1.0), vad.isSpeech);

      // Buffer the audio while drains tell us what segments closed. We always
      // buffer (even during silence) so we can slice [start_s..end_s] out of
      // it when an end-event arrives.
      // Memory bound: keep at most 60 s of trailing audio in the buffer.
      vadSpeechBuf.push(pcm);
      const maxSamples = SAMPLE_RATE * 60;
      let totalBuf = 0;
      for (const b of vadSpeechBuf) totalBuf += b.length;
      while (totalBuf > maxSamples && vadSpeechBuf.length > 1) {
        const dropped = vadSpeechBuf.shift().length;
        totalBuf -= dropped;
        vadSpeechStart += dropped / SAMPLE_RATE;
      }

      // Walk frame events to find segment closes.
      const frames = vad.drainFrames(512);
      for (const f of frames) {
        if (f.is_speech_end) {
          // Segments come from drainSegments — they have absolute timestamps
          // relative to the VAD's `samples_in` counter.
          // We already track audioTime which mirrors that — close the segment
          // using the segment queue.
        }
      }
      const segs = vad.drainSegments(32);
      for (const seg of segs) {
        const startS = seg.start_s;
        const endS   = seg.end_s;
        // Chain onto the serial ASR queue so concurrent segments don't race.
        asrQueue = asrQueue.then(() => dispatchVadSegment(startS, endS))
                           .catch((err) => console.error('ASR queue error', err));
      }
    }

    async function dispatchVadSegment(startS, endS) {
      // Find the slice of pcm covering [startS, endS] using vadSpeechStart as
      // the buffer origin in seconds.
      const bufStart = vadSpeechStart;
      const sliceStart = Math.max(0, Math.floor((startS - bufStart) * SAMPLE_RATE));
      const sliceEnd   = Math.max(sliceStart, Math.floor((endS - bufStart) * SAMPLE_RATE));
      if (sliceEnd <= sliceStart) return;

      // Concatenate buffered chunks into a flat slice.
      const flat = new Float32Array(sliceEnd - sliceStart);
      let off = 0, srcOff = 0, written = 0;
      for (const b of vadSpeechBuf) {
        const next = srcOff + b.length;
        if (next <= sliceStart) { srcOff = next; continue; }
        if (srcOff >= sliceEnd) break;
        const from = Math.max(0, sliceStart - srcOff);
        const to   = Math.min(b.length, sliceEnd - srcOff);
        flat.set(b.subarray(from, to), off);
        off += to - from;
        srcOff = next;
        written += to - from;
        if (srcOff >= sliceEnd) break;
      }
      if (written < SAMPLE_RATE * 0.2) { return; } // <200 ms — skip

      const twoPass = els.twoPass && els.twoPass.checked;
      const useLlm  = els.useLlm.checked;
      const audioS  = written / SAMPLE_RATE;
      const startLabel = formatTime(startS);
      const endLabel   = formatTime(endS);

      rs.reset();
      rs.pushAudio(flat);

      if (twoPass) {
        // Pass 1: CTC greedy (fast).
        rs.setUseLlm(false);
        const t0 = performance.now();
        let ctcResult;
        try { ctcResult = await rs.process(); }
        catch (err) { console.error('ASR process failed', err); rs.reset(); return; }
        const ctcInferS = (performance.now() - t0) / 1000;
        const ctcRtf    = audioS > 0 ? ctcInferS / audioS : 0;

        // Render CTC as a partial line we can upgrade in place.
        const line = appendTranscript(startLabel, endLabel,
                                      ctcResult.text || '(no speech)',
                                      { kind: 'ctc' });

        // Pass 2: LLM rescore.
        rs.setUseLlm(true);
        const t1 = performance.now();
        let llmResult;
        try { llmResult = await rs.redecode(); }
        catch (err) { console.error('ASR redecode failed', err); rs.reset(); return; }
        const llmInferS = (performance.now() - t1) / 1000;
        const totalRtf  = audioS > 0 ? (ctcInferS + llmInferS) / audioS : 0;
        maybeWarnRtf(totalRtf);

        if (llmResult.status > 0 && llmResult.text) {
          updateTranscriptLine(line, startLabel, endLabel, llmResult.text,
                               { kind: 'llm' });
        }
      } else {
        if (!useLlm) rs.setUseLlm(false);
        else rs.setUseLlm(true);
        const t0 = performance.now();
        let result;
        try { result = await rs.process(); }
        catch (err) { console.error('ASR process failed', err); rs.reset(); return; }
        const inferS = (performance.now() - t0) / 1000;
        const rtf    = audioS > 0 ? inferS / audioS : 0;
        maybeWarnRtf(rtf);
        if (result.status > 0 && result.text) {
          appendTranscript(startLabel, endLabel, result.text);
        }
      }
      els.partial.textContent = '';
    }

    function processWithEnergyGate(buffer) {
      const { threshold, silenceMax } = vadParams();
      const chunkSize = 512;

      const merged = new Float32Array(carry.length + buffer.length);
      merged.set(carry, 0);
      merged.set(buffer, carry.length);

      let offset = 0;
      for (; offset + chunkSize <= merged.length; offset += chunkSize) {
        const chunk = merged.subarray(offset, offset + chunkSize);
        const energy = rmsEnergy(chunk);
        updateVadDisplay(Math.min(energy / 0.05, 1.0), energy >= threshold);
        audioTime += chunkSize / SAMPLE_RATE;

        if (energy >= threshold) {
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
          if (silenceFrames >= silenceMax) {
            finalizeEnergySegment();
          }
        }
      }
      carry = merged.slice(offset);
    }

    function rmsEnergy(pcm) {
      let sum = 0;
      for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
      return Math.sqrt(sum / pcm.length);
    }

    async function finalizeEnergySegment() {
      if (speechBuffer.length === 0) return;
      const buf = speechBuffer;
      const segStart = speechStartTime;
      const segEnd = audioTime;
      inSpeech = false;
      speechBuffer = [];
      silenceFrames = 0;

      let pcm = new Float32Array(buf);
      if (pcm.length < SAMPLE_RATE * 0.2) { rs.reset(); els.partial.textContent = ''; return; }

      rs.pushAudio(pcm);
      let result;
      try { result = await rs.process(); }
      catch (err) { console.error('ASR process failed', err); rs.reset(); return; }

      if (result.status > 0 && result.text) {
        appendTranscript(formatTime(segStart), formatTime(segEnd), result.text);
      }
      rs.reset();
      els.partial.textContent = '';
    }

    // Track recent RTFs — warn the user once the running average passes 1.
    const rtfHistory = [];
    let rtfWarned = false;
    function maybeWarnRtf(rtf) {
      if (!isFinite(rtf) || rtf <= 0) return;
      rtfHistory.push(rtf);
      if (rtfHistory.length > 5) rtfHistory.shift();
      const avg = rtfHistory.reduce((a, b) => a + b, 0) / rtfHistory.length;
      els.info.textContent = `${rs.archName} | ${rs.sampleRate} Hz | RTF ${avg.toFixed(2)}`;
      if (els.rtfWarn) {
        if (avg > 1.0 && !rtfWarned) {
          rtfWarned = true;
          const hint = els.useLlm.checked
            ? '关闭 "Use LLM" 可显著提升速度（仅 CTC 解码）。'
            : '当前已关闭 LLM；浏览器 WebGPU 内核调度开销限制了 encoder 吞吐。';
          els.rtfWarn.textContent = `⚠ RTF ≈ ${avg.toFixed(2)} — 无法实时跟上音频。${hint}`;
          els.rtfWarn.style.display = 'block';
        } else if (avg <= 0.9 && rtfWarned) {
          rtfWarned = false;
          els.rtfWarn.style.display = 'none';
        }
      }
    }

    function appendTranscript(start, end, text, opts = {}) {
      const line = document.createElement('div');
      line.className = 'transcript-line';
      if (opts.kind) line.dataset.kind = opts.kind;
      renderLine(line, start, end, text, opts.kind);
      els.transcript.appendChild(line);
      els.transcript.scrollTop = els.transcript.scrollHeight;
      return line;
    }

    function updateTranscriptLine(line, start, end, text, opts = {}) {
      if (!line) return;
      if (opts.kind) line.dataset.kind = opts.kind;
      renderLine(line, start, end, text, opts.kind);
      els.transcript.scrollTop = els.transcript.scrollHeight;
    }

    function renderLine(line, start, end, text, kind) {
      // CTC partial gets a tag + greyed text; LLM result replaces it.
      const tag = kind === 'ctc' ? ' <span class="tag tag-ctc">CTC</span>'
                : kind === 'llm' ? ' <span class="tag tag-llm">LLM</span>'
                : '';
      const cls = kind === 'ctc' ? ' style="opacity:0.65; font-style:italic;"' : '';
      line.innerHTML = `<span class="timestamp">[${start} → ${end}]</span>${tag} <span${cls}>${escapeHtml(text)}</span>`;
    }

    function stopMic() {
      if (!running) return;
      running = false;
      els.startBtn.disabled = false;
      els.stopBtn.disabled  = true;

      if (!vad || !vad.isReady) {
        if (inSpeech && speechBuffer.length > 0) finalizeEnergySegment();
      }
      carry = new Float32Array(0);

      if (workletNode) { try { workletNode.disconnect(); } catch {} workletNode = null; }
      if (scriptNode)  { try { scriptNode.disconnect();  } catch {} scriptNode  = null; }
      if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
      if (audioCtx && audioCtx.state !== 'closed') { audioCtx.close().catch(() => {}); audioCtx = null; }

      setStatus('Stopped.', 'idle');
      updateVadDisplay(0, false);
    }

    function clearTranscript() {
      els.transcript.innerHTML = '';
      els.partial.textContent = '';
      audioTime = 0;
    }

    function updateVadDisplay(prob, isSpeech) {
      els.vadBar.style.width = (prob * 100).toFixed(1) + '%';
      els.vadBar.className = 'vad-fill' + (isSpeech ? ' speech' : '');
      els.vadLabel.textContent = isSpeech ? 'SPEECH' : 'silence';
      els.vadLabel.className = 'vad-label' + (isSpeech ? ' speech' : '');
    }
  }

  window.RSPageAsrOnline = { init };
})();
