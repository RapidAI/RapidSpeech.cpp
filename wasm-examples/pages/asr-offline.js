/**
 * ASR Offline — upload a file or record a clip, transcribe the whole thing.
 * Optional neural VAD (silero-vad / firered-vad) segments long clips into
 * speech regions and transcribes each region separately.
 *
 * Exposes window.RSPageAsrOffline.init({root, createInstance}).
 */
(function () {
  const { decodeFileToMono16k, escapeHtml, statusSetter, TARGET_SR } = window.RSUtils;

  function $(root, id) { return root.querySelector(`[data-id="${id}"]`); }

  async function init({ root, createInstance }) {
    const els = {
      modelUrl: $(root, 'model-url'),
      nThreads: $(root, 'n-threads'),
      useLlm:   $(root, 'use-llm'),
      loadBtn:  $(root, 'load-btn'),
      status:   $(root, 'status'),
      progress: $(root, 'progress-container'),
      progressBar: $(root, 'progress-bar'),

      vadUrl:        $(root, 'vad-url'),
      vadThreshold:  $(root, 'vad-threshold'),
      vadThresholdVal: $(root, 'vad-threshold-val'),
      vadMinSeg:     $(root, 'vad-min-seg'),
      vadLoadBtn:    $(root, 'vad-load-btn'),
      vadStatus:     $(root, 'vad-status'),
      vadProgress:   $(root, 'vad-progress-container'),
      vadProgressBar:$(root, 'vad-progress-bar'),

      sourceMode: $(root, 'source-mode'),
      uploadRow:  $(root, 'upload-row'),
      recordRow:  $(root, 'record-row'),
      audioFile:  $(root, 'audio-file'),
      recBtn:     $(root, 'rec-btn'),
      recTime:    $(root, 'rec-time'),

      runBtn:      $(root, 'run-btn'),
      redecodeBtn: $(root, 'redecode-btn'),
      transcript:  $(root, 'transcript'),
    };

    const setStatus    = statusSetter(els.status);
    const setVadStatus = statusSetter(els.vadStatus);

    let rs = null;
    let vad = null;
    let pendingPcm = null;   // Float32Array @ 16k, ready to run
    let mediaStream = null;
    let mediaRecorder = null;
    let recChunks = [];
    let recStartTime = 0;
    let recTimer = null;
    let recording = false;

    els.vadThreshold.addEventListener('input', () => {
      const v = parseFloat(els.vadThreshold.value);
      els.vadThresholdVal.textContent = v.toFixed(2);
      if (vad && vad.isReady) vad.setThreshold(v);
    });
    els.vadThresholdVal.textContent = parseFloat(els.vadThreshold.value).toFixed(2);

    els.sourceMode.addEventListener('change', () => {
      const isRecord = els.sourceMode.value === 'record';
      els.uploadRow.style.display = isRecord ? 'none' : '';
      els.recordRow.style.display = isRecord ? '' : 'none';
    });

    els.loadBtn.addEventListener('click', loadModel);
    els.vadLoadBtn.addEventListener('click', loadVad);
    els.audioFile.addEventListener('change', onFilePicked);
    els.recBtn.addEventListener('click', toggleRecord);
    els.runBtn.addEventListener('click', runTranscribe);
    els.redecodeBtn.addEventListener('click', runRedecode);

    async function loadModel() {
      const url = els.modelUrl.value.trim();
      if (!url) { setStatus('Please enter a model URL.', 'error'); return; }

      setStatus('Downloading model...', 'loading');
      els.loadBtn.disabled = true;
      els.progress.style.display = 'block';

      try {
        rs = await createInstance();
        const nThreads = parseInt(els.nThreads.value, 10) || 4;
        await rs.initAsr(url, nThreads, (p) => {
          els.progressBar.style.width = (p * 100).toFixed(1) + '%';
        });
        rs.setUseLlm(els.useLlm.checked);
        els.useLlm.addEventListener('change', () => rs.setUseLlm(els.useLlm.checked));
        els.progress.style.display = 'none';
        els.recBtn.disabled = false;
        if (pendingPcm) els.runBtn.disabled = false;
        setStatus(`Ready — ${rs.archName} @ ${rs.sampleRate}Hz`, 'ready');
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
        // The VAD shares the same WASM module as the ASR context. Each tab
        // already has its own MODULARIZE'd module, so the VAD lives next to
        // the rs context on that module.
        if (!rs) {
          rs = await createInstance();
        }
        vad = new window.RapidSpeechVAD(rs._mod);
        await vad.load(url, 2, (p) => {
          els.vadProgressBar.style.width = (p * 100).toFixed(1) + '%';
        });
        vad.setThreshold(parseFloat(els.vadThreshold.value));
        els.vadProgress.style.display = 'none';
        setVadStatus(`Ready — ${vad.arch}`, 'ready');
      } catch (err) {
        setVadStatus('Failed: ' + err.message, 'error');
        els.vadProgress.style.display = 'none';
        vad = null;
      } finally {
        els.vadLoadBtn.disabled = false;
      }
    }

    async function onFilePicked() {
      const f = els.audioFile.files[0];
      if (!f) return;
      setStatus('Decoding audio...', 'loading');
      try {
        pendingPcm = await decodeFileToMono16k(f);
        setStatus(`Decoded ${f.name} — ${(pendingPcm.length / TARGET_SR).toFixed(2)} s`, 'ready');
        if (rs) els.runBtn.disabled = false;
      } catch (err) {
        setStatus('Decode failed: ' + err.message, 'error');
      }
    }

    async function toggleRecord() {
      if (recording) {
        stopRecording();
        return;
      }
      try {
        mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: { ideal: TARGET_SR } }
        });
      } catch { setStatus('Microphone access denied.', 'error'); return; }

      recChunks = [];
      mediaRecorder = new MediaRecorder(mediaStream);
      mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recChunks.push(e.data); };
      mediaRecorder.onstop = async () => {
        const blob = new Blob(recChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        setStatus('Decoding recording...', 'loading');
        try {
          pendingPcm = await decodeFileToMono16k(blob);
          setStatus(`Recorded ${(pendingPcm.length / TARGET_SR).toFixed(2)} s`, 'ready');
          if (rs) els.runBtn.disabled = false;
        } catch (err) {
          setStatus('Decode failed: ' + err.message, 'error');
        }
      };
      mediaRecorder.start();
      recording = true;
      recStartTime = performance.now();
      els.recBtn.textContent = '■ Stop';
      recTimer = setInterval(() => {
        const t = (performance.now() - recStartTime) / 1000;
        const m = Math.floor(t / 60);
        const s = (t % 60).toFixed(1);
        els.recTime.textContent = String(m).padStart(2, '0') + ':' + String(s).padStart(4, '0');
      }, 100);
    }

    function stopRecording() {
      if (!recording) return;
      recording = false;
      els.recBtn.textContent = '● Record';
      clearInterval(recTimer);
      mediaRecorder.stop();
      mediaStream.getTracks().forEach(t => t.stop());
      mediaStream = null;
    }

    // Detect speech segments using the loaded VAD.
    function detectSegments(pcm) {
      vad.reset();
      vad.setThreshold(parseFloat(els.vadThreshold.value));
      // Push in chunks so we don't fight a huge single allocation.
      const STRIDE = 16000;
      for (let off = 0; off < pcm.length; off += STRIDE) {
        const end = Math.min(off + STRIDE, pcm.length);
        vad.pushAudio(pcm.subarray(off, end));
      }
      const segments = vad.drainSegments(1024);
      // If the clip ends mid-speech, close out a final segment so we don't
      // drop the trailing audio.
      if (vad.isSpeech) {
        const lastEnd = pcm.length / TARGET_SR;
        const lastStart = segments.length > 0 ? segments[segments.length - 1].end_s : 0;
        segments.push({ start_s: Math.max(lastStart, 0), end_s: lastEnd });
      }
      const minSeg = parseFloat(els.vadMinSeg.value) || 0;
      return segments.filter(s => s.end_s - s.start_s >= minSeg);
    }

    async function runTranscribe() {
      if (!rs || !pendingPcm) return;
      els.runBtn.disabled = true;
      els.redecodeBtn.disabled = true;

      const useVad = vad && vad.isReady;
      const audioDur = pendingPcm.length / TARGET_SR;

      if (!useVad) {
        setStatus('Running ASR (full clip)...', 'loading');
        rs.reset();
        rs.pushAudio(pendingPcm);
        const t0 = performance.now();
        let result;
        try { result = await rs.process(); }
        catch (err) { setStatus('ASR failed: ' + err.message, 'error'); els.runBtn.disabled = false; return; }
        const dt = (performance.now() - t0) / 1000;
        const rtf = dt / audioDur;
        if (result.status > 0 && result.text) {
          appendLine(0, audioDur, result.text, rtf);
          els.redecodeBtn.disabled = false;
        } else {
          appendLine(0, audioDur, '(no speech detected)', rtf);
        }
        setStatus(`Done in ${dt.toFixed(2)} s (RTF ${rtf.toFixed(2)})`, 'ready');
        els.runBtn.disabled = false;
        return;
      }

      setStatus('Running VAD...', 'loading');
      const segments = detectSegments(pendingPcm);
      if (segments.length === 0) {
        appendLine(0, audioDur, '(no speech detected by VAD)', 0);
        setStatus('VAD found no speech.', 'ready');
        els.runBtn.disabled = false;
        return;
      }

      const t0 = performance.now();
      for (let i = 0; i < segments.length; ++i) {
        const seg = segments[i];
        setStatus(`Transcribing segment ${i+1}/${segments.length}...`, 'loading');
        const s0 = Math.max(0, Math.floor(seg.start_s * TARGET_SR));
        const s1 = Math.min(pendingPcm.length, Math.ceil(seg.end_s * TARGET_SR));
        if (s1 <= s0) continue;
        const slice = pendingPcm.subarray(s0, s1);
        rs.reset();
        rs.pushAudio(slice);
        let result;
        try { result = await rs.process(); }
        catch (err) { appendLine(seg.start_s, seg.end_s, `(error: ${err.message})`, null); continue; }
        const text = (result.status > 0 && result.text) ? result.text : '(no speech)';
        const segDur = seg.end_s - seg.start_s;
        appendLine(seg.start_s, seg.end_s, text, null, segDur);
      }
      const dt = (performance.now() - t0) / 1000;
      const rtf = dt / audioDur;
      els.redecodeBtn.disabled = false;
      setStatus(`Done — ${segments.length} segment(s) in ${dt.toFixed(2)} s (RTF ${rtf.toFixed(2)})`, 'ready');
      els.runBtn.disabled = false;
    }

    async function runRedecode() {
      if (!rs) return;
      els.redecodeBtn.disabled = true;
      setStatus('Re-decoding...', 'loading');
      const t0 = performance.now();
      let result;
      try { result = await rs.redecode(); }
      catch (err) { setStatus('Redecode failed: ' + err.message, 'error'); els.redecodeBtn.disabled = false; return; }
      const dt = (performance.now() - t0) / 1000;
      if (result.status > 0 && result.text) {
        appendLine(null, null, '↻ ' + result.text, null);
      }
      setStatus(`Re-decoded in ${dt.toFixed(2)} s`, 'ready');
      els.redecodeBtn.disabled = false;
    }

    function appendLine(startS, endS, text, rtf, segDur) {
      const line = document.createElement('div');
      line.className = 'transcript-line';
      let meta = '';
      if (startS != null && endS != null) {
        meta = `[${startS.toFixed(2)}s → ${endS.toFixed(2)}s]`;
        if (rtf != null) meta += ` RTF ${rtf.toFixed(2)}`;
      } else if (rtf != null) {
        meta = `[RTF ${rtf.toFixed(2)}]`;
      }
      line.innerHTML = `<span class="timestamp">${meta}</span> ${escapeHtml(text)}`;
      els.transcript.appendChild(line);
      els.transcript.scrollTop = els.transcript.scrollHeight;
    }
  }

  window.RSPageAsrOffline = { init };
})();
