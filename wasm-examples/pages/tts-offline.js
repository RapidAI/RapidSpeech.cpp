/**
 * TTS Offline — text → audio, optional voice cloning via reference audio.
 * Exposes window.RSPageTtsOffline.init({root, createInstance}).
 */
(function () {
  const { decodeFileToMonoAt, encodeWav, statusSetter } = window.RSUtils;

  function $(root, id) { return root.querySelector(`[data-id="${id}"]`); }

  async function init({ root, createInstance }) {
    const els = {
      modelUrl: $(root, 'model-url'),
      nThreads: $(root, 'n-threads'),
      loadBtn:  $(root, 'load-btn'),
      status:   $(root, 'status'),
      progress: $(root, 'progress-container'),
      progressBar: $(root, 'progress-bar'),

      instruct: $(root, 'instruct'),
      language: $(root, 'language'),
      seed:     $(root, 'seed'),
      steps:    $(root, 'steps'),
      stepsVal: $(root, 'steps-val'),

      refAudio: $(root, 'ref-audio'),
      refText:  $(root, 'ref-text'),

      text:     $(root, 'text'),
      genBtn:   $(root, 'gen-btn'),
      clearBtn: $(root, 'clear-btn'),

      audioOutput: $(root, 'audio-output'),
      audio:       $(root, 'audio'),
      download:    $(root, 'download'),
      genInfo:     $(root, 'gen-info'),
    };

    const setStatus = statusSetter(els.status);

    let rs = null;
    let refPcm = null;     // Float32Array at refSr
    let refSr  = 16000;
    let lastObjectUrl = null;

    els.steps.addEventListener('input', () => {
      els.stepsVal.textContent = els.steps.value;
    });
    els.stepsVal.textContent = els.steps.value;

    els.loadBtn.addEventListener('click', loadModel);
    els.refAudio.addEventListener('change', onRefPicked);
    els.genBtn.addEventListener('click', generate);
    els.clearBtn.addEventListener('click', clearOutput);

    async function loadModel() {
      const url = els.modelUrl.value.trim();
      if (!url) { setStatus('Please enter a model URL.', 'error'); return; }

      setStatus('Downloading model...', 'loading');
      els.loadBtn.disabled = true;
      els.progress.style.display = 'block';

      try {
        rs = await createInstance();
        const nThreads = parseInt(els.nThreads.value, 10) || 4;
        // Use generic init with TTS_OFFLINE task type so the engine batches the
        // whole utterance instead of streaming chunks.
        await rs.init(url, RS_TASK.TTS_OFFLINE, nThreads, (p) => {
          els.progressBar.style.width = (p * 100).toFixed(1) + '%';
        });
        refSr = rs.sampleRate || 24000;
        els.progress.style.display = 'none';
        els.genBtn.disabled = false;
        setStatus(`Ready — ${rs.archName} @ ${rs.sampleRate}Hz`, 'ready');
      } catch (err) {
        setStatus('Failed: ' + err.message, 'error');
        els.progress.style.display = 'none';
      } finally {
        els.loadBtn.disabled = false;
      }
    }

    async function onRefPicked() {
      const f = els.refAudio.files[0];
      if (!f) { refPcm = null; return; }
      setStatus('Decoding reference audio...', 'loading');
      try {
        // Decode at the model's native rate (24k for OmniVoice, but we let the
        // engine resample if it differs). We use rs.sampleRate if loaded, else
        // a sensible default.
        refPcm = await decodeFileToMonoAt(f, refSr);
        setStatus(`Reference loaded (${(refPcm.length / refSr).toFixed(2)} s)`, 'ready');
      } catch (err) {
        setStatus('Decode failed: ' + err.message, 'error');
        refPcm = null;
      }
    }

    async function generate() {
      if (!rs) return;
      const text = els.text.value.trim();
      if (!text) { setStatus('Enter some text first.', 'error'); return; }

      els.genBtn.disabled = true;
      setStatus('Synthesizing...', 'loading');

      // Push voice-clone reference if provided.
      if (refPcm) {
        rs.pushReferenceAudio(refPcm, refSr);
        const refT = els.refText.value.trim();
        if (refT) rs.pushReferenceText(refT);
      }
      rs.setTtsParams({
        instruct: els.instruct.value,
        language: els.language.value,
        seed:     parseInt(els.seed.value, 10) || 42,
      });
      rs.setDiffusionSteps(parseInt(els.steps.value, 10) || 32);

      const t0 = performance.now();
      let pcm;
      try { pcm = await rs.synthesize(text); }
      catch (err) {
        setStatus('Synthesis failed: ' + err.message, 'error');
        els.genBtn.disabled = false;
        return;
      }
      const dt = (performance.now() - t0) / 1000;
      const dur = pcm.length / rs.sampleRate;
      const rtf = dt / Math.max(dur, 1e-6);

      const blob = encodeWav(pcm, rs.sampleRate);
      if (lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
      lastObjectUrl = URL.createObjectURL(blob);
      els.audio.src = lastObjectUrl;
      els.download.href = lastObjectUrl;
      els.audioOutput.hidden = false;
      els.genInfo.textContent =
        `${dur.toFixed(2)} s synthesized in ${dt.toFixed(2)} s — RTF ${rtf.toFixed(2)}`;
      setStatus('Done.', 'ready');
      els.genBtn.disabled = false;
    }

    function clearOutput() {
      els.text.value = '';
      els.audioOutput.hidden = true;
      if (lastObjectUrl) { URL.revokeObjectURL(lastObjectUrl); lastObjectUrl = null; }
    }
  }

  window.RSPageTtsOffline = { init };
})();
