/**
 * Tab router for the multi-task RapidSpeech web demo.
 *
 * Loads each per-tab module exactly once with a fresh RapidSpeechWASM
 * instance factory. Pages are isolated — switching tabs does not reset
 * loaded models on the inactive tabs.
 */
(function () {
  const GPU_PREF_KEY = 'rs_gpu_pref';

  // ── GPU preference select ──────────────────────────────────
  const gpuPrefEl = document.getElementById('gpu-pref');
  if (gpuPrefEl) {
    const saved = localStorage.getItem(GPU_PREF_KEY) || '';
    gpuPrefEl.value = saved;
    gpuPrefEl.addEventListener('change', () => {
      localStorage.setItem(GPU_PREF_KEY, gpuPrefEl.value);
      probeRuntime();
    });
  }

  function getGpuPref() {
    return (gpuPrefEl && gpuPrefEl.value) || localStorage.getItem(GPU_PREF_KEY) || '';
  }

  // Factory: each call returns a brand-new RapidSpeechWASM instance, which
  // owns its own Emscripten module (MODULARIZE=1 returns a fresh instance).
  async function createInstance() {
    // Catch the missing-isolation case before Emscripten's worker.postMessage
    // throws DataCloneError. The error there is opaque ("Failed to execute
    // 'postMessage' on 'Worker'"); a friendly message saves users a trip to
    // DevTools.
    if (typeof self !== 'undefined' && self.crossOriginIsolated === false) {
      throw new Error(
        '页面未启用 cross-origin isolation —— SharedArrayBuffer / pthreads 不可用。' +
        '请检查服务器是否返回 COOP=same-origin 与 COEP=require-corp/credentialless，' +
        '并直接通过 *.hf.space 域名访问 Space。');
    }
    const pref = getGpuPref();
    const mod = await RapidSpeechModule({
      // Inject the preference as an env var so ggml-webgpu's RequestAdapter
      // call picks it up via std::getenv("GGML_WEBGPU_POWER_PREFERENCE").
      preRun: [m => {
        if (pref) m.ENV['GGML_WEBGPU_POWER_PREFERENCE'] = pref;
      }],
    });
    return new RapidSpeechWASM(mod);
  }

  // Probe runtime capabilities and surface them in the header. We check two
  // things up-front because both are deal-breakers for the WASM module:
  //   1. crossOriginIsolated — required for SharedArrayBuffer / pthreads.
  //      Without it the worker postMessage throws DataCloneError before the
  //      user has a chance to wonder why "Load model" never finishes.
  //   2. WebGPU adapter — determines whether inference runs on GPU or falls
  //      back to WASM SIMD CPU.
  async function probeRuntime() {
    const badge = document.getElementById('webgpu-badge');
    if (!badge) return;
    const setBadge = (html, cls) => { badge.innerHTML = html; badge.className = 'webgpu-badge ' + cls; };

    // ── Cross-origin isolation (pthreads + SharedArrayBuffer) ──
    const isolated = (typeof self !== 'undefined' && self.crossOriginIsolated === true);
    const hasSAB   = (typeof SharedArrayBuffer !== 'undefined');
    if (!isolated || !hasSAB) {
      // The coi-serviceworker shim reloads the page on first visit; during
      // that window the page is briefly not isolated. Detect that case so we
      // show a "registering…" hint instead of a hard error.
      const swPending = ('serviceWorker' in navigator) &&
                        navigator.serviceWorker &&
                        !navigator.serviceWorker.controller;
      if (swPending) {
        setBadge('⏳ 正在注册 Service Worker 以启用 cross-origin isolation… 页面即将自动刷新。', 'warn');
      } else {
        setBadge(
          '⛔ 当前页面未启用 cross-origin isolation —— 加载模型会失败 ' +
          '(SharedArrayBuffer / pthreads 不可用)。' +
          '若服务器无法设置 COOP/COEP，请确认 <code>coi-serviceworker.js</code> ' +
          '已加载（HTTPS 必需），或者改用支持自定义 Header 的部署方式。',
          'error');
      }
      return;
    }

    // ── WebGPU ──
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      setBadge('⚠ WebGPU 不可用 — 将回退到 CPU (WASM SIMD)。请使用 Chrome/Edge 113+ 或 Safari TP。', 'warn');
      return;
    }
    try {
      const pref = getGpuPref();
      const opts = pref ? { powerPreference: pref } : {};
      const adapter = await navigator.gpu.requestAdapter(opts);
      if (!adapter) {
        setBadge('⚠ WebGPU API 可用但未获取到适配器 — 将回退到 CPU。', 'warn');
        return;
      }
      let info = '';
      try {
        const i = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
        if (i) {
          const parts = [i.vendor, i.architecture, i.device].filter(Boolean).map(s => String(s).trim()).filter(Boolean);
          if (parts.length) info = ' · ' + parts.join(' / ');
        }
      } catch {}
      const prefLabel = pref === 'high-performance' ? ' [高性能]' : pref === 'low-power' ? ' [低功耗]' : '';
      setBadge('✓ WebGPU + pthreads 已启用 — GPU 加速' + info + prefLabel, 'ok');
    } catch (err) {
      setBadge('⚠ WebGPU 探测失败: ' + (err && err.message ? err.message : err) + ' — 回退到 CPU。', 'warn');
    }
  }
  probeRuntime();

  const tabs = [
    { id: 'asr-offline', mod: window.RSPageAsrOffline },
    { id: 'asr-online',  mod: window.RSPageAsrOnline  },
    { id: 'tts-offline', mod: window.RSPageTtsOffline },
  ];

  function showTab(id) {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === id);
    });
    document.querySelectorAll('section[data-page]').forEach(s => {
      s.hidden = s.dataset.page !== id;
    });
    if (location.hash.slice(1) !== id) {
      history.replaceState(null, '', '#' + id);
    }
  }

  // Wire tab clicks.
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => showTab(btn.dataset.tab));
  });

  // Initialize every page once (so DOM listeners and state are bound).
  for (const { id, mod } of tabs) {
    if (!mod) { console.warn('Missing page module for', id); continue; }
    const root = document.querySelector(`section[data-page="${id}"]`);
    if (!root) { console.warn('Missing root for', id); continue; }
    mod.init({ root, createInstance });
  }

  // Honor URL hash on load.
  const initial = (location.hash || '').slice(1);
  const valid = tabs.find(t => t.id === initial);
  showTab(valid ? initial : 'asr-offline');
})();


  const tabs = [
    { id: 'asr-offline', mod: window.RSPageAsrOffline },
    { id: 'asr-online',  mod: window.RSPageAsrOnline  },
    { id: 'tts-offline', mod: window.RSPageTtsOffline },
  ];

  function showTab(id) {
    document.querySelectorAll('.tab-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.tab === id);
    });
    document.querySelectorAll('section[data-page]').forEach(s => {
      s.hidden = s.dataset.page !== id;
    });
    if (location.hash.slice(1) !== id) {
      history.replaceState(null, '', '#' + id);
    }
  }

  // Wire tab clicks.
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => showTab(btn.dataset.tab));
  });

  // Initialize every page once (so DOM listeners and state are bound).
  for (const { id, mod } of tabs) {
    if (!mod) { console.warn('Missing page module for', id); continue; }
    const root = document.querySelector(`section[data-page="${id}"]`);
    if (!root) { console.warn('Missing root for', id); continue; }
    mod.init({ root, createInstance });
  }

  // Honor URL hash on load.
  const initial = (location.hash || '').slice(1);
  const valid = tabs.find(t => t.id === initial);
  showTab(valid ? initial : 'asr-offline');
})();
