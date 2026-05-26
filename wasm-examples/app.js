/**
 * Tab router for the multi-task RapidSpeech web demo.
 *
 * Loads each per-tab module exactly once with a fresh RapidSpeechWASM
 * instance factory. Pages are isolated — switching tabs does not reset
 * loaded models on the inactive tabs.
 */
(function () {
  // Factory: each call returns a brand-new RapidSpeechWASM instance, which
  // owns its own Emscripten module (MODULARIZE=1 returns a fresh instance).
  async function createInstance() {
    const mod = await RapidSpeechModule();
    return new RapidSpeechWASM(mod);
  }

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
