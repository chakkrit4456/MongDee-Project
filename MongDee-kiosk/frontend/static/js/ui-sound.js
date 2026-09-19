// Button-press sounds, synthesized with Web Audio (no audio files — the kiosk
// runs offline). One delegated listener covers every button/link on the page.
// On by default; the ⚙ menu on the kiosk page has a switch, stored in
// localStorage under "mongdee_ui_sound" ("off" disables).
(function () {
  const STORAGE_KEY = "mongdee_ui_sound";
  const CLICKABLE = "button, a[href], [role='button'], summary, .showcase-card";

  // freq start -> end (Hz), duration (s), peak gain, oscillator type
  const SOUNDS = {
    click: [1000, 640, 0.07, 0.14, "triangle"],
    soft: [820, 600, 0.05, 0.09, "sine"],
    close: [700, 300, 0.12, 0.14, "sine"],
  };

  let context = null;

  function enabled() {
    try { return localStorage.getItem(STORAGE_KEY) !== "off"; } catch (e) { return true; }
  }

  function setEnabled(on) {
    try { localStorage.setItem(STORAGE_KEY, on ? "on" : "off"); } catch (e) { /* private mode */ }
  }

  function audio() {
    if (!context) {
      const Ctor = window.AudioContext || window.webkitAudioContext;
      if (!Ctor) return null;
      context = new Ctor();
    }
    if (context.state === "suspended") context.resume();
    return context;
  }

  function play(kind) {
    const spec = SOUNDS[kind] || SOUNDS.click;
    const ctx = audio();
    if (!ctx) return;
    const [from, to, duration, peak, type] = spec;
    const now = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(from, now);
    osc.frequency.exponentialRampToValueAtTime(to, now + duration);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(peak, now + 0.005);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + duration);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + duration + 0.02);
  }

  function kindFor(target) {
    const names = " " + (target.className || "") + " ";
    if (names.includes(" popup-done ")) return "close";
    if (/ (secondary|chip|active|gallery-thumb|showcase-card) /.test(names)) return "soft";
    return "click";
  }

  document.addEventListener("click", (event) => {
    if (!enabled() || !event.target || !event.target.closest) return;
    const target = event.target.closest(CLICKABLE);
    if (!target || target.disabled || target.getAttribute("aria-disabled") === "true") return;
    play(kindFor(target));
  }, true);

  const toggle = document.getElementById("uiSoundToggle");
  if (toggle) {
    const label = () => { toggle.textContent = enabled() ? "เสียงปุ่มกด: เปิด" : "เสียงปุ่มกด: ปิด"; };
    label();
    toggle.addEventListener("click", () => {
      setEnabled(!enabled());
      label();
      if (enabled()) play("click");
    });
  }

  window.playUiSound = play;
})();
