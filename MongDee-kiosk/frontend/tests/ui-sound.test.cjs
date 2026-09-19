const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

let oscillators = 0;
class FakeParam { setValueAtTime() {} exponentialRampToValueAtTime() {} }
class FakeAudioContext {
  constructor() { this.currentTime = 0; this.state = 'running'; this.destination = {}; }
  createOscillator() { oscillators++; return { frequency: new FakeParam(), connect(node) { return node; }, start() {}, stop() {} }; }
  createGain() { return { gain: new FakeParam(), connect(node) { return node; } }; }
  resume() {}
}

const listeners = [];
const storage = new Map();
const toggle = { textContent: '', clicks: [], addEventListener(type, fn) { this.clicks.push(fn); } };
const context = vm.createContext({
  document: {
    addEventListener(type, fn, capture) { listeners.push({ type, fn, capture }); },
    getElementById: id => (id === 'uiSoundToggle' ? toggle : null),
  },
  window: { AudioContext: FakeAudioContext },
  localStorage: { getItem: k => (storage.has(k) ? storage.get(k) : null), setItem: (k, v) => storage.set(k, v) },
});
vm.runInContext(fs.readFileSync('frontend/static/js/ui-sound.js', 'utf8'), context);

const click = listeners.find(l => l.type === 'click');
assert(click && click.capture, 'a capturing click listener is installed');

const target = (props = {}) => ({ className: '', disabled: false, getAttribute: () => null, ...props });
const press = el => click.fn({ target: { closest: () => el } });

press(target());
assert.equal(oscillators, 1, 'a button press plays a sound');

press(target({ disabled: true }));
assert.equal(oscillators, 1, 'a disabled button is silent');

press(target({ getAttribute: name => (name === 'aria-disabled' ? 'true' : null) }));
assert.equal(oscillators, 1, 'an aria-disabled control is silent');

click.fn({ target: { closest: () => null } });
assert.equal(oscillators, 1, 'clicking non-interactive content is silent');

assert.equal(toggle.textContent, 'เสียงปุ่มกด: เปิด', 'toggle label starts as on');
toggle.clicks[0]();
assert.equal(toggle.textContent, 'เสียงปุ่มกด: ปิด', 'toggle switches sounds off');
assert.equal(storage.get('mongdee_ui_sound'), 'off', 'the choice is remembered');
const before = oscillators;
press(target());
assert.equal(oscillators, before, 'no sound while switched off');

toggle.clicks[0]();
assert.equal(toggle.textContent, 'เสียงปุ่มกด: เปิด', 'toggle switches sounds back on');
assert.equal(oscillators, before + 1, 'turning sounds on plays a confirmation click');

console.log('PASS: button sounds play, skip disabled/non-buttons, and can be switched off');
