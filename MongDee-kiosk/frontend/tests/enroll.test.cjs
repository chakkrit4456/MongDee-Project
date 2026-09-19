const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
class Element {
  constructor() { this.children = []; this.style = {}; this.handlers = {}; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  addEventListener(event, handler) { this.handlers[event] = handler; }
  querySelector() { return submit; }
  reset() {}
  focus() {}
  scrollIntoView() {}
}
const submit = new Element();
const ids = new Map();
function element(id) { if (!ids.has(id)) ids.set(id, new Element()); return ids.get(id); }
let failCapture = false;
let confirmCalls = 0;
let catalog = [{ id: 1, name: 'Existing' }];
const batch = { session_id: 'scan-session', frames: [1, 2, 3].map(id => ({ frame_id: String(id), thumbnail_data_url: 'data:image/jpeg;base64,AA==' })) };
const context = vm.createContext({
  document: { getElementById: element, createElement: () => new Element() },
  URLSearchParams, location: { search: '' }, console,
  setInterval: () => 0, clearInterval: () => {},
  FormData: class { entries() { return Object.entries({ name: 'New product', price: '', production_date: '', expiry_date: '' }); } },
  fetch: async (url, options) => {
    if (url === '/api/camera/devices') return { ok: true, json: async () => ({ devices: [], current_index: 0 }) };
    if (url === '/api/camera/status') return { ok: true, json: async () => ({ camera_open: true, has_reference: true, state: 'empty', box_status: 'none' }) };
    if (url === '/api/calibration') return { ok: true, json: async () => ({ roi: { x: 0.2, y: 0.15, w: 0.6, h: 0.75 }, presence_on_ratio: 0.12, presence_off_ratio: 0.04 }) };
    if (url === '/api/products') return { ok: true, json: async () => catalog };
    if (url === '/api/enroll/start') return { ok: !failCapture, json: async () => failCapture ? { detail: 'Camera unavailable' } : batch };
    if (url.endsWith('/confirm')) {
      confirmCalls++;
      assert.equal(JSON.parse(options.body).selected_frame_ids.length, 3);
      const product = { id: 2, name: 'New product' };
      catalog = [...catalog, product];
      return { ok: true, json: async () => product };
    }
    return { ok: true, json: async () => ({}) };
  },
});
vm.runInContext(fs.readFileSync('frontend/static/js/detect-box.js', 'utf8'), context);
vm.runInContext(fs.readFileSync('frontend/static/js/price.js', 'utf8'), context);
vm.runInContext(fs.readFileSync('frontend/static/js/enroll.js', 'utf8'), context);
(async () => {
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(element('enrolledProducts').children.length, 1);
  await context.runCapture('/api/enroll/start', true);
  assert.equal(element('frameGrid').children.length, 3);
  failCapture = true;
  await context.runCapture('/api/enroll/start', true);
  assert.equal(element('frameGrid').children.length, 3, 'failed rescan must preserve previous frames');
  await element('productForm').handlers.submit({ preventDefault() {} });
  assert.equal(confirmCalls, 1);
  assert.equal(element('enrolledProducts').children.length, 2, 'catalog must include both old and new products');
  assert.equal(element('savedResult').style.display, 'block');
  assert(element('savedMessage').textContent.includes('New product'));
  await element('productForm').handlers.submit({ preventDefault() {} });
  assert.equal(confirmCalls, 1, 'saved session cannot be submitted again');
  console.log('PASS: scan/review/save, failed rescan recovery, appended catalog, persistent success, no duplicate submission');
})().catch(error => { console.error(error); process.exitCode = 1; });
