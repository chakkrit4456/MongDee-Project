const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') { this.tag = tag; this.children = []; this.style = {}; this.className = ''; this.textContent = ''; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
}
const ids = new Map();
const element = id => { if (!ids.has(id)) ids.set(id, new Element()); return ids.get(id); };
const storage = new Map();
const requested = [];

const data = {
  all: {
    overview: { successful: 20, unmatched: 5, corrected: 2, orphaned: 871, success_rate: 0.8, products_total: 2, products_scanned: 1 },
    summary: [{ product_id: 1, name: 'ลำโพง', scan_count: 20 }, { product_id: 2, name: 'ตลับเมตร', scan_count: 0 }],
    hourly: [{ hour: 11, count: 12 }, { hour: 14, count: 8 }],
    daily: [{ day: '2026-09-17', count: 5 }, { day: '2026-09-18', count: 0 }, { day: '2026-09-19', count: 15 }],
  },
  today: {
    overview: { successful: 3, unmatched: 0, corrected: 0, orphaned: 0, success_rate: 1, products_total: 2, products_scanned: 1 },
    summary: [{ product_id: 2, name: 'ตลับเมตร', scan_count: 3 }, { product_id: 1, name: 'ลำโพง', scan_count: 0 }],
    hourly: [{ hour: 9, count: 3 }],
    daily: [{ day: '2026-09-19', count: 3 }],
  },
  '7d': {
    overview: { successful: 0, unmatched: 0, corrected: 0, orphaned: 0, success_rate: null, products_total: 2, products_scanned: 0 },
    summary: [{ product_id: 1, name: 'ลำโพง', scan_count: 0 }],
    hourly: [],
    daily: [{ day: '2026-09-19', count: 0 }],
  },
};

let confirmAnswer = true;
const methods = [];
const applyReset = on => {
  for (const r of Object.values(data)) { r.overview.day_reset = on; r.overview.day_start = on ? 1789754700 : 1789700000; }
};
applyReset(false);

const context = vm.createContext({
  document: { getElementById: element, createElement: tag => new Element(tag) },
  localStorage: { getItem: k => (storage.has(k) ? storage.get(k) : null), setItem: (k, v) => storage.set(k, v) },
  confirm: () => confirmAnswer,
  fetch: async (url, opts = {}) => {
    requested.push(url);
    if (url === '/api/analytics/reset-day') {
      methods.push(opts.method);
      applyReset(opts.method === 'POST');
      return { ok: true, json: async () => ({}) };
    }
    const [, kind, range] = /\/api\/analytics\/(\w+)\?range=(\w+)/.exec(url);
    return { ok: true, json: async () => data[range][kind] };
  },
  setInterval() {}, Date, Intl, Math, console,
});
vm.runInContext(fs.readFileSync('frontend/static/js/dashboard.js', 'utf8'), context);

const settle = async () => { for (let i = 0; i < 12; i++) await new Promise(r => setImmediate(r)); };
const text = id => String(element(id).textContent);  // the real DOM stringifies textContent

(async () => {
  await settle();
  const tabs = () => element('rangeTabs').children;
  assert.equal(tabs().length, 4, 'four range buttons');
  assert.equal(tabs().find(t => t.className === 'chip active').textContent, 'ทั้งหมด', 'defaults to all time');
  for (const kind of ['overview', 'summary', 'hourly', 'daily']) {
    assert(requested.includes(`/api/analytics/${kind}?range=all`), `${kind} is requested for the selected range`);
  }

  assert.equal(text('statTotalScans'), '20');
  assert.equal(text('statUnmatched'), '5');
  assert.equal(text('statRate'), '80%');
  assert.equal(text('statProducts'), '1/2');
  assert.equal(text('statCorrected'), '2');
  assert.equal(text('statTopProduct'), 'ลำโพง');
  assert(text('dashNote').includes('871'), 'the note says how many old scans are left out');
  assert(text('peakNote').includes('11:00'), 'busiest hour is named');
  assert.equal(element('hourlyChart').children.find(b => b.className === 'hourly-bar peak').children[0].textContent, '12', 'peak bar is marked and labelled');
  assert.equal(element('dailyChart').children.length, 3, 'one bar per day');

  // Switching the range refetches, remembers the choice and re-renders.
  tabs().find(t => t.textContent === 'วันนี้').onclick();
  await settle();
  assert(requested.includes('/api/analytics/overview?range=today'));
  assert.equal(storage.get('mongdee_report_range'), 'today');
  assert.equal(tabs().find(t => t.className === 'chip active').textContent, 'วันนี้');
  assert.equal(text('statTotalScans'), '3');
  assert.equal(text('statTopProduct'), 'ตลับเมตร');
  assert.equal(text('dashNote'), '', 'no note when nothing was left out');

  // A period without scans shows empty states, not zero-height charts.
  tabs().find(t => t.textContent === '7 วันล่าสุด').onclick();
  await settle();
  assert.equal(text('statRate'), '-', 'no success rate without any attempts');
  assert.equal(text('statTopProduct'), '-');
  assert.equal(element('hourlyChart').children[0].className, 'hint', 'hourly chart shows an empty message');
  assert.equal(element('dailyChart').children[0].className, 'hint', 'daily chart shows an empty message');

  // Export link follows the selected range.
  assert.equal(element('exportLink').href, '/api/analytics/export?range=7d', 'export link uses the current range');
  tabs().find(t => t.textContent === 'ทั้งหมด').onclick();
  await settle();
  assert.equal(element('exportLink').href, '/api/analytics/export?range=all');

  // "Start a new day": asks first, then restarts today without touching other ranges.
  assert.equal(element('dayNote').children.length, 0, 'no note before a reset');
  confirmAnswer = false;
  await element('resetDayBtn').onclick();
  assert.deepEqual(methods, [], 'declining the confirmation does nothing');
  confirmAnswer = true;
  await element('resetDayBtn').onclick();
  await settle();
  assert.deepEqual(methods, ['POST']);
  assert.equal(tabs().find(t => t.className === 'chip active').textContent, 'วันนี้', 'switches to today so the fresh counters are visible');
  const note = element('dayNote').children;
  assert(note[0].textContent.includes('เริ่มนับตั้งแต่'), 'the note says when today restarted');
  assert.equal(note[1].textContent, 'ยกเลิกการรีเซ็ต');

  await note[1].onclick();
  await settle();
  assert.deepEqual(methods, ['POST', 'DELETE']);
  assert.equal(element('dayNote').children.length, 0, 'note disappears after undoing the reset');

  console.log('PASS: report ranges, overview cards, hourly/daily charts, empty states, export link, new-day reset');
})().catch(e => { console.error(e); process.exitCode = 1; });
