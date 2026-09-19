const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

const calls = [];
let pinSet = true;
let serverPin = '4821';
let lockedOnce = false;

const json = (status, body) => ({ ok: status >= 200 && status < 300, status, json: async () => body });
const context = vm.createContext({
  document: {},
  fetch: async (url, opts = {}) => {
    calls.push({ url, method: opts.method || 'GET', headers: opts.headers, body: opts.body });
    if (url === '/api/admin/pin' && !opts.method) return json(200, { is_set: pinSet });
    if (url === '/api/admin/pin' && opts.method === 'POST') {
      const body = JSON.parse(opts.body);
      if (pinSet && body.current_pin !== serverPin) return json(401, { detail: { code: 'pin_invalid', message: 'รหัสไม่ถูกต้อง' } });
      pinSet = true; serverPin = body.new_pin;
      return json(200, { is_set: true });
    }
    // a protected delete
    if (lockedOnce) { lockedOnce = false; return json(429, { detail: { code: 'pin_locked', message: 'ใส่รหัสผิดหลายครั้ง กรุณารอ 60 วินาที' } }); }
    if (opts.headers['X-Admin-Pin'] !== serverPin) return json(401, { detail: { code: 'pin_invalid', message: 'รหัสไม่ถูกต้อง' } });
    return json(200, { ok: true });
  },
  console, Object, JSON,
});
vm.runInContext(fs.readFileSync('frontend/static/js/admin-pin.js', 'utf8'), context);

// Stand in for the modal: each call takes the next scripted answer (null = cancel).
const asked = [];
const script = answers => {
  asked.length = 0;
  context.pinDialog = async options => { asked.push(options); return answers.shift() ?? null; };
};
const deletes = () => calls.filter(c => c.method === 'DELETE');
const reset = () => { calls.length = 0; };
const target = { title: 'ลบสินค้า', message: 'ลบ?' };

(async () => {
  // Right PIN -> DELETE carries it in the header.
  script([{ pin: '4821' }]);
  let res = await context.protectedDelete('/api/products/1', target);
  assert.equal(res.ok, true);
  assert.equal(deletes().length, 1);
  assert.equal(deletes()[0].headers['X-Admin-Pin'], '4821', 'PIN travels in the X-Admin-Pin header');
  assert.equal(asked[0].danger, true, 'delete prompt is styled as dangerous');

  // Wrong PIN -> asked again with the server's message, then succeeds.
  reset();
  script([{ pin: '0000' }, { pin: '4821' }]);
  res = await context.protectedDelete('/api/products/1', target);
  assert.equal(res.ok, true);
  assert.equal(asked.length, 2, 'prompted a second time after a wrong PIN');
  assert.equal(asked[1].error, 'รหัสไม่ถูกต้อง', 'the error is shown on the retry');
  assert.equal(deletes().length, 2);

  // Temporarily locked -> message shown, and cancelling stops cleanly.
  reset();
  lockedOnce = true;
  script([{ pin: '4821' }, null]);
  res = await context.protectedDelete('/api/products/1', target);
  assert.equal(res, null);
  assert(asked[1].error.includes('รอ 60 วินาที'), 'lock-out message is shown');

  // Cancelling the prompt never sends a DELETE.
  reset();
  script([null]);
  res = await context.protectedDelete('/api/products/1', target);
  assert.equal(res, null);
  assert.equal(deletes().length, 0, 'no request is made when staff cancel');

  // No PIN yet: staff must choose one first (mismatch is rejected), then the delete proceeds.
  reset();
  pinSet = false; serverPin = '';
  script([{ pin: '1234', again: '1235' }, { pin: '1234', again: '1234' }, { pin: '1234' }]);
  res = await context.protectedDelete('/api/products/1', target);
  assert.equal(res.ok, true);
  assert.equal(asked[0].title, 'ตั้งรหัสผ่านแอดมิน');
  assert.equal(asked[1].error, 'รหัสสองครั้งไม่ตรงกัน', 'mismatching entries are rejected');
  const setup = calls.filter(c => c.url === '/api/admin/pin' && c.method === 'POST');
  assert.equal(setup.length, 1, 'only the matching entry is sent');
  assert.equal(JSON.parse(setup[0].body).new_pin, '1234');
  assert.equal(deletes().length, 1);

  // Cancelling the first-time setup deletes nothing.
  reset();
  pinSet = false; serverPin = '';
  script([null]);
  assert.equal(await context.protectedDelete('/api/products/1', target), null);
  assert.equal(deletes().length, 0);

  // Changing the PIN sends the current and the new one; a wrong current PIN is reported.
  reset();
  pinSet = true; serverPin = '1234';
  script([{ current: '0000', pin: '5555', again: '5555' }, { current: '1234', pin: '5555', again: '5555' }]);
  assert.equal(await context.changeAdminPin(), true);
  assert.equal(asked[1].error, 'รหัสไม่ถูกต้อง', 'wrong current PIN is explained');
  const change = calls.filter(c => c.method === 'POST').at(-1);
  assert.deepEqual(JSON.parse(change.body), { new_pin: '5555', current_pin: '1234' });
  assert.equal(serverPin, '5555');

  console.log('PASS: PIN prompt before deleting, retry on wrong PIN, first-time setup, cancel, change PIN');
})().catch(e => { console.error(e); process.exitCode = 1; });
