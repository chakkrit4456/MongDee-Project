const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.children = [];
    this.style = {};
    this.classList = { toggle() {}, contains() { return false; }, remove() {}, add() {} };
    this.open = false;
  }
  append(...nodes) {
    for (const n of nodes) { if (n && typeof n === 'object') n.parentElement = this; }
    this.children.push(...nodes);
  }
  replaceChildren(...nodes) {
    for (const n of nodes) { if (n && typeof n === 'object') n.parentElement = this; }
    this.children = nodes;
  }
  addEventListener() {}
  querySelectorAll(selector) {
    const tag = selector.toLowerCase();
    const out = [];
    const walk = node => { for (const c of node.children) { if (c.tag === tag) out.push(c); walk(c); } };
    walk(this);
    return out;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
  getBoundingClientRect() { return { left: 0, right: 0, top: 0, bottom: 0 }; }
  pause() {}
  get childElementCount() { return this.children.length; }
  showModal() { this.open = true; }
  close() { this.open = false; }
}
const ids = new Map();
const element = id => { if (!ids.has(id)) ids.set(id, new Element()); return ids.get(id); };

const products = [
  { id: 1, name: 'กระเป๋าสาน', price: 1290, category: 'คราฟต์', story: 'เรื่องเล่า', video_path: 'captures/videos/a.mp4' },
  { id: 2, name: 'ผ้าทอ', price: 850, origin: 'น่าน' },
];
const context = vm.createContext({
  document: { getElementById: element, createElement: tag => new Element(tag), addEventListener() {} },
  localStorage: { getItem() { return null; }, setItem() {} },
  fetch: async () => ({ ok: true, json: async () => products }),
  Intl, URL, console,
});
vm.runInContext(fs.readFileSync('frontend/static/js/spin-viewer.js', 'utf8'), context);
vm.runInContext(fs.readFileSync('frontend/static/js/price.js', 'utf8'), context);
vm.runInContext(fs.readFileSync('frontend/static/js/products.js', 'utf8'), context);

(async () => {
  await new Promise(r => setImmediate(r));
  const grid = element('showcaseGrid');
  assert.equal(grid.children.length, 2, 'one card per product');
  assert(grid.children[0].children.some(n => n.children?.some(c => c.textContent === '▶ วิดีโอ')),
    'a product with a video shows the video badge');

  // Tapping a card opens the detail popup with the product's fields.
  context.openDetail(products[0]);
  const popup = element('productPopup');
  assert.equal(popup.open, true, 'popup opens on card tap');
  const flat = element('popupContent').children.flatMap(n => [n, ...n.children]);
  assert(flat.some(n => n.textContent === 'กระเป๋าสาน'), 'popup shows the product name');
  assert(flat.some(n => n.tag === 'video'), 'popup embeds the product video');

  context.closeDetail();
  assert.equal(popup.open, false, 'popup closes');

  // Lifting the product / closing the popup must stop an embedded
  // YouTube/Vimeo/Facebook iframe — closing the <dialog> alone does not.
  context.openDetail({ id: 6, name: 'G', video_link: 'https://www.youtube.com/watch?v=xyz789' });
  const iframe = element('popupContent').querySelector('iframe');
  assert(iframe && iframe.src.includes('xyz789'), 'the video link is embedded as an iframe');
  context.closeDetail();
  assert.equal(iframe.src, 'about:blank', 'the iframe is blanked so its audio actually stops');

  // Search + category filter.
  const search = element('searchInput');
  search.value = 'ผ้า';
  context.render();
  assert.equal(grid.children.length, 1, 'search narrows the grid');
  search.value = 'ผ้า น่าน';
  context.render();
  assert.equal(grid.children.length, 1, 'every typed word must match');
  search.value = 'ผ้า กระเป๋า';
  context.render();
  assert.equal(grid.children[0].className, 'showcase-empty', 'no match shows the empty state');
  grid.children[0].children.at(-1).onclick();
  assert.equal(search.value, '', 'clear button empties the search');
  assert.equal(grid.children.length, 2, 'clearing restores every product');

  const chips = element('categoryChips').children;  // ทั้งหมด, คราฟต์, ไม่ระบุหมวด
  assert.equal(chips.length, 3, 'one chip per category plus "all"');
  chips[1].onclick();
  assert.equal(grid.children.length, 1, 'category chip filters the grid');
  element('categoryChips').children[0].onclick();
  assert.equal(grid.children.length, 2, '"all" chip shows everything again');
  // Popup thumbnails: hand-added photos replace the auto-captured angle frames.
  const angles = Array.from({ length: 10 }, (_, i) => `captures/frames/f${i}.jpg`);
  const thumbsFor = product => {
    context.openDetail(product);
    const strip = element('popupContent').children.find(n => n.className === 'popup-thumbs');
    context.closeDetail();
    return strip ? strip.children.length : 0;
  };
  assert.equal(thumbsFor({ id: 3, name: 'auto', cover_image_path: 'captures/covers/c.jpg', spin_frames: angles }), 10,
    '360 + cover + 8 sampled angles when nothing was added by hand');
  assert.equal(thumbsFor({ id: 4, name: 'manual', cover_image_path: 'captures/covers/c.jpg', spin_frames: angles,
    gallery: [{ id: 1, path: 'captures/gallery/a.jpg' }, { id: 2, path: 'captures/gallery/b.jpg' }] }), 4,
    '360 + cover + only the 2 hand-added photos');
  assert.equal(thumbsFor({ id: 5, name: 'plain', cover_image_path: 'captures/covers/c.jpg' }), 0, 'a single image needs no thumbnail row');
  console.log('PASS: showcase grid, video badge, detail popup, search and category filter, hand-added gallery');
})().catch(e => { console.error(e); process.exitCode = 1; });
