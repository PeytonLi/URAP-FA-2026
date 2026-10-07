// Fictional UI records only. No browser, HTTP request, photo or human-review file is touched.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const script = fs.readFileSync(path.join(__dirname, '..', 'target_review.html'), 'utf8').match(/<script>([\s\S]*?)<\/script>/)[1];
const created = [], elements = new Map(), storage = new Map();
class Element {
  constructor(tag) { this.tag = tag; this.value = ''; this.children = []; this.listeners = {}; this.isConnected = true; created.push(this); }
  append(...values) { this.children.push(...values); }
  prepend(...values) { this.children.unshift(...values); }
  replaceChildren(...values) { this.children.forEach(e => e.isConnected = false); this.children = values; values.forEach(e => e.isConnected = true); }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  setAttribute(name, value) { this[name] = value; }
}
const element = id => { if (!elements.has(id)) elements.set(id, new Element('div')); return elements.get(id); };
element('filter').value = 'pending'; element('split').value = 'all';
const context = vm.createContext({
  document: { getElementById: element, createElement: tag => new Element(tag), addEventListener() {} },
  window: { addEventListener() {} },
  localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
  Option: class { constructor(label, value) { this.textContent = label; this.value = value; } },
  fetch: () => new Promise(() => {}), confirm: () => false, setTimeout: () => 1, clearTimeout() {}, assert
});
vm.runInContext(script, context); // Includes JavaScript syntax/initialization checking.
const run = source => vm.runInContext(source, context);
run(`busy=false; data={required:2, words:['Smiling','Happy','Cheerful','Angry','Negative','Sadness'], decisions:{}, drafts:[
 {image_id:'fictional-smile',split:'train',chosen:'Smiling, Happy, Cheerful',rejected:'Angry, Negative, Sadness',visible_evidence:'Fixture only.',rejected_reason:'Fixture only.',teacher_expression:'positive',subjective_confidence:0.8,review_flags:['emotion_words_are_tentative'],draft_source:'codex_direct_visual_review',draft_revision_sha256:'fixture-1',target_unresolved:false},
 {image_id:'fictional-uncertain',split:'development',chosen:'',rejected:'',visible_evidence:'Fixture only.',rejected_reason:'Fixture only.',teacher_expression:'uncertain',subjective_confidence:0.2,review_flags:['no_neutral_vocabulary'],draft_source:'codex_direct_visual_review',draft_revision_sha256:'fixture-2',target_unresolved:true}
 ]}; render();
 assert.equal(selection().length,2);
 assert.throws(()=>validateAnswers('Smiling, Happy, Cheerful','Cheerful, Smiling, Happy','approved',data.words));
 assert.throws(()=>validateAnswers('Smiling, Smiling, Happy','Angry, Negative, Sadness','approved',data.words));
 assert.throws(()=>validateAnswers('','','approved',data.words));
 validateAnswers('','','needs_revision',data.words);
 assert.equal($('export').disabled,true);
 assert.equal($('count').textContent,'0 / 2');
 dirty=true; move(1); assert.equal(page,0); dirty=false;
`);
const latest = (tag, condition = () => true) => created.filter(e => e.tag === tag && condition(e)).at(-1);
const checkboxes = () => created.filter(e => e.tag === 'input' && e.type === 'checkbox').slice(-2);
async function main() {
  let approve = latest('button', e => e.textContent === 'Approve & next');
  assert.equal(approve.disabled, true);
  element('reviewer').value = 'FICTIONAL-TEST-REVIEWER';
  checkboxes().forEach(e => e.checked = true);
  run('updateActions();'); assert.equal(approve.disabled, true); // Photo not loaded yet.
  latest('img').onload(); assert.equal(approve.disabled, false);
  context.fetch = async (route, options) => {
    assert.equal(route, '/api/decision');
    return { ok: true, json: async () => ({ ...JSON.parse(options.body), reviewed_at: '2026-10-05T00:00:00Z' }) };
  };
  await approve.onclick();
  run(`assert.equal(Object.keys(data.decisions).length,1);assert.equal(data.decisions['fictional-smile'].status,'approved');assert.equal(selection()[page].image_id,'fictional-uncertain');assert.equal($('export').disabled,true);`);
  approve = latest('button', e => e.textContent === 'Approve & next');
  checkboxes().forEach(e => e.checked = true);
  latest('img').onload(); run('updateActions();');
  assert.equal(approve.disabled, true); // Blank targets can never be approved.
  await latest('button', e => e.textContent === 'Save as unsure & next').onclick();
  run(`assert.equal(data.decisions['fictional-uncertain'].status,'needs_revision');assert.equal(data.decisions['fictional-uncertain'].chosen,'');assert.equal(selection().length,1);assert.equal($('export').disabled,true);`);
  const oldImage = latest('img'); oldImage.isConnected = false;
  run('render();'); oldImage.onload();
  run('assert.equal(photoReady,false);'); // A late old-photo load cannot unlock this photo.
  run(`$('filter').value='unresolved';page=0;render();assert.equal(selection().length,1);$('filter').value='approved';render();assert.equal(selection().length,1);`);
  assert.equal(JSON.parse(storage.get('urap-review-place')).image_id, 'fictional-smile');
  run(`data.decisions['fictional-smile'].reviewer_kind='ai';render();assert.equal($('progress').textContent,'1 AI-approved · 0 human-approved');assert.equal($('export').disabled,true);`);
  assert.equal(latest('span', e => e.textContent === 'AI self-approved').textContent, 'AI self-approved');
  console.log('Photo-review logic passes: validation, explicit approval, unsure retention, navigation, resume and late-photo guard. Fixtures only.');
}
main().catch(error => { console.error(error); process.exitCode = 1; });
