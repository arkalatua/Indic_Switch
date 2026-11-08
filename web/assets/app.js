// IndicSwitch web app. No framework, no inline code (strict CSP), no innerHTML with data:
// every server or user string goes in through textContent.

const $ = (sel) => document.querySelector(sel);
const LIMITS = { maxSeconds: 600, maxText: 5000, maxBytes: 25 * 1024 * 1024 };
const TARGET_SR = 16000;
const EXAMPLE = 'main kal office nhi ja raha, bohot kaam hai yaar. मेरी मीटिंग कल है, please Ravi ko call karo.';
const TAG_NAME = { HI: 'Hindi', EN: 'English', OTH: 'Other' };

const state = { mode: 'upload', wav: null, wavSeconds: 0, objectUrl: null, result: null, busy: false };

/* ------------------------------------------------------------------ config */
async function loadConfig() {
  try {
    const res = await fetch('api/config', { headers: { Accept: 'application/json' } });
    if (!res.ok) throw new Error();
    const cfg = await res.json();
    const sel = $('#lid');
    sel.replaceChildren(...cfg.lid_models.map((m) => new Option(m.label, m.id, m.id === cfg.default_lid, m.id === cfg.default_lid)));
    if (cfg.default_model && [...$('#model').options].some((o) => o.value === cfg.default_model)) $('#model').value = cfg.default_model;
    if (cfg.limits) {
      LIMITS.maxSeconds = cfg.limits.max_audio_seconds;
      LIMITS.maxText = cfg.limits.max_text_chars;
      LIMITS.maxBytes = cfg.limits.max_upload_mb * 1024 * 1024;
    }
  } catch {
    $('#lid').replaceChildren(new Option('Script rule', 'rule', true, true));
    setStatus('The server is not answering. Start it, then reload this page.', true);
  }
}

/* ------------------------------------------------------------------ tabs */
const tabs = [...document.querySelectorAll('.app-tabs [role="tab"]')];
function selectTab(tab, focus = false) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute('aria-selected', String(on));
    t.tabIndex = on ? 0 : -1;
    document.getElementById(t.getAttribute('aria-controls')).hidden = !on;
  }
  if (focus) tab.focus();
  state.mode = tab.id.replace('tab-', '');
  const audio = state.mode !== 'text';
  document.querySelectorAll('[data-audio-only]').forEach((el) => { el.hidden = !audio; });
  $('#run').textContent = audio ? 'Transcribe audio' : 'Tag the text';
  setStatus('');
}
tabs.forEach((tab, i) => {
  tab.addEventListener('click', () => selectTab(tab));
  tab.addEventListener('keydown', (e) => {
    const k = e.key;
    if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(k)) return;
    e.preventDefault();
    const n = k === 'Home' ? 0 : k === 'End' ? tabs.length - 1 : (i + (k === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    selectTab(tabs[n], true);
  });
});

/* ------------------------------------------------------------------ status */
function setStatus(msg, isError = false) {
  const el = $('#status');
  el.textContent = msg;
  el.classList.toggle('is-error', isError);
}
function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}

/* ------------------------------------------------------------------ audio: decode anything, send 16 kHz mono WAV */
async function toWav16k(blob) {
  const buf = await blob.arrayBuffer();
  const Ctx = window.AudioContext || window.webkitAudioContext;
  const ctx = new Ctx();
  let decoded;
  try {
    decoded = await ctx.decodeAudioData(buf);
  } finally {
    ctx.close();
  }
  if (decoded.duration > LIMITS.maxSeconds) throw new Error(`The clip is longer than ${LIMITS.maxSeconds / 60} minutes.`);
  if (decoded.duration < 0.1) throw new Error('The clip is empty.');
  const frames = Math.ceil(decoded.duration * TARGET_SR);
  const off = new OfflineAudioContext(1, frames, TARGET_SR);
  const src = off.createBufferSource();
  src.buffer = decoded;
  src.connect(off.destination);
  src.start();
  const rendered = await off.startRendering();
  return { blob: encodeWav(rendered.getChannelData(0), TARGET_SR), seconds: decoded.duration };
}
function encodeWav(samples, sr) {
  const view = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const str = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); view.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sr, true); view.setUint32(28, sr * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  str(36, 'data'); view.setUint32(40, samples.length * 2, true);
  for (let i = 0, o = 44; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: 'audio/wav' });
}

async function acceptAudio(blob, name) {
  setStatus('Reading the audio…');
  try {
    const { blob: wav, seconds } = await toWav16k(blob);
    if (wav.size > LIMITS.maxBytes) throw new Error('The clip is too large.');
    state.wav = wav;
    state.wavSeconds = seconds;
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = URL.createObjectURL(blob);
    $('#clip-audio').src = state.objectUrl;
    $('#clip-name').textContent = name;
    $('#clip-dur').textContent = fmtTime(seconds);
    $('#clip').hidden = false;
    $('#drop').hidden = true;
    setStatus(`Ready: ${fmtTime(seconds)} of audio.`);
    return true;
  } catch (err) {
    state.wav = null;
    setStatus(err && err.message && !/decode|Unable|EncodingError/i.test(err.message)
      ? err.message : 'This file could not be read as audio. Try a WAV or MP3.', true);
    return false;
  }
}

// upload
const fileInput = $('#file');
fileInput.addEventListener('change', () => { if (fileInput.files[0]) acceptAudio(fileInput.files[0], fileInput.files[0].name); });
const drop = $('#drop');
['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('is-over'); }));
['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove('is-over')));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f) acceptAudio(f, f.name);
});
$('#clip-clear').addEventListener('click', () => {
  state.wav = null;
  fileInput.value = '';
  $('#clip').hidden = true;
  $('#drop').hidden = false;
  setStatus('');
  fileInput.focus();
});

// record
const rec = { media: null, stream: null, chunks: [], start: 0, timer: 0, ctx: null, raf: 0 };
async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    setStatus('This browser cannot record audio. Upload a file instead.', true);
    return;
  }
  try {
    rec.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    setStatus('Microphone access was blocked. Allow it in the browser, or upload a file.', true);
    return;
  }
  rec.chunks = [];
  rec.media = new MediaRecorder(rec.stream);
  rec.media.addEventListener('dataavailable', (e) => { if (e.data.size) rec.chunks.push(e.data); });
  rec.media.addEventListener('stop', onRecorded);
  rec.media.start();
  rec.start = performance.now();
  $('#rec-btn').setAttribute('aria-pressed', 'true');
  $('#rec-label').textContent = 'Stop recording';
  setStatus('Recording…');
  rec.timer = setInterval(() => {
    const s = (performance.now() - rec.start) / 1000;
    $('#rec-time').textContent = fmtTime(s);
    if (s >= LIMITS.maxSeconds) stopRecording();
  }, 250);
  // level meter
  rec.ctx = new (window.AudioContext || window.webkitAudioContext)();
  const an = rec.ctx.createAnalyser();
  an.fftSize = 512;
  rec.ctx.createMediaStreamSource(rec.stream).connect(an);
  const data = new Uint8Array(an.fftSize);
  const level = $('#rec-level');
  const tick = () => {
    an.getByteTimeDomainData(data);
    let peak = 0;
    for (const v of data) peak = Math.max(peak, Math.abs(v - 128));
    level.style.transform = `scaleX(${Math.min(1, peak / 90).toFixed(3)})`;
    rec.raf = requestAnimationFrame(tick);
  };
  tick();
}
function stopRecording() {
  clearInterval(rec.timer);
  cancelAnimationFrame(rec.raf);
  $('#rec-level').style.transform = 'scaleX(0)';
  if (rec.ctx) rec.ctx.close();
  if (rec.media && rec.media.state !== 'inactive') rec.media.stop();
  rec.stream?.getTracks().forEach((t) => t.stop());
  $('#rec-btn').setAttribute('aria-pressed', 'false');
  $('#rec-label').textContent = 'Record again';
}
async function onRecorded() {
  const blob = new Blob(rec.chunks, { type: rec.media.mimeType || 'audio/webm' });
  if (await acceptAudio(blob, 'Recording')) $('#rec-note').textContent = 'Recorded. Press Transcribe audio, or record again.';
}
$('#rec-btn').addEventListener('click', () => {
  if (rec.media && rec.media.state === 'recording') stopRecording();
  else startRecording();
});

/* ------------------------------------------------------------------ settings */
const conf = $('#conf');
conf.addEventListener('input', () => { $('#conf-out').textContent = Number(conf.value).toFixed(2); });
function settings() {
  return {
    lid: $('#lid').value || 'rule',
    min_confidence: Number(conf.value),
    canonicalize: $('#canon').checked,
    collapse_loops: $('#loops').checked,
    target_script: $('#script').value,
    protected_words: $('#protected').value.split(',').map((w) => w.trim()).filter(Boolean).slice(0, 50),
  };
}
$('#use-example').addEventListener('click', () => {
  $('#text').value = EXAMPLE;
  $('#text').focus();
});

/* ------------------------------------------------------------------ run */
async function apiError(res) {
  try {
    const body = await res.json();
    if (body && typeof body.error === 'string') return body.error;
  } catch { /* fall through */ }
  return res.status === 429 ? 'Too many requests. Please wait a moment.' : 'Something went wrong. Please try again.';
}

$('#run-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (state.busy) return;
  const textErr = $('#text-error');
  textErr.hidden = true;
  $('#text').removeAttribute('aria-invalid');
  let req;
  if (state.mode === 'text') {
    const text = $('#text').value.trim();
    if (!text) {
      textErr.textContent = 'Write or paste some text to tag.';
      textErr.hidden = false;
      $('#text').setAttribute('aria-invalid', 'true');
      $('#text').focus();
      setStatus('1 field needs attention.', true);
      return;
    }
    req = fetch('api/text', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ text, ...settings() }) });
  } else {
    if (!state.wav) {
      setStatus(state.mode === 'record' ? 'Record a clip first.' : 'Choose an audio file first.', true);
      (state.mode === 'record' ? $('#rec-btn') : fileInput).focus();
      return;
    }
    const fd = new FormData();
    fd.append('file', state.wav, 'clip.wav');
    fd.append('settings', JSON.stringify({ ...settings(), model: $('#model').value, language: $('#language').value,
      block_urdu: $('#block-urdu').checked }));
    req = fetch('api/audio/stream', { method: 'POST', body: fd, headers: { Accept: 'text/event-stream' } });
  }
  state.busy = true;
  $('#run').disabled = true;
  $('#result').setAttribute('aria-busy', 'true');
  const slow = setTimeout(() => setStatus(state.mode === 'text' ? 'Tagging…' : 'Uploading…'), 300);
  const t0 = performance.now();
  try {
    const res = await req;
    if (!res.ok) throw new Error(await apiError(res));
    const data = state.mode === 'text' ? await res.json() : await readStream(res);
    render(data);
    const secs = (performance.now() - t0) / 1000;
    setStatus(secs < 0.1 ? 'Done.' : `Done in ${secs.toFixed(1)} s.`);
  } catch (err) {
    setStatus(err instanceof TypeError ? 'The server could not be reached. Check that it is running.' : err.message, true);
  } finally {
    clearTimeout(slow);
    state.busy = false;
    $('#run').disabled = false;
    $('#result').setAttribute('aria-busy', 'false');
  }
});

/* ------------------------------------------------------------------ streaming: show text while Whisper works */
async function readStream(res) {
  const live = $('#live');
  const liveText = $('#live-text');
  const fill = $('#live-fill');
  const bar = $('#live-bar');
  const time = $('#live-time');
  liveText.replaceChildren();
  fill.style.transform = 'scaleX(0)';
  bar.setAttribute('aria-valuenow', '0');
  $('#empty').hidden = true;
  $('#out').hidden = true;
  live.hidden = false;
  if (matchMedia('(max-width: 1023px)').matches) $('#result').scrollIntoView({ block: 'start' });

  let total = state.wavSeconds || 0;
  const progress = (sec) => {
    const p = total ? Math.min(1, sec / total) : 0;
    fill.style.transform = `scaleX(${p.toFixed(3)})`;
    bar.setAttribute('aria-valuenow', String(Math.round(p * 100)));
    time.textContent = `${fmtTime(sec)} of ${fmtTime(total)}`;
  };
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let result = null;
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf('\n\n')) >= 0) {
        const block = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        const line = block.split('\n').find((l) => l.startsWith('data: '));
        if (!line) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === 'start') {
          total = ev.audio_seconds || total;
          progress(0);
          setStatus('Transcribing…');
        } else if (ev.type === 'segment') {
          if (liveText.childNodes.length) liveText.append(' ');
          liveText.append(el('span', '', ev.text));
          progress(ev.end);
        } else if (ev.type === 'result') {
          progress(total);
          result = ev.result;
        } else if (ev.type === 'error') {
          throw new Error(ev.error || 'Something went wrong. Please try again.');
        }
      }
    }
  } finally {
    live.hidden = true;
  }
  if (!result) throw new Error('The connection closed before the transcript was finished. Please try again.');
  return result;
}

/* ------------------------------------------------------------------ render */
const NO_SPACE_BEFORE = /^[,.!?;:।)\]’”%]$/;
const PUNCT = /^[\p{P}\p{S}]+$/u;
function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
}
function appendWords(target, parts) {
  // parts: [{text, cls?, title?}], joined with spaces except before closing punctuation
  parts.forEach((p, i) => {
    if (i > 0 && !NO_SPACE_BEFORE.test(p.text)) target.append(' ');
    const span = el('span', p.cls || '', p.text);
    if (p.title) span.title = p.title;
    target.append(span);
  });
}

function render(data) {
  state.result = data;
  const edits = new Map(data.edits.map((e) => [e.index, e]));
  const tokens = data.tokens;

  // corrected text: tokens with edits applied, fixes underlined
  const outText = $('#out-text');
  outText.replaceChildren();
  const fixed = [];
  tokens.forEach((w, i) => {
    const ed = edits.get(i);
    if (ed && ed.after === '') return;
    fixed.push(ed ? { text: ed.after, cls: 'is-fixed', title: `Was “${ed.before}”` } : { text: w });
  });
  appendWords(outText, fixed);
  if (!fixed.length) outText.textContent = 'No speech was recognised.';

  // raw whisper output: removed words struck, changed words brighter
  const raw = $('#out-raw');
  raw.replaceChildren();
  appendWords(raw, tokens.map((w, i) => {
    const ed = edits.get(i);
    if (!ed) return { text: w };
    return ed.after === '' ? { text: w, cls: 'is-removed', title: 'Removed: repeated loop' } : { text: w, cls: 'is-changed', title: `Changed to “${ed.after}”` };
  }));
  if (!tokens.length) raw.textContent = 'Whisper returned no text.';

  $('#raw-title').textContent = data.audio_seconds ? 'What Whisper wrote' : 'What you pasted';

  // legend + tokens (punctuation is always "other" and is left out of the word strip)
  const s = data.stats;
  const minConf = Number(conf.value);
  // the strip describes the corrected transcript: words a fix removed are left out, changed words show their new form
  const words = tokens.map((w, i) => ({ w: edits.get(i)?.after ?? w, t: data.language_tags[i], c: data.confidences[i] }))
    .filter((x) => x.w !== '' && !PUNCT.test(x.w));
  const count = (tag) => words.filter((x) => x.t === tag).length;
  if (!words.length) {
    outText.textContent = data.audio_seconds
      ? 'Whisper heard no words in this clip. Try the Small model under Settings, or a clearer recording.'
      : 'There are no words to tag in this text.';
  }
  const legend = $('#legend');
  legend.replaceChildren();
  [['HI', count('HI')], ['EN', count('EN')], ['OTH', count('OTH')]].forEach(([t, n]) => {
    const item = el('span', `lg-${t.toLowerCase()}`);
    item.append(el('i'), `${TAG_NAME[t]} ${n}`);
    legend.append(item);
  });
  legend.append(el('span', '', `${s.switches} ${s.switches === 1 ? 'switch' : 'switches'}`));
  if (words.some((x) => x.c < minConf)) {
    const unsure = el('span', 'lg-unsure');
    unsure.append(el('i'), 'Dashed: too unsure to fix');
    legend.append(unsure);
  }

  const tokBox = $('#tokens');
  tokBox.replaceChildren();
  const MAX_CHIPS = 1500; // a ten-minute clip stays light to render
  const more = $('#tokens-more');
  more.hidden = words.length <= MAX_CHIPS;
  more.textContent = `Showing the first ${MAX_CHIPS.toLocaleString()} of ${words.length.toLocaleString()} words. The download has them all.`;
  words.slice(0, MAX_CHIPS).forEach(({ w, t, c }) => {
    const chip = el('span', `tok tok--${t}${c < minConf ? ' is-unsure' : ''}`);
    chip.setAttribute('role', 'listitem');
    chip.setAttribute('aria-label', `${w}: ${TAG_NAME[t]}, confidence ${c.toFixed(2)}`);
    chip.title = `${TAG_NAME[t]} · confidence ${c.toFixed(2)}`;
    chip.append(el('span', 'tok__w', w), el('span', 'tok__t', t.toLowerCase()));
    tokBox.append(chip);
  });

  // stats
  const pct = (n) => (words.length ? `${Math.round((100 * n) / words.length)}%` : '0%');
  const statRows = [['Words', String(words.length)], ['Hindi', pct(count('HI'))], ['English', pct(count('EN'))], ['Switches', String(s.switches)], ['Fixes', String(s.edits)]];
  if (typeof data.rtf === 'number') statRows.push(['Speed', `${data.rtf.toFixed(2)}×`]);
  const stats = $('#stats');
  stats.replaceChildren(...statRows.map(([k, v]) => {
    const d = el('div');
    d.append(el('dt', '', k), el('dd', '', v));
    return d;
  }));
  if (typeof data.rtf === 'number') stats.lastChild.title = 'Processing time divided by audio length; under 1 is faster than real time';

  // edits
  const list = $('#edits');
  list.replaceChildren();
  const RULE = { collapse_loop: 'repeated loop', canonicalize: 'spelling', en_backtransliterate: 'English word', romanize: 'script', to_native: 'script' };
  const shown = data.edits.filter((e, idx, all) => !(e.rule === 'collapse_loop' && idx > 0 && all[idx - 1].rule === 'collapse_loop' && all[idx - 1].index === e.index - 1));
  const loopCount = data.edits.filter((e) => e.rule === 'collapse_loop').length;
  for (const e of shown.slice(0, 60)) {
    const li = el('li');
    const words = el('span', 'edit__words');
    if (e.after === '') {
      words.append(el('del', '', e.before), ' removed');
    } else {
      words.append(el('del', '', e.before), ' → ', el('span', '', e.after));
    }
    const why = e.rule === 'collapse_loop' ? `repeated loop · ${loopCount} words in all` : `${RULE[e.rule] || e.rule} · ${e.tag.toLowerCase()} ${e.confidence.toFixed(2)}`;
    li.append(words, el('span', 'edit__why mono', why));
    list.append(li);
  }

  $('#score-table').hidden = true;
  $('#empty').hidden = true;
  $('#out').hidden = false;
  if (matchMedia('(max-width: 1023px)').matches) $('#result').scrollIntoView({ block: 'start' });
}

/* ------------------------------------------------------------------ tools */
function correctedText() {
  return $('#out-text').textContent;
}
$('#copy').addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(correctedText());
    setStatus('Copied the corrected text.');
  } catch {
    setStatus('Copying is blocked here. Select the text and copy it by hand.', true);
  }
});
$('#download').addEventListener('click', () => {
  if (!state.result) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.result, null, 2)], { type: 'application/json' }));
  const a = el('a');
  a.href = url;
  a.download = 'indicswitch-result.json';
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

$('#score-btn').addEventListener('click', async () => {
  const ref = $('#ref');
  const err = $('#ref-error');
  err.hidden = true;
  ref.removeAttribute('aria-invalid');
  if (!ref.value.trim()) {
    err.textContent = 'Write the correct transcript to compare against.';
    err.hidden = false;
    ref.setAttribute('aria-invalid', 'true');
    ref.focus();
    return;
  }
  try {
    const res = await fetch('api/score', { method: 'POST', headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ reference: ref.value, raw: state.result.raw_transcript, normalized: correctedText() }) });
    if (!res.ok) throw new Error(await apiError(res));
    const data = await res.json();
    const tbody = $('#score-table tbody');
    const fmt = (x) => `${(100 * x).toFixed(1)}%`;
    tbody.replaceChildren(...[['Before fixes', data.raw], ['After fixes', data.normalized]].map(([name, r]) => {
      const tr = el('tr');
      tr.append(el('th', '', name), el('td', 'mono', fmt(r.wer)), el('td', 'mono', fmt(r.cer)), el('td', 'mono', fmt(r.wer_script_agnostic)));
      tr.firstChild.scope = 'row';
      return tr;
    }));
    $('#score-table').hidden = false;
  } catch (e) {
    err.textContent = e.message;
    err.hidden = false;
  }
});

loadConfig();
