"use strict";

const $ = id => document.getElementById(id);
const state = {runs: [], record: null, episode: 0, frame: 0, playing: false,
  elapsed: 0, last: null, request: null};
const canvas = $("flight-canvas"), ctx = canvas.getContext("2d");
const descriptions = {
  baseline: "Deterministic position and velocity guidance. No model inference.",
  assisted: "Laya proposes controls. Every disagreement is replaced with guidance.",
  laya: "Model proposals execute directly. No guidance overrides are applied.",
};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function route() {
  const page = ["flight", "compare", "about"].includes(location.hash.slice(1))
    ? location.hash.slice(1) : "flight";
  for (const key of ["flight", "compare", "about"]) $(key + "-page").hidden = page !== key;
  document.querySelectorAll("nav a").forEach(a => a.classList.toggle("active", a.dataset.page === page));
  $("page-title").textContent = {flight: "Flight deck", compare: "Compare pilots", about: "How it works"}[page];
  if (page !== "flight") { state.playing = false; updatePlay(); }
}

function summaryCards() {
  $("comparison-cards").replaceChildren();
  $("comparison-table").replaceChildren();
  for (const run of state.runs) {
    const count = run.summaries.length;
    const landed = run.summaries.filter(s => s.status === "landed").length;
    const decisions = run.summaries.reduce((sum, s) => sum + s.decisions, 0);
    const overrides = run.summaries.reduce((sum, s) => sum + s.interventions, 0);
    const card = element("article", undefined, "card");
    card.append(element("div", run.trained ? "FINE-TUNED / LOCAL MLX" : run.mode.toUpperCase(), "eyebrow"));
    card.append(element("h3", run.label));
    const stat = element("div", `${landed}/${count}`, "compare-stat");
    stat.append(element("small", " landings"));
    card.append(stat, element("p", `${decisions.toLocaleString()} decisions · ${overrides.toLocaleString()} overrides`));
    const button = element("button", "Explore flights ↗");
    button.onclick = () => { $("pilot").value = run.id; location.hash = "flight"; loadRun(run); };
    card.append(button);
    $("comparison-cards").append(card);
    for (const s of run.summaries) {
      const tr = element("tr");
      for (const value of [run.label, `${s.seed} / ${s.target + 1}`, s.status.replaceAll("_", " "),
        s.score, `${s.interventions}/${s.decisions}`, `${s.latency_p50_ms.toFixed(2)} ms`]) {
        tr.append(element("td", String(value)));
      }
      $("comparison-table").append(tr);
    }
  }
}

async function loadRun(run) {
  if (state.request) state.request.abort();
  const request = new AbortController();
  state.request = request;
  state.playing = false;
  state.record = null;
  $("play").disabled = $("restart").disabled = $("timeline").disabled = true;
  $("episode").disabled = true;
  $("download").hidden = true;
  $("notice").textContent = `Loading ${run.label.toLowerCase()}…`;
  updatePlay();
  try {
    const response = await fetch(run.file, {signal: request.signal});
    if (!response.ok) throw new Error(`Recording request failed (${response.status})`);
    const record = await response.json();
    if (request !== state.request) return;
    if (record.schema_version !== 1 || !record.episodes?.length ||
        record.episodes.some(e => !e.frames?.length)) throw new Error("Unsupported or empty flight recording");
    state.record = record;
    $("pilot-name").textContent = run.label;
    $("pilot-description").textContent = descriptions[run.mode];
    $("engine-tag").textContent = run.mode === "baseline" ? "FEEDBACK" :
      `MLX / ${(run.inference?.bits ?? 16) === 16 ? "FP16" : "Q" + run.inference.bits}`;
    $("scene-mode").textContent = run.mode === "assisted" ? "GUIDANCE ASSISTANCE ACTIVE" : "RECORDED SIMULATION";
    $("episode").replaceChildren();
    record.episodes.forEach((e, index) => {
      const option = element("option", `Seed ${e.summary.seed} · Pad ${e.summary.target + 1} · ${e.summary.status}`);
      option.value = String(index);
      $("episode").append(option);
    });
    $("episode").value = "0";
    $("download").href = run.file;
    $("download").hidden = false;
    $("play").disabled = $("restart").disabled = $("timeline").disabled = false;
    $("episode").disabled = false;
    $("notice").textContent = "";
    selectEpisode();
  } catch (error) {
    if (error.name !== "AbortError" && request === state.request) {
      $("notice").textContent = `${error.message}. Rebuild the app or select another recording.`;
    }
  }
}

function selectEpisode() {
  if (!state.record) return;
  state.episode = Number($("episode").value);
  state.frame = 0;
  state.elapsed = 0;
  state.last = null;
  state.playing = false;
  $("timeline").max = currentEpisode().frames.length;
  render();
}

function currentEpisode() { return state.record.episodes[state.episode]; }
function updatePlay() { $("play").textContent = state.playing ? "Pause" : "▶ Play"; }
function togglePlay() {
  if (!state.record) return;
  if (state.frame >= currentEpisode().frames.length) { state.frame = 0; state.elapsed = 0; }
  state.playing = !state.playing;
  state.last = null;
  render();
}
function command(c) { return `${["LEFT", "HOLD", "RIGHT"][c.turn + 1]} / ${Math.round(c.throttle * 100)}%`; }

function probabilities(answers) {
  $("probabilities").replaceChildren();
  if (!answers) {
    $("probabilities").append(element("p", "Guidance controls this flight directly. No model probability distribution."));
    return;
  }
  for (const [name, answer] of Object.entries(answers)) {
    const group = element("div", undefined, "prob-group");
    group.append(element("div", name === "rotation" ? "ROTATION" : "MAIN ENGINE", "prob-title"));
    for (const [label, p] of Object.entries(answer.probabilities)) {
      const row = element("div", undefined, "prob-row" + (label === answer.choice ? " selected" : ""));
      const bar = element("div", undefined, "prob-bar"), fill = element("i");
      fill.style.width = `${Math.max(0, Math.min(100, p * 100))}%`;
      bar.append(fill);
      row.append(element("span", label), bar, element("span", `${(p * 100).toFixed(0)}%`));
      group.append(row);
    }
    $("probabilities").append(group);
  }
}

function render() {
  if (!state.record) return;
  const episode = currentEpisode(), end = state.frame >= episode.frames.length;
  const frame = episode.frames[Math.min(state.frame, episode.frames.length - 1)];
  const s = end ? episode.summary : frame.before, d = frame.decision;
  const pad = state.record.world.pads[episode.summary.target];
  $("mission-label").textContent = `SEED ${episode.summary.seed} / PAD 0${episode.summary.target + 1}`;
  $("altitude").textContent = `${Math.max(0, s.y - pad[2] - 8).toFixed(0)} m`;
  $("vertical").textContent = `${s.vy.toFixed(1)} m/s`;
  $("horizontal").textContent = `${s.vx.toFixed(1)} m/s`;
  $("fuel").textContent = `${s.fuel.toFixed(1)}%`;
  $("flight-status").textContent = s.status.replaceAll("_", " ").toUpperCase();
  $("mission-time").textContent = `T+ ${s.time.toFixed(2)} s`;
  $("tilt").textContent = `TILT ${s.angle.toFixed(1)}°`;
  $("proposed").textContent = command(d.proposed);
  $("executed").textContent = command(d.executed);
  $("latency").textContent = `${d.latency_ms.toFixed(1)} ms`;
  $("prompt").textContent = d.prompt;
  $("outcome").textContent = episode.summary.status.replaceAll("_", " ");
  $("score").textContent = episode.summary.score;
  $("override-count").textContent = episode.summary.interventions;
  $("decision-count").textContent = episode.summary.decisions;
  $("frame-label").textContent = `FRAME ${state.frame} / ${episode.frames.length}`;
  $("timeline").value = state.frame;
  $("intervention").textContent = d.intervened ? "GUIDANCE OVERRIDE · Proposal replaced" :
    state.record.pilot.mode === "assisted" ? "MODEL AGREES WITH GUIDANCE" : "DIRECT EXECUTION · No override";
  $("intervention").classList.toggle("warning", d.intervened);
  probabilities(d.answers);
  updatePlay();
  draw(s, d, episode, end);
}

function draw(s, decision, episode, end) {
  const W = canvas.width, H = canvas.height, scale = W / 1100;
  const px = x => 50 * scale + x * scale;
  const py = y => H - 38 * scale - y * (H - 58 * scale) / 750;
  ctx.fillStyle = "#070e17";
  ctx.fillRect(0, 0, W, H);
  for (let i = 0; i < 100; i++) {
    ctx.fillStyle = i % 7 ? "#263847" : "#687e8c";
    ctx.fillRect((i * 173 + 19) % W, (i * 79 + 31) % (H - 100), 1.4, 1.4);
  }
  ctx.strokeStyle = "#142333";
  ctx.lineWidth = 1;
  for (let x = 0; x <= 1000; x += 100) {
    ctx.beginPath();ctx.moveTo(px(x), 0);ctx.lineTo(px(x), H - 38);ctx.stroke();
  }
  for (let y = 100; y <= 700; y += 100) {
    ctx.beginPath();ctx.moveTo(px(0), py(y));ctx.lineTo(px(1000), py(y));ctx.stroke();
    ctx.fillStyle = "#3b5366";ctx.font = "10px monospace";ctx.fillText(String(y), 13, py(y) + 3);
  }
  function line(points, color, width) {
    ctx.beginPath();ctx.strokeStyle = color;ctx.lineWidth = width;
    points.forEach(([x, y], i) => i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y)));
    ctx.stroke();
  }
  line(episode.frames.slice(0, state.frame + 1).map(f => [f.before.x, f.before.y]), "#316459", 1.5);
  const terrain = state.record.world.terrain;
  ctx.beginPath();ctx.moveTo(px(0), H);
  terrain.forEach(([x, y]) => ctx.lineTo(px(x), py(y)));
  ctx.lineTo(px(1000), H);ctx.closePath();ctx.fillStyle = "#14212d";ctx.fill();
  line(terrain, "#7b929e", 1.5);
  state.record.world.pads.forEach((p, i) => {
    const selected = i === episode.summary.target;
    line([[p[0], p[2]], [p[1], p[2]]], selected ? "#90edd0" : "#678397", selected ? 4 : 2);
    ctx.font = "11px monospace";ctx.fillStyle = selected ? "#90edd0" : "#6d8596";
    ctx.fillText(`0${i + 1} / ${p[3]}×`, px(p[0]), py(p[2]) + 22);
  });
  const x = px(s.x), y = py(s.y);
  ctx.strokeStyle = "#244c46";ctx.setLineDash([3, 7]);ctx.beginPath();ctx.moveTo(x, y + 24);ctx.lineTo(x, py(padHeight(s.x)));ctx.stroke();ctx.setLineDash([]);
  ctx.save();ctx.translate(x, y);ctx.rotate(s.angle * Math.PI / 180);
  ctx.strokeStyle = s.status === "crashed" || s.status === "out_of_bounds" ? "#e7ac83" : "#d9fff0";
  ctx.lineWidth = 1.8;ctx.beginPath();ctx.moveTo(0, -10);ctx.lineTo(7, -4);ctx.lineTo(6, 4);ctx.lineTo(-6, 4);ctx.lineTo(-7, -4);ctx.closePath();
  ctx.moveTo(-5, 4);ctx.lineTo(-6, 8);ctx.lineTo(-8, 8);ctx.moveTo(5, 4);ctx.lineTo(6, 8);ctx.lineTo(8, 8);ctx.stroke();
  if (!end && s.fuel > 0 && decision.executed.throttle > 0) {
    ctx.strokeStyle = "#edbf7f";ctx.beginPath();ctx.moveTo(-3, 6);ctx.lineTo(0, 10 + 15 * decision.executed.throttle);ctx.lineTo(3, 6);ctx.stroke();
  }
  ctx.restore();
  ctx.fillStyle = "#829eab";ctx.font = "10px monospace";
  ctx.fillText("LUNAR SURFACE / VECTOR TELEMETRY", 24, H - 12);
}

function padHeight(x) {
  const points = state.record.world.terrain;
  for (let i = 1; i < points.length; i++) {
    if (x <= points[i][0]) {
      const [x0, y0] = points[i - 1], [x1, y1] = points[i];
      return y0 + (y1 - y0) * Math.max(0, x - x0) / (x1 - x0);
    }
  }
  return points.at(-1)[1];
}

function tick(now) {
  let changed = false;
  if (state.last !== null && state.playing && state.record) {
    state.elapsed += Math.min((now - state.last) / 1000, 0.1) * Number($("speed").value);
    const frames = currentEpisode().frames;
    while (state.frame < frames.length) {
      const duration = frames[state.frame].after.time - frames[state.frame].before.time;
      if (state.elapsed < duration) break;
      state.elapsed -= duration;state.frame++;changed = true;
    }
    if (state.frame === frames.length) state.playing = false;
  }
  state.last = now;
  if (changed) render();
  requestAnimationFrame(tick);
}

$("play").onclick = togglePlay;
$("restart").onclick = () => { selectEpisode();state.playing = true;updatePlay(); };
$("episode").onchange = selectEpisode;
$("pilot").onchange = () => loadRun(state.runs.find(r => r.id === $("pilot").value));
$("timeline").oninput = () => { state.frame = Number($("timeline").value);state.playing = false;state.elapsed = 0;render(); };
window.addEventListener("hashchange", route);
window.addEventListener("keydown", event => {
  if (event.code === "Space" && !["INPUT", "SELECT", "BUTTON", "TEXTAREA", "SUMMARY", "A"].includes(event.target.tagName)
      && !$("flight-page").hidden) { event.preventDefault();togglePlay(); }
});

async function init() {
  route();
  try {
    const response = await fetch("manifest.json");
    if (!response.ok) throw new Error(`Manifest request failed (${response.status})`);
    const manifest = await response.json();
    if (manifest.version !== 1 || !manifest.runs?.length) throw new Error("No compatible recordings in manifest");
    state.runs = manifest.runs;
    for (const run of state.runs) {
      const option = element("option", run.label);option.value = run.id;$("pilot").append(option);
    }
    summaryCards();
    const initial = state.runs.find(r => r.trained) || state.runs.find(r => r.mode === "assisted") || state.runs[0];
    $("pilot").value = initial.id;
    await loadRun(initial);
  } catch (error) {
    $("notice").textContent = `${error.message}. Serve the built dist/web directory with a local HTTP server; see web/README.md.`;
  }
  requestAnimationFrame(tick);
}
init();
