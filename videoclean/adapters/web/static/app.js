const $ = (id) => document.getElementById(id);
const view = location.pathname.startsWith("/config") ? "config" : "work";
const AUTH_KEY = "vc-auth";

function authHeader() {
  const raw = sessionStorage.getItem(AUTH_KEY);
  return raw ? { Authorization: "Basic " + raw } : {};
}

let state = {
  jobs: [],
  models: {},
  options: {},
  ollama: {},
  doctor: {},
  selected: null,
  file: null,
  srcUrl: null,
  showing: "src",
  pvJobId: null,
  pvTargets: [],
};

document.querySelectorAll(".nav-link").forEach((a) => {
  a.classList.toggle("is-on", a.dataset.view === view);
});
$("view-work").classList.toggle("hidden", view !== "work");
$("view-config").classList.toggle("hidden", view !== "config");

async function api(path, opts) {
  opts = opts || {};
  const headers = { ...(opts.headers || {}), ...authHeader() };
  const url = new URL(path, location.origin);
  const res = await fetch(url, { ...opts, headers });
  if (res.status === 401) {
    sessionStorage.removeItem(AUTH_KEY);
    $("login").classList.remove("hidden");
    throw new Error("нужен логин");
  }
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { detail: text };
  }
  if (!res.ok) {
    const detail = (data && data.detail) || res.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function fillSelect(el, values, current) {
  const vals = values || [];
  const sig = vals.join("\0");
  const want = current || el.value;
  if (el.dataset.sig === sig && el.options.length) {
    if (want && [...el.options].some((o) => o.value === want)) el.value = want;
    return;
  }
  el.dataset.sig = sig;
  el.innerHTML = "";
  vals.forEach((v) => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = v;
    el.appendChild(opt);
  });
  if (want && [...el.options].some((o) => o.value === want)) el.value = want;
}

function modelRefs(kind, backend) {
  const rows = (state.models[kind] || []).filter((m) => m.state === "ready");
  const filtered = backend ? rows.filter((m) => m.backend === backend || !m.backend) : rows;
  const refs = filtered.map((m) => m.model_ref).filter((r) => r && r !== "(builtin)");
  return [...new Set(refs)];
}

function applyOptions() {
  const opt = state.options || {};
  $("device-pill").textContent = opt.device ? `device ${opt.device}` : "";
  if (view !== "work") return;
  fillSelect($("device"), opt.devices || ["cpu", "cuda", "mps"], opt.device);
  fillSelect($("detector"), opt.detectors && opt.detectors.length ? opt.detectors : ["grounding-dino"]);
  fillSelect($("segmenter"), opt.segmenters && opt.segmenters.length ? opt.segmenters : ["sam2"]);
  fillSelect($("inpainter"), opt.inpainters && opt.inpainters.length ? opt.inpainters : ["opencv-telea"]);
  fillSelect($("llm_model"), opt.llm_models || []);
  fillSelect($("detector_model"), modelRefs("detector", $("detector").value));
  fillSelect($("segmenter_model"), modelRefs("segmenter"));
  $("max-btn").disabled = !opt.max_quality_ready;
}

function renderJobs() {
  const filt = $("job-filter").value;
  const list = $("job-list");
  list.innerHTML = "";
  const rows = state.jobs.filter((j) => filt === "all" || j.state === filt);
  if (!rows.length) {
    const empty = document.createElement("li");
    empty.className = "px-3.5 py-2 text-sm text-mute";
    empty.textContent = "Пока пусто.";
    list.appendChild(empty);
    return;
  }
  if (!state.selected && rows[0]) state.selected = rows[0].id;
  rows.forEach((job) => {
    const li = document.createElement("li");
    li.className =
      job.id === state.selected
        ? "grid gap-0.5 px-3.5 py-2.5 cursor-pointer border-l-2 border-accent bg-white/[0.04]"
        : "grid gap-0.5 px-3.5 py-2.5 cursor-pointer border-l-2 border-transparent hover:bg-white/[0.03]";
    li.innerHTML = `<span class="font-mono text-[11px] text-mute">${job.id}</span><span class="text-[13px] leading-snug">${esc(job.prompt || "—")}</span><span class="font-mono text-[11px] st-${job.state}">${job.state}</span>`;
    li.addEventListener("click", () => selectJob(job.id));
    list.appendChild(li);
  });
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function currentJob() {
  return state.jobs.find((j) => j.id === state.selected) || state.jobs.find((j) => j.state === "RUNNING") || null;
}

function renderLive() {
  const job = currentJob();
  const line = $("live-line");
  const bar = $("frac-bar");
  const stages = $("stages");
  const dl = $("dl-btn");
  const dlLive = $("dl-live");
  if (!job) {
    line.textContent = "Нет активной задачи.";
    bar.style.width = "0%";
    stages.innerHTML = "";
    dl.hidden = true;
    dlLive.hidden = true;
    return;
  }
  const pct = Math.round((job.fraction || 0) * 1000) / 10;
  const bits = [`${job.id}`, job.state, `${pct}%`];
  if (job.detail) bits.push(job.detail);
  if (job.eta) bits.push(`ETA ${job.eta}`);
  line.textContent = bits.join("  ·  ");
  bar.style.width = `${pct}%`;
  stages.innerHTML = (job.stages || [])
    .map((s) => `<li class="${s.mark}">${esc(s.title)}</li>`)
    .join("");
  const errEl = $("live-error");
  if (errEl) {
    if (job.error) {
      errEl.classList.remove("hidden");
      errEl.textContent = "Ошибка: " + job.error;
    } else {
      errEl.classList.add("hidden");
      errEl.textContent = "";
    }
  }
  dl.hidden = !job.has_output;
  dlLive.hidden = !job.has_output;
  if (job.has_output) {
    dl.href = job.output_url;
    dl.setAttribute("download", "cleaned.mp4");
    dlLive.href = job.output_url;
    dlLive.setAttribute("download", "cleaned.mp4");
  }
  if (state.file) return;
  if (state.showing === "out" && job.has_output) setPlayer(job.output_url);
  else if (state.showing === "src" && job.has_input) setPlayer(job.input_url);
}

function setPlayer(url) {
  const player = $("player");
  const drop = $("drop");
  if (!url) {
    player.classList.add("hidden");
    drop.classList.remove("hidden");
    return;
  }
  if (player.src !== new URL(url, location.href).href) player.src = url;
  player.classList.remove("hidden");
  drop.classList.add("hidden");
}

function selectJob(id) {
  state.selected = id;
  renderJobs();
  renderLive();
  const job = currentJob();
  if (state.showing === "src" && job && job.has_input) setPlayer(job.input_url);
  if (state.showing === "out" && job && job.has_output) setPlayer(job.output_url);
}

let configSig = "";
function renderConfig() {
  const sig = JSON.stringify(state.models) + JSON.stringify(state.doctor) + JSON.stringify(state.ollama);
  if (sig === configSig) return;
  configSig = sig;
  const doc = $("doctor");
  const facts = state.doctor || {};
  doc.innerHTML = ["python", "ffmpeg", "ffprobe", "opencv", "torch", "cuda", "mps"]
    .map(
      (k) =>
        `<div class="border-b border-line pb-2 min-w-0"><dt class="text-xs text-mute">${k}</dt><dd class="m-0 font-mono text-xs truncate" title="${esc(facts[k] || "—")}">${esc(facts[k] || "—")}</dd></div>`
    )
    .join("");

  const ollama = state.ollama || {};
  $("providers").innerHTML = `
    <article class="border border-line rounded-lg px-4 py-3">
      <h2 class="m-0 mb-1 text-base font-medium">Ollama</h2>
      <p class="m-0 text-sm text-mute">${ollama.ok ? "на связи, " + (ollama.models || []).length + " моделей" : "не отвечает. запусти ollama serve."}</p>
      <p class="m-0 mt-1 text-sm text-mute">OpenAI-совместимый корень: <code class="font-mono text-ink">${esc(ollama.base_url || "")}</code>. Другие провайдеры подключим тем же base_url + ключ.</p>
    </article>`;

  const catalog = $("catalog");
  const titles = {
    detector: "Детектор",
    segmenter: "Сегментатор",
    inpainter: "Инпейнтер",
    llm: "LLM",
  };
  const backends = {
    detector: ["grounding-dino"],
    segmenter: ["sam2"],
    inpainter: ["propainter", "lama"],
    llm: ["ollama"],
  };
  catalog.innerHTML = "";
  ["detector", "segmenter", "inpainter", "llm"].forEach((kind) => {
    const rows = state.models[kind] || [];
    const wrap = document.createElement("section");
    wrap.className = "flex flex-col gap-3";
    const addClass =
      kind === "llm"
        ? "grid gap-2 sm:grid-cols-[1fr_auto] items-end"
        : "grid gap-2 sm:grid-cols-[10rem_minmax(0,1fr)_auto] items-end";
    const backendOpts = backends[kind].map((b) => `<option value="${b}">${b}</option>`).join("");
    wrap.innerHTML = `
      <h2 class="m-0 text-lg font-medium">${titles[kind]}</h2>
      <div class="overflow-x-auto">
      <table class="w-full text-sm border-collapse">
        <thead><tr class="text-mute font-medium text-left">
          <th class="py-2 pr-3 border-b border-line">модель</th>
          <th class="py-2 pr-3 border-b border-line">backend</th>
          <th class="py-2 pr-3 border-b border-line">вес</th>
          <th class="py-2 pr-3 border-b border-line">статус</th>
          <th class="py-2 border-b border-line"></th>
        </tr></thead>
        <tbody>${
          rows
            .map((m) => {
              const pct = Math.round((m.progress || 0) * 100);
              const bar =
                m.state === "downloading"
                  ? `<span class="mt-1 block h-[3px] bg-line rounded-full overflow-hidden"><i class="block h-full bg-accent" style="width:${pct}%"></i></span>`
                  : "";
              const btn = m.downloadable
                ? `<button type="button" class="btn-ghost py-1 px-2 text-xs" data-dl="${esc(m.id)}">${
                    m.state === "downloading" ? pct + "%" : m.state === "ready" ? "ещё раз" : "скачать"
                  }</button>`
                : "";
              const st =
                m.state === "ready" ? "COMPLETED" : m.state === "error" ? "FAILED" : "QUEUED";
              return `<tr>
                <td class="py-2 pr-3 border-b border-line align-middle"><strong class="font-medium text-ink">${esc(m.title)}</strong><br><span class="font-mono text-[11px] text-mute">${esc(m.model_ref)}</span>${bar}</td>
                <td class="py-2 pr-3 border-b border-line align-middle">${esc(m.backend)}</td>
                <td class="py-2 pr-3 border-b border-line align-middle text-mute">${esc(m.size_hint)}</td>
                <td class="py-2 pr-3 border-b border-line align-middle st-${st}">${esc(m.state)}<br><span class="text-mute font-normal">${esc(m.message || "")}</span></td>
                <td class="py-2 border-b border-line align-middle">${btn}</td>
              </tr>`;
            })
            .join("") || `<tr><td class="py-3 text-mute" colspan="5">пусто</td></tr>`
        }</tbody>
      </table>
      </div>
      <form class="${addClass}" data-kind="${kind}">
        ${kind === "llm" ? "" : `<label class="lbl">backend<select class="field" name="backend">${backendOpts}</select></label>`}
        <label class="lbl">${kind === "llm" ? "тег Ollama" : "Hugging Face id"}
          <input class="field" name="model_ref" required placeholder="${kind === "llm" ? "llama3.2" : "org/name"}">
        </label>
        <button type="submit" class="btn-ghost">Добавить и скачать</button>
      </form>`;
    catalog.appendChild(wrap);
  });

  catalog.querySelectorAll("[data-dl]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      try {
        await api("/api/models/download", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: btn.dataset.dl }),
        });
      } catch (err) {
        alert(err.message);
      }
    });
  });
  catalog.querySelectorAll("form[data-kind]").forEach((form) => {
    form.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const kind = form.dataset.kind;
      const fd = new FormData(form);
      const body = {
        kind,
        backend: fd.get("backend") || (kind === "llm" ? "ollama" : ""),
        model_ref: fd.get("model_ref"),
        download: true,
      };
      try {
        await api("/api/models/custom", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        form.reset();
      } catch (err) {
        alert(err.message);
      }
    });
  });
}

async function tick() {
  try {
    const data = await api("/api/poll");
    $("login").classList.add("hidden");
    state.jobs = data.jobs || [];
    state.models = data.models || {};
    state.options = data.options || {};
    state.ollama = data.ollama || {};
    state.doctor = data.doctor || {};
    applyOptions();
    if (view === "work") {
      renderJobs();
      renderLive();
    } else {
      renderConfig();
    }
  } catch (err) {
    if (view === "work") $("form-msg").hidden = false, ($("form-msg").textContent = err.message);
  }
}

if (view === "work") {
  $("detector").addEventListener("change", () => {
    fillSelect($("detector_model"), modelRefs("detector", $("detector").value));
  });
  $("job-filter").addEventListener("change", renderJobs);
  $("tab-src").addEventListener("click", () => {
    state.showing = "src";
    $("tab-src").classList.add("is-on");
    $("tab-out").classList.remove("is-on");
    if (state.srcUrl) setPlayer(state.srcUrl);
    else {
      const job = currentJob();
      if (job && job.has_input) setPlayer(job.input_url);
      else if (state.file) {
        state.srcUrl = URL.createObjectURL(state.file);
        setPlayer(state.srcUrl);
      } else setPlayer(null);
    }
  });
  $("tab-out").addEventListener("click", () => {
    state.showing = "out";
    $("tab-out").classList.add("is-on");
    $("tab-src").classList.remove("is-on");
    const job = currentJob();
    if (job && job.has_output) setPlayer(job.output_url);
  });
  const drop = $("drop");
  const file = $("file");
  function takeFile(f) {
    if (!f) return;
    state.file = f;
    if (state.srcUrl) URL.revokeObjectURL(state.srcUrl);
    state.srcUrl = URL.createObjectURL(f);
    $("drop-label").textContent = f.name;
    state.showing = "src";
    $("tab-src").classList.add("is-on");
    $("tab-out").classList.remove("is-on");
    setPlayer(state.srcUrl);
  }
  file.addEventListener("change", () => takeFile(file.files[0]));
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add("is-hot");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove("is-hot");
    })
  );
  drop.addEventListener("drop", (e) => takeFile(e.dataTransfer.files[0]));

  $("max-btn").addEventListener("click", () => {
    $("device").value = "cuda";
    if ([...$("detector").options].some((o) => o.value === "grounding-dino")) $("detector").value = "grounding-dino";
    if ([...$("segmenter").options].some((o) => o.value === "sam2-video")) $("segmenter").value = "sam2-video";
    if ([...$("inpainter").options].some((o) => o.value === "propainter")) $("inpainter").value = "propainter";
  });

  $("clean-form").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const msg = $("form-msg");
    msg.hidden = true;
    if (!state.file) {
      msg.hidden = false;
      msg.classList.add("text-fail");
      msg.textContent = "Сначала видео.";
      return;
    }
    const fd = new FormData();
    fd.append("video", state.file, state.file.name);
    fd.append("prompt", $("prompt").value);
    [
      "device",
      "detector",
      "segmenter",
      "inpainter",
      "llm_place",
      "llm_model",
      "fmt",
      "detector_model",
      "segmenter_model",
      "detector_threshold",
      "detector_keyframes",
      "detector_nms_iou",
      "detector_max_box_area",
      "tracker_min_score",
      "tracker_max_template_area",
      "mask_dilate_px",
      "telea_radius",
      "prompt_frame_stride",
      "prompt_frame_max",
      "parse_chunk_frames",
      "vision_batch",
      "propainter_mask_dilation",
      "propainter_ref_stride",
      "propainter_neighbor_length",
      "propainter_subvideo_length",
      "propainter_raft_iter",
    ].forEach((id) => {
      const el = $(id);
      if (el.value !== "") fd.append(id, el.value);
    });
    fd.append("verify", $("verify").checked ? "true" : "false");
    try {
      const job = await api("/api/jobs", { method: "POST", body: fd });
      state.selected = job.id;
      msg.hidden = false;
      msg.classList.remove("text-fail");
      msg.textContent = `в очереди ${job.id}`;
      tick();
    } catch (err) {
      msg.hidden = false;
      msg.classList.add("text-fail");
      msg.textContent = err.message;
    }
  });

  $("cancel-btn").addEventListener("click", async () => {
    if (!state.selected) return;
    await api(`/api/jobs/${state.selected}/cancel`, { method: "POST" });
    tick();
  });
  $("retry-btn").addEventListener("click", async () => {
    if (!state.selected) return;
    const job = await api(`/api/jobs/${state.selected}/retry`, { method: "POST" });
    state.selected = job.id;
    tick();
  });
  $("delete-btn").addEventListener("click", async () => {
    if (!state.selected) return;
    await api(`/api/jobs/${state.selected}`, { method: "DELETE" });
    state.selected = null;
    tick();
  });

  // ---- Preview panel ----
  function previewFormFields() {
    const ids = ["device", "detector", "detector_model", "detector_threshold", "detector_keyframes",
      "detector_nms_iou", "detector_max_box_area", "tracker_min_score", "tracker_max_template_area",
      "segmenter", "segmenter_model", "mask_dilate_px", "prompt_frame_stride", "prompt_frame_max", "vision_batch"];
    const fd = new FormData();
    ids.forEach((id) => {
      const el = $(id);
      if (el && el.value !== "") fd.append(id, el.value);
    });
    return fd;
  }

  function previewFrameData() {
    const indices = $("pv-indices").value.trim();
    if (indices) return { indices };
    return { start: $("pv-start").value, count: $("pv-count").value, stride: $("pv-stride").value };
  }

  async function submitPreview(mode, targets) {
    const msg = $("form-msg");
    if (!state.file) {
      msg.hidden = false;
      msg.classList.add("text-fail");
      msg.textContent = "Сначала видео.";
      return;
    }
    const fd = previewFormFields();
    fd.append("video", state.file, state.file.name);
    fd.append("mode", mode);
    fd.append("prompt", $("prompt").value);
    Object.entries(previewFrameData()).forEach(([k, v]) => {
      if (v !== "" && v !== null) fd.append(k, v);
    });
    if (targets) fd.append("targets", JSON.stringify(targets));
    const job = await api("/api/preview", { method: "POST", body: fd });
    state.pvJobId = job.id;
    state.selected = job.id;
    msg.hidden = false;
    msg.classList.remove("text-fail");
    msg.textContent = `превью в очереди ${job.id}`;
    tick();
  }

  $("pv-parse-btn").addEventListener("click", async () => {
    try {
      await submitPreview("parse", null);
    } catch (err) {
      $("form-msg").hidden = false;
      $("form-msg").classList.add("text-fail");
      $("form-msg").textContent = err.message;
    }
  });

  $("pv-detect-btn").addEventListener("click", async () => {
    try {
      const targets = readTargetsFromDom();
      if (!targets.length) {
        $("form-msg").hidden = false;
        $("form-msg").classList.add("text-fail");
        $("form-msg").textContent = "Нет таргетов: сначала сделай разбор.";
        return;
      }
      await submitPreview("detect", targets);
    } catch (err) {
      $("form-msg").hidden = false;
      $("form-msg").classList.add("text-fail");
      $("form-msg").textContent = err.message;
    }
  });

  $("pv-run-full").addEventListener("click", async () => {
    try {
      const targets = readTargetsFromDom();
      if (!targets.length) return;
      const fd = new FormData();
      fd.append("video", state.file, state.file.name);
      ["device", "detector", "segmenter", "inpainter", "llm_place", "llm_model", "fmt",
        "detector_model", "segmenter_model", "detector_threshold", "detector_keyframes",
        "detector_nms_iou", "detector_max_box_area", "tracker_min_score", "tracker_max_template_area",
        "mask_dilate_px", "telea_radius", "vision_batch", "propainter_mask_dilation",
        "propainter_ref_stride", "propainter_neighbor_length", "propainter_subvideo_length",
        "propainter_raft_iter"].forEach((id) => {
        const el = $(id);
        if (el && el.value !== "") fd.append(id, el.value);
      });
      fd.append("targets_override", JSON.stringify(targets));
      fd.append("prompt", $("prompt").value);
      const job = await api("/api/jobs", { method: "POST", body: fd });
      state.selected = job.id;
      tick();
    } catch (err) {
      $("form-msg").hidden = false;
      $("form-msg").classList.add("text-fail");
      $("form-msg").textContent = err.message;
    }
  });

  function readTargetsFromDom() {
    const rows = document.querySelectorAll("#pv-targets .pv-target");
    const out = [];
    rows.forEach((row) => {
      const q = row.querySelector(".pv-query").value.trim();
      if (!q) return;
      out.push({
        kind: row.querySelector(".pv-kind").value,
        query: q,
        where: row.querySelector(".pv-where").value || null,
        motion: "any",
      });
    });
    return out;
  }

  function renderTargetsFromPreview(preview) {
    const wrap = $("pv-targets");
    wrap.innerHTML = "";
    const targets = (preview.targets || []).map((t) => ({
      kind: t.kind || "object",
      query: t.query || "",
      where: t.where || "",
    }));
    if (!targets.length) {
      const p = document.createElement("p");
      p.className = "m-0 text-xs text-mute";
      p.textContent = "LLM не нашёл таргетов. Впиши запросы руками или перепарси.";
      wrap.appendChild(p);
    }
    targets.forEach((t) => wrap.appendChild(targetRow(t)));
    const add = document.createElement("button");
    add.type = "button";
    add.className = "btn-ghost text-xs py-1 px-2 justify-self-start";
    add.textContent = "+ таргет";
    add.addEventListener("click", () => wrap.appendChild(targetRow({ kind: "object", query: "", where: "" })));
    wrap.appendChild(add);
    $("pv-run-full").classList.toggle("hidden", !targets.length);
  }

  function targetRow(t) {
    const row = document.createElement("div");
    row.className = "pv-target grid grid-cols-[7rem_minmax(0,1fr)_7rem] gap-1.5";
    row.innerHTML = `
      <select class="field pv-kind text-xs">
        ${["object", "text_overlay", "watermark"].map((k) => `<option value="${k}" ${k === t.kind ? "selected" : ""}>${k}</option>`).join("")}
      </select>
      <input class="field pv-query text-xs" placeholder="запрос (en)" value="${esc(t.query)}">
      <select class="field pv-where text-xs">
        <option value="" ${!t.where ? "selected" : ""}>—</option>
        ${["top", "bottom", "left", "right", "top-left", "top-right", "bottom-left", "bottom-right"]
          .map((w) => `<option value="${w}" ${w === t.where ? "selected" : ""}>${w}</option>`)
          .join("")}
      </select>`;
    return row;
  }

  function renderPreviewGallery(job) {
    let host = $("pv-gallery");
    if (!host) {
      host = document.createElement("div");
      host.id = "pv-gallery";
      host.className = "grid gap-2 pt-2";
      $("preview-panel").querySelector(".grid").appendChild(host);
    }
    host.innerHTML = "";
    const meta = document.createElement("p");
    meta.className = "m-0 text-xs text-mute";
    meta.textContent = `превью ${job.id} · ${job.state}`;
    host.appendChild(meta);
    if (job.state === "COMPLETED") {
      api(`/api/jobs/${job.id}/preview/preview.json`)
        .then((pv) => {
          if (pv.state === "COMPLETED") renderTargetsFromPreview(pv);
          const grid = document.createElement("div");
          grid.className = "grid grid-cols-4 gap-1.5";
          (pv.frames || []).forEach((f) => {
            const idx = String(f.index).padStart(6, "0");
            const cell = document.createElement("div");
            cell.className = "grid gap-0.5";
            cell.innerHTML = `
              <img class="w-full rounded border border-line cursor-zoom-in" title="кадр ${f.index}, покрытие ${(f.maskCoverage * 100).toFixed(1)}%" src="/api/jobs/${job.id}/preview/${idx}_mask.jpg" loading="lazy">
              <span class="font-mono text-[10px] text-mute">${f.index} · ${(f.maskCoverage * 100).toFixed(1)}%</span>`;
            cell.querySelector("img").addEventListener("click", () => {
              window.open(`/api/jobs/${job.id}/preview/${idx}_boxes.jpg`, "_blank");
            });
            grid.appendChild(cell);
          });
          host.appendChild(grid);
        })
        .catch(() => {});
    }
  }

  const pvHook = setInterval(() => {
    if (view !== "work") return;
    const panel = $("preview-panel");
    if (!panel || !panel.open || !state.pvJobId) return;
    const job = state.jobs.find((j) => j.id === state.pvJobId);
    if (!job) return;
    if (job.state === "COMPLETED" && !$("pv-gallery")) renderPreviewGallery(job);
    else if (job.state !== "COMPLETED" && job.state !== "FAILED") {
      const host = $("pv-gallery");
      if (host) host.innerHTML = `<p class="m-0 text-xs text-mute">превью ${job.id}: ${job.state}…</p>`;
    } else if (job.state === "FAILED") renderPreviewGallery(job);
  }, 1000);
}

$("login-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const user = $("login-user").value.trim() || "admin";
  const pass = $("login-pass").value;
  sessionStorage.setItem(AUTH_KEY, btoa(`${user}:${pass}`));
  try {
    await api("/api/poll");
    $("login").classList.add("hidden");
    $("login-msg").hidden = true;
    tick();
  } catch (err) {
    $("login-msg").hidden = false;
    $("login-msg").textContent = "не пускает: проверь VIDEOCLEAN_UI_USER / PASSWORD";
  }
});

tick();
setInterval(tick, 1000);
