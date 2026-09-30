const state = {
  projects: [],
  project: null,
  videoId: null,
  detail: null,
  activeTab: "tracking",
  videoView: "source",
  rawKind: "tracks",
  pollTimer: null,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatBytes(bytes) {
  if (!Number.isFinite(Number(bytes))) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = Number(bytes);
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit > 1 ? 1 : 0)} ${units[unit]}`;
}

function formatTime(seconds) {
  const value = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(value / 60);
  const rest = Math.floor(value % 60);
  return `${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function toast(title, message = "", type = "") {
  const element = document.createElement("div");
  element.className = `toast ${type}`;
  element.innerHTML = `<strong>${escapeHtml(title)}</strong>${escapeHtml(message)}`;
  $("#toastStack").append(element);
  setTimeout(() => element.remove(), 5000);
}

async function api(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail || message;
    } catch (_) {}
    throw new Error(message);
  }
  return response.json();
}

async function bootstrap() {
  bindEvents();
  try {
    const health = await api("/api/health");
    const ready = health.configuration?.ready;
    $("#systemState").classList.add(ready ? "ready" : "error");
    $("#systemState span:last-child").textContent = ready
      ? "模型环境已就绪"
      : "服务在线 · 配置待检查";
    if (!ready) toast("服务在线，但模型路径不完整", "可在参数与日志中继续检查。", "error");
  } catch (error) {
    $("#systemState").classList.add("error");
    $("#systemState span:last-child").textContent = "后端连接失败";
    toast("无法连接后端", error.message, "error");
    return;
  }
  await loadProjects();
  if (state.projects.length) {
    await selectProject(state.projects[0].project_id);
  } else {
    await createProject("篮球视频实验");
  }
  startPolling();
}

function bindEvents() {
  const input = $("#fileInput");
  $("#selectFilesButton").addEventListener("click", (event) => {
    event.stopPropagation();
    input.click();
  });
  $("#addMoreButton").addEventListener("click", () => input.click());
  $("#dropZone").addEventListener("click", () => input.click());
  $("#dropZone").addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") input.click();
  });
  input.addEventListener("change", () => uploadFiles([...input.files]));
  ["dragenter", "dragover"].forEach((name) =>
    $("#dropZone").addEventListener(name, (event) => {
      event.preventDefault();
      $("#dropZone").classList.add("dragging");
    }),
  );
  ["dragleave", "drop"].forEach((name) =>
    $("#dropZone").addEventListener(name, (event) => {
      event.preventDefault();
      $("#dropZone").classList.remove("dragging");
    }),
  );
  $("#dropZone").addEventListener("drop", (event) =>
    uploadFiles([...event.dataTransfer.files]),
  );
  $("#newProjectButton").addEventListener("click", async () => {
    const name = window.prompt("给这次实验起一个名字", "篮球视频实验");
    if (name !== null) await createProject(name);
  });
  $("#optionsButton").addEventListener("click", () =>
    $("#optionsPanel").classList.toggle("hidden"),
  );
  $("#runFullButton").addEventListener("click", () => runCurrent("full"));
  $("#batchRunButton").addEventListener("click", batchRun);
  $$(".module-buttons button").forEach((button) =>
    button.addEventListener("click", () => runCurrent(button.dataset.target)),
  );
  $$("#viewSwitcher button").forEach((button) =>
    button.addEventListener("click", () => setVideoView(button.dataset.view)),
  );
  $$("#tabs button").forEach((button) =>
    button.addEventListener("click", () => setTab(button.dataset.tab)),
  );
}

async function loadProjects() {
  state.projects = await api("/api/projects");
  renderProjects();
}

async function createProject(name) {
  try {
    const project = await api("/api/projects", {
      method: "POST",
      body: JSON.stringify({ name: name || null }),
    });
    await loadProjects();
    await selectProject(project.project_id);
    toast("已新建实验", project.name);
  } catch (error) {
    toast("新建失败", error.message, "error");
  }
}

function renderProjects() {
  $("#projectList").innerHTML = state.projects
    .map(
      (project) => `
        <button class="project-item ${state.project?.project_id === project.project_id ? "active" : ""}"
                data-project-id="${escapeHtml(project.project_id)}">
          <strong>${escapeHtml(project.name)}</strong>
          <span>${project.videos?.length || 0} CLIPS · ${escapeHtml(project.project_id.slice(0, 15))}</span>
        </button>`,
    )
    .join("");
  $$(".project-item").forEach((button) =>
    button.addEventListener("click", () => selectProject(button.dataset.projectId)),
  );
}

async function selectProject(projectId, preserveVideo = false) {
  try {
    const previousVideo = preserveVideo ? state.videoId : null;
    state.project = await api(`/api/projects/${projectId}`);
    $("#projectTitle").textContent = state.project.name;
    renderProjects();
    renderClipStrip();
    renderCrossClipIndex();
    const videos = state.project.videos || [];
    $("#workspace").classList.toggle("hidden", videos.length === 0);
    if (videos.length) {
      const selected = videos.some((item) => item.video_id === previousVideo)
        ? previousVideo
        : videos[0].video_id;
      await selectVideo(selected, false);
    } else {
      state.videoId = null;
      state.detail = null;
    }
  } catch (error) {
    toast("加载实验失败", error.message, "error");
  }
}

function renderClipStrip() {
  const videos = state.project?.videos || [];
  $("#clipStrip").innerHTML = videos
    .map((video) => {
      const stages = Object.values(video.stages || {});
      const progress = stages
        .map((stage) => `<i class="${escapeHtml(stage.status)} ${stage.status === "completed" ? "done" : ""}"></i>`)
        .join("");
      return `
        <button class="clip-card ${state.videoId === video.video_id ? "active" : ""}"
                data-video-id="${escapeHtml(video.video_id)}">
          <div class="clip-card-top">
            <span class="eyebrow">CLIP ${String(videos.indexOf(video) + 1).padStart(2, "0")}</span>
            <i class="mini-status ${escapeHtml(video.status)}"></i>
          </div>
          <strong title="${escapeHtml(video.filename)}">${escapeHtml(video.filename)}</strong>
          <small>${formatBytes(video.size_bytes)} · ${escapeHtml(video.status.toUpperCase())}</small>
          <div class="clip-progress">${progress}</div>
        </button>`;
    })
    .join("");
  $$(".clip-card").forEach((button) =>
    button.addEventListener("click", () => selectVideo(button.dataset.videoId)),
  );
}

async function selectVideo(videoId, resetTab = true) {
  state.videoId = videoId;
  if (resetTab) state.activeTab = "tracking";
  renderClipStrip();
  await refreshCurrentVideo();
}

async function refreshCurrentVideo() {
  if (!state.project || !state.videoId) return;
  try {
    state.detail = await api(
      `/api/projects/${state.project.project_id}/videos/${state.videoId}`,
    );
    renderInspector();
  } catch (error) {
    toast("刷新片段失败", error.message, "error");
  }
}

function renderInspector() {
  if (!state.detail) return;
  const { video, artifacts, job_active: active } = state.detail;
  $("#clipTitle").textContent = video.filename;
  const meta = artifacts.tracking.video_meta;
  const pieces = [formatBytes(video.size_bytes)];
  if (meta) pieces.push(`${meta.width}×${meta.height}`, `${Number(meta.fps).toFixed(2)} FPS`, `${meta.processed_frames} FRAMES`);
  if (video.error) pieces.push(`ERROR: ${video.error}`);
  $("#clipMeta").textContent = pieces.join("  ·  ");
  $("#clipMeta").classList.toggle("error-text", Boolean(video.error));
  $("#clipStatus").textContent = (video.active_stage || video.status).toUpperCase();
  $("#clipStatus").className = `status-pill ${video.status}`;
  $("#runFullButton").disabled = active;
  $("#runFullButton small").textContent = active ? "PIPELINE ACTIVE" : "RUN PIPELINE";
  $("#runFullButton span:last-child").lastChild.textContent = active
    ? ` ${stageName(video.active_stage)}`
    : "处理完整流程";
  renderStageRail(video.stages);
  renderMetrics(artifacts);
  setVideoView(state.videoView, false);
  renderTab();
}

function stageName(stage) {
  return {
    tracking: "正在追踪人物",
    identity: "正在提取身份特征",
    action: "正在识别动作",
    link: "正在连接人物事件",
    render: "正在渲染结果",
  }[stage] || "等待 GPU 队列";
}

function renderStageRail(stages = {}) {
  $$("#stageRail .stage").forEach((element) => {
    element.classList.remove("pending", "running", "completed", "failed", "skipped");
    element.classList.add(stages[element.dataset.stage]?.status || "pending");
  });
}

function renderMetrics(artifacts) {
  const tracking = artifacts.tracking.summary || {};
  const identity = artifacts.identity.summary || {};
  const actions = artifacts.action.index || {};
  const frames = artifacts.tracking.video_meta?.processed_frames;
  const metrics = [
    ["TRACKS", tracking.track_count ?? "—", "MOTIP raw IDs"],
    ["PEOPLE", identity.identity_count ?? identity.selected_track_count ?? "—", "clip-local archive"],
    ["EVENTS", actions.event_count ?? "—", "selected + candidates"],
    ["FRAMES", frames ?? "—", "processed source"],
  ];
  $("#metricsGrid").innerHTML = metrics
    .map(([label, value, note]) => `<div class="metric-card"><small>${label}</small><strong>${escapeHtml(value)}</strong><span>${escapeHtml(note)}</span></div>`)
    .join("");
}

function setVideoView(view, rerender = true) {
  state.videoView = view;
  $$("#viewSwitcher button").forEach((button) =>
    button.classList.toggle("active", button.dataset.view === view),
  );
  if (!state.detail) return;
  const media = state.detail.artifacts.media || {};
  const url = media[view];
  const video = $("#resultVideo");
  const empty = $("#videoEmpty");
  $("#videoModeLabel").textContent = `${view.toUpperCase()} VIDEO`;
  if (url) {
    empty.classList.add("hidden");
    video.classList.remove("hidden");
    if (video.dataset.url !== url) {
      video.dataset.url = url;
      video.src = `${url}?v=${encodeURIComponent(state.detail.video.updated_at)}`;
      video.load();
    }
  } else {
    video.pause();
    video.removeAttribute("src");
    video.dataset.url = "";
    video.classList.add("hidden");
    empty.classList.remove("hidden");
  }
  if (rerender) renderInspector();
}

function setTab(tab) {
  state.activeTab = tab;
  $$("#tabs button").forEach((button) =>
    button.classList.toggle("active", button.dataset.tab === tab),
  );
  renderTab();
}

function renderTab() {
  if (!state.detail) return;
  $$("#tabs button").forEach((button) =>
    button.classList.toggle("active", button.dataset.tab === state.activeTab),
  );
  if (state.activeTab === "tracking") renderTracking();
  else if (state.activeTab === "identity") renderIdentity();
  else if (state.activeTab === "events") renderEvents();
  else if (state.activeTab === "raw") renderRaw();
  else if (state.activeTab === "logs") renderLogs();
}

function emptyState(title, detail) {
  return `<div class="empty-state"><div><strong>${escapeHtml(title)}</strong>${escapeHtml(detail)}</div></div>`;
}

function renderTracking() {
  const summary = state.detail.artifacts.tracking.summary;
  if (!summary?.tracks?.length) {
    $("#tabContent").innerHTML = emptyState("还没有轨迹结果", "先运行人物追踪模块，轨迹覆盖率和检测分数会出现在这里。");
    return;
  }
  const rows = summary.tracks
    .map((track) => `
      <tr>
        <td><strong>T${escapeHtml(track.track_id)}</strong></td>
        <td>${escapeHtml(track.start_frame)} – ${escapeHtml(track.end_frame)}</td>
        <td>${escapeHtml(track.observations)}</td>
        <td>${Number(track.duration_s).toFixed(2)} s</td>
        <td><span class="score-bar"><i style="width:${Math.round(Number(track.observation_coverage) * 100)}%"></i></span>${(Number(track.observation_coverage) * 100).toFixed(0)}%</td>
        <td>${Number(track.mean_det_score).toFixed(3)}</td>
      </tr>`)
    .join("");
  $("#tabContent").innerHTML = `
    <table class="data-table">
      <thead><tr><th>RAW ID</th><th>FRAME SPAN</th><th>OBSERVATIONS</th><th>DURATION</th><th>COVERAGE</th><th>MEAN DET</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function renderIdentity() {
  const people = state.detail.artifacts.identity.people || [];
  if (!people.length) {
    $("#tabContent").innerHTML = emptyState("还没有人物档案", "KPR 完成后会展示封面、代表帧和轨迹内一致性。这里不会自动虚构球员姓名。");
    return;
  }
  $("#tabContent").innerHTML = `<div class="identity-grid">${people
    .map((person) => {
      const cover = person.cover || {};
      const image = cover.crop_url
        ? `<img src="${escapeHtml(cover.crop_url)}" alt="${escapeHtml(person.person_id)}" loading="lazy" />`
        : `<div class="no-image">NO COVER</div>`;
      return `<article class="identity-card">
        <div class="identity-image">${image}</div>
        <div class="identity-body">
          <div class="identity-head"><strong>${escapeHtml(person.person_id)}</strong><span>TRACK ${escapeHtml(person.raw_track_ids?.join(", "))}</span></div>
          <div class="identity-stats"><span>${escapeHtml(person.sample_count)} samples</span><span>${escapeHtml(person.observation_count)} observations</span><span>${escapeHtml(person.status)}</span></div>
        </div>
      </article>`;
    })
    .join("")}</div>`;
}

function renderEvents() {
  const index = state.detail.artifacts.action.index;
  if (!index?.people?.length) {
    $("#tabContent").innerHTML = emptyState("还没有人物事件", "动作结果与身份档案连接后，会按人物排列事件时间、类别和分数。");
    return;
  }
  $("#tabContent").innerHTML = index.people
    .map((person) => {
      const identity = person.identity;
      const image = identity?.cover?.crop_url;
      const events = person.events
        .map((event) => `<div class="event-row">
          <time>${formatTime(event.timestamp_s)}</time>
          <strong class="${event.below_threshold ? "candidate" : ""}">${escapeHtml(event.below_threshold ? `top? ${event.label}` : event.label)}</strong>
          <span>${Number(event.score).toFixed(3)}</span>
          <span>frame ${escapeHtml(event.frame_idx)} · T${escapeHtml(event.track_id)}</span>
        </div>`)
        .join("");
      return `<section class="event-person">
        ${image ? `<img class="event-avatar" src="${escapeHtml(image)}" alt="" loading="lazy" />` : `<div class="event-avatar"></div>`}
        <div class="event-person-title"><strong>${escapeHtml(person.person_id || `Track ${person.raw_track_ids.join(",")}`)}</strong><span>${person.event_count} events · IDs ${escapeHtml(person.raw_track_ids.join(", "))}</span></div>
        <div class="event-list">${events}</div>
      </section>`;
    })
    .join("");
}

async function renderRaw() {
  const kinds = ["tracks", "identities", "actions", "linked_actions", "pairs"];
  $("#tabContent").innerHTML = `
    <div class="raw-toolbar">${kinds.map((kind) => `<button data-kind="${kind}" class="${state.rawKind === kind ? "active" : ""}">${kind}</button>`).join("")}</div>
    <pre class="code-view">Loading ${escapeHtml(state.rawKind)}…</pre>`;
  $$(".raw-toolbar button").forEach((button) =>
    button.addEventListener("click", () => {
      state.rawKind = button.dataset.kind;
      renderRaw();
    }),
  );
  try {
    const data = await api(
      `/api/projects/${state.project.project_id}/videos/${state.videoId}/records/${state.rawKind}?limit=200`,
    );
    if (state.activeTab === "raw") $(".code-view").textContent = JSON.stringify(data.records, null, 2);
  } catch (error) {
    if (state.activeTab === "raw") $(".code-view").textContent = error.message;
  }
}

async function renderLogs() {
  $("#tabContent").innerHTML = `<pre class="log-view">Loading log…</pre>`;
  try {
    const response = await fetch(
      `/api/projects/${state.project.project_id}/videos/${state.videoId}/log`,
    );
    const text = await response.text();
    if (state.activeTab === "logs") {
      const view = $(".log-view");
      view.textContent = text;
      view.scrollTop = view.scrollHeight;
    }
  } catch (error) {
    if (state.activeTab === "logs") $(".log-view").textContent = error.message;
  }
}

function renderCrossClipIndex() {
  const entries = state.project?.people_index?.entries || [];
  if (!entries.length) {
    $("#crossClipGrid").innerHTML = emptyState("等待身份与事件结果", "处理多个片段后，这里会形成跨片段浏览视图。");
    return;
  }
  $("#crossClipGrid").innerHTML = entries
    .map((entry) => {
      const crop = entry.identity?.cover?.crop_path;
      const url = crop
        ? `/api/projects/${state.project.project_id}/videos/${entry.video_id}/artifacts/identity/${crop}`
        : null;
      const topLabel = Object.keys(entry.labels || {})[0] || "no event";
      return `<article class="cross-card" data-video-id="${escapeHtml(entry.video_id)}">
        ${url ? `<img src="${escapeHtml(url)}" alt="" loading="lazy" />` : `<div class="cross-placeholder">NO IMAGE</div>`}
        <div><strong>${escapeHtml(entry.person_id || `T${entry.raw_track_ids?.[0]}`)}</strong><span title="${escapeHtml(entry.filename)}">${escapeHtml(entry.filename)}</span><b>${entry.event_count} events · ${escapeHtml(topLabel)}</b></div>
      </article>`;
    })
    .join("");
  $$(".cross-card").forEach((card) =>
    card.addEventListener("click", () => selectVideo(card.dataset.videoId)),
  );
}

function runPayload(target) {
  const maxFrames = $("#maxFrames").value.trim();
  return {
    target,
    force: $("#forceRun").checked,
    max_frames: maxFrames ? Number(maxFrames) : null,
    tracking_det_threshold: Number($("#trackingThreshold").value),
    identity_samples: Number($("#identitySamples").value),
    identity_min_det_score: Number($("#identityThreshold").value),
    prompt_mode: "none",
    action_threshold: Number($("#actionThreshold").value),
    action_min_det_score: 0.3,
  };
}

async function runCurrent(target) {
  if (!state.project || !state.videoId) return;
  try {
    await api(`/api/projects/${state.project.project_id}/videos/${state.videoId}/run`, {
      method: "POST",
      body: JSON.stringify(runPayload(target)),
    });
    toast("任务已进入队列", `${state.detail.video.filename} · ${target}`);
    await refreshCurrentVideo();
  } catch (error) {
    toast("无法启动任务", error.message, "error");
  }
}

async function batchRun() {
  const videos = state.project?.videos || [];
  if (!videos.length) return;
  let queued = 0;
  for (const video of videos) {
    if (["running", "queued"].includes(video.status)) continue;
    try {
      await api(`/api/projects/${state.project.project_id}/videos/${video.video_id}/run`, {
        method: "POST",
        body: JSON.stringify(runPayload("full")),
      });
      queued += 1;
    } catch (error) {
      toast(`未能加入 ${video.filename}`, error.message, "error");
    }
  }
  toast("批量任务已提交", `${queued} 个片段将按顺序使用 GPU。`);
  await selectProject(state.project.project_id, true);
}

async function uploadFiles(files) {
  const videos = files.filter((file) => file.type.startsWith("video/") || /\.(mp4|mov|m4v|avi|mkv|webm)$/i.test(file.name));
  if (!videos.length) {
    toast("没有可上传的视频", "请选择常见视频格式。", "error");
    return;
  }
  if (!state.project) await createProject("篮球视频实验");
  const data = new FormData();
  videos.forEach((file) => data.append("files", file));
  const progress = $("#uploadProgress");
  progress.classList.remove("hidden");
  $("#uploadLabel").textContent = `正在上传 ${videos.length} 个视频`;
  try {
    const response = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `/api/projects/${state.project.project_id}/videos`);
      xhr.upload.onprogress = (event) => {
        if (!event.lengthComputable) return;
        const percent = Math.round((event.loaded / event.total) * 100);
        $("#uploadPercent").textContent = `${percent}%`;
        $("#uploadBar").style.width = `${percent}%`;
      };
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) resolve(JSON.parse(xhr.responseText));
        else {
          try { reject(new Error(JSON.parse(xhr.responseText).detail)); }
          catch (_) { reject(new Error(`Upload failed (${xhr.status})`)); }
        }
      };
      xhr.onerror = () => reject(new Error("Network error during upload"));
      xhr.send(data);
    });
    toast("上传完成", `${videos.length} 个片段已加入实验。`);
    $("#fileInput").value = "";
    await selectProject(state.project.project_id, true);
    const uploaded = response.videos || [];
    if (uploaded.length) await selectVideo(uploaded[uploaded.length - 1].video_id);
  } catch (error) {
    toast("上传失败", error.message, "error");
  } finally {
    setTimeout(() => {
      progress.classList.add("hidden");
      $("#uploadPercent").textContent = "0%";
      $("#uploadBar").style.width = "0%";
    }, 700);
  }
}

function startPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    if (!state.project) return;
    const hasActiveJob =
      state.detail?.job_active ||
      (state.project.videos || []).some((video) =>
        ["queued", "running"].includes(video.status),
      );
    if (!hasActiveJob) return;
    const projectId = state.project.project_id;
    try {
      state.project = await api(`/api/projects/${projectId}`);
      renderClipStrip();
      renderCrossClipIndex();
      if (state.videoId) await refreshCurrentVideo();
      if (state.activeTab === "logs" && state.detail?.job_active) renderLogs();
    } catch (_) {}
  }, 2500);
}

bootstrap();
