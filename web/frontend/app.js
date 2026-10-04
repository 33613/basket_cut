const state = {
  projects: [],
  project: null,
  videoId: null,
  detail: null,
  activeTab: "events",
  videoView: "source",
  rawKind: "tracks",
  pollTimer: null,
  personFilter: "",
  eventFilter: "",
  reviewData: null,
  reviewDirty: false,
  projectRequest: 0,
  detailRequest: 0,
  reviewRequest: 0,
  overlayRequest: 0,
  overlayAnimation: null,
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
  const rest = (value % 60).toFixed(2).padStart(5, "0");
  return `${String(minutes).padStart(2, "0")}:${rest}`;
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
  $("#importForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = $("#importForm button");
    button.disabled = true;
    button.textContent = "正在建立索引…";
    try {
      const project = await api("/api/import-results", {
        method: "POST", body: JSON.stringify({result_dir: $("#importPath").value.trim()}),
      });
      await loadProjects();
      state.videoView = "final";
      await selectProject(project.project_id);
      toast("已导入，只读展示", `${project.videos.length} 条视频；没有重跑模型或修改原结果。`);
    } catch (error) {
      toast("导入失败", error.message, "error");
    } finally {
      button.disabled = false;
      button.textContent = "导入结果 ↗";
    }
  });
  $("#resultVideo").addEventListener("error", () => {
    $("#mediaHint").textContent = "视频无法播放：可能是 MP4 编码不受浏览器支持。运行 cli.import_web_results --prepare-media 生成 H.264 预览后，再刷新页面。";
  });
  $("#batchRunButton").addEventListener("click", batchRun);
  $("#batchReviewButton").addEventListener("click", loadBatchReview);
  window.addEventListener("beforeunload", (event) => {
    if (state.reviewDirty) { event.preventDefault(); event.returnValue = ""; }
  });
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
  if (state.reviewDirty && !window.confirm("复核尚未保存，确定放弃修改并切换吗？")) return;
  const request = ++state.projectRequest;
  try {
    const previousVideo = preserveVideo ? state.videoId : null;
    const project = await api(`/api/projects/${projectId}`);
    if (request !== state.projectRequest) return;
    state.project = project;
    state.reviewDirty = false;
    state.reviewData = null;
    state.detail = null;
    state.videoId = null;
    $("#batchReviewSummary").classList.add("hidden");
    $("#reviewBundleDownload").href = `/api/projects/${projectId}/track-reviews.zip`;
    if (!preserveVideo) state.videoView = state.project.read_only ? "final" : "source";
    const readOnly = Boolean(state.project.read_only);
    document.body.classList.toggle("inspection-mode", readOnly);
    $("#readOnlyBadge").classList.toggle("hidden", !readOnly);
    $("#batchRunButton").disabled = readOnly;
    $("#addMoreButton").disabled = readOnly;
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
  if (videoId !== state.videoId) {
    if (state.reviewDirty && !window.confirm("复核尚未保存，确定放弃修改并切换片段吗？")) return;
    state.personFilter = "";
    state.eventFilter = "";
    state.reviewData = null;
    state.reviewDirty = false;
    state.detail = null;
    ++state.reviewRequest;
    clearReviewOverlay();
  }
  state.videoId = videoId;
  if (resetTab) state.activeTab = "events";
  renderClipStrip();
  await refreshCurrentVideo();
}

async function refreshCurrentVideo() {
  if (!state.project || !state.videoId) return;
  const projectId = state.project.project_id, videoId = state.videoId;
  const request = ++state.detailRequest;
  try {
    const detail = await api(
      `/api/projects/${projectId}/videos/${videoId}`,
    );
    if (request !== state.detailRequest || projectId !== state.project?.project_id || videoId !== state.videoId) return;
    state.detail = detail;
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
  if (video.read_only) pieces.push("只读导入 · 原结果未修改");
  $("#clipMeta").textContent = pieces.join("  ·  ");
  $("#clipMeta").classList.toggle("error-text", Boolean(video.error));
  $("#clipStatus").textContent = (video.active_stage || video.status).toUpperCase();
  $("#clipStatus").className = `status-pill ${video.status}`;
  $("#runFullButton").disabled = active || video.read_only;
  $("#runFullButton").classList.toggle("hidden", Boolean(video.read_only));
  $("#optionsButton").classList.toggle("hidden", Boolean(video.read_only));
  if (video.read_only) $("#optionsPanel").classList.add("hidden");
  $$(".module-buttons button").forEach((button) => { button.disabled = active || video.read_only; });
  $("#inspectionDownload").href = `/api/projects/${state.project.project_id}/videos/${state.videoId}/inspection.zip`;
  $("#runFullButton small").textContent = active ? "PIPELINE ACTIVE" : "RUN PIPELINE";
  $("#runFullButton span:last-child").lastChild.textContent = active
    ? ` ${stageName(video.active_stage)}`
    : video.read_only ? "已导入 · 不重跑" : "处理完整流程";
  renderStageRail(video.stages);
  renderMetrics(artifacts);
  setVideoView(state.videoView, false);
  renderTab();
}

function stageName(stage) {
  return {
    tracking: "正在追踪人物",
    quality: "正在检查轨迹质量",
    identity: "正在提取身份特征",
    resolution: "正在归并人物档案",
    action: "正在识别动作",
    link: "正在连接人物事件",
    aggregate: "正在聚合事件区间",
    render: "正在渲染结果",
    review: "正在准备复核证据",
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
  const quality = artifacts.tracking.quality;
  const metrics = [
    ["人物轨迹", quality ? `${quality.raw_track_count} → ${quality.retained_track_count}` : tracking.track_count ?? "—", quality ? "原始 → 有时序支持的轨迹（非准确率）" : "原始 track ID 数量"],
    ["身份档案", identity.identity_count ?? identity.selected_track_count ?? "—", "片段内档案，不等于真人数"],
    ["事件区间", artifacts.available.events ? actions.event_count : "—", "聚合结果，不包含低分候选"],
    ["处理帧数", frames ?? "—", "来源视频帧数"],
  ];
  $("#metricsGrid").innerHTML = metrics
    .map(([label, value, note]) => `<div class="metric-card"><small>${label}</small><strong>${escapeHtml(value)}</strong><span>${escapeHtml(note)}</span></div>`)
    .join("");
  const evaluation = artifacts.evaluation || {};
  const idf1 = evaluation.tracking?.headline?.value;
  const messages = [];
  if (idf1 != null) messages.push(`轨迹 IDF1 ${(Number(idf1) * 100).toFixed(1)}% · 仅限评估文件声明的范围`);
  const identityF1 = evaluation.identity?.headline?.value;
  if (identityF1 != null) messages.push(`身份归并 F1 ${(Number(identityF1) * 100).toFixed(1)}% · 仅限独立标注的轨迹对`);
  if (quality) messages.push(`质量检查：${quality.tentative_track_count} 条证据不足，${quality.review_track_count} 条需复核；未确认的档案不能当成真人数。`);
  if (identity.missing_archive_tracks?.length) messages.push(`身份缺失：${identity.missing_archive_tracks.length} 条稳定轨迹没有 KPR 档案，其动作未写入人物事件。`);
  if (!evaluation.events) messages.push("未加载事件评估结果：这里的计数和 raw_score 不是准确率。");
  if (!artifacts.available.events) messages.push("缺少 events.jsonl：不会把动作预测点冒充事件区间。");
  if (artifacts.warnings?.length) messages.push(`数据警告 ${artifacts.warnings.length} 条：存在无法解析的事件，当前计数不完整。`);
  $("#evaluationNote").textContent = messages.join("\n");
}

function setVideoView(view, rerender = true) {
  clearReviewOverlay();
  state.videoView = view;
  $$("#viewSwitcher button").forEach((button) =>
    button.classList.toggle("active", button.dataset.view === view),
  );
  if (!state.detail) return;
  const media = state.detail.artifacts.media || {};
  const url = media[view];
  $("#videoDownload").classList.toggle("hidden", !url);
  if (url) {
    $("#videoDownload").href = url;
    $("#videoDownload").download = `${state.detail.video.filename.replace(/\.[^.]+$/, "")}-${view}.mp4`;
  }
  const video = $("#resultVideo");
  const empty = $("#videoEmpty");
  $("#videoModeLabel").textContent = `${view.toUpperCase()} VIDEO`;
  if (url) {
    empty.classList.add("hidden");
    video.classList.remove("hidden");
    if (video.dataset.url !== url) {
      $("#mediaHint").textContent = "点击事件时间区间可跳转；end 为不包含的结束时间。预览可能不含音频。";
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
  if (state.reviewDirty && tab !== "review") {
    if (!window.confirm("请先保存复核。确定放弃未保存的修改吗？")) return;
    state.reviewDirty = false;
    state.reviewData = null;
  }
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
  else if (state.activeTab === "review") { if (!state.reviewDirty) renderReview(); }
  else if (state.activeTab === "identity") renderIdentity();
  else if (state.activeTab === "events") renderEvents();
  else if (state.activeTab === "points") renderActionPoints();
  else if (state.activeTab === "raw") renderRaw();
  else if (state.activeTab === "logs") renderLogs();
}

function emptyState(title, detail) {
  return `<div class="empty-state"><div><strong>${escapeHtml(title)}</strong>${escapeHtml(detail)}</div></div>`;
}

function renderTracking() {
  const tracking = state.detail.artifacts.tracking;
  const summary = tracking.quality_tracks?.length ? {tracks: tracking.quality_tracks} : tracking.summary;
  if (!summary?.tracks?.length) {
    $("#tabContent").innerHTML = emptyState("还没有轨迹结果", "先运行人物追踪模块，轨迹覆盖率和检测分数会出现在这里。");
    return;
  }
  const rows = summary.tracks
    .map((track) => `
      <tr>
        <td><strong>T${escapeHtml(track.track_id)}</strong></td>
        <td>${escapeHtml(track.status || "raw")}<small>${escapeHtml((track.reasons || []).join(", "))}</small></td>
        <td>${escapeHtml(track.start_frame)} – ${escapeHtml(track.end_frame)}</td>
        <td>${escapeHtml(track.observations)}</td>
        <td>${Number(track.duration_s).toFixed(2)} s</td>
        <td><span class="score-bar"><i style="width:${Math.round(Number(track.observation_coverage) * 100)}%"></i></span>${(Number(track.observation_coverage) * 100).toFixed(0)}%</td>
        <td>${Number(track.mean_det_score).toFixed(3)}</td>
      </tr>`)
    .join("");
  $("#tabContent").innerHTML = `
    <table class="data-table">
      <thead><tr><th>RAW ID</th><th>QUALITY</th><th>FRAME SPAN</th><th>OBSERVATIONS</th><th>DURATION</th><th>COVERAGE</th><th>MEAN DET</th></tr></thead>
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
      const events = state.detail.artifacts.action.index.people.find((p) => p.person_id === person.person_id)?.event_count || 0;
      const status = {human_confirmed: "人工确认归并", auto_merged: "模型候选归并 · 待验证", needs_review: "有风险 · 需复核", unresolved: "未确认身份", unlabeled: "未确认身份"}[person.status] || person.status;
      const audit = state.detail.artifacts.evaluation.track_review?.archives?.groups.find((group) => group.person_id === person.person_id);
      const auditText = audit ? {conflict: "复核发现混人 / 错误归并，不能作为真人档案使用", consistent: "当前来源轨迹完整复核一致（不是独立模型准确率）", unverified: "仍有来源轨迹未完整复核"}[audit.status] : "尚无真人一致性复核";
      const image = cover.crop_url
        ? `<img src="${escapeHtml(cover.crop_url)}" alt="${escapeHtml(person.person_id)}" loading="lazy" />`
        : `<div class="no-image">NO COVER</div>`;
      return `<article class="identity-card">
        <div class="identity-image">${image}</div>
        <div class="identity-body">
          <div class="identity-head"><strong>${escapeHtml(person.person_id)}</strong><span>TRACK ${escapeHtml(person.raw_track_ids?.join(", "))}</span></div>
          ${person.identity_label ? `<p>${escapeHtml(person.identity_label)}</p>` : ""}
          <div class="identity-stats"><span>${escapeHtml(status)}</span><span>${events} 个事件区间</span><span>${person.raw_track_ids?.length || 1} 条来源轨迹</span></div>
          <div class="identity-stats"><span>${escapeHtml(person.sample_count)} samples</span><span>${escapeHtml(person.observation_count)} observations</span></div>
          <p class="archive-warning">封面来自 T${escapeHtml(person.representative_raw_track_id ?? cover.raw_track_id ?? person.raw_track_ids?.[0])}，不能证明整份档案始终同人。</p>
          <p class="archive-warning ${audit?.status === "conflict" ? "error-text" : ""}">${escapeHtml(auditText)}</p>
          <div class="exemplar-strip">${(person.exemplars || []).map((item) => item.crop_url ? `<img src="${escapeHtml(item.crop_url)}" alt="T${escapeHtml(item.raw_track_id)} 帧 ${escapeHtml(item.frame_idx)}" title="T${escapeHtml(item.raw_track_id)} / frame ${escapeHtml(item.frame_idx)}" loading="lazy" />` : "").join("")}</div>
          <details class="archive-sources"><summary>逐条查看来源轨迹 (${person.raw_track_ids?.length || 1})</summary>${(person.source_tracks || []).map((source) => `<div class="archive-source">${source.cover?.crop_url ? `<a href="${escapeHtml(source.cover.context_url || source.cover.crop_url)}" target="_blank" rel="noopener"><img src="${escapeHtml(source.cover.crop_url)}" alt="T${escapeHtml(source.raw_track_id)}" loading="lazy" /></a>` : ""}<div><strong>T${escapeHtml(source.raw_track_id)}</strong><small>${formatTime(source.start)} → ${formatTime(source.end)} · ${escapeHtml(source.observation_count)} 次观测</small><small>${escapeHtml((source.review_reasons || []).join(" / ") || (source.status === "human_confirmed" || audit?.status === "consistent" ? "已人工确认来源同人" : "未人工验证"))}</small></div></div>`).join("") || '<p class="archive-warning">旧档案缺少来源证据，请重跑归并阶段；不会虚构归并后的平均距离。</p>'}</details>
          <button class="ghost-button person-events-button" data-person="${escapeHtml(person.person_id)}">查看这个档案的事件 →</button>
        </div>
      </article>`;
    })
    .join("")}</div>`;
  $$(".person-events-button").forEach((button) => button.addEventListener("click", () => {
    state.personFilter = button.dataset.person;
    setTab("events");
  }));
}

function reviewSummary(metrics) {
  const full = metrics.scopes.full_track.raw, sampled = metrics.scopes.sampled.raw, stable = metrics.scopes.full_track.retained;
  const fragments = metrics.fragmentation;
  const stableName = metrics.quality_available === false ? "未加载质量报告（同原始范围）" : "质量过滤后";
  const percent = (value) => value == null ? "尚无分母" : `${(value * 100).toFixed(1)}%`;
  return `完整复核：${full.assessed_tracks}/${full.total_tracks} 条（${percent(full.review_coverage)}）。原始球员轨迹同人率 ${full.pure}/${full.player_tracks_assessed}（${percent(full.pure_track_rate)}），混人 ${full.mixed} 条，非球员 ${full.non_player} 条。\n${stableName}：完整复核 ${stable.assessed_tracks}/${stable.total_tracks} 条，同人率 ${stable.pure}/${stable.player_tracks_assessed}（${percent(stable.pure_track_rate)}），混人 ${stable.mixed} 条。\n仅抽样：同人候选 ${sampled.pure}，已发现混人 ${sampled.mixed}；不能据此声称全轨迹正确。\n身份已标注的完整同人轨迹：${fragments.labeled_pure_tracks} 条 / ${fragments.reviewed_people} 人；多 ID 人物 ${fragments.people_with_multiple_track_ids} 人，额外 ID ${fragments.extra_track_ids} → 当前处理范围 ${fragments.retained_extra_track_ids} 个。未标注、混人、未检出的人不在碎片统计内。`;
}

async function renderReview() {
  const projectId = state.project.project_id, videoId = state.videoId;
  const request = ++state.reviewRequest;
  if (!state.reviewData) {
    $("#tabContent").innerHTML = emptyState("正在加载复核记录", "先看同一条轨迹的不同时间，再判断是否始终同人。");
    try {
      const data = await api(`/api/projects/${projectId}/videos/${videoId}/track-review`);
      if (request !== state.reviewRequest || projectId !== state.project?.project_id || videoId !== state.videoId) return;
      state.reviewData = data;
    } catch (error) {
      if (request === state.reviewRequest && state.activeTab === "review") $("#tabContent").innerHTML = emptyState("复核暂不可用", error.message);
      return;
    }
  }
  if (state.activeTab !== "review") return;
  const data = state.reviewData, byId = Object.fromEntries(data.review.tracks.map((row) => [row.raw_track_id, row]));
  const options = [["unreviewed", "未复核"], ["pure", "同一人"], ["mixed", "混人 / 中途换人"], ["non_player", "非球员 / 误检"], ["uncertain", "看不清 / 不确定"]];
  $("#tabContent").innerHTML = `<div class="review-toolbar"><div><h3>原始轨迹复核 · 包括已过滤的碎片</h3><p>点“播放区间”查看整条轨迹。只看几张图请保持“仅抽样”。同人轨迹填写相同的队伍＋号码，才可统计同人多 ID；混人轨迹不要整条归给某个号码。</p></div><button class="control-button" id="saveReviewButton">保存复核</button><a class="ghost-button" id="mergeReviewDownload" href="/api/projects/${projectId}/videos/${videoId}/merge-review">导出确认归并 ↓</a></div>
    ${data.stale_review ? '<div class="evaluation-note error-text">原轨迹已变化，旧复核未计入。请重新核查并保存。</div>' : ""}
    <div class="evaluation-note" id="reviewMetrics">${escapeHtml(reviewSummary(data.metrics))}</div>
    <div class="review-grid">${data.index.tracks.map((track) => {
      const row = byId[track.raw_track_id] || {verdict: "unreviewed", scope: "sampled"};
      return `<article class="review-card" data-track="${track.raw_track_id}"><div class="review-heading"><strong>T${track.raw_track_id}</strong><span>${track.retained ? "保留" : "已过滤"} · ${track.observation_count} 次观测 · ${formatTime(track.start)} → ${formatTime(track.end)}</span><button class="ghost-button review-play" data-time="${track.start}" data-end="${track.end}">播放区间 ↗</button></div><p class="archive-warning">自动告警：${escapeHtml(track.quality_reasons.join(" / ") || "无告警不等于正确")}</p>
        <div class="review-samples">${track.samples.map((sample) => `<figure><a href="${escapeHtml(sample.context_url)}" target="_blank" rel="noopener"><img src="${escapeHtml(sample.crop_url)}" alt="T${track.raw_track_id} 帧 ${sample.frame_idx}" loading="lazy" /></a><figcaption>${formatTime(sample.timestamp_s)} · f${sample.frame_idx}</figcaption></figure>`).join("") || '<p class="archive-warning">未生成时序截图，仍可播放原视频复核。运行 cli.prepare_track_review 补充证据。</p>'}</div>
        <div class="review-fields"><label>轨迹是否同人<select data-field="verdict">${options.map(([value, label]) => `<option value="${value}" ${row.verdict === value ? "selected" : ""}>${label}</option>`).join("")}</select></label><label>检查范围<select data-field="scope"><option value="sampled" ${row.scope === "sampled" ? "selected" : ""}>仅抽样</option><option value="full_track" ${row.scope === "full_track" ? "selected" : ""}>已检查完整轨迹区间</option></select></label><label>片段内真人标签<input data-field="identity_label" maxlength="120" value="${escapeHtml(row.identity_label || "")}" placeholder="如：白队#15" ${row.verdict !== "pure" ? "disabled" : ""} /></label><label>问题备注<input data-field="note" maxlength="1000" value="${escapeHtml(row.note || "")}" placeholder="如：6.3秒换成另一位球员" /></label></div></article>`;
    }).join("")}</div>`;
  $$(".review-card input, .review-card select").forEach((element) => element.addEventListener("input", () => {
    state.reviewDirty = true;
    $("#saveReviewButton").textContent = "保存复核（未保存）";
    $("#mergeReviewDownload").classList.add("hidden");
    if (element.dataset.field === "verdict") {
      const label = element.closest(".review-card").querySelector('[data-field="identity_label"]');
      label.disabled = element.value !== "pure";
      if (label.disabled) label.value = "";
    }
  }));
  $$(".review-play").forEach((button) => button.addEventListener("click", () => playReviewTrack(button)));
  $("#saveReviewButton").addEventListener("click", saveTrackReview);
}

function clearReviewOverlay() {
  ++state.overlayRequest;
  if (state.overlayAnimation) cancelAnimationFrame(state.overlayAnimation);
  state.overlayAnimation = null;
  $("#reviewOverlay")?.classList.add("hidden");
}

async function playReviewTrack(button) {
  if (!state.detail.artifacts.media.source) { toast("缺少原视频", "不能进行完整轨迹复核。", "error"); return; }
  setVideoView("source", false);
  const request = state.overlayRequest, projectId = state.project.project_id, videoId = state.videoId;
  const tid = Number(button.closest(".review-card").dataset.track);
  try {
    const data = await api(`/api/projects/${projectId}/videos/${videoId}/track-review/${tid}/observations`);
    if (request !== state.overlayRequest || projectId !== state.project?.project_id || videoId !== state.videoId) return;
    const boxes = new Map(data.observations.map((row) => [row.frame_idx, row.bbox_xyxy]));
    const player = $("#resultVideo"), canvas = $("#reviewOverlay"), context = canvas.getContext("2d");
    const start = Number(button.dataset.time), end = Number(button.dataset.end);
    canvas.classList.remove("hidden");
    const draw = () => {
      if (request !== state.overlayRequest) return;
      const width = player.clientWidth, height = player.clientHeight;
      canvas.width = width; canvas.height = height;
      if (player.currentTime >= end && !player.paused) player.pause();
      const box = boxes.get(Math.floor(player.currentTime * data.fps + 1e-4));
      if (box && player.videoWidth && player.currentTime >= start && player.currentTime < end) {
        const scale = Math.min(width / player.videoWidth, height / player.videoHeight);
        const dx = (width - player.videoWidth * scale) / 2, dy = (height - player.videoHeight * scale) / 2;
        context.strokeStyle = "#16c79a"; context.lineWidth = 3;
        context.strokeRect(dx + box[0] * scale, dy + box[1] * scale, (box[2] - box[0]) * scale, (box[3] - box[1]) * scale);
        context.font = "bold 15px sans-serif"; context.fillStyle = "#16c79a";
        context.fillText(`RAW T${tid}`, dx + box[0] * scale, Math.max(18, dy + box[1] * scale - 6));
      }
      state.overlayAnimation = requestAnimationFrame(draw);
    };
    const seek = () => {
      if (request !== state.overlayRequest) return;
      player.currentTime = start; player.play().catch(() => {}); draw();
    };
    if (player.readyState >= 1) seek(); else player.addEventListener("loadedmetadata", seek, {once: true});
    player.scrollIntoView({behavior: "smooth", block: "center"});
    toast(`正在单独复核原始轨迹 T${tid}`, `${formatTime(start)} → ${formatTime(end)}；没观测的帧不补框，播放结束会暂停。`);
  } catch (error) { if (request === state.overlayRequest) toast("复核播放失败", error.message, "error"); }
}

async function saveTrackReview() {
  const projectId = state.project.project_id, videoId = state.videoId;
  const data = state.reviewData;
  const tracks = $$(".review-card").map((card) => {
    const value = (field) => card.querySelector(`[data-field="${field}"]`).value;
    return {raw_track_id: Number(card.dataset.track), verdict: value("verdict"), scope: value("scope"), identity_label: value("identity_label") || null, note: value("note")};
  });
  $("#saveReviewButton").disabled = true;
  $$(".review-card input, .review-card select").forEach((element) => { element.disabled = true; });
  try {
    const saved = await api(`/api/projects/${projectId}/videos/${videoId}/track-review`, {method: "PUT", body: JSON.stringify({expected_revision: data.revision, review: {...data.review, tracks}})});
    if (projectId !== state.project?.project_id || videoId !== state.videoId) return;
    state.reviewData = saved;
    state.reviewDirty = false;
    state.detail.artifacts.evaluation.track_review = saved.metrics;
    toast("复核已保存", "刷新或重新打开仍保留；不会改变原模型结果。归并修正需要单独生成新结果。 ");
    await renderReview();
  } catch (error) { toast("保存失败", error.message, "error"); }
  finally {
    if (projectId === state.project?.project_id && videoId === state.videoId && $("#saveReviewButton")) {
      $("#saveReviewButton").disabled = false;
      $$(".review-card input, .review-card select").forEach((element) => {
        element.disabled = element.dataset.field === "identity_label" && element.closest(".review-card").querySelector('[data-field="verdict"]').value !== "pure";
      });
    }
  }
}

async function loadBatchReview() {
  const projectId = state.project?.project_id;
  if (!projectId) return;
  try {
    const summary = await api(`/api/projects/${projectId}/track-reviews`);
    if (projectId !== state.project?.project_id) return;
    const subset = summary.completed_subset;
    const text = reviewSummary(subset);
    $("#batchReviewSummary").textContent = `${summary.complete ? "复核数据读取完成" : "数据不完整，以下仅为已读取子集"}：${summary.completed_clips}/${summary.expected_clips} 条视频，${summary.failed_clips.length} 条读取失败。\n${text}\n混人和非球员档案须先修正；这不是漏检召回率或事件准确率。`;
    $("#batchReviewSummary").classList.remove("hidden");
  } catch (error) { toast("汇总失败", error.message, "error"); }
}

function renderEvents() {
  const index = state.detail.artifacts.action.index;
  if (!state.detail.artifacts.available.events) {
    $("#tabContent").innerHTML = emptyState("尚未生成事件区间", "动作预测点可以在单独的标签页检查；请先运行事件聚合。");
    return;
  }
  const allPeople = index.people || [];
  const labels = Object.keys(index.label_counts || {});
  const filteredCount = allPeople.filter(person => !state.personFilter || person.person_id === state.personFilter).reduce((sum, person) => sum + person.events.filter(event => !state.eventFilter || event.event === state.eventFilter).length, 0);
  const filters = `<div class="event-filters">
    <label>人物档案<select id="personFilter" aria-label="人物档案"><option value="">全部档案</option>${allPeople.map((person) => `<option value="${escapeHtml(person.person_id)}" ${state.personFilter === person.person_id ? "selected" : ""}>${escapeHtml(person.person_id)}</option>`).join("")}</select></label>
    <label>事件类型<select id="eventFilter" aria-label="事件类型"><option value="">全部类型</option>${labels.map((label) => `<option value="${escapeHtml(label)}" ${state.eventFilter === label ? "selected" : ""}>${escapeHtml(label)}</option>`).join("")}</select></label>
    <span>筛选 ${filteredCount} / 总 ${index.event_count} 个区间 · raw_score 未校准</span>
  </div>`;
  const meta = state.detail.artifacts.tracking.video_meta || {};
  const duration = Math.max(Number(meta.processed_frames || 0) / Number(meta.fps || 1), 0.01);
  const people = allPeople.filter((person) => !state.personFilter || person.person_id === state.personFilter);
  const sections = people.map((person) => {
    const events = person.events.filter((event) => !state.eventFilter || event.event === state.eventFilter);
    if (!events.length) return "";
    const displayed = events.slice(0, 300);
    const image = person.identity?.cover?.crop_url;
    return `<section class="interval-person">
      <div class="interval-person-heading">${image ? `<img class="event-avatar" src="${escapeHtml(image)}" alt="${escapeHtml(person.person_id)}" />` : ""}<div><strong>${escapeHtml(person.person_id)}</strong><small>${events.length} 个事件区间 · 原轨迹 ${escapeHtml(person.raw_track_ids.join(", "))}</small></div></div>
      ${displayed.map((event) => `<button class="interval-row" data-start="${Number(event.start)}" data-end="${Number(event.end)}" title="跳转到 ${Number(event.start).toFixed(2)} 秒">
        <div><strong>${escapeHtml(event.event.replace(/^basketball_/, ""))}</strong><small>${formatTime(event.start)} → ${formatTime(event.end)}</small></div>
        <div class="timeline-track"><span style="left:${Math.min(100, Math.max(0, Number(event.start) / duration * 100))}%;width:${Math.min(100, Math.max(0.5, (Number(event.end) - Number(event.start)) / duration * 100))}%"></span></div>
        <div class="interval-score"><strong>${Number(event.raw_score).toFixed(3)}</strong><small>${Number(event.support_count)} 个支持点 · ${(Number(event.end) - Number(event.start)).toFixed(2)}s</small></div><span>↗</span>
      </button>`).join("")}
      ${events.length > displayed.length ? `<p class="scope-warning">只展示前 300 个；完整记录在原始数据与诊断包中。</p>` : ""}
    </section>`;
  }).join("");
  $("#tabContent").innerHTML = filters + (sections || emptyState("没有符合条件的事件", index.event_count ? "清除筛选后再查看。" : "聚合文件存在但没有事件；这不代表视频中没有真实动作。"));
  ["person", "event"].forEach((kind) => $(`#${kind}Filter`).addEventListener("change", (event) => {
    state[`${kind}Filter`] = event.target.value;
    renderEvents();
  }));
  $$(".interval-row").forEach((button) => button.addEventListener("click", () => {
    const media = state.detail.artifacts.media || {};
    if (!media[state.videoView]) setVideoView(media.source ? "source" : "final", false);
    const player = $("#resultVideo");
    const seek = () => { player.currentTime = Number(button.dataset.start); };
    if (player.readyState >= 1) seek();
    else player.addEventListener("loadedmetadata", seek, {once: true});
    $(".video-panel").scrollIntoView({behavior: "smooth", block: "center"});
  }));
}

function renderActionPoints() {
  const index = state.detail.artifacts.action.points;
  if (!index?.people?.length) {
    $("#tabContent").innerHTML = emptyState("还没有动作预测点", "采样点不是事件区间。低于阈值的候选只用于诊断。");
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
        <div class="event-person-title"><strong>${escapeHtml(person.person_id || `Track ${person.raw_track_ids.join(",")}`)}</strong><span>${person.event_count} points · IDs ${escapeHtml(person.raw_track_ids.join(", "))}</span></div>
        <div class="event-list">${events}</div>
      </section>`;
    })
    .join("");
}

async function renderRaw() {
  const kinds = ["events", "tracks", "quality", "quality_observations", "identities", "resolution", "actions", "linked_actions", "pairs", "samples", "sampling"];
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
    identity_merge_distance: $("#identityMergeDistance").value.trim() ? Number($("#identityMergeDistance").value) : null,
    prompt_mode: "none",
    action_threshold: Number($("#actionThreshold").value),
    action_min_det_score: 0.3,
  };
}

async function runCurrent(target) {
  if (state.reviewDirty) { toast("请先保存复核", "处理前保存当前修改，避免轨迹重跑后旧复核失效。", "error"); return; }
  state.reviewData = null;
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
  if (state.reviewDirty) { toast("请先保存复核", "批量处理前保存当前修改。", "error"); return; }
  state.reviewData = null;
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
  if (state.project?.read_only) {
    toast("当前实验为只读导入", "请用左侧 ＋ 新建实验，再上传需要重跑的视频。", "error");
    return;
  }
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
      const project = await api(`/api/projects/${projectId}`);
      if (projectId !== state.project?.project_id) return;
      state.project = project;
      renderClipStrip();
      renderCrossClipIndex();
      if (state.videoId) await refreshCurrentVideo();
      if (state.activeTab === "logs" && state.detail?.job_active) renderLogs();
    } catch (_) {}
  }, 2500);
}

bootstrap();
