const state = {
  projects: [],
  project: null,
  videoId: null,
  detail: null,
  activeTab: "workflow",
  identityView: "resolved",
  configuration: null,
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
  queue: null, queueOffset: 0, queueFilter: 'all', queueSearch: '', queueRequest: 0,
  materialOffset: 0, materialView: 'resolved', materialSearch: '', materialRequest: 0,
  reviewOffset: 0, reviewFilter: 'retained', reviewSearch: '', reviewDrafts: new Map(),
  reviewBaseRevision: null, savingReview: false,
  identityOffset: 0, identitySearch: '', trackingOffset: 0, eventOffset: 0,
  evidenceOffset: 0, evidenceTrack: '', evidenceRequest: 0,
  assignmentRequest: 0,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function preferences() {
  try { return JSON.parse(localStorage.getItem('courtvision.inspection.preferences') || '{}'); }
  catch (_) { return {}; }
}

function rememberPosition() {
  try { localStorage.setItem('courtvision.inspection.preferences', JSON.stringify({
    projectId: state.project?.project_id, videoId: state.videoId, activeTab: state.activeTab,
    queueOffset: state.queueOffset, queueFilter: state.queueFilter, queueSearch: state.queueSearch,
    reviewOffset: state.reviewOffset, reviewSearch: state.reviewSearch,
    reviewFilter: state.reviewFilter, identityView: state.identityView,
  })); } catch (_) {}
}

function pager(id, offset, limit, total, change) {
  const element = $(`#${id}`);
  if (!element) return;
  element.innerHTML = `<span>${total ? offset + 1 : 0}–${Math.min(total, offset + limit)} / ${total}</span><button data-page="prev" ${offset <= 0 ? 'disabled' : ''}>← 上一页</button><button data-page="next" ${offset + limit >= total ? 'disabled' : ''}>下一页 →</button>`;
  element.querySelector('[data-page="prev"]').addEventListener('click', () => change(Math.max(0, offset - limit)));
  element.querySelector('[data-page="next"]').addEventListener('click', () => change(offset + limit));
}

function debounce(handler, delay = 250) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => handler(...args), delay); };
}

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
    state.configuration = await api('/api/configuration');
    const health = await api("/api/health");
    const ready = health.configuration?.ready;
    $("#systemState").classList.add(ready ? "ready" : "error");
    $("#systemState span:last-child").textContent = ready
      ? "服务在线 · 模型路径已配置"
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
    const prefs = preferences();
    state.queueOffset = Math.max(0, Number(prefs.queueOffset) || 0);
    state.queueFilter = ['all','remaining','complete','risk','uncertain','failed'].includes(prefs.queueFilter) ? prefs.queueFilter : 'all';
    state.queueSearch = String(prefs.queueSearch || '').slice(0, 120);
    state.reviewFilter = ['all','remaining','retained','filtered','risk','uncertain','full'].includes(prefs.reviewFilter) ? prefs.reviewFilter : 'retained';
    state.identityView = prefs.identityView === 'raw' ? 'raw' : 'resolved';
    const projectId = state.projects.some(p => p.project_id === prefs.projectId) ? prefs.projectId : state.projects[0].project_id;
    await selectProject(projectId);
    if (state.project?.videos.some(v => v.video_id === prefs.videoId)) await selectVideo(prefs.videoId, false);
    if (state.videoId === prefs.videoId) {
      state.reviewOffset = Math.max(0, Number(prefs.reviewOffset) || 0);
      state.reviewSearch = String(prefs.reviewSearch || '').slice(0, 120);
    }
    if (['workflow','sampling','jersey','identity','review','tracking','events','points','raw','logs'].includes(prefs.activeTab)) setTab(prefs.activeTab);
  } else {
    await createProject("篮球比赛");
  }
  startPolling();
  $('#systemState').addEventListener('click', () => {
    const config = state.configuration.settings;
    const missing = Object.entries(state.configuration.readiness.paths).filter(([, v]) => !v.exists).map(([key, v]) => `${key}: ${v.path}`).join('\n');
    alert(`网页项目目录：${config.output_root}\n允许导入：${config.import_root}\n原视频根目录：${config.source_root}\n复核截图 Python：${config.review_python}\n\n缺失配置：\n${missing || '无'}\n\n模型路径不完整不妨碍浏览已有结果。记录持久保存在索引目录，重启需读取同一目录。`);
  });
}

function bindEvents() {
  bindPersonLibrary();
  $('#refreshMatchButton').addEventListener('click', refreshMatch);
  const input = $("#fileInput");
  $('#toggleImportButton').addEventListener('click', () => {
    const hidden = $('#importSection').classList.toggle('hidden');
    $('#toggleImportButton').setAttribute('aria-expanded', String(!hidden));
    if (!hidden) $('#importPath').focus();
  });
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
    if (state.savingReview) return;
    const name = window.prompt("新建一场比赛；同场片段共用一个球员库", "篮球比赛");
    if (name !== null) await createProject(name);
  });
  $("#optionsButton").addEventListener("click", () =>
    $("#optionsPanel").classList.toggle("hidden"),
  );
  $("#runFullButton").addEventListener("click", () => runCurrent("full"));
  $("#importForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (state.savingReview) return;
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
  $('#nextReviewButton').addEventListener('click', nextReviewVideo);
  $('#refreshQueueButton').addEventListener('click', () => { loadReviewQueue(); renderCrossClipIndex(); });
  $('#queueFilter').addEventListener('change', event => { state.queueFilter = event.target.value; state.queueOffset = 0; loadReviewQueue(); });
  $('#queueSearch').addEventListener('input', debounce(event => { state.queueSearch = event.target.value; state.queueOffset = 0; loadReviewQueue(); }));
  $('#materialView').addEventListener('change', event => { state.materialView = event.target.value; state.materialOffset = 0; renderCrossClipIndex(); });
  $('#materialSearch').addEventListener('input', debounce(event => { state.materialSearch = event.target.value; state.materialOffset = 0; renderCrossClipIndex(); }));
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
    toast("已新建比赛", project.name);
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
          <span>${project.videos?.length || 0} 个片段 · ${project.read_only ? 'CLI 结果' : '网页处理'}</span>
        </button>`,
    )
    .join("");
  $$(".project-item").forEach((button) =>
    button.addEventListener("click", () => selectProject(button.dataset.projectId)),
  );
}

async function selectProject(projectId, preserveVideo = false) {
  if (libraryState.editing) { toast('球员库正在更新', '请等待保存完成后再切换比赛。'); return; }
  if (state.savingReview) { toast('复核正在保存', '请等待保存完成后再切换比赛。'); return; }
  if (state.reviewDirty && !window.confirm("复核尚未保存，确定放弃修改并切换吗？")) return;
  const request = ++state.projectRequest;
  try {
    const previousVideo = preserveVideo ? state.videoId : null;
    const project = await api(`/api/projects/${projectId}?include_people=false`);
    if (request !== state.projectRequest) return;
    state.project = project;
    state.reviewDirty = false;
    state.reviewData = null;
    state.reviewDrafts.clear();
    state.detail = null;
    state.videoId = null;
    $("#batchReviewSummary").classList.add("hidden");
    $("#reviewBundleDownload").href = `/api/projects/${projectId}/track-reviews.zip`;
    if (!preserveVideo) state.videoView = state.project.read_only ? "final" : "source";
    const readOnly = Boolean(state.project.read_only);
    document.body.classList.toggle("inspection-mode", readOnly);
    $('#uploadSection').classList.toggle('hidden', readOnly);
    $('#inspectionIntro').classList.remove('hidden');
    $('#matchContext').textContent = `${project.name} · ${project.videos.length} 个可浏览片段 · ${readOnly ? 'CLI 结果，追加处理后点击“刷新比赛”更新' : '上传本场比赛片段，顺序处理并持续更新球员库'}`;
    $('#importSection').classList.toggle('hidden', readOnly);
    $('#toggleImportButton').setAttribute('aria-expanded', 'false');
    $("#readOnlyBadge").classList.toggle("hidden", !readOnly);
    $("#batchRunButton").disabled = readOnly;
    $("#addMoreButton").disabled = readOnly;
    $("#projectTitle").textContent = state.project.name;
    renderProjects();
    state.queue = null;
    resetPersonLibrary();
    $('#queueFilter').value = state.queueFilter;
    $('#queueSearch').value = state.queueSearch;
    await loadReviewQueue();
    renderCrossClipIndex();
    const importWarnings = state.project.import_warnings || [];
    $('#importWarnings').classList.toggle('hidden', importWarnings.length === 0);
    $('#importWarnings').textContent = importWarnings.length
      ? `本批次计划 ${state.project.expected_video_count ?? '未知'} 条，当前可浏览 ${state.project.videos.length} 条。${importWarnings.join('；')}`
      : '';
    const videos = state.project.videos || [];
    $("#workspace").classList.toggle("hidden", videos.length === 0);
    if (videos.length) {
      const selected = videos.some((item) => item.video_id === previousVideo)
        ? previousVideo
        : state.queue?.items?.[0]?.video_id || videos[0].video_id;
      await selectVideo(selected, false);
    } else {
      state.videoId = null;
      state.detail = null;
    }
  } catch (error) {
    toast("加载比赛失败", error.message, "error");
  }
}

async function refreshMatch() {
  if (!state.project || state.savingReview || libraryState.editing) return;
  if (state.reviewDirty) { toast('请先保存轨迹复核', '保存后再刷新比赛结果。', 'error'); return; }
  const project = state.project, opened = libraryState.opened;
  const button = $('#refreshMatchButton'); button.disabled = true;
  try {
    if (project.read_only && project.imported_from) {
      await api('/api/import-results', {method: 'POST', body: JSON.stringify({result_dir: project.imported_from, name: project.name})});
    }
    if (project.project_id !== state.project?.project_id) return;
    await loadProjects();
    await selectProject(project.project_id, true);
    if (opened) await openPersonLibrary();
    toast('比赛结果已刷新', `${state.project.videos.length} 个片段`);
  } catch (error) { toast('刷新失败', error.message, 'error'); }
  finally { button.disabled = false; }
}

function renderClipStrip() {
  const videos = state.queue?.items || [];
  $("#clipStrip").innerHTML = videos
    .map((video) => {
      const label = {unreviewed:'未复核', partial:'部分已复核', complete:'完整复核已完成', uncertain:'不确定', stale:'旧复核已失效', unavailable:'无法复核', processing:'处理中', empty:'零轨迹 · 待核查漏检'}[video.review_status] || video.review_status;
      return `
        <button class="clip-card ${state.videoId === video.video_id ? "active" : ""}"
                data-video-id="${escapeHtml(video.video_id)}">
          <div class="clip-card-top">
            <span class="eyebrow">VIDEO ${String(video.ordinal).padStart(3, "0")}</span>
            <i class="mini-status ${escapeHtml(video.status)}"></i>
          </div>
          <strong title="${escapeHtml(video.filename)}">${escapeHtml(video.filename)}</strong>
          <small>${formatTime(video.duration_s)} · ${formatBytes(video.size_bytes)} · ${escapeHtml(String(video.status).toUpperCase())}</small>
          <span class="review-chip ${escapeHtml(video.review_status)} ${video.mixed_tracks ? 'risk' : ''}">${escapeHtml(label)} · ${video.full_assessed}/${video.raw_tracks}</span>
          <span class="queue-detail">保留 ${video.retained_tracks} · 待完整检查 ${video.remaining_tracks} · 抽样 ${video.sampled_tracks} · 不确定 ${video.uncertain_tracks}<br />告警 ${video.risk_tracks} · 混人 ${video.mixed_tracks} · 档案 ${video.raw_archives ?? '—'} → ${video.resolved_archives}</span>
        </button>`;
    })
    .join("");
  if (!videos.length) $('#clipStrip').innerHTML = emptyState('没有符合条件的视频', '清除搜索或切换审查队列。');
  $$(".clip-card").forEach((button) =>
    button.addEventListener("click", () => selectVideo(button.dataset.videoId)),
  );
  if (state.queue) pager('queuePagination', state.queue.offset, state.queue.limit, state.queue.total, offset => { state.queueOffset = offset; loadReviewQueue(); });
}

async function loadReviewQueue() {
  if (!state.project) return;
  const projectId = state.project.project_id, request = ++state.queueRequest;
  try {
    const query = new URLSearchParams({offset: state.queueOffset, limit: 12, status: state.queueFilter, q: state.queueSearch});
    const data = await api(`/api/projects/${projectId}/review-queue?${query}`);
    if (projectId !== state.project?.project_id || request !== state.queueRequest) return;
    if (data.total && state.queueOffset >= data.total) { state.queueOffset = Math.floor((data.total - 1) / 12) * 12; return loadReviewQueue(); }
    state.queue = data;
    const summary = data.summary, processing = summary.processing_statuses;
    $('#batchProgress').innerHTML = [[`${processing.completed || 0} / ${summary.expected_clips}`, '片段处理完成', `可浏览 ${summary.available_clips} · 尚缺 ${summary.missing_clips}`], [summary.raw_tracks, 'MOTIP 原始轨迹', `人工完整检查 ${summary.full_assessed}`], [(processing.running || 0) + (processing.queued || 0), '处理或排队中的片段', `失败 ${processing.failed || 0}`], [summary.remaining_tracks, '可继续人工检查', `不确定 ${summary.uncertain_tracks} · 混人 ${summary.mixed_tracks}`]].map(([value,label,note]) => `<div class="batch-stat"><strong>${escapeHtml(value)}</strong><span>${label}</span><small>${escapeHtml(note)}</small></div>`).join('');
    renderClipStrip(); rememberPosition();
  } catch (error) { if (projectId === state.project?.project_id) toast('审查队列加载失败', error.message, 'error'); }
}

async function nextReviewVideo() {
  if (state.savingReview) return;
  if (!state.project) return;
  if (state.reviewDirty) { toast('请先保存复核', '使用“保存并下一条视频”可保存后继续。', 'error'); return; }
  const projectId = state.project.project_id;
  try {
    const status = ['risk', 'uncertain'].includes(state.queueFilter) ? state.queueFilter : 'remaining';
    const data = await api(`/api/projects/${projectId}/next-review?${new URLSearchParams({after:state.videoId || '', q:state.queueSearch, status})}`);
    if (projectId !== state.project?.project_id) return;
    if (!data.video_id) { toast('当前搜索 / 告警筛选范围没有待复核视频', '零轨迹、无法读取和失败视频仍需单独排查漏检或运行问题。'); return; }
    await selectVideo(data.video_id, false); setTab('review');
    $('#inspector').scrollIntoView({behavior:'smooth', block:'start'});
  } catch (error) { toast('无法切换待复核视频', error.message, 'error'); }
}

async function selectVideo(videoId, resetTab = true) {
  if (state.savingReview) { toast('复核正在保存', '请等待保存完成后再切换视频。'); return; }
  if (videoId !== state.videoId) {
    if (state.reviewDirty && !window.confirm("复核尚未保存，确定放弃修改并切换片段吗？")) return;
    state.personFilter = "";
    state.eventFilter = "";
    state.reviewData = null;
    state.reviewDirty = false;
    state.reviewDrafts.clear();
    state.reviewOffset = 0; state.reviewSearch = ''; state.reviewBaseRevision = null;
    state.identityOffset = 0; state.identitySearch = ''; state.trackingOffset = 0; state.eventOffset = 0;
    state.evidenceOffset = 0; state.evidenceTrack = ''; ++state.evidenceRequest; ++state.assignmentRequest;
    state.detail = null;
    ++state.reviewRequest;
    clearReviewOverlay();
  }
  state.videoId = videoId;
  if (resetTab) state.activeTab = "workflow";
  rememberPosition();
  renderClipStrip();
  await refreshCurrentVideo();
  if (resetTab && state.videoId === videoId) $('#inspector').scrollIntoView({behavior:'smooth', block:'start'});
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
    resolution: "正在整理轨迹档案",
    players: "正在更新球员库",
    jersey: '正在读取号码候选',
    action: "正在识别动作",
    link: "正在连接人物事件",
    aggregate: "正在聚合事件区间",
    render: "正在渲染结果",
    review: "正在准备复核证据",
  }[stage] || "等待 GPU 队列";
}

function renderStageRail(stages = {}) {
  renderWorkflowRail(stages);
}

function renderMetrics(artifacts) {
  const tracking = artifacts.tracking.summary || {};
  const identity = artifacts.identity.summary || {};
  const actions = artifacts.action.index || {};
  const frames = artifacts.tracking.video_meta?.processed_frames;
  const quality = artifacts.tracking.quality;
  const metrics = [
    ["人物轨迹", quality ? `${quality.raw_track_count ?? tracking.track_count ?? '—'} → ${quality.retained_track_count ?? '—'}` : tracking.track_count ?? "—", quality ? "原始 → 有时序支持的轨迹（非准确率）" : "原始 track ID 数量"],
    ["片段身份档案", artifacts.identity.resolved_count, "质量采样后建立的局部轨迹档案"],
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
  if (quality) messages.push(`质量检查：${quality.tentative_track_count ?? '未记录'} 条证据不足，${quality.review_track_count ?? '未记录'} 条需复核。`);
  if (identity.missing_archive_tracks?.length) messages.push(`身份缺失：${identity.missing_archive_tracks.length} 条保留轨迹没有 KPR 档案，事件可能暂未关联球员身份。`);
  if (!messages.length) messages.push('处理状态表示阶段执行结果；球员候选和动作事件可继续按原始证据检查。');
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
  if (state.savingReview) return;
  if (state.reviewDirty && tab !== "review") {
    if (!window.confirm("请先保存复核。确定放弃未保存的修改吗？")) return;
    state.reviewDirty = false;
    state.reviewData = null;
    state.reviewDrafts.clear();
  }
  if (state.activeTab !== tab) { state.evidenceOffset = 0; state.evidenceTrack = ''; ++state.evidenceRequest; }
  state.activeTab = tab;
  rememberPosition();
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
  if (state.activeTab === "workflow") renderWorkflow();
  else if (state.activeTab === "sampling") renderIdentityEvidence('kpr');
  else if (state.activeTab === "jersey") renderIdentityEvidence('jersey');
  else if (state.activeTab === "tracking") renderTracking();
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
  if (state.trackingOffset >= summary.tracks.length) state.trackingOffset = 0;
  const rows = summary.tracks.slice(state.trackingOffset, state.trackingOffset + 24)
    .map((track) => `
      <tr>
        <td><strong>T${escapeHtml(track.track_id)}</strong><button class="ghost-button track-review-link" data-track="${escapeHtml(track.track_id)}">复核此轨迹</button></td>
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
    </table><div class="pagination" id="trackingPagination"></div>`;
  pager('trackingPagination', state.trackingOffset, 24, summary.tracks.length, offset => { state.trackingOffset = offset; renderTracking(); });
  $$('.track-review-link').forEach(button => button.addEventListener('click', () => openTrackReview(Number(button.dataset.track))));
}

async function openTrackReview(tid) {
  if (state.savingReview) return;
  state.reviewFilter = 'all'; state.reviewSearch = `T${tid}`; state.reviewOffset = 0;
  state.activeTab = 'review';
  await renderReview();
  document.querySelector(`.review-card[data-track="${tid}"]`)?.scrollIntoView({behavior: 'smooth', block: 'center'});
}

function jerseyBadge(person) {
  const jersey = person.jersey;
  if (person.identity_label) return `<p class="jersey-badge">人工标签：${escapeHtml(person.identity_label)}</p>`;
  if (!jersey) return '<p class="archive-warning">号码未运行 / 无证据</p>';
  const text = jersey.number != null ? `号码候选 #${jersey.number} · 未人工确认` : jersey.status === 'conflict' ? '号码读数冲突 · 不能作为身份' : '号码不确定 / 不可读';
  const evidence = (jersey.candidates || []).map(candidate => `<div><strong>#${escapeHtml(candidate.number)}</strong> · ${candidate.support_frames} 个独立帧<div class="exemplar-strip">${candidate.evidence.map(row => row.crop_url ? `<a href="${escapeHtml(row.crop_url)}" target="_blank" rel="noopener"><img src="${escapeHtml(row.crop_url)}" alt="号码证据帧 ${row.frame_idx}" loading="lazy" /></a>` : '').join('')}</div></div>`).join('');
  const readings = jersey.readings || (jersey.tracks || []).flatMap(t => t.readings || []);
  const details = readings.map(r => `<span>${r.crop_url ? `<a href="${escapeHtml(r.crop_url)}" target="_blank" rel="noopener"><img src="${escapeHtml(r.crop_url)}" alt="Qwen读号帧 ${escapeHtml(r.frame_idx)}" loading="lazy" /></a>` : ''}<small>#${escapeHtml(r.text)}<br />${escapeHtml((r.rejection_reasons || []).join(' / ') || '候选支持，尚未确认')}</small></span>`).join('');
  return `<p class="jersey-badge ${jersey.status === 'conflict' ? 'error-text' : ''}">${escapeHtml(text)}</p>${evidence || details ? `<details><summary>号码识别证据</summary>${evidence}<div class="number-readings">${details}</div></details>` : ''}`;
}

function renderIdentity() {
  const identity = state.detail.artifacts.identity;
  const raw = state.identityView === 'raw';
  const allPeople = raw ? identity.raw_people || [] : identity.people || [];
  const selected = allPeople.filter(person => !state.identitySearch || `${person.person_id} ${person.identity_label || ''} ${person.raw_track_ids.join(' ')} ${person.jersey?.number ?? ''}`.toLowerCase().includes(state.identitySearch.toLowerCase()));
  if (state.identityOffset >= selected.length) state.identityOffset = Math.max(0, Math.floor((selected.length - 1) / 12) * 12);
  const people = selected.slice(state.identityOffset, state.identityOffset + 12);
  const rawTracks = state.detail.artifacts.tracking.quality?.raw_track_count ?? state.detail.artifacts.tracking.summary?.track_count ?? '—';
  const retained = state.detail.artifacts.tracking.quality?.retained_track_count ?? '未检查';
  const toolbar = `<div class="archive-comparison"><div><h3>片段档案 · KPR 外观与轨迹来源</h3><p>原始轨迹 ${rawTracks} · 质量保留 ${retained} · KPR 原始档案 ${identity.raw_count ?? '未记录'} · 局部档案 ${identity.resolved_count}。当前流程先保留单轨迹档案，由比赛球员库建立同场身份；旧结果中的归并档案仍可浏览。</p></div><div class="archive-switch"><button class="${raw ? 'active' : ''}" data-archive-view="raw">KPR 原始档案</button><button class="${raw ? '' : 'active'}" data-archive-view="resolved">片段身份档案</button></div></div><section class="assignment-section"><h3>当前比赛球员归属</h3><div id="clipAssignments">正在读取球员库…</div></section><div class="queue-controls archive-view-controls"><label class="search-field">查找本视频档案<input id="identitySearch" type="search" value="${escapeHtml(state.identitySearch)}" placeholder="档案 ID / 原轨迹 / 标签 / 号码候选" /></label><span class="queue-caveat">每页 12 份；完整数据仍保留</span></div>`;
  if (!people.length) {
    $("#tabContent").innerHTML = toolbar + emptyState("还没有这一类档案", raw ? "本片段没有合格 KPR 样本，或旧结果未保留原始档案。" : "完成质量采样与 KPR 特征提取后查看片段档案。");
    bindArchiveSwitch();
    renderClipAssignments();
    return;
  }
  $("#tabContent").innerHTML = toolbar + `<div class="identity-grid">${people
    .map((person) => {
      const cover = person.cover || {};
      const events = raw ? null : state.detail.artifacts.action.index.people.find((p) => p.person_id === person.person_id)?.event_count || 0;
      const status = {human_confirmed: "人工确认归并", auto_merged: "模型候选归并 · 待验证", needs_review: "有风险 · 需复核", unresolved: "未确认身份", unlabeled: "未确认身份"}[person.status] || person.status;
      const audit = raw ? null : state.detail.artifacts.evaluation.track_review?.archives?.groups.find((group) => group.person_id === person.person_id);
      const auditText = audit ? {conflict: "复核发现混人 / 错误归并，不能作为真人档案使用", consistent: "当前来源轨迹完整复核一致（不是独立模型准确率）", unverified: "仍有来源轨迹未完整复核"}[audit.status] : "尚无真人一致性复核";
      const image = cover.crop_url
        ? `<img src="${escapeHtml(cover.crop_url)}" alt="${escapeHtml(person.person_id)}" loading="lazy" />`
        : `<div class="no-image">NO COVER</div>`;
      return `<article class="identity-card">
        <div class="identity-image">${image}</div>
        <div class="identity-body">
          <div class="identity-head"><strong>${escapeHtml(person.person_id)}</strong><span>TRACK ${escapeHtml(person.raw_track_ids?.join(", "))}</span></div>
          ${jerseyBadge(person)}
          <div class="identity-stats"><span>${escapeHtml(status)}</span><span>${raw ? '原始单轨迹档案' : events + ' 个事件区间'}</span><span>${person.raw_track_ids?.length || 1} 条来源轨迹</span></div>
          <div class="identity-stats"><span>${escapeHtml(person.sample_count)} samples</span><span>${escapeHtml(person.observation_count)} observations</span></div>
          <p class="archive-warning">封面来自 T${escapeHtml(person.representative_raw_track_id ?? cover.raw_track_id ?? person.raw_track_ids?.[0])}，不能证明整份档案始终同人。</p>
          <p class="archive-warning ${audit?.status === "conflict" ? "error-text" : ""}">${escapeHtml(auditText)}</p>
          <div class="exemplar-strip">${(person.exemplars || []).map((item) => item.crop_url ? `<img src="${escapeHtml(item.crop_url)}" alt="T${escapeHtml(item.raw_track_id)} 帧 ${escapeHtml(item.frame_idx)}" title="T${escapeHtml(item.raw_track_id)} / frame ${escapeHtml(item.frame_idx)}" loading="lazy" />` : "").join("")}</div>
          ${raw ? `<p class="archive-warning">轨迹内平均 KPR 距离：${person.within_track?.mean_distance == null ? '无足够采样' : Number(person.within_track.mean_distance).toFixed(3)}。距离不是同人概率，须查看不同时间的观测。</p>` : `<details class="archive-sources"><summary>逐条查看来源轨迹 (${person.raw_track_ids?.length || 1})</summary>${(person.source_tracks || []).map((source) => `<div class="archive-source">${source.cover?.crop_url ? `<a href="${escapeHtml(source.cover.context_url || source.cover.crop_url)}" target="_blank" rel="noopener"><img src="${escapeHtml(source.cover.crop_url)}" alt="T${escapeHtml(source.raw_track_id)}" loading="lazy" /></a>` : ""}<div><strong>T${escapeHtml(source.raw_track_id)}</strong><small>${formatTime(source.start)} → ${formatTime(source.end)} · ${escapeHtml(source.observation_count)} 次观测</small><small>${escapeHtml((source.review_reasons || []).join(" / ") || (source.status === "human_confirmed" || audit?.status === "consistent" ? "已人工确认来源同人" : "未人工验证"))}</small></div></div>`).join("") || '<p class="archive-warning">旧档案缺少时序来源证据，请检查原始轨迹与采样图。</p>'}</details>`}
          ${raw ? `<button class="ghost-button raw-review-link" data-track="${person.raw_track_ids?.[0]}">复核原始轨迹 T${person.raw_track_ids?.[0]} →</button>` : `<button class="ghost-button person-events-button" data-person="${escapeHtml(person.person_id)}">查看这个档案的事件 →</button>`}
        </div>
      </article>`;
    })
    .join("")}</div><div class="pagination" id="identityPagination"></div>`;
  bindArchiveSwitch();
  renderClipAssignments();
  pager('identityPagination', state.identityOffset, 12, selected.length, offset => { state.identityOffset = offset; renderIdentity(); });
  $$('.raw-review-link').forEach(button => button.addEventListener('click', () => openTrackReview(Number(button.dataset.track))));
  $$(".person-events-button").forEach((button) => button.addEventListener("click", () => {
    state.personFilter = button.dataset.person;
    setTab("events");
  }));
}

function bindArchiveSwitch() {
  $('#identitySearch')?.addEventListener('input', debounce(event => { state.identitySearch = event.target.value; state.identityOffset = 0; renderIdentity(); }));
  $$('[data-archive-view]').forEach(button => button.addEventListener('click', () => {
    state.identityView = button.dataset.archiveView;
    state.identityOffset = 0; rememberPosition();
    renderIdentity();
  }));
}

function reviewSummary(metrics) {
  const full = metrics.scopes.full_track.raw, sampled = metrics.scopes.sampled.raw, stable = metrics.scopes.full_track.retained;
  const fragments = metrics.fragmentation;
  const stableName = metrics.quality_available === false ? "未加载质量报告（同原始范围）" : "质量过滤后";
  const percent = (value) => value == null ? "尚无分母" : `${(value * 100).toFixed(1)}%`;
  return `完整复核：${full.assessed_tracks}/${full.total_tracks} 条（${percent(full.review_coverage)}）。原始球员轨迹同人率 ${full.pure}/${full.player_tracks_assessed}（${percent(full.pure_track_rate)}），混人 ${full.mixed} 条，非球员 ${full.non_player} 条。\n${stableName}：完整复核 ${stable.assessed_tracks}/${stable.total_tracks} 条，同人率 ${stable.pure}/${stable.player_tracks_assessed}（${percent(stable.pure_track_rate)}），混人 ${stable.mixed} 条。\n仅抽样：同人候选 ${sampled.pure}，已发现混人 ${sampled.mixed}；不能据此声称全轨迹正确。\n身份已标注的完整同人轨迹：${fragments.labeled_pure_tracks} 条 / ${fragments.reviewed_people} 人；多 ID 人物 ${fragments.people_with_multiple_track_ids} 人，额外 ID ${fragments.extra_track_ids} → 当前处理范围 ${fragments.retained_extra_track_ids} 个。未标注、混人、未检出的人不在碎片统计内。\n归并 F1（仅已完整复核、标注身份的纯轨迹对）：${metrics.identity_pairs?.f1 == null ? "尚无有效分母" : (metrics.identity_pairs.f1 * 100).toFixed(1) + "%"}，错并 ${metrics.identity_pairs?.fp ?? 0} 对，漏并 ${metrics.identity_pairs?.fn ?? 0} 对。人工修正后使用同一标签复测不算独立准确率。`;
}

async function renderReview() {
  const projectId = state.project.project_id, videoId = state.videoId;
  const request = ++state.reviewRequest;
  if (!state.reviewData) {
    $("#tabContent").innerHTML = emptyState("正在加载复核记录", "先看同一条轨迹的不同时间，再判断是否始终同人。");
  }
    try {
      const query = new URLSearchParams({offset:state.reviewOffset, limit:6, status:state.reviewFilter, q:state.reviewSearch});
      const data = await api(`/api/projects/${projectId}/videos/${videoId}/track-review?${query}`);
      if (request !== state.reviewRequest || projectId !== state.project?.project_id || videoId !== state.videoId) return;
      if (state.reviewDirty && state.reviewData?.index.fingerprint !== data.index.fingerprint) {
        toast('原始轨迹已变化', '当前草稿未覆盖新结果。请先导出记录或放弃草稿后重新加载。', 'error'); return;
      }
      if (data.pagination.total && state.reviewOffset >= data.pagination.total) { state.reviewOffset = Math.floor((data.pagination.total - 1) / 6) * 6; return renderReview(); }
      state.reviewData = data;
      if (!state.reviewDirty) state.reviewBaseRevision = data.revision;
    } catch (error) {
      if (request === state.reviewRequest && state.activeTab === "review") {
        $("#tabContent").innerHTML = emptyState("复核数据读取失败", `${error.message}。若提示质量报告不覆盖原始轨迹，请导入未过滤的原始基线，不能用旧过滤副本冒充完整评估。点击右上角查看实际索引和资源目录。`) + '<button class="control-button" id="retryReview">重试加载</button>';
        $('#retryReview').addEventListener('click', () => { state.reviewData = null; renderReview(); });
      }
      return;
    }
  if (state.activeTab !== "review") return;
  const data = state.reviewData, byId = Object.fromEntries(data.review.tracks.map((row) => [row.raw_track_id, row]));
  for (const [tid, row] of state.reviewDrafts) byId[tid] = row;
  const options = [["unreviewed", "未复核"], ["pure", "同一人"], ["mixed", "混人 / 中途换人"], ["non_player", "非球员 / 误检"], ["uncertain", "看不清 / 不确定"]];
  $("#tabContent").innerHTML = `<div class="review-toolbar"><div><h3>逐页复核 · 原始轨迹始终保留</h3><p>点“播放区间”查看整条轨迹。只看几张图请保持“仅抽样”。同人轨迹填写相同队伍＋号码；混人轨迹不要整条归给某个号码。分页和筛选不会丢失本视频草稿，也不会自动判断任何轨迹。</p></div><button class="control-button" id="saveReviewButton" ${state.savingReview ? 'disabled' : ''}>${state.reviewDirty ? '保存复核（未保存）' : '保存复核'}</button><button class="control-button" id="saveNextReviewButton" ${state.savingReview ? 'disabled' : ''}>保存并下一条视频 →</button><button class="ghost-button" id="prepareEvidence">生成 / 更新时序截图（CPU）</button><a class="ghost-button ${state.reviewDirty ? 'hidden' : ''}" id="mergeReviewDownload" href="/api/projects/${projectId}/videos/${videoId}/merge-review">导出确认归并 ↓</a></div>
    <div class="evaluation-note">截图状态：${escapeHtml(data.evidence_job?.status || (data.evidence_available ? '已有证据' : '尚未生成'))}。${escapeHtml(data.evidence_job?.error || '')} 无截图仍可播放原始轨迹；生成截图不重新跑模型，也不改变预测。</div>
    ${data.stale_review ? '<div class="evaluation-note error-text">原轨迹已变化，旧复核未计入。请重新核查并保存。</div>' : ""}
    <details class="review-summary-details"><summary>查看本视频复核统计及分母定义</summary><div class="evaluation-note" id="reviewMetrics">${escapeHtml(reviewSummary(data.metrics))}</div></details>
    <div class="review-controls"><div class="queue-controls"><label>轨迹范围<select id="reviewFilter">${[['retained','质量保留轨迹'],['remaining','尚未完整判断'],['all','全部原始轨迹'],['filtered','已过滤碎片'],['risk','自动告警 / 已发现混人'],['uncertain','看不清 / 不确定'],['full','完整判断已完成']].map(([value,label]) => `<option value="${value}" ${state.reviewFilter === value ? 'selected' : ''}>${label}</option>`).join('')}</select></label><label class="search-field">查找轨迹 / 身份<input id="reviewSearch" type="search" maxlength="120" value="${escapeHtml(state.reviewSearch)}" placeholder="T12 / 白队#15 / 备注" /></label><span class="queue-caveat">完整判断 ${data.metrics.scopes.full_track.raw.assessed_tracks}/${data.pagination.raw_total} · 当前筛选 ${data.pagination.total}</span></div><div class="draft-note" id="reviewDraftCount">${state.reviewDrafts.size ? `${state.reviewDrafts.size} 条修改尚未保存（包含其他页）` : '已保存的数据不会被当前分页覆盖'}</div><div class="pagination" id="reviewPagination"></div></div>
    <div class="review-grid">${data.index.tracks.map((track) => {
      const row = byId[track.raw_track_id] || {verdict: "unreviewed", scope: "sampled"};
      return `<article class="review-card" data-track="${track.raw_track_id}"><div class="review-heading"><strong>T${track.raw_track_id}</strong><span>${track.retained ? "保留" : "已过滤"} · ${track.observation_count} 次观测 · ${formatTime(track.start)} → ${formatTime(track.end)}</span><button class="ghost-button review-play" data-time="${track.start}" data-end="${track.end}">播放区间 ↗</button></div><p class="archive-warning">自动告警：${escapeHtml(track.quality_reasons.join(" / ") || "无告警不等于正确")}</p>
        <div class="review-samples">${track.samples.map((sample) => `<figure><a href="${escapeHtml(sample.context_url)}" target="_blank" rel="noopener"><img src="${escapeHtml(sample.crop_url)}" alt="T${track.raw_track_id} 帧 ${sample.frame_idx}" loading="lazy" /></a><figcaption>${formatTime(sample.timestamp_s)} · f${sample.frame_idx}</figcaption></figure>`).join("") || '<p class="archive-warning">未生成时序截图，仍可播放原视频复核。运行 cli.prepare_track_review 补充证据。</p>'}</div>
        <div class="review-fields"><label>轨迹是否同人<select data-field="verdict">${options.map(([value, label]) => `<option value="${value}" ${row.verdict === value ? "selected" : ""}>${label}</option>`).join("")}</select></label><label>检查范围<select data-field="scope"><option value="sampled" ${row.scope === "sampled" ? "selected" : ""}>仅抽样</option><option value="full_track" ${row.scope === "full_track" ? "selected" : ""}>已检查完整轨迹区间</option></select></label><label>片段内真人标签<input data-field="identity_label" maxlength="120" value="${escapeHtml(row.identity_label || "")}" placeholder="如：白队#15" ${row.verdict !== "pure" ? "disabled" : ""} /></label><label>问题备注<input data-field="note" maxlength="1000" value="${escapeHtml(row.note || "")}" placeholder="如：6.3秒换成另一位球员" /></label></div></article>`;
    }).join("")}</div>`;
  $$(".review-card input, .review-card select").forEach((element) => element.addEventListener("input", () => {
    state.reviewDirty = true;
    if (!state.reviewDrafts.size) state.reviewBaseRevision = data.revision;
    $("#saveReviewButton").textContent = "保存复核（未保存）";
    $("#mergeReviewDownload").classList.add("hidden");
    if (element.dataset.field === "verdict") {
      const label = element.closest(".review-card").querySelector('[data-field="identity_label"]');
      label.disabled = element.value !== "pure";
      if (label.disabled) label.value = "";
    }
    const card = element.closest('.review-card'), value = field => card.querySelector(`[data-field="${field}"]`).value;
    const tid = Number(card.dataset.track);
    state.reviewDrafts.set(tid, {raw_track_id:tid, verdict:value('verdict'), scope:value('scope'), identity_label:value('identity_label') || null, note:value('note')});
    $('#reviewDraftCount').textContent = `${state.reviewDrafts.size} 条修改尚未保存（包含其他页）`;
  }));
  $$(".review-play").forEach((button) => button.addEventListener("click", () => playReviewTrack(button)));
  $("#saveReviewButton").addEventListener("click", saveTrackReview);
  $('#saveNextReviewButton').addEventListener('click', async () => { if (await saveTrackReview()) await nextReviewVideo(); });
  $('#prepareEvidence').addEventListener('click', prepareReviewEvidence);
  $('#reviewFilter').addEventListener('change', event => { state.reviewFilter = event.target.value; state.reviewOffset = 0; rememberPosition(); renderReview(); });
  $('#reviewSearch').addEventListener('input', debounce(event => { state.reviewSearch = event.target.value; state.reviewOffset = 0; renderReview(); }));
  pager('reviewPagination', data.pagination.offset, data.pagination.limit, data.pagination.total, offset => { state.reviewOffset = offset; renderReview(); });
  rememberPosition();
  syncReviewSaveLock();
}

function syncReviewSaveLock() {
  // A page response can rebuild controls during a save. Reapply the lock to
  // freshly rendered controls instead of only disabling the previous DOM.
  const controls = $$('.review-card input, .review-card select, #reviewFilter, #reviewSearch, #reviewPagination button, #saveReviewButton, #saveNextReviewButton, #prepareEvidence');
  for (const element of controls) {
    if (state.savingReview) {
      if (!element.hasAttribute('data-save-lock')) {
        element.dataset.saveLock = element.disabled ? 'disabled' : 'enabled';
      }
      element.disabled = true;
    } else if (element.hasAttribute('data-save-lock')) {
      element.disabled = element.dataset.saveLock === 'disabled';
      delete element.dataset.saveLock;
    }
  }
}

async function prepareReviewEvidence() {
  if (state.savingReview) return;
  if (state.reviewDirty) { toast('请先保存复核', '生成截图不会修改预测，但刷新会丢失尚未保存的表单。', 'error'); return; }
  const projectId = state.project.project_id, videoId = state.videoId;
  $('#prepareEvidence').disabled = true;
  try {
    await api(`/api/projects/${projectId}/videos/${videoId}/review-evidence`, {method: 'POST', body: '{}'});
    toast('截图任务已提交', '使用配置的复核 Python，在独立缓存中生成，不修改导入结果。');
    const poll = async () => {
      if (projectId !== state.project?.project_id || videoId !== state.videoId) return;
      try {
        const data = await api(`/api/projects/${projectId}/videos/${videoId}/track-review?limit=1`);
        if (projectId !== state.project?.project_id || videoId !== state.videoId) return;
        if (!state.reviewDirty) { state.reviewData = data; if (state.activeTab === 'review') renderReview(); }
        if (['queued', 'running'].includes(data.evidence_job?.status)) setTimeout(poll, 2000);
        else toast(data.evidence_job?.status === 'completed' ? '时序截图已生成' : '截图生成失败', data.evidence_job?.error || '', data.evidence_job?.status === 'completed' ? '' : 'error');
      } catch (error) { toast('截图状态读取失败', error.message, 'error'); }
    };
    setTimeout(poll, 1000);
  } catch (error) { toast('无法生成截图', error.message, 'error'); }
  finally { if (projectId === state.project?.project_id && videoId === state.videoId && $('#prepareEvidence')) $('#prepareEvidence').disabled = false; }
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
  if (state.savingReview) return false;
  if (!state.reviewDirty) return true;
  const projectId = state.project.project_id, videoId = state.videoId;
  const data = state.reviewData;
  const tracks = [...state.reviewDrafts.values()].map(row => ({...row}));
  const submitted = new Map(tracks.map(row => [row.raw_track_id, JSON.stringify(row)]));
  state.savingReview = true;
  syncReviewSaveLock();
  try {
    const saved = await api(`/api/projects/${projectId}/videos/${videoId}/track-review`, {method: "PATCH", body: JSON.stringify({expected_revision: state.reviewBaseRevision, review: {...data.review, tracks}})});
    if (projectId !== state.project?.project_id || videoId !== state.videoId) return;
    state.reviewData = null;
    // Never discard a draft which was not part of this exact request.
    for (const [tid, value] of submitted) {
      if (JSON.stringify(state.reviewDrafts.get(tid)) === value) state.reviewDrafts.delete(tid);
    }
    state.reviewBaseRevision = saved.revision;
    state.reviewDirty = state.reviewDrafts.size > 0;
    state.detail.artifacts.evaluation.track_review = saved.metrics;
    toast("复核已保存", "轨迹检查记录已保存；球员身份归属请到比赛球员库确认或拆出。 ");
    await renderReview();
    await loadReviewQueue();
    return true;
  } catch (error) { toast("保存失败 · 草稿仍保留", `${error.message}。如果是另一窗口已保存，请刷新并核对冲突，不要覆盖。`, "error"); return false; }
  finally {
    state.savingReview = false;
    if (projectId === state.project?.project_id && videoId === state.videoId && $("#saveReviewButton")) {
      syncReviewSaveLock();
      $("#saveReviewButton").disabled = false;
      $('#saveNextReviewButton').disabled = false;
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
  if (state.eventOffset >= filteredCount) state.eventOffset = Math.max(0, Math.floor((filteredCount - 1) / 60) * 60);
  let before = 0;
  const sections = people.map((person) => {
    const events = person.events.filter((event) => !state.eventFilter || event.event === state.eventFilter);
    if (!events.length) return "";
    const displayed = events.slice(Math.max(0, state.eventOffset - before), Math.max(0, state.eventOffset + 60 - before));
    before += events.length;
    if (!displayed.length) return '';
    const image = person.identity?.cover?.crop_url;
    return `<section class="interval-person">
      <div class="interval-person-heading">${image ? `<img class="event-avatar" src="${escapeHtml(image)}" alt="${escapeHtml(person.person_id)}" />` : ""}<div><strong>${escapeHtml(person.person_id)}</strong><small>${events.length} 个事件区间 · 原轨迹 ${escapeHtml(person.raw_track_ids.join(", "))}</small></div></div>
      ${displayed.map((event) => `<button class="interval-row" data-start="${Number(event.start)}" data-end="${Number(event.end)}" title="跳转到 ${Number(event.start).toFixed(2)} 秒">
        <div><strong>${escapeHtml(event.event.replace(/^basketball_/, ""))}</strong><small>${formatTime(event.start)} → ${formatTime(event.end)}</small></div>
        <div class="timeline-track"><span style="left:${Math.min(100, Math.max(0, Number(event.start) / duration * 100))}%;width:${Math.min(100, Math.max(0.5, (Number(event.end) - Number(event.start)) / duration * 100))}%"></span></div>
        <div class="interval-score"><strong>${Number(event.raw_score).toFixed(3)}</strong><small>${Number(event.support_count)} 个支持点 · ${(Number(event.end) - Number(event.start)).toFixed(2)}s</small></div><span>↗</span>
      </button>`).join("")}
    </section>`;
  }).join("");
  $("#tabContent").innerHTML = filters + (sections || emptyState("没有符合条件的事件", index.event_count ? "清除筛选后再查看。" : "聚合文件存在但没有事件；这不代表视频中没有真实动作。")) + '<div class="pagination" id="eventPagination"></div>';
  pager('eventPagination', state.eventOffset, 60, filteredCount, offset => { state.eventOffset = offset; renderEvents(); });
  ["person", "event"].forEach((kind) => $(`#${kind}Filter`).addEventListener("change", (event) => {
    state[`${kind}Filter`] = event.target.value;
    state.eventOffset = 0;
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
  const kinds = ["events", "tracks", "quality", "quality_observations", "raw_identities", "identities", "jersey_tracks", "jersey_people", "resolution", "actions", "linked_actions", "pairs", "samples", "sampling"];
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

async function renderCrossClipIndex() {
  if (!state.project) return;
  const projectId = state.project.project_id, request = ++state.materialRequest;
  try {
    const query = new URLSearchParams({offset:state.materialOffset,limit:18,view:state.materialView,q:state.materialSearch});
    const data = await api(`/api/projects/${projectId}/materials?${query}`);
    if (projectId !== state.project?.project_id || request !== state.materialRequest) return;
    if (data.total && state.materialOffset >= data.total) { state.materialOffset = Math.floor((data.total - 1) / 18) * 18; return renderCrossClipIndex(); }
    const entries = data.items;
  $("#crossClipGrid").innerHTML = entries
    .map((entry) => {
      const crop = entry.cover?.crop_path;
      const url = crop
        ? `/api/projects/${projectId}/videos/${entry.video_id}/artifacts/${entry.media_prefix}/${crop}`
        : null;
      const label = entry.identity_label || (entry.jersey?.number != null ? `号码候选 #${entry.jersey.number}` : '身份尚未确认');
      return `<button class="cross-card" data-video-id="${escapeHtml(entry.video_id)}" data-person="${escapeHtml(entry.person_id)}">
        ${url ? `<img src="${escapeHtml(url)}" alt="" loading="lazy" />` : `<div class="cross-placeholder">NO IMAGE</div>`}
        <div><strong>${escapeHtml(entry.person_id || `T${entry.raw_track_ids?.[0]}`)}</strong><span title="${escapeHtml(entry.filename)}">${escapeHtml(entry.filename)}</span><b>${escapeHtml(label)}</b><small>${entry.raw_track_ids?.length || 0} 条来源轨迹 · 本视频局部档案</small></div>
      </button>`;
    })
    .join("");
  if (!entries.length) $('#crossClipGrid').innerHTML = emptyState('尚无符合条件的人物素材', '检查身份模块结果或清除搜索；不需要事件识别才能归档人物素材。');
  pager('materialPagination', data.offset, data.limit, data.total, offset => { state.materialOffset = offset; renderCrossClipIndex(); });
  $$(".cross-card").forEach((card) =>
    card.addEventListener("click", async () => { await selectVideo(card.dataset.videoId, false); if (state.videoId !== card.dataset.videoId) return; state.identityView = state.materialView; state.identitySearch = card.dataset.person; state.identityOffset = 0; setTab('identity'); $('#inspector').scrollIntoView({behavior:'smooth',block:'start'}); }),
  );
  } catch (error) { if (projectId === state.project?.project_id) toast('素材索引加载失败', error.message, 'error'); }
}

function runPayload(target) {
  const maxFrames = $("#maxFrames").value.trim();
  return {
    target,
    force: $("#forceRun").checked,
    jersey_qwen: $('#jerseyQwen').checked,
    max_frames: maxFrames ? Number(maxFrames) : null,
    tracking_det_threshold: Number($("#trackingThreshold").value),
    identity_samples: Number($("#identitySamples").value),
    identity_min_det_score: Number($("#identityThreshold").value),
    player_match_distance: Number($("#playerMatchDistance").value),
    player_novelty_distance: Number($("#playerNoveltyDistance").value),
    prompt_mode: "none",
    action_threshold: Number($("#actionThreshold").value),
    action_min_det_score: 0.3,
  };
}

async function runCurrent(target) {
  if (state.savingReview) return;
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
  if (state.savingReview) return;
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
  if (state.savingReview) return;
  if (state.project?.read_only) {
    toast("当前比赛为 CLI 结果导入", "追加片段请在 CLI 处理后刷新比赛；网页上传需先新建比赛。", "error");
    return;
  }
  const videos = files.filter((file) => file.type.startsWith("video/") || /\.(mp4|mov|m4v|avi|mkv|webm)$/i.test(file.name));
  if (!videos.length) {
    toast("没有可上传的视频", "请选择常见视频格式。", "error");
    return;
  }
  if (!state.project) await createProject("篮球比赛");
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
    toast("上传完成", `${videos.length} 个片段已加入本场比赛。`);
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
      const project = await api(`/api/projects/${projectId}?include_people=false`);
      if (projectId !== state.project?.project_id) return;
      state.project = project;
      loadReviewQueue();
      renderCrossClipIndex();
      if (state.videoId) await refreshCurrentVideo();
      if (state.activeTab === "logs" && state.detail?.job_active) renderLogs();
    } catch (_) {}
  }, 2500);
}

bootstrap();
