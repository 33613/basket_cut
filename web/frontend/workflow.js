/* Inspect the real serial workflow; the map describes its two data branches. */
const workflowStages = [
  ['tracking', 'MOTIP', '人物追踪', 'tracking'],
  ['quality', 'QUALITY', '轨迹质量', 'tracking'],
  ['action', 'SLOWFAST', '动作识别', 'points'],
  ['aggregate', 'EVENTS', '事件区间', 'events'],
  ['identity', 'KPR', '身份特征', 'sampling'],
  ['resolution', 'ARCHIVE', '片段档案', 'identity'],
  ['jersey', 'QWEN VL', '号码证据', 'jersey'],
  ['players', 'PLAYER MEMORY', '更新球员库', 'players'],
  ['link', 'IDENTITY × EVENT', '事件身份关联', 'events'],
  ['render', 'VIDEO', '结果可视化', 'render'],
  ['review', 'REVIEW', '轨迹复核证据', 'review'],
];
const workflowStatus = {pending: '未运行', queued: '排队中', running: '处理中', completed: '已完成',
  failed: '失败', skipped: '已跳过'};
const playerStatus = {candidate: '候选身份', needs_review: '待定 / 证据不足',
  manual_grouping: '人工确认', non_player: '非球员'};
const jerseyStatus = {candidate_consensus: '多帧一致候选', conflict: '读数冲突',
  insufficient_evidence: '支持帧不足', unreadable: '不可读 / 未取得样本'};

function artifactHref(path, projectId = state.project.project_id, videoId = state.videoId) {
  return `/api/projects/${encodeURIComponent(projectId)}/videos/${encodeURIComponent(videoId)}/artifacts/`
    + path.split('/').map(encodeURIComponent).join('/');
}

function bindWorkflowLinks(container) {
  container.querySelectorAll('[data-workflow-stage]').forEach(button => button.addEventListener('click', () => {
    const stage = workflowStages.find(row => row[0] === button.dataset.workflowStage);
    if (!stage) return;
    if (stage[3] === 'players') openPersonLibrary();
    else if (stage[3] === 'render') { setVideoView('final', false); setTab('workflow'); }
    else setTab(stage[3]);
  }));
}

function renderWorkflowRail(stages) {
  const outputs = state.detail?.artifacts.workflow?.stages || {};
  $('#stageRail').innerHTML = workflowStages.map(([key, model, label], index) => {
    const status = stages[key]?.status || 'pending';
    const missing = status === 'completed' && outputs[key] && !outputs[key].primary_exists;
    return `<button type="button" class="stage ${escapeHtml(status)} ${missing ? 'missing-output' : ''}"
      data-stage="${key}" data-workflow-stage="${key}" title="打开${label}结果">
      <span>${String(index + 1).padStart(2, '0')}</span><div><small>${model}</small>
      <strong>${label}</strong><em>${missing ? '完成，但产物缺失' : workflowStatus[status] || escapeHtml(status)}</em></div></button>`;
  }).join('');
  bindWorkflowLinks($('#stageRail'));
}

function workflowCounts(artifacts) {
  const quality = artifacts.tracking.quality || {}, kpr = artifacts.workflow?.kpr_summary || {};
  const qwen = artifacts.identity.jersey_summary || {}, registration = artifacts.workflow?.registration;
  const amount = value => value == null ? '尚无记录' : String(value);
  return {
    tracking: `${amount(quality.raw_track_count ?? artifacts.tracking.summary?.track_count)} 条原始轨迹`,
    quality: `${amount(quality.retained_track_count)} 条保留 · ${amount(quality.tentative_track_count)} 条证据不足`,
    action: artifacts.workflow?.stages.action.primary_exists ? `${amount(artifacts.action.points?.event_count)} 个类别预测点` : '动作结果尚未生成',
    aggregate: artifacts.workflow?.stages.aggregate.primary_exists ? `${amount(artifacts.action.index?.event_count)} 个事件区间` : '事件区间尚未生成',
    identity: `${amount(kpr.sample_count)} 个质量样本 · ${amount(artifacts.identity.raw_count)} 份轨迹档案`,
    resolution: `${amount(artifacts.identity.resolved_count)} 份局部档案 · 保留原始轨迹对应关系`,
    jersey: qwen.backend ? `${amount(qwen.sample_count)} 张躯干图 · 实际推理 ${amount(qwen.inference_count)} 次 · 缓存 ${amount(qwen.cache_hit_count)} 次` : '号码证据尚未生成',
    players: registration ? `已登记至比赛 ${registration.match_id}` : '本片段尚未登记球员库',
    link: '为动作预测保留片段身份与来源轨迹', render: '原视频 / 轨迹 / 身份与动作叠加',
    review: '时序截图、原轨迹播放与人工记录',
  };
}

function renderWorkflow() {
  const {video, artifacts} = state.detail, counts = workflowCounts(artifacts);
  const summaries = artifacts.workflow?.stages || {};
  const node = key => {
    const stage = workflowStages.find(row => row[0] === key), status = video.stages[key]?.status || 'pending';
    return `<button class="workflow-node ${escapeHtml(status)}" data-workflow-stage="${key}"><small>${stage[1]} · ${workflowStatus[status] || escapeHtml(status)}</small><strong>${stage[2]}</strong><span>${escapeHtml(counts[key])}</span></button>`;
  };
  $('#tabContent').innerHTML = `<div class="workflow-heading"><div><h3>从片段到球员与事件</h3><p>点击节点检查对应结果。上方阶段栏按实际执行顺序排列，单张 GPU 上依次处理。</p></div><button class="ghost-button" id="workflowRefresh">刷新当前片段</button></div>
    <div class="workflow-map"><div class="workflow-shared">${node('tracking')}${node('quality')}</div>
    <div class="workflow-branches"><section><h4>动作支路</h4>${node('action')}${node('aggregate')}</section>
    <section><h4>身份支路</h4>${node('identity')}${node('resolution')}${node('jersey')}${node('players')}</section></div>
    <div class="workflow-shared workflow-final">${node('link')}${node('render')}${node('review')}</div></div>
    <section class="assignment-section"><h3>本片段 → 比赛球员 ID</h3><p>原始轨迹 T* → 片段档案 P* → 本场比赛球员 PL-*。这里读取球员库的当前归属。</p><div id="clipAssignments">正在读取球员登记…</div></section>
    <details class="workflow-files"><summary>检查各阶段产物与下载</summary><div class="workflow-output-grid">${workflowStages.map(([key, model, label]) => `<article><strong>${label}</strong><ul>${(summaries[key]?.files || []).map(file => `<li>${file.exists ? `<a href="${escapeHtml(artifactHref(file.path))}" target="_blank" rel="noopener">${escapeHtml(file.path)}</a><small>${formatBytes(file.size_bytes)}</small>` : `<span>${escapeHtml(file.path)}</span><small>未生成</small>`}</li>`).join('') || '<li>旧结果未记录产物清单</li>'}</ul></article>`).join('')}</div></details>`;
  bindWorkflowLinks($('#tabContent'));
  $('#workflowRefresh').addEventListener('click', refreshCurrentVideo);
  renderClipAssignments();
}

async function renderClipAssignments() {
  const projectId = state.project.project_id, videoId = state.videoId, request = ++state.assignmentRequest;
  try {
    const data = await api(`/api/projects/${projectId}/videos/${videoId}/player-assignments`);
    if (request !== state.assignmentRequest || projectId !== state.project?.project_id || videoId !== state.videoId || !$('#clipAssignments')) return;
    $('#clipAssignments').innerHTML = data.available ? (data.items.map(row => `<button class="assignment-row" data-player="${escapeHtml(row.global_person_id)}"><span>${escapeHtml((row.raw_track_ids || []).map(tid => `T${tid}`).join(', '))}</span><span>${escapeHtml(row.local_person_id)}</span><strong>${escapeHtml(row.identity_label || row.global_person_id)}<small>${escapeHtml(row.global_person_id)}</small></strong><span>${escapeHtml(playerStatus[row.status] || row.status)}<small>${row.gallery_eligible ? '已进入外观图库' : '未进入外观图库'}</small></span></button>`).join('') || emptyState('本片段没有可登记的身份样本', '轨迹与动作结果仍可分别检查。')) : emptyState('球员库尚未建立', '完成身份和球员登记阶段后，查看比赛级 ID。');
    $$('#clipAssignments [data-player]').forEach(button => button.addEventListener('click', async () => {
      await openPersonLibrary();
      if (projectId === state.project?.project_id && libraryState.available) selectLibraryPerson(button.dataset.player);
    }));
  } catch (error) {
    if (request === state.assignmentRequest && videoId === state.videoId && $('#clipAssignments')) $('#clipAssignments').innerHTML = emptyState('球员归属暂不可读', error.message);
  }
}

function samplingComponents(row) {
  const names = {sharpness: '清晰度', size: '尺寸', detection: '检测', completeness: '裁剪完整度',
    exposure: '曝光', stability: '框稳定性', overlap_fraction: '重叠占比'};
  return Object.entries(row.sampling_components || {}).map(([key, value]) =>
    `<span>${escapeHtml(names[key] || key)} <b>${Number(value).toFixed(2)}</b></span>`).join('');
}

function seekEvidence(timestamp) {
  if (!state.detail?.artifacts.media.source) { toast('缺少原视频', '无法跳转到采样帧。', 'error'); return; }
  setVideoView('source', false);
  const player = $('#resultVideo'), projectId = state.project.project_id, videoId = state.videoId;
  const seek = () => { if (projectId === state.project?.project_id && videoId === state.videoId) { player.currentTime = Number(timestamp); player.pause(); } };
  if (player.readyState >= 1) seek(); else player.addEventListener('loadedmetadata', seek, {once: true});
  player.scrollIntoView({behavior: 'smooth', block: 'center'});
}

async function renderIdentityEvidence(purpose) {
  const projectId = state.project.project_id, videoId = state.videoId, request = ++state.evidenceRequest;
  const tab = purpose === 'kpr' ? 'sampling' : 'jersey';
  const artifacts = state.detail.artifacts;
  const summary = purpose === 'kpr' ? artifacts.workflow?.kpr_summary : artifacts.identity.jersey_summary;
  const tracks = artifacts.tracking.quality_tracks || artifacts.tracking.summary?.tracks || [];
  const description = purpose === 'kpr' ? '人物图用于 KPR 特征提取；按质量和时间间隔选择，展示实际进入模型的样本。' : '躯干图用于本地 Qwen 读号；保留不可读、拒绝和冲突结果，号码候选辅助球员匹配。';
  $('#tabContent').innerHTML = `<div class="evidence-heading"><h3>${purpose === 'kpr' ? 'KPR 人物质量采样' : 'Qwen 多帧号码证据'}</h3><p>${description}</p></div>
    <div class="queue-controls"><label>原始轨迹<select id="evidenceTrack"><option value="">全部轨迹</option>${tracks.map(row => `<option value="${row.track_id}" ${String(row.track_id) === state.evidenceTrack ? 'selected' : ''}>T${row.track_id}</option>`).join('')}</select></label><span class="queue-caveat">${purpose === 'jersey' && summary ? `实际推理 ${summary.inference_count ?? '未记录'} 次 · 缓存 ${summary.cache_hit_count ?? '未记录'} 次 · 最少 ${summary.thresholds?.min_support ?? '未记录'} 个支持帧` : `采样策略：${escapeHtml(summary?.sampling_strategy || '旧结果未记录')}`}</span></div>
    <div id="evidenceDiagnostics"></div><div class="evidence-grid" id="identityEvidenceGrid">正在读取采样证据…</div><div class="pagination" id="evidencePagination"></div>`;
  $('#evidenceTrack').addEventListener('change', event => { state.evidenceTrack = event.target.value; state.evidenceOffset = 0; renderIdentityEvidence(purpose); });
  try {
    const query = new URLSearchParams({purpose, offset: state.evidenceOffset, limit: 24});
    if (state.evidenceTrack !== '') query.set('track_id', state.evidenceTrack);
    const data = await api(`/api/projects/${projectId}/videos/${videoId}/identity-evidence?${query}`);
    if (request !== state.evidenceRequest || projectId !== state.project?.project_id || videoId !== state.videoId || state.activeTab !== tab) return;
    if (data.total && state.evidenceOffset >= data.total) { state.evidenceOffset = 0; return renderIdentityEvidence(purpose); }
    $('#evidenceDiagnostics').innerHTML = data.tracks.map(row => `<span class="evidence-track-state"><strong>T${escapeHtml(row.raw_track_id ?? row.track_id)}</strong> ${escapeHtml(purpose === 'jersey' ? jerseyStatus[row.status] || row.status : row.status === 'excluded' ? '未取得合格样本' : '已采样')}${row.number != null ? ` · #${escapeHtml(row.number)}` : ''}${row.exclusion_reasons?.length ? ` · ${escapeHtml(row.exclusion_reasons.join(' / '))}` : ''}</span>`).join('');
    $('#identityEvidenceGrid').innerHTML = data.items.map((row, index) => {
      const tid = row.raw_track_id ?? row.track_id;
      const image = row.crop_path ? artifactHref(`${data.media_prefix}/${row.crop_path}`, projectId, videoId) : null;
      const context = row.context_path ? artifactHref(`${data.media_prefix}/${row.context_path}`, projectId, videoId) : image;
      const rejected = row.rejection_reasons || [];
      return `<article class="evidence-card">${image ? `<a href="${escapeHtml(context)}" target="_blank" rel="noopener"><img src="${escapeHtml(image)}" alt="T${escapeHtml(tid)} 帧 ${escapeHtml(row.frame_idx)} ${purpose === 'kpr' ? '人物图' : '躯干图'}" loading="lazy" /></a>` : '<div class="no-image">无采样图片</div>'}<div class="evidence-card-body"><div><strong>T${escapeHtml(tid)} · 帧 ${escapeHtml(row.frame_idx)}</strong><button class="ghost-button" data-evidence-seek="${index}">${formatTime(row.timestamp_s)} ↗</button></div>
        <p>质量 ${row.sampling_quality == null ? '旧结果未记录' : Number(row.sampling_quality).toFixed(3)}${purpose === 'kpr' ? ` · 可见部位 ${row.visible_parts ?? '未记录'}` : ''}</p><div class="quality-components">${samplingComponents(row)}</div>
        ${purpose === 'jersey' ? `<p class="jersey-badge ${rejected.length ? 'error-text' : ''}">${row.number == null ? '未读取号码' : `本帧读数 #${escapeHtml(row.number)}`} · ${escapeHtml(rejected.join(' / ') || '候选支持帧')}</p><details><summary>Qwen 原始回答</summary><pre>${escapeHtml(row.raw_response || '未记录')}</pre></details>` : `<p>KPR 距离 ${row.prototype_distance == null ? '未记录' : Number(row.prototype_distance).toFixed(3)} · 用于外观比对</p>`}</div></article>`;
    }).join('') || emptyState(data.available ? '当前范围没有合格采样图' : '该阶段尚无证据文件', data.available ? '质量不足时保留空结果；可回到轨迹页查看来源。' : '完成对应处理阶段后再查看。');
    $$('[data-evidence-seek]').forEach(button => button.addEventListener('click', () => seekEvidence(data.items[Number(button.dataset.evidenceSeek)].timestamp_s)));
    pager('evidencePagination', data.offset, data.limit, data.total, offset => { state.evidenceOffset = offset; renderIdentityEvidence(purpose); });
  } catch (error) {
    if (request === state.evidenceRequest && state.activeTab === tab && $('#identityEvidenceGrid')) $('#identityEvidenceGrid').innerHTML = emptyState('证据读取失败', error.message);
  }
}
