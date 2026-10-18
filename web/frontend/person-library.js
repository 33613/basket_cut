/* Match library browsing is separate from clip-local inspection. */
const libraryState = {offset: 0, q: '', status: 'active', person: null, event: '', eventOffset: 0,
  eventQuery: '', minScore: 0.2, revision: null, request: 0, eventRequest: 0, personRequest: 0,
  editing: false, available: false, opened: false};

function libraryJersey(jersey, showEvidence = false) {
  if (!jersey) return '';
  const label = jersey.number != null ? `Qwen 号码候选 #${jersey.number}` :
    jersey.status === 'conflict' ? '号码读数冲突' :
    jersey.candidate_numbers?.length === 1 ? `弱证据号码候选 #${jersey.candidate_numbers[0]} · 未确认` : '号码未知 / 证据不足';
  const readings = [...(jersey.readings || []), ...(jersey.tracks || []).flatMap(t => t.readings || [])];
  return `<p class="jersey-badge">${escapeHtml(label)}</p>${showEvidence && readings.length ?
    `<details><summary>读号中间结果（${readings.length} 帧）</summary><div class="number-readings">${readings.map(r =>
      `<span>${r.crop_url ? `<a href="${escapeHtml(r.crop_url)}" target="_blank" rel="noopener"><img src="${escapeHtml(r.crop_url)}" alt="Qwen 身份证据裁剪" loading="lazy" /></a>` : ''}<small>${formatTime(r.source_timestamp_s ?? r.timestamp_s)} · 帧 ${escapeHtml(r.frame_idx)} · ${(r.text ?? r.number) == null ? '不可读' : '#' + escapeHtml(r.text ?? r.number)}<br />质量 ${(r.sampling_quality ?? r.quality_score) == null ? '未记录' : Number(r.sampling_quality ?? r.quality_score).toFixed(3)}${r.usable == null ? '' : r.usable ? ' · 清晰无遮挡球员' : ' · 未通过人物检查'}<br />${escapeHtml((r.rejection_reasons || []).join(' / ') || '候选支持帧')}</small><details><summary>原始回答</summary><pre>${escapeHtml(r.raw_response || '未记录')}</pre></details></span>`).join('')}</div></details>` : ''}`;
}

function bindPersonLibrary() {
  $('#openPersonLibraryButton').addEventListener('click', openPersonLibrary);
  $('#closePersonLibrary').addEventListener('click', closePersonLibrary);
  $('#refreshPersonLibrary').addEventListener('click', loadPersonLibrary);
  $('#reprocessPlayerMemory').addEventListener('click', () => editPersonLibrary('reprocess'));
  $('#rollbackPlayerMemory').addEventListener('click', () => { if (window.confirm('撤销最近一次图库更新，恢复身份归属和模板？原始轨迹与动作保留。')) editPersonLibrary('rollback_memory'); });
  $('#librarySearch').addEventListener('input', debounce(event => {
    libraryState.q = event.target.value; libraryState.offset = 0; loadPersonLibrary();
  }));
  $('#libraryStatus').addEventListener('change', event => {
    libraryState.status = event.target.value; libraryState.offset = 0; loadPersonLibrary();
  });
  $('#libraryEventType').addEventListener('change', event => {
    libraryState.event = event.target.value; libraryState.eventOffset = 0; loadLibraryEvents();
  });
  $('#libraryMinScore').addEventListener('change', event => {
    const value = Number(event.target.value);
    if (!Number.isFinite(value) || value < 0 || value > 1) { toast('分数应在0–1之间', '', 'error'); return; }
    libraryState.minScore = value; libraryState.eventOffset = 0; loadLibraryEvents();
  });
  $('#libraryEventSearch').addEventListener('input', debounce(event => {
    libraryState.eventQuery = event.target.value; libraryState.eventOffset = 0; loadLibraryEvents();
  }));
  $('#libraryAllPeople').addEventListener('click', () => {
    libraryState.person = null; libraryState.eventOffset = 0;
    ++libraryState.personRequest; $('#libraryPersonDetail').classList.add('hidden');
    $('#librarySelectedPerson').textContent = '全部人物'; loadLibraryEvents();
  });

}

function resetPersonLibrary() {
  ++libraryState.request; ++libraryState.eventRequest; ++libraryState.personRequest;
  Object.assign(libraryState, {offset: 0, q: '', status: 'active', person: null, event: '', eventOffset: 0,
    eventQuery: '', revision: null, available: false, opened: false});
  $('#personLibrarySection').classList.add('hidden'); $('#libraryPersonDetail').classList.add('hidden');
  $('#librarySearch').value = ''; $('#libraryStatus').value = 'active'; $('#libraryEventSearch').value = '';
  $('#rollbackPlayerMemory').disabled = true;
}

async function openPersonLibrary() {
  if (!state.project || libraryState.editing) return;
  libraryState.opened = true;
  $('#personLibrarySection').classList.remove('hidden'); $('#workspace').classList.add('hidden');
  $('#uploadSection').classList.add('hidden');
  $('#personLibrarySection').scrollIntoView({behavior: 'smooth', block: 'start'});
  await loadPersonLibrary();
  await loadMatchReport();
}

async function loadMatchReport() {
  const projectId = state.project?.project_id;
  if (!projectId) return;
  try {
    const r = await api(`/api/projects/${projectId}/match-report`);
    if (projectId !== state.project?.project_id) return;
    const s = r.segmentation, l = r.library, e = r.possession_evaluation;
    const metric = (label, value) => `<p><span>${escapeHtml(label)}</span><strong>${escapeHtml(value ?? '未记录')}</strong></p>`;
    $('#matchEvaluationSidebar').innerHTML = `<h3>流程统计</h3>${metric('回合候选', s.coarse_candidates)}${metric('已导出片段', s.exported_clips)}${metric('待复核边界', s.unresolved_boundaries)}${metric('完成处理', `${r.completed_clips}/${r.planned_processing_clips}`)}${metric('失败片段', r.failed_clips)}${metric('初始化短窗口', r.bootstrap.window_count ?? r.bootstrap.windows_completed ?? 0)}${metric('清晰轨迹 / 保留图片', `${r.bootstrap.tracklet_count ?? 0} / ${r.bootstrap.accepted_clear_samples ?? 0}`)}${metric('未入库图片', r.bootstrap.discarded_samples ?? 0)}${metric('比赛候选身份', l.global_person_count)}${metric('待定观察', l.held_archive_count)}${metric('未关联事件', l.unmapped_event_count)}<h3>回合切分评估</h3>${e.accuracy_evaluated ? e.ranges.map(m => `<section><strong>${escapeHtml(m.name)} · ${m.stage === 'coarse' ? '粗看候选' : '实际切点'}</strong>${metric('TP / FP / FN', `${m.TP} / ${m.FP} / ${m.FN}`)}${metric('Precision', m.precision == null ? '无预测' : (100*m.precision).toFixed(1)+'%')}${metric('Recall', m.recall == null ? '无参考回合' : (100*m.recall).toFixed(1)+'%')}${metric('FP / 分钟', m.FP_per_minute?.toFixed(2))}${metric('匹配误差 MAE（秒）', m.matched_midpoint_mae_s?.toFixed(2))}</section>`).join('') : '<p>待补齐人工边界标注，尚不能计算切分准确率。</p>'}<p class="scope-warning">人物数量、成功运行和事件数量都是流程统计。当前视频没有跟踪、身份和动作真值，相关准确率未评估。</p><details><summary>各片段阶段耗时</summary>${r.clips.map(c => `<section><strong>${escapeHtml(c.clip_id)}</strong>${Object.entries(c.stage_metrics || {}).map(([name,m]) => metric(name, `${m.elapsed_s ?? '未知'} 秒${m.reused ? '（缓存）' : ''}`)).join('')}</section>`).join('')}</details>`;
  } catch (error) {
    if (projectId === state.project?.project_id) $('#matchEvaluationSidebar').textContent = `统计暂不可用：${error.message}`;
  }
}

async function closePersonLibrary() {
  libraryState.opened = false;
  $('#personLibrarySection').classList.add('hidden');
  $('#workspace').classList.toggle('hidden', !state.project?.videos.length);
  $('#uploadSection').classList.toggle('hidden', Boolean(state.project?.read_only));
  if (!state.videoId && state.project?.videos.length) await selectVideo(state.project.videos[0].video_id, false);
}

async function loadPersonLibrary() {
  if (!state.project) return;
  const projectId = state.project.project_id, request = ++libraryState.request;
  $('#personLibraryWarning').textContent = '正在读取比赛球员库…';
  try {
    const query = new URLSearchParams({offset: libraryState.offset, limit: 12, q: libraryState.q, status: libraryState.status});
    const data = await api(`/api/projects/${projectId}/person-library?${query}`);
    if (projectId !== state.project?.project_id || request !== libraryState.request) return;
    libraryState.available = data.available; libraryState.revision = data.revision || null;
    if (!data.available) {
      $('#personLibrarySummary').innerHTML = ''; $('#personLibraryGrid').innerHTML = emptyState('球员库尚未建立', data.message);
      $('#personLibraryWarning').textContent = '每个片段完成身份处理后更新本场比赛球员库。';
      $('#personLibraryPagination').innerHTML = ''; $('#libraryEvents').innerHTML = ''; $('#libraryPersonDetail').classList.add('hidden'); return;
    }
    if (data.total && libraryState.offset >= data.total) {
      libraryState.offset = Math.floor((data.total - 1) / 12) * 12; return loadPersonLibrary();
    }
    const s = data.summary;
    loadMatchReport();
    const statuses = s.player_status_counts || {};
    $('#rollbackPlayerMemory').disabled = !s.memory_rollback_available;
    $('#personLibrarySummary').innerHTML = [[s.expected_clips, '已登记片段', `本场比赛：${s.match_id}`],
      [s.candidate_person_count ?? s.global_person_count, '已建候选 / 确认球员', `候选 ${statuses.candidate || 0} · 人工确认 ${statuses.manual_grouping || 0} · 非球员 ${statuses.non_player || 0}`],
      [statuses.needs_review || 0, '待定身份', `模板 ${s.gallery_template_count ?? s.gallery_observation_count ?? '未记录'} · 淘汰 ${s.memory_status_counts?.rejected ?? 0} · 重复 ${s.memory_status_counts?.duplicate ?? 0}`],
      [s.event_count, '可检索事件区间', `暂未关联身份 ${s.unmapped_event_count ?? 0}`]].map(([value, label, note]) =>
      `<div class="batch-stat"><strong>${escapeHtml(value)}</strong><span>${label}</span><small>${escapeHtml(note)}</small></div>`).join('');
    $('#personLibraryWarning').textContent = s.memory_policy ? `本场比赛 ${s.match_id} · 外观严格距离 ${s.memory_policy.strong_distance} · 有一致号码候选时距离 ${s.memory_policy.supported_distance} · 每人最多 ${s.memory_policy.gallery_limit} 个模板。新增观察对照更新前的图库匹配，统一提交；待定观察保留，候选身份仍需核验。` : `本场比赛 ${s.match_id} · 匹配距离 ${s.settings.max_distance ?? '未记录'} · 新身份距离 ${s.settings.novelty_distance ?? '未记录'}。${s.jersey_constraints ? 'KPR 外观与 Qwen 号码证据联合匹配。' : 'KPR 外观匹配。'}图库持续追加合格观察，待定观察暂不进入图库。`;
    const labels = {candidate: '候选身份', needs_review: '待检查 / 未自动匹配', manual_grouping: '人工确认', non_player: '非球员'};
    $('#personLibraryGrid').innerHTML = data.items.map(p => `<button class="library-card ${libraryState.person === p.global_person_id ? 'selected' : ''}" data-library-person="${escapeHtml(p.global_person_id)}">
      <div class="library-cover">${p.cover?.crop_url ? `<img src="${escapeHtml(p.cover.crop_url)}" alt="候选人物代表图" loading="lazy" />` : '<span>暂无代表图</span>'}<span class="library-status">${labels[p.status] || escapeHtml(p.status)}</span></div>
      <div class="library-card-copy"><strong>${escapeHtml(p.identity_label || p.global_person_id)}</strong><small>${escapeHtml(p.global_person_id)}</small>${libraryJersey(p.jersey)}<small>${escapeHtml((p.team_ids || []).join(' / '))}</small><p>${p.clip_count} 个片段 · ${p.bootstrap_tracklet_count || 0} 条初始化轨迹 · 模板 ${p.gallery_template_count ?? 0} · ${p.event_count} 个事件</p>
      <div class="library-preview-row">${p.previews.filter(c => c.crop_url).map(c => `<img src="${escapeHtml(c.crop_url)}" alt="来源外观" loading="lazy" />`).join('')}</div><span class="library-card-link">核查来源 · 查看人物事件 ↗</span></div></button>`).join('') || emptyState('没有符合条件的人物', '清除搜索或切换筛选条件。');
    $$('#personLibraryGrid [data-library-person]').forEach(button => button.addEventListener('click', () => selectLibraryPerson(button.dataset.libraryPerson)));
    pager('personLibraryPagination', data.offset, data.limit, data.total, offset => { libraryState.offset = offset; loadPersonLibrary(); });
    const select = $('#libraryEventType');
    select.innerHTML = '<option value="">全部事件</option>' + Object.keys(s.event_types).sort().map(label => `<option value="${escapeHtml(label)}">${escapeHtml(label)} (${s.event_types[label]})</option>`).join('');
    select.value = libraryState.event;
    if (select.value !== libraryState.event) libraryState.event = '';
    await loadLibraryEvents();
  } catch (error) { if (projectId === state.project?.project_id) { $('#personLibraryWarning').textContent = error.message; toast('人物库读取失败', error.message, 'error'); } }
}

async function selectLibraryPerson(personId) {
  const projectId = state.project?.project_id, request = ++libraryState.personRequest;
  if (!projectId || libraryState.editing) return;
  try {
    const p = await api(`/api/projects/${projectId}/person-library/people/${encodeURIComponent(personId)}`);
    if (projectId !== state.project?.project_id || request !== libraryState.personRequest) return;
    libraryState.person = personId; libraryState.revision = p.revision || null; libraryState.eventOffset = 0;
    $$('#personLibraryGrid [data-library-person]').forEach(button =>
      button.classList.toggle('selected', button.dataset.libraryPerson === personId));
    $('#librarySelectedPerson').textContent = p.identity_label || personId;
    const detail = $('#libraryPersonDetail'); detail.classList.remove('hidden');
    detail.innerHTML = `<div class="panel-title"><div><span class="eyebrow">PLAYER IDENTITY & EVIDENCE</span><h3>${escapeHtml(p.identity_label || personId)}</h3><p>${escapeHtml(personId)} · ${escapeHtml(playerStatus[p.status] || p.status)} · ${p.clip_count} 个片段 · ${p.event_count} 个事件</p></div></div>
      <div class="library-label-editor"><label>球队 / 号码 / 姓名标签<input id="libraryIdentityLabel" maxlength="120" value="${escapeHtml(p.identity_label || '')}" placeholder="如：白队 #23 / 球员姓名" /></label><button class="secondary-button" id="labelLibraryPerson">保存标签 / 合并同标签</button><button class="secondary-button" id="confirmLibraryPerson">确认当前身份分组</button><button class="ghost-button" id="excludeLibraryPerson">标记为非球员</button></div><p class="scope-warning">同标签身份会人工合并并保留 ID 别名；同时出现的不同轨迹不能合并。可拆出错误成员，混人轨迹回到轨迹页检查。</p>
      <div class="library-members">${p.members.map(m => `<article class="library-member">
        ${m.cover.crop_url ? `<img src="${escapeHtml(m.cover.crop_url)}" alt="局部档案来源" loading="lazy" />` : ''}
        <div><strong>${escapeHtml(m.filename)}</strong><small>${m.evidence_kind === 'bootstrap_photo' ? `历史初始化照片 · 原视频 ${formatTime(m.timestamp_s)}` : m.evidence_kind === 'bootstrap_tracklet' ? `初始化短窗口 ${escapeHtml(m.clip_name)} · 连续轨迹 ${escapeHtml(m.raw_track_ids.join(', '))}` : `片段 ${escapeHtml(m.clip_id || m.clip_name)} · 原轨迹 ${escapeHtml(m.raw_track_ids.join(', '))}`} · ${m.sample_count} 个 KPR 样本</small>${m.source_video_id && m.evidence_kind !== 'bootstrap_photo' ? `<small>原视频 ${escapeHtml(m.source_video_id)} · ${formatTime(m.source_clip_start_s)} → ${formatTime(m.source_clip_end_s)}</small>` : ''}<p class="member-decision">${escapeHtml({matched: '匹配已有球员', new_candidate: '建立新候选身份', pending: '待定观察', memory_matched: '图库候选关联', accepted: '候选关联', rejected: '已淘汰', duplicate: '重复观察'}[m.decision] || m.decision || '旧结果未记录匹配决策')} · ${m.gallery_eligible ? '已进入外观图库' : '未进入外观图库'}</p><p>${escapeHtml((m.hold_reasons || []).join(' / ') || '查看来源外观确认身份')}</p><p>${escapeHtml(m.team_id || '队伍待定')} · ${escapeHtml(m.team_evidence?.kit_color || '颜色未知')} · 多帧队伍证据 ${m.team_evidence?.support_frames || 0}</p>${libraryJersey(m.jersey, true)}${m.memory ? `<p>质量 ${Number(m.memory.quality || 0).toFixed(3)} · ${escapeHtml({accepted:'候选关联',pending:'待核验',rejected:'已淘汰',duplicate:'重复观察'}[m.memory.status] || m.memory.status)} · 已选模板 ${(m.memory.templates || []).length}</p>${m.memory.association_only ? '<p>已关联身份；图片质量不足，未加入外观模板。</p>' : ''}${m.memory.match_dependencies?.observations?.length ? `<p>关联依据：${escapeHtml(m.memory.match_dependencies.basis === 'team_number' ? '多帧队伍和号码' : '历史外观模板')} · 依赖 ${m.memory.match_dependencies.observations.length} 条历史观察 · 图库版本 ${escapeHtml(m.memory.matched_against_revision || '未记录')}</p>` : ''}${m.memory.dependency_invalidated ? '<p>依赖证据已撤销，身份关联等待重新核验。</p>' : ''}<div class="library-preview-row">${(m.memory.templates || []).filter(t => t.crop_url).map(t => `<a href="${escapeHtml(t.crop_url)}" target="_blank" rel="noopener" title="查看原尺寸图片"><img src="${escapeHtml(t.crop_url)}" alt="${escapeHtml(t.view || 'unknown')} 朝向模板" loading="lazy" /><small>${escapeHtml(({front:'正面',back:'背面',left:'左侧',right:'右侧',unknown:'朝向待定'})[t.view] || '朝向待定')}</small></a>`).join('')}</div>` : ''}
        <div class="library-preview-row">${m.exemplars.filter(s => s.crop_url).map(s => `<img src="${escapeHtml(s.crop_url)}" alt="时序外观证据" loading="lazy" />`).join('')}</div>
        <button class="ghost-button" data-memory-reject="${escapeHtml(m.node_id)}" data-operation="${m.memory?.manual_rejected ? 'reinstate_observation' : 'reject_observation'}">${m.memory?.manual_rejected ? '重新检查此观察' : '从图库淘汰此观察'}</button>${m.window_url ? `<a class="ghost-button" href="${escapeHtml(m.window_url)}">下载原分辨率初始化短窗口</a>` : ''}${m.web_video_id ? `<button class="ghost-button" data-library-open="${escapeHtml(m.node_id)}">检查原片段 ↗</button>` : ''}<button class="ghost-button" data-library-detach="${escapeHtml(m.node_id)}" data-operation="${m.manually_detached ? 'restore' : 'detach'}">${m.manually_detached ? '撤销拆出' : '从人物组拆出'}</button></div></article>`).join('')}</div>
      <details class="library-pair-evidence"><summary>最近匹配证据（${p.pairs.length} 对；距离不是置信度）</summary>${p.pairs.map(pair => `<div class="library-pair"><div>${[pair.left, pair.right].map(m => `<span>${m.cover.crop_url ? `<img src="${escapeHtml(m.cover.crop_url)}" alt="匹配来源" loading="lazy" />` : ''}${escapeHtml(m.filename)} / ${escapeHtml(m.local_person_id)}</span>`).join('')}</div><p>距离 ${pair.distance == null ? '无可比部位' : Number(pair.distance).toFixed(3)} · ${pair.same_global_person ? '当前同组' : '当前分开'} · ${escapeHtml(pair.auto_block_reasons.join(' / ') || '满足候选条件')}</p></div>`).join('') || '<p>没有可用匹配对。</p>'}</details>`;
    $('#confirmLibraryPerson').addEventListener('click', () => editPersonLibrary('confirm', {person_id: personId, label: p.identity_label}));
    $('#excludeLibraryPerson').addEventListener('click', () => { if (window.confirm('标记整组为非球员并停止用于自动匹配？')) editPersonLibrary('exclude', {person_id: personId}); });
    $('#labelLibraryPerson').addEventListener('click', () => {
      const label = $('#libraryIdentityLabel').value.trim();
      if (window.confirm(`保存标签“${label || '无标签'}”？同标签身份会合并。`)) editPersonLibrary('label_group', {person_id: personId, label});
    });
    detail.querySelectorAll('[data-library-open]').forEach(button => button.addEventListener('click', async () => {
      const m = p.members.find(n => n.node_id === button.dataset.libraryOpen);
      await selectVideo(m.web_video_id, false); if (state.videoId !== m.web_video_id) return;
      closePersonLibrary(); state.identityView = 'resolved'; state.identitySearch = m.local_person_id;
      state.identityOffset = 0; setTab('identity'); $('#inspector').scrollIntoView({behavior: 'smooth'});
    }));
    detail.querySelectorAll('[data-library-detach]').forEach(button => button.addEventListener('click', () => {
      if (window.confirm('修改这个局部档案的跨片段归组？原轨迹、片段内档案和事件预测不变。')) editPersonLibrary(button.dataset.operation, {node_id: button.dataset.libraryDetach});
    }));
    detail.querySelectorAll('[data-memory-reject]').forEach(button => button.addEventListener('click', () => editPersonLibrary(button.dataset.operation, {node_id: button.dataset.memoryReject})));
    await loadLibraryEvents(); detail.scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) { if (projectId === state.project?.project_id) toast('人物来源读取失败', error.message, 'error'); }
}

async function loadLibraryEvents() {
  if (!state.project || !libraryState.available) return;
  const projectId = state.project.project_id, request = ++libraryState.eventRequest;
  const query = new URLSearchParams({person_id: libraryState.person || '', event: libraryState.event,
    q: libraryState.eventQuery, min_score: libraryState.minScore, offset: libraryState.eventOffset, limit: 20});
  try {
    const data = await api(`/api/projects/${projectId}/person-library/events?${query}`);
    if (projectId !== state.project?.project_id || request !== libraryState.eventRequest) return;
    if (data.total && libraryState.eventOffset >= data.total) { libraryState.eventOffset = Math.floor((data.total - 1) / 20) * 20; return loadLibraryEvents(); }
    $('#libraryEvents').innerHTML = data.items.map((e, index) => `<article class="library-event-row"><button class="library-event" data-library-event="${index}"><div><strong>${escapeHtml(e.event)}</strong><span>${escapeHtml(e.filename)}</span><small>球员 ${escapeHtml(e.global_person_id)} · 片段 ${escapeHtml(e.clip_id || e.clip_name)} · 轨迹 ${escapeHtml((e.raw_track_ids || []).join(', '))}</small></div><div><strong>${formatTime(e.start)} → ${formatTime(e.end)}</strong>${e.source_start_s != null ? `<small>原视频 ${formatTime(e.source_start_s)} → ${formatTime(e.source_end_s)}</small>` : ''}<small>raw_score ${Number(e.raw_score).toFixed(3)} · 跳转片段 ↗</small></div></button>${e.global_person_id ? `<div class="event-export-actions"><button class="secondary-button" data-export-event="${index}" data-scope="event">下载动作 ±1秒</button><button class="ghost-button" data-export-event="${index}" data-scope="clip">下载对应片段</button></div>` : ''}</article>`).join('') || emptyState('没有符合条件的事件', '切换人物 / 事件类型或降低原始分数筛选。无事件不代表该人物没有做动作。');
    $('#libraryEvents').querySelectorAll('[data-library-event]').forEach(button => button.addEventListener('click', async () => {
      const e = data.items[Number(button.dataset.libraryEvent)];
      await selectVideo(e.web_video_id, false); if (state.videoId !== e.web_video_id) return;
      closePersonLibrary(); state.personFilter = e.local_person_id; state.eventFilter = e.event;
      setVideoView('source', false); setTab('events');
      const player = $('#resultVideo'), seek = () => { player.currentTime = e.start; player.play().catch(() => {}); };
      if (player.readyState >= 1) seek(); else player.addEventListener('loadedmetadata', seek, {once: true});
      $('#inspector').scrollIntoView({behavior: 'smooth', block: 'start'});
    }));
    $('#libraryEvents').querySelectorAll('[data-export-event]').forEach(button => button.addEventListener('click', async () => {
      const row = data.items[Number(button.dataset.exportEvent)];
      button.disabled = true;
      const label = button.textContent; button.textContent = '正在剪辑…';
      try {
        const result = await api(`/api/projects/${projectId}/person-library/export`, {method: 'POST',
          body: JSON.stringify({player_id: row.global_person_id, event_id: row.event_id,
            expected_revision: data.revision || libraryState.revision, scope: button.dataset.scope})});
        if (projectId !== state.project?.project_id) return;
        const link = document.createElement('a'); link.href = result.download_url; link.download = '';
        document.body.appendChild(link); link.click(); link.remove();
      } catch (error) { toast('剪辑失败', error.message, 'error'); }
      finally { button.disabled = false; button.textContent = label; }
    }));
    pager('libraryEventsPagination', data.offset, data.limit, data.total, offset => { libraryState.eventOffset = offset; loadLibraryEvents(); });
  } catch (error) { if (projectId === state.project?.project_id) toast('事件检索失败', error.message, 'error'); }
}

async function editPersonLibrary(operation, extra = {}) {
  if (!state.project || libraryState.editing) return;
  const projectId = state.project.project_id;
  libraryState.editing = true; $('#personLibrarySection').classList.add('library-busy');
  toast('正在更新人物库', '仅重新计算缓存特征的归组与索引，不运行GPU模型。');
  try {
    const result = await api(`/api/projects/${projectId}/person-library/review`, {method: 'POST',
      body: JSON.stringify({operation, expected_revision: libraryState.revision, ...extra})});
    if (projectId !== state.project?.project_id) return;
    libraryState.revision = result.revision; libraryState.person = null; libraryState.eventOffset = 0;
    ++libraryState.personRequest; $('#libraryPersonDetail').classList.add('hidden');
    $('#librarySelectedPerson').textContent = '全部人物'; await loadPersonLibrary();
    toast('人物库已更新', '归属与模板已更新；原始轨迹、特征和动作证据保留。');
  } catch (error) { if (projectId === state.project?.project_id) toast('归组未修改', error.message, 'error'); }
  finally {
    libraryState.editing = false; $('#personLibrarySection').classList.remove('library-busy');
  }
}
