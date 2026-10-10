/* Match library browsing is separate from clip-local inspection. */
const libraryState = {offset: 0, q: '', status: 'all', person: null, event: '', eventOffset: 0,
  eventQuery: '', minScore: 0.2, revision: null, request: 0, eventRequest: 0, personRequest: 0,
  editing: false, available: false, opened: false};

function libraryJersey(jersey, showEvidence = false) {
  if (!jersey) return '';
  const label = jersey.number != null ? `Qwen 号码候选 #${jersey.number}` :
    jersey.status === 'conflict' ? '号码读数冲突' : '号码未知 / 证据不足';
  const readings = (jersey.tracks || []).flatMap(t => t.readings || []);
  return `<p class="jersey-badge">${escapeHtml(label)}</p>${showEvidence && readings.length ?
    `<details><summary>读号中间结果（${readings.length} 帧）</summary><div class="number-readings">${readings.map(r =>
      `<span>${r.crop_url ? `<a href="${escapeHtml(r.crop_url)}" target="_blank" rel="noopener"><img src="${escapeHtml(r.crop_url)}" alt="Qwen 躯干裁剪" loading="lazy" /></a>` : ''}<small>${formatTime(r.timestamp_s)} · 帧 ${escapeHtml(r.frame_idx)} · ${r.text == null ? '不可读' : '#' + escapeHtml(r.text)}<br />质量 ${r.sampling_quality == null ? '未记录' : Number(r.sampling_quality).toFixed(3)}<br />${escapeHtml((r.rejection_reasons || []).join(' / ') || '候选支持帧')}</small><details><summary>原始回答</summary><pre>${escapeHtml(r.raw_response || '未记录')}</pre></details></span>`).join('')}</div></details>` : ''}`;
}

function bindPersonLibrary() {
  $('#openPersonLibraryButton').addEventListener('click', openPersonLibrary);
  $('#closePersonLibrary').addEventListener('click', closePersonLibrary);
  $('#refreshPersonLibrary').addEventListener('click', loadPersonLibrary);
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
  Object.assign(libraryState, {offset: 0, q: '', status: 'all', person: null, event: '', eventOffset: 0,
    eventQuery: '', revision: null, available: false, opened: false});
  $('#personLibrarySection').classList.add('hidden'); $('#libraryPersonDetail').classList.add('hidden');
  $('#librarySearch').value = ''; $('#libraryStatus').value = 'all'; $('#libraryEventSearch').value = '';
}

async function openPersonLibrary() {
  if (!state.project || libraryState.editing) return;
  libraryState.opened = true;
  $('#personLibrarySection').classList.remove('hidden'); $('#workspace').classList.add('hidden');
  $('#uploadSection').classList.add('hidden');
  $('#personLibrarySection').scrollIntoView({behavior: 'smooth', block: 'start'});
  await loadPersonLibrary();
}

function closePersonLibrary() {
  libraryState.opened = false;
  $('#personLibrarySection').classList.add('hidden');
  $('#workspace').classList.toggle('hidden', !state.project?.videos.length);
  $('#uploadSection').classList.toggle('hidden', Boolean(state.project?.read_only));
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
    const statuses = s.player_status_counts || {};
    $('#personLibrarySummary').innerHTML = [[s.expected_clips, '已登记片段', `本场比赛：${s.match_id}`],
      [s.global_person_count, '比赛级身份', `候选 ${statuses.candidate || 0} · 人工确认 ${statuses.manual_grouping || 0} · 非球员 ${statuses.non_player || 0}`],
      [statuses.needs_review || 0, '待定身份', `图库观察 ${s.gallery_observation_count ?? '未记录'} · 局部档案 ${s.local_archive_count}`],
      [s.event_count, '可检索事件区间', `暂未关联身份 ${s.unmapped_event_count ?? 0}`]].map(([value, label, note]) =>
      `<div class="batch-stat"><strong>${escapeHtml(value)}</strong><span>${label}</span><small>${escapeHtml(note)}</small></div>`).join('');
    $('#personLibraryWarning').textContent = `本场比赛 ${s.match_id} · 匹配距离 ${s.settings.max_distance ?? '未记录'} · 新身份距离 ${s.settings.novelty_distance ?? '未记录'}。${s.jersey_constraints ? 'KPR 外观与 Qwen 号码证据联合匹配。' : 'KPR 外观匹配。'}图库持续追加合格观察，待定观察暂不进入图库。`;
    const labels = {candidate: '候选身份', needs_review: '待检查 / 未自动匹配', manual_grouping: '人工确认', non_player: '非球员'};
    $('#personLibraryGrid').innerHTML = data.items.map(p => `<button class="library-card ${libraryState.person === p.global_person_id ? 'selected' : ''}" data-library-person="${escapeHtml(p.global_person_id)}">
      <div class="library-cover">${p.cover?.crop_url ? `<img src="${escapeHtml(p.cover.crop_url)}" alt="候选人物代表图" loading="lazy" />` : '<span>暂无代表图</span>'}<span class="library-status">${labels[p.status] || escapeHtml(p.status)}</span></div>
      <div class="library-card-copy"><strong>${escapeHtml(p.identity_label || p.global_person_id)}</strong><small>${escapeHtml(p.global_person_id)}</small>${libraryJersey(p.jersey)}<p>${p.clip_count} 个片段 · ${p.local_archive_count} 个局部档案 · ${p.event_count} 个事件</p>
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
        <div><strong>${escapeHtml(m.filename)}</strong><small>${escapeHtml(m.local_person_id)} · 原轨迹 ${escapeHtml(m.raw_track_ids.join(', '))} · ${m.sample_count} 个 KPR 样本</small><p class="member-decision">${escapeHtml({matched: '匹配已有球员', new_candidate: '建立新候选身份', pending: '待定观察'}[m.decision] || m.decision || '旧结果未记录匹配决策')} · ${m.gallery_eligible ? '已进入外观图库' : '未进入外观图库'}</p><p>${escapeHtml((m.hold_reasons || []).join(' / ') || '查看原轨迹与时序外观确认身份')}</p>${libraryJersey(m.jersey, true)}
        <div class="library-preview-row">${m.exemplars.filter(s => s.crop_url).map(s => `<img src="${escapeHtml(s.crop_url)}" alt="时序外观证据" loading="lazy" />`).join('')}</div>
        <button class="ghost-button" data-library-open="${escapeHtml(m.node_id)}">检查原片段 ↗</button><button class="ghost-button" data-library-detach="${escapeHtml(m.node_id)}" data-operation="${m.manually_detached ? 'restore' : 'detach'}">${m.manually_detached ? '撤销拆出' : '从人物组拆出'}</button></div></article>`).join('')}</div>
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
    $('#libraryEvents').innerHTML = data.items.map((e, index) => `<button class="library-event" data-library-event="${index}"><div><strong>${escapeHtml(e.event)}</strong><span>${escapeHtml(e.filename)}</span><small>${escapeHtml(e.global_person_id)} · 局部 ${escapeHtml(e.local_person_id)}</small></div><div><strong>${formatTime(e.start)} → ${formatTime(e.end)}</strong><small>raw_score ${Number(e.raw_score).toFixed(3)} · ${(e.end - e.start).toFixed(2)}s · 跳转原片段 ↗</small></div></button>`).join('') || emptyState('没有符合条件的事件', '切换人物 / 事件类型或降低原始分数筛选。无事件不代表该人物没有做动作。');
    $('#libraryEvents').querySelectorAll('[data-library-event]').forEach(button => button.addEventListener('click', async () => {
      const e = data.items[Number(button.dataset.libraryEvent)];
      await selectVideo(e.web_video_id, false); if (state.videoId !== e.web_video_id) return;
      closePersonLibrary(); state.personFilter = e.local_person_id; state.eventFilter = e.event;
      setVideoView('source', false); setTab('events');
      const player = $('#resultVideo'), seek = () => { player.currentTime = e.start; player.play().catch(() => {}); };
      if (player.readyState >= 1) seek(); else player.addEventListener('loadedmetadata', seek, {once: true});
      $('#inspector').scrollIntoView({behavior: 'smooth', block: 'start'});
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
    toast('人物库已更新', '人工修正独立保存；原始模型输出保持不变。');
  } catch (error) { if (projectId === state.project?.project_id) toast('归组未修改', error.message, 'error'); }
  finally {
    libraryState.editing = false; $('#personLibrarySection').classList.remove('library-busy');
  }
}
