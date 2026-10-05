/* Match library browsing is separate from clip-local inspection. */
const libraryState = {offset: 0, q: '', status: 'all', person: null, event: '', eventOffset: 0,
  eventQuery: '', minScore: 0.2, revision: null, request: 0, eventRequest: 0, personRequest: 0,
  editing: false, available: false, opened: false};

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
  $('#resetLibraryReview').addEventListener('click', () => {
    if (window.confirm('恢复自动结果？这会清空本批人物库的人工归组和拆出设置，不改原始轨迹。')) editPersonLibrary('reset');
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
  $('#personLibrarySection').scrollIntoView({behavior: 'smooth', block: 'start'});
  await loadPersonLibrary();
}

function closePersonLibrary() {
  libraryState.opened = false;
  $('#personLibrarySection').classList.add('hidden');
  $('#workspace').classList.toggle('hidden', !state.project?.videos.length);
}

async function loadPersonLibrary() {
  if (!state.project) return;
  const projectId = state.project.project_id, request = ++libraryState.request;
  $('#personLibraryWarning').textContent = '正在读取比赛级人物库…';
  try {
    const query = new URLSearchParams({offset: libraryState.offset, limit: 12, q: libraryState.q, status: libraryState.status});
    const data = await api(`/api/projects/${projectId}/person-library?${query}`);
    if (projectId !== state.project?.project_id || request !== libraryState.request) return;
    libraryState.available = data.available; libraryState.revision = data.revision || null;
    $('#resetLibraryReview').disabled = !data.available || libraryState.editing;
    if (!data.available) {
      $('#personLibrarySummary').innerHTML = ''; $('#personLibraryGrid').innerHTML = emptyState('人物库尚未建立', data.message);
      $('#personLibraryWarning').textContent = '先完成所有选定片段，再构建比赛级人物库；不会把片段内同名ID当成同一人。';
      $('#personLibraryPagination').innerHTML = ''; $('#libraryEvents').innerHTML = ''; return;
    }
    if (data.total && libraryState.offset >= data.total) {
      libraryState.offset = Math.floor((data.total - 1) / 12) * 12; return loadPersonLibrary();
    }
    const s = data.summary;
    $('#personLibrarySummary').innerHTML = [[s.expected_clips, '来源片段', '仅限本批次 / 本场比赛'],
      [`${s.local_archive_count} → ${s.global_person_count}`, '局部档案 → 候选人物', '减少数量不是准确率'],
      [s.merged_group_count, '跨片段人物组', `待复核局部档案 ${s.held_archive_count}`],
      [s.event_count, '可检索事件区间', '动作预测未经人工验证']].map(([value, label, note]) =>
      `<div class="batch-stat"><strong>${escapeHtml(value)}</strong><span>${label}</span><small>${escapeHtml(note)}</small></div>`).join('');
    $('#personLibraryWarning').textContent = `跨片段阈值 ${s.settings.max_distance ?? '未开启'} · 候选归并不是确定身份；不会强行压成10人。时间为各原片段内的秒数，不是整场时间轴。${data.stale_review ? ' 旧人工复核已失效，请恢复自动归并后重新核查。' : ''}${s.warnings?.length ? ` 数据警告 ${s.warnings.length} 条。` : ''}`;
    const labels = {candidate: '候选身份', needs_review: '待检查 / 未自动匹配', manual_grouping: '人工归组 · 非准确率'};
    $('#personLibraryGrid').innerHTML = data.items.map(p => `<button class="library-card ${libraryState.person === p.global_person_id ? 'selected' : ''}" data-library-person="${escapeHtml(p.global_person_id)}">
      <div class="library-cover">${p.cover?.crop_url ? `<img src="${escapeHtml(p.cover.crop_url)}" alt="候选人物代表图" loading="lazy" />` : '<span>暂无代表图</span>'}<span class="library-status">${labels[p.status] || escapeHtml(p.status)}</span></div>
      <div class="library-card-copy"><strong>${escapeHtml(p.identity_label || p.global_person_id)}</strong><small>${escapeHtml(p.global_person_id)}</small><p>${p.clip_count} 个片段 · ${p.local_archive_count} 个局部档案 · ${p.event_count} 个事件</p>
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
    detail.innerHTML = `<div class="panel-title"><div><span class="eyebrow">MATCHING EVIDENCE</span><h3>${escapeHtml(p.identity_label || personId)}</h3><p>${p.clip_count} 个片段 · ${p.event_count} 个事件。请看来源，不要只看代表图。</p></div><button class="secondary-button" id="labelLibraryPerson">标记 / 合并到同名人物</button></div>
      <p class="scope-warning">输入相同的“球队＋号码”标签可人工归组；不能合并同一片段中的不同局部人物。错误成员可拆出，混人轨迹须回到轨迹页处理。</p>
      <div class="library-members">${p.members.map(m => `<article class="library-member">
        ${m.cover.crop_url ? `<img src="${escapeHtml(m.cover.crop_url)}" alt="局部档案来源" loading="lazy" />` : ''}
        <div><strong>${escapeHtml(m.filename)}</strong><small>${escapeHtml(m.local_person_id)} · 原轨迹 ${escapeHtml(m.raw_track_ids.join(', '))} · ${m.sample_count} 个KPR样本</small><p>${escapeHtml(m.hold_reasons.join(' / ') || '自动匹配证据通过；仍需人工核查')}</p>
        <div class="library-preview-row">${m.exemplars.filter(s => s.crop_url).map(s => `<img src="${escapeHtml(s.crop_url)}" alt="时序外观证据" loading="lazy" />`).join('')}</div>
        <button class="ghost-button" data-library-open="${escapeHtml(m.node_id)}">检查原片段 ↗</button><button class="ghost-button" data-library-detach="${escapeHtml(m.node_id)}" data-operation="${m.manually_detached ? 'restore' : 'detach'}">${m.manually_detached ? '恢复自动匹配' : '从人物组拆出'}</button></div></article>`).join('')}</div>
      <details class="library-pair-evidence"><summary>最近匹配证据（${p.pairs.length} 对；距离不是置信度）</summary>${p.pairs.map(pair => `<div class="library-pair"><div>${[pair.left, pair.right].map(m => `<span>${m.cover.crop_url ? `<img src="${escapeHtml(m.cover.crop_url)}" alt="匹配来源" loading="lazy" />` : ''}${escapeHtml(m.filename)} / ${escapeHtml(m.local_person_id)}</span>`).join('')}</div><p>距离 ${pair.distance == null ? '无可比部位' : Number(pair.distance).toFixed(3)} · ${pair.same_global_person ? '当前同组' : '当前分开'} · ${escapeHtml(pair.auto_block_reasons.join(' / ') || '满足候选条件')}</p></div>`).join('') || '<p>没有可用匹配对。</p>'}</details>`;
    $('#labelLibraryPerson').addEventListener('click', () => {
      const label = window.prompt('输入球队＋号码等标签。同标签会人工归组；留空移除该组标签。此操作不是独立准确率评估。', p.identity_label || '');
      if (label !== null) editPersonLibrary('label_group', {person_id: personId, label});
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
    $('#resetLibraryReview').disabled = !libraryState.available;
  }
}
