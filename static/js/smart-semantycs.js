(function () {
  'use strict';

  var currentSessionId = null;

  var $ = function (id) { return document.getElementById(id); };
  var chatMessages = $('chat-messages');
  var chatWelcome = $('chat-welcome');
  var chatInput = $('chat-input');
  var chatSendBtn = $('chat-send-btn');
  var btnLoadVideo = $('btn-load-video');
  var fileVideo = $('file-video');
  var sourceChip = $('source-chip');
  var stateBadge = $('state-badge');
  var countersEl = $('ss-counters');
  var logsBody = $('ss-logs-body');
  var videoFrame = $('video-frame');
  var videoEmpty = $('video-empty');
  var videoEmptyText = $('video-empty-text');
  var videoControls = $('video-controls');
  var videoPrompt = $('video-prompt');
  var btnToggle = $('btn-toggle');
  var btnReset = $('btn-reset');
  var btnStop = $('btn-stop');
  var btnNew = $('btn-new-session');
  var sessionLabel = $('session-label');
  var sessionsList = $('ss-sessions-list');
  var sessionsToggle = $('ss-sessions-toggle');
  var sessionsCaret = $('ss-sessions-caret');

  function toggleSessionsOpen(force) {
    var open = sessionsList.hidden;
    if (force === true) open = true;
    if (force === false) open = false;
    sessionsList.hidden = !open;
    sessionsCaret.textContent = open ? '▴' : '▾';
  }

  var typingBubble = null;

  var STATE_UI = {
    no_video: 'No video',
    video: 'Video',
    prompted: 'Ready',
    running: 'Running',
    stopped: 'Stopped'
  };

  function autoResize() {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    updateSendBtn();
  }

  function updateSendBtn() {
    chatSendBtn.disabled = !chatInput.value.trim() || chatInput.disabled;
  }

  function api(path, opts) {
    return fetch(path, opts).then(function (r) {
      return r.json().then(function (d) { return { status: r.status, data: d }; });
    });
  }

  function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = (text == null ? '' : String(text));
    return div.innerHTML;
  }

  function shortId(id) {
    return id ? id.slice(0, 8) : '';
  }

  function hasState(state) {
    return state === 'prompted' || state === 'running' || state === 'stopped';
  }

  // ── Messages ─────────────────────────────────────────────────────
  function addMessage(content, role, kind) {
    chatWelcome.style.display = 'none';
    var div = document.createElement('div');
    div.className = 'message ' + (role === 'model' ? 'assistant' : 'user');
    var avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : '\u2728';
    div.appendChild(avatar);
    var bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    if (role === 'model' && kind === 'skill') {
      bubble.innerHTML = '<strong>\u2713 ' + escapeHtml(content) + '</strong>';
    } else {
      bubble.textContent = content;
    }
    div.appendChild(bubble);
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function showTypingBubble() {
    chatWelcome.style.display = 'none';
    typingBubble = document.createElement('div');
    typingBubble.className = 'message assistant typing';
    var avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '\u2728';
    typingBubble.appendChild(avatar);
    var bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    var dots = document.createElement('div');
    dots.className = 'typing-dots';
    for (var i = 0; i < 3; i++) {
      var s = document.createElement('span');
      s.className = 'typing-dot';
      dots.appendChild(s);
    }
    bubble.appendChild(dots);
    var label = document.createElement('span');
    label.className = 'typing-label';
    label.textContent = 'Interpreting...';
    bubble.appendChild(label);
    typingBubble.appendChild(bubble);
    chatMessages.appendChild(typingBubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function removeTypingBubble() {
    if (typingBubble && typingBubble.parentNode) {
      typingBubble.parentNode.removeChild(typingBubble);
      typingBubble = null;
    }
  }

  function clearChat() {
    while (chatMessages.firstChild) chatMessages.removeChild(chatMessages.firstChild);
    chatWelcome.style.display = 'flex';
    removeTypingBubble();
  }

  function loadMessages(sessionId) {
    api('/api/semantycs/sessions/' + sessionId).then(function (res) {
      if (res.status !== 200) return;
      clearChat();
      (res.data.session.messages || []).forEach(function (m) {
        addMessage(m.content, m.role, m.kind);
      });
    });
  }

  // ── Sessions history ─────────────────────────────────────────────
  function renderSessions(sessions) {
    if (!sessions || sessions.length === 0) {
      sessionsList.innerHTML = '<div class="ss-sessions-empty">No sessions yet</div>';
      return;
    }
    sessionsList.innerHTML = sessions.map(function (s) {
      var active = s.id === currentSessionId ? ' ss-session-active' : '';
      var title = s.prompt || s.title || ('Session ' + shortId(s.id));
      var time = (s.updated_at || '').split(' ')[1] || '';
      var dot = ((s.id === currentSessionId && s.state === 'running') ? 'running' : 'off');
      return '<div class="ss-session-item' + active + '" data-sid="' + s.id + '">' +
        '<span class="ss-session-dot ' + escapeHtml(dot) + '"></span>' +
        '<span class="ss-session-item-title">' + escapeHtml(title) + '</span>' +
        '<span class="ss-session-item-time">' + escapeHtml(time) + '</span>' +
        '</div>';
    }).join('');
    sessionsList.querySelectorAll('.ss-session-item').forEach(function (row) {
      row.addEventListener('click', function () { openSession(row.dataset.sid); });
    });
  }

  function refreshSessions() {
    return api('/api/semantycs/sessions').then(function (res) {
      if (res.status !== 200) return;
      renderSessions(res.data.sessions || []);
    });
  }

  function openSession(sessionId) {
    currentSessionId = sessionId;
    sessionLabel.textContent = 'Session ' + shortId(sessionId);
    loadMessages(sessionId);
    refreshSessions();
    refreshState();
    toggleSessionsOpen(false);
  }

  function resetView() {
    sessionLabel.textContent = 'Session';
    stateBadge.textContent = '';
    stateBadge.className = 'ss-state-badge';
    sourceChip.hidden = true;
    videoFrame.hidden = true;
    videoFrame.removeAttribute('src');
    videoControls.hidden = true;
    videoPrompt.hidden = true;
    videoEmpty.hidden = false;
    videoEmptyText.textContent = 'Load a video to begin.';
    countersEl.innerHTML = '<div class="ss-counters-empty">No counters</div>';
    logsBody.innerHTML = '<div class="ss-logs-empty">No events</div>';
    btnToggle.textContent = '▶ Play';
    btnStop.disabled = true;
    btnReset.disabled = true;
  }

  function newSession() {
    api('/api/semantycs/sessions', { method: 'POST' }).then(function (res) {
      if (res.status !== 201) return;
      currentSessionId = res.data.session_id;
      sessionLabel.textContent = 'Session ' + shortId(currentSessionId);
      resetView();
      clearChat();
      chatInput.value = '';
      chatInput.disabled = false;
      autoResize();
      toggleSessionsOpen(false);
      refreshSessions();
      refreshState();
    });
  }

  function initialize() {
    api('/api/semantycs/sessions').then(function (res) {
      var sessions = (res.status === 200 && res.data.sessions) ? res.data.sessions : [];
      renderSessions(sessions);
      if (sessions.length) {
        openSession(sessions[0].id);
      } else {
        newSession();
      }
    });
  }

  // ── New session button ───────────────────────────────────────────
  btnNew.addEventListener('click', function () {
    newSession();
  });

  sessionsToggle.addEventListener('click', function () {
    toggleSessionsOpen();
  });

  // ── Video source ────────────────────────────────────────────────
  btnLoadVideo.addEventListener('click', function () { fileVideo.click(); });

  fileVideo.addEventListener('change', function () {
    var f = fileVideo.files && fileVideo.files[0];
    if (!f || !currentSessionId) return;
    var fd = new FormData();
    fd.append('video', f);
    fetch('/api/sources/upload-video', { method: 'POST', body: fd })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d.path) { alert(d.error || 'Upload failed'); return null; }
        return api('/api/semantycs/sessions/' + currentSessionId + '/video', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: d.path, type: 'video' })
        });
      })
      .then(function () { refreshState(); refreshSessions(); })
      .catch(function () { alert('Upload failed'); });
  });

  // ── Interpret (with auto-start) ─────────────────────────────────
  function interpret() {
    var prompt = chatInput.value.trim();
    if (!prompt || !currentSessionId || chatInput.disabled) return;

    chatInput.value = '';
    autoResize();
    addMessage(prompt, 'user');
    showTypingBubble();
    chatInput.disabled = true;
    updateSendBtn();

    api('/api/semantycs/sessions/' + currentSessionId + '/interpret', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: prompt })
    }).then(function (res) {
      removeTypingBubble();
      chatInput.disabled = false;
      if (res.status === 429) {
        addMessage('\u23F1\uFE0F ' + (res.data.error || 'Wait a moment.'), 'model');
      } else if (res.status !== 200) {
        addMessage('\u26A0\uFE0F ' + (res.data.error || 'Error'), 'model');
      }
      return refreshState();
    }).then(function (st) {
      if (st && st.state === 'prompted') {
        // Auto-start: begin detection as soon as the skill is ready.
        api('/api/semantycs/sessions/' + currentSessionId + '/start', { method: 'POST' })
          .then(function (rs) {
            if (rs.status !== 200) { alert(rs.data.error || 'Error starting'); return; }
            refreshState();
          })
          .then(refreshSessions);
      }
    }).catch(function () {
      removeTypingBubble();
      chatInput.disabled = false;
      addMessage('\u26A0\uFE0F Connection error.', 'model');
    }).finally(function () {
      updateSendBtn();
    });
  }

  chatSendBtn.addEventListener('click', interpret);
  chatInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); interpret(); }
  });
  chatInput.addEventListener('input', autoResize);

  // ── Controls ─────────────────────────────────────────────────────
  btnToggle.addEventListener('click', function () {
    if (!currentSessionId) return;
    api('/api/semantycs/sessions/' + currentSessionId + '/state').then(function (res) {
      if (res.status !== 200) return;
      var st = res.data;
      var op = st.running ? (st.paused ? 'resume' : 'pause') : 'start';
      api('/api/semantycs/sessions/' + currentSessionId + '/' + op, { method: 'POST' })
        .then(function (r) {
          if (r.status !== 200) { alert(r.data.error || 'Error'); return; }
          if (op === 'start') refreshSessions();
          refreshState();
        });
    });
  });

  btnReset.addEventListener('click', function () {
    if (!currentSessionId) return;
    if (!confirm('Reset detection? Counters, logs and captures of this session will be cleared.')) return;
    api('/api/semantycs/sessions/' + currentSessionId + '/reset', { method: 'POST' })
      .then(refreshState);
  });

  btnStop.addEventListener('click', function () {
    if (!currentSessionId) return;
    if (!confirm('Stop detection? You can press Play to run it again from the start.')) return;
    api('/api/semantycs/sessions/' + currentSessionId + '/stop', { method: 'POST' })
      .then(function () { refreshState(); refreshSessions(); });
  });

  // ── Render ───────────────────────────────────────────────────────
  function renderCounters(counters) {
    if (!counters || counters.length === 0) {
      countersEl.innerHTML = '<div class="ss-counters-empty">No counters</div>';
      return;
    }
    countersEl.innerHTML = counters.map(function (c) {
      return '<div class="ss-counter" style="border-left-color:' + escapeHtml(c.color || '#22C55E') + '">' +
        '<div class="ss-counter-value">' + c.value + '</div>' +
        '<div class="ss-counter-label">' + escapeHtml(c.label) + '</div>' +
        '</div>';
    }).join('');
  }

  function renderLogs(logs) {
    if (!logs || logs.length === 0) {
      logsBody.innerHTML = '<div class="ss-logs-empty">No events</div>';
      return;
    }
    logsBody.innerHTML = logs.map(function (l) {
      var time = (l.created_at || '').split(' ')[1] || '';
      return '<div class="ss-log-row" data-log-id="' + l.id + '">' +
        '<span class="ss-log-prio ' + escapeHtml(l.priority) + '"></span>' +
        '<span class="ss-log-time">' + escapeHtml(time) + '</span>' +
        '<span class="ss-log-label">' + escapeHtml(l.label) + '</span>' +
        '</div>';
    }).join('');
    logsBody.querySelectorAll('.ss-log-row').forEach(function (row) {
      row.addEventListener('click', function () {
        openLogEvidence(parseInt(row.dataset.logId, 10));
      });
    });
  }

  function openLogEvidence(logRowId) {
    api('/api/semantycs/sessions/' + currentSessionId + '/logs/' + logRowId)
      .then(function (res) {
        if (res.status !== 200) return;
        var log = res.data.log;
        document.getElementById('ev-title').textContent = 'Smart Semantycs · Evidence';
        var cap = log.capture_path || '';
        if (cap.indexOf('static/') === 0) cap = '/' + cap;
        document.getElementById('ev-image').src = cap;
        document.getElementById('ev-module').textContent = 'Smart Semantycs';
        document.getElementById('ev-source').textContent = shortId(currentSessionId);
        var parts = (log.created_at || '').split(' ');
        document.getElementById('ev-date').textContent = parts[0] || '';
        document.getElementById('ev-time').textContent = parts[1] || '';
        document.getElementById('ev-type').textContent = log.priority || '';
        document.getElementById('ev-description').textContent = log.label || '';
        document.getElementById('ev-checklist').hidden = true;
        document.getElementById('ev-extra').hidden = true;
        document.getElementById('ev-modal').hidden = false;
      });
  }

  function setToggleUI(st) {
    videoControls.hidden = !hasState(st.state);
    if (st.running) {
      btnToggle.textContent = st.paused ? '▶ Play' : '⏸ Pause';
      btnToggle.title = st.paused ? 'Resume' : 'Pause';
    } else {
      btnToggle.textContent = '▶ Play';
      btnToggle.title = 'Play';
    }
    btnStop.disabled = !st.running;
    btnReset.disabled = !st.running;
  }

  function refreshState() {
    if (!currentSessionId) return Promise.resolve(null);
    return api('/api/semantycs/sessions/' + currentSessionId + '/state').then(function (res) {
      if (res.status !== 200) return null;
      var st = res.data;
      stateBadge.textContent = STATE_UI[st.state] || st.state;
      stateBadge.className = 'ss-state-badge ' + (st.state || '');
      chatInput.disabled = st.state === 'running';
      updateSendBtn();

      if (st.video_path) {
        sourceChip.hidden = false;
        sourceChip.textContent = (st.video_type === 'stream' ? '🔗 ' : '🎞 ') +
          (st.video_path.split('/').pop() || st.video_path);
      } else {
        sourceChip.hidden = true;
      }

      if (st.running) {
        videoEmpty.hidden = true;
        videoFrame.hidden = false;
        videoFrame.src = '/api/semantycs/sessions/' + currentSessionId + '/stream';
      } else {
        videoFrame.hidden = true;
        videoFrame.removeAttribute('src');
        videoEmpty.hidden = false;
        if (st.state === 'no_video') {
          videoEmptyText.textContent = 'Load a video to begin.';
        } else if (st.state === 'video') {
          videoEmptyText.textContent = 'Video linked. Type a prompt to configure detection.';
        } else if (st.state === 'prompted') {
          videoEmptyText.textContent = 'Detection ready. Press Play.';
        } else if (st.state === 'stopped') {
          videoEmptyText.textContent = 'Detection stopped. Press Play to run it again from the start.';
        }
      }

      setToggleUI(st);

      videoPrompt.hidden = !st.prompt;
      if (st.prompt) videoPrompt.textContent = st.prompt;

      renderCounters(st.counters);
      renderLogs(st.logs);
      return st;
    });
  }

  // ── Polling ──────────────────────────────────────────────────────
  setInterval(function () { refreshState(); }, 1500);
  setInterval(function () { refreshSessions(); }, 5000);

  initialize();
})();