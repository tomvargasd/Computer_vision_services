(function () {
  'use strict';

  var currentSessionId = null;
  var pollTimer = null;

  var $ = function (id) { return document.getElementById(id); };
  var chatMessages = $('chat-messages');
  var chatWelcome = $('chat-welcome');
  var chatInput = $('chat-input');
  var chatSendBtn = $('chat-send-btn');
  var btnStart = $('btn-start');
  var btnLoadVideo = $('btn-load-video');
  var fileVideo = $('file-video');
  var sourceChip = $('source-chip');
  var stateBadge = $('state-badge');
  var countersEl = $('ss-counters');
  var logsBody = $('ss-logs-body');
  var videoFrame = $('video-frame');
  var videoEmpty = $('video-empty');
  var videoEmptyText = $('video-empty-text');
  var videoOverlay = $('video-overlay');
  var videoPrompt = $('video-prompt');
  var btnPause = $('btn-pause');
  var btnResume = $('btn-resume');
  var btnReset = $('btn-reset');
  var btnStop = $('btn-stop');
  var sessionLabel = $('session-label');

  var typingBubble = null;

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
    div.textContent = text || '';
    return div.innerHTML;
  }

  function shortId(id) {
    return id ? id.slice(0, 8) : '';
  }

  // ── Mensajes del chat ────────────────────────────────────────────────
  function addMessage(role, content, kind) {
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
      var d = document.createElement('span');
      d.className = 'typing-dot';
      dots.appendChild(d);
    }
    bubble.appendChild(dots);
    var label = document.createElement('span');
    label.className = 'typing-label';
    label.textContent = 'Interpretando...';
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

  function clearMessages() {
    while (chatMessages.firstChild) chatMessages.removeChild(chatMessages.firstChild);
    chatWelcome.style.display = 'flex';
  }

  function loadMessages(sessionId) {
    api('/api/semantycs/sessions/' + sessionId).then(function (res) {
      if (res.status !== 200) return;
      clearMessages();
      (res.data.session.messages || []).forEach(function (m) {
        addMessage(m.role, m.content, m.kind);
      });
      applyStateHint(res.data.session);
    });
  }

  // ── Sesión ──────────────────────────────────────────────────────────
  function newSession() {
    api('/api/semantycs/sessions', { method: 'POST' }).then(function (res) {
      if (res.status === 201) {
        currentSessionId = res.data.session_id;
        sessionLabel.textContent = 'Sesión ' + shortId(currentSessionId);
        clearMessages();
        chatInput.value = '';
        autoResize();
        refreshState();
      }
    });
  }

  function boot() {
    api('/api/semantycs/sessions').then(function (res) {
      if (res.status !== 200) return;
      var sessions = res.data.sessions || [];
      if (sessions.length) {
        currentSessionId = sessions[0].id;
        sessionLabel.textContent = 'Sesión ' + shortId(currentSessionId);
        loadMessages(currentSessionId);
        refreshState();
      } else {
        newSession();
      }
    });
  }

  // ── Video ───────────────────────────────────────────────────────────
  btnLoadVideo.addEventListener('click', function () { fileVideo.click(); });

  fileVideo.addEventListener('change', function () {
    var f = fileVideo.files && fileVideo.files[0];
    if (!f || !currentSessionId) return;
    var fd = new FormData();
    fd.append('video', f);
    fetch('/api/sources/upload-video', { method: 'POST', body: fd })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (!d.path) { alert(d.error || 'Upload failed'); return; }
        return api('/api/semantycs/sessions/' + currentSessionId + '/video', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ path: d.path, type: 'video' }),
        });
      })
      .then(function () { refreshState(); });
  });

  // ── Interpretar ─────────────────────────────────────────────────────
  function interpret() {
    var prompt = chatInput.value.trim();
    if (!prompt || !currentSessionId || chatInput.disabled) return;

    chatInput.value = '';
    autoResize();
    addMessage('user', prompt);
    showTypingBubble();
    chatInput.disabled = true;
    updateSendBtn();

    api('/api/semantycs/sessions/' + currentSessionId + '/interpret', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: prompt }),
    }).then(function (res) {
      removeTypingBubble();
      chatInput.disabled = false;
      if (res.status === 429) {
        addMessage('model', '\u23F1\uFE0F ' + (res.data.error || 'Espera un momento.'));
      } else if (res.status !== 200) {
        addMessage('model', '\u26A0\uFE0F ' + (res.data.error || 'Error'));
      }
      loadMessages(currentSessionId);
      refreshState();
    }).catch(function () {
      removeTypingBubble();
      chatInput.disabled = false;
      addMessage('model', '\u26A0\uFE0F Error de conexión.');
    }).finally(function () {
      updateSendBtn();
    });
  }

  chatSendBtn.addEventListener('click', interpret);
  chatInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      interpret();
    }
  });
  chatInput.addEventListener('input', autoResize);

  // ── Controles ───────────────────────────────────────────────────────
  btnStart.addEventListener('click', function () {
    if (!currentSessionId) return;
    api('/api/semantycs/sessions/' + currentSessionId + '/start', { method: 'POST' })
      .then(function (res) {
        if (res.status !== 200) { alert(res.data.error || 'Error al iniciar'); return; }
        refreshState();
      });
  });

  btnPause.addEventListener('click', function () {
    api('/api/semantycs/sessions/' + currentSessionId + '/pause', { method: 'POST' })
      .then(refreshState);
  });

  btnResume.addEventListener('click', function () {
    api('/api/semantycs/sessions/' + currentSessionId + '/resume', { method: 'POST' })
      .then(refreshState);
  });

  btnReset.addEventListener('click', function () {
    if (!confirm('¿Reiniciar la detección? Se limpiarán contadores, logs y capturas de esta sesión.')) return;
    api('/api/semantycs/sessions/' + currentSessionId + '/reset', { method: 'POST' })
      .then(refreshState);
  });

  btnStop.addEventListener('click', function () {
    api('/api/semantycs/sessions/' + currentSessionId + '/stop', { method: 'POST' })
      .then(refreshState);
  });

  document.getElementById('btn-new-session').addEventListener('click', function () {
    if (confirm('¿Iniciar una nueva sesión? La sesión actual se conservará en el historial.')) {
      newSession();
    }
  });

  // ── Render de estado ────────────────────────────────────────────────
  function applyStateHint(session) {
    var state = session.state;
    var locked = state === 'running';
    chatInput.disabled = locked;
    updateSendBtn();
    btnStart.hidden = state !== 'prompted';
    btnLoadVideo.hidden = locked || state !== 'no_video';
  }

  function renderCounters(counters) {
    if (!counters || counters.length === 0) {
      countersEl.innerHTML = '<div class="ss-counters-empty">Sin contadores</div>';
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
      logsBody.innerHTML = '<div class="ss-logs-empty">Sin eventos</div>';
      return;
    }
    logsBody.innerHTML = logs.map(function (l) {
      var t = l.created_at || '';
      var time = t.split(' ')[1] || t;
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
        document.getElementById('ev-title').textContent = 'Smart Semantycs · Evidencia';
        var cap = log.capture_path || '';
        if (cap && cap.indexOf('static/') === 0) cap = '/' + cap;
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

  function refreshState() {
    if (!currentSessionId) return;
    api('/api/semantycs/sessions/' + currentSessionId + '/state').then(function (res) {
      if (res.status !== 200) return;
      var st = res.data;

      stateBadge.textContent = st.state;
      stateBadge.className = 'ss-state-badge ' + (st.state || '');

      var locked = st.state === 'running';
      chatInput.disabled = locked;
      updateSendBtn();
      btnStart.hidden = st.state !== 'prompted';
      btnLoadVideo.hidden = locked || st.state !== 'no_video';

      if (st.video_path) {
        sourceChip.hidden = false;
        sourceChip.textContent = (st.video_type === 'stream' ? '🔗 ' : '🎞 ') +
          (st.video_path.split('/').pop() || st.video_path);
      } else {
        sourceChip.hidden = true;
      }

      // Video
      if (st.running) {
        videoEmpty.hidden = true;
        videoFrame.hidden = false;
        videoFrame.src = '/api/semantycs/sessions/' + currentSessionId + '/stream';
        videoOverlay.hidden = false;
        btnPause.hidden = !!st.paused;
        btnResume.hidden = !st.paused;
        videoPrompt.hidden = false;
        videoPrompt.textContent = st.prompt || '';
      } else {
        videoFrame.hidden = true;
        videoFrame.removeAttribute('src');
        videoOverlay.hidden = true;
        videoPrompt.hidden = true;
        videoEmpty.hidden = false;
        if (st.state === 'no_video') {
          videoEmptyText.textContent = 'Sube o vincula un video o stream para comenzar.';
        } else if (st.state === 'video') {
          videoEmptyText.textContent = 'Video vinculado. Escribe un prompt para configurar la detección.';
        } else if (st.state === 'prompted') {
          videoEmptyText.textContent = 'Detección lista. Pulsa Iniciar.';
        } else if (st.state === 'stopped') {
          videoEmptyText.textContent = 'Detección detenida. Crea una nueva sesión si deseas continuar.';
        }
      }

      renderCounters(st.counters);
      renderLogs(st.logs);
    });
  }

  // ── Polling ─────────────────────────────────────────────────────────
  setInterval(refreshState, 1500);

  chatInput.addEventListener('focus', function () {});
  boot();
})();