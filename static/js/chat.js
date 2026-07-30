(function () {
  'use strict';

  let currentSessionId = null;
  let isLoading = false;

  const chatMessages = document.getElementById('chat-messages');
  const chatWelcome = document.getElementById('chat-welcome');
  const chatInput = document.getElementById('chat-input');
  const chatSendBtn = document.getElementById('chat-send-btn');
  const chatThinking = document.getElementById('chat-thinking');
  const suggestions = document.getElementById('chat-suggestions');
  const sessionDropdown = document.getElementById('session-dropdown');
  const sessionList = document.getElementById('session-list');
  const sessionCountBadge = document.getElementById('session-count-badge');
  const btnSessionHistory = document.getElementById('btn-session-history');
  const btnNewSession = document.getElementById('btn-new-session');

  function autoResize() {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
    chatSendBtn.disabled = !chatInput.value.trim();
  }
  chatInput.addEventListener('input', autoResize);
  chatInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
      return marked.parse(text, { breaks: true, gfm: true });
    }
    return text.replace(/\n/g, '<br>');
  }

  function addMessage(role, content) {
    chatWelcome.style.display = 'none';

    var div = document.createElement('div');
    div.className = 'message ' + role;

    var avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : '\u2728';
    div.appendChild(avatar);

    var bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (role === 'assistant') {
      bubble.innerHTML = renderMarkdown(content);
    } else {
      bubble.textContent = content;
    }

    div.appendChild(bubble);
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function sendMessage() {
    var prompt = chatInput.value.trim();
    if (!prompt || isLoading) return;

    if (!currentSessionId) {
      try {
        var r = await fetch('/api/chat/session', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: prompt.slice(0, 60) }),
        });
        var data = await r.json();
        currentSessionId = data.session_id;
      } catch (e) {
        console.error('Error creating session:', e);
        return;
      }
    }

    chatInput.value = '';
    autoResize();
    addMessage('user', prompt);

    isLoading = true;
    chatThinking.hidden = false;
    chatSendBtn.disabled = true;

    try {
      var r = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt, session_id: currentSessionId }),
      });
      var data = await r.json();

      if (!r.ok) {
        addMessage('assistant', '\u26A0\uFE0F ' + (data.error || 'Error desconocido'));
        return;
      }

      addMessage('assistant', data.reply);
      if (data.session_id) currentSessionId = data.session_id;
      loadSessionList();
    } catch (e) {
      addMessage('assistant', '\u26A0\uFE0F Error de conexi\u00F3n con el servidor.');
    } finally {
      isLoading = false;
      chatThinking.hidden = true;
      chatSendBtn.disabled = !chatInput.value.trim();
    }
  }

  chatSendBtn.addEventListener('click', sendMessage);

  suggestions.addEventListener('click', function (e) {
    var chip = e.target.closest('.suggestion-chip');
    if (chip) {
      chatInput.value = chip.dataset.prompt;
      autoResize();
      sendMessage();
    }
  });

  async function loadSessionList() {
    try {
      var r = await fetch('/api/chat/sessions');
      if (!r.ok) return;
      var data = await r.json();
      var sessions = data.sessions || [];

      sessionCountBadge.textContent = sessions.length;

      if (sessions.length === 0) {
        sessionList.innerHTML = '<div style="padding:20px;text-align:center;color:var(--muted);font-size:0.78rem">No hay sesiones a\u00FAn</div>';
        return;
      }

      sessionList.innerHTML = sessions.map(function (s) {
        var active = s.id === currentSessionId ? ' active' : '';
        var date = s.updated_at || s.created_at || '';
        var dateStr = date.slice(0, 16).replace('T', ' ');
        return '<div class="session-item' + active + '" data-session-id="' + s.id + '">' +
          '<div class="session-item-icon">\uD83D\uDCAC</div>' +
          '<div class="session-item-body">' +
          '<div class="session-item-title">' + escapeHtml(s.title || 'Nueva sesi\u00F3n') + '</div>' +
          '<div class="session-item-date">' + dateStr + '</div>' +
          '</div>' +
          '<button class="session-item-delete" data-session-id="' + s.id + '" title="Eliminar sesi\u00F3n">\u2715</button>' +
          '</div>';
      }).join('');

      sessionList.querySelectorAll('.session-item').forEach(function (item) {
        item.addEventListener('click', function (e) {
          if (e.target.closest('.session-item-delete')) return;
          var sid = item.dataset.sessionId;
          if (sid && sid !== currentSessionId) {
            switchSession(sid);
          }
        });
      });

      sessionList.querySelectorAll('.session-item-delete').forEach(function (btn) {
        btn.addEventListener('click', async function (e) {
          e.stopPropagation();
          var sid = btn.dataset.sessionId;
          if (!sid) return;
          try {
            await fetch('/api/chat/session/' + sid, { method: 'DELETE' });
            if (sid === currentSessionId) {
              currentSessionId = null;
              clearMessages();
            }
            loadSessionList();
          } catch (err) {
            console.error('Error deleting session:', err);
          }
        });
      });
    } catch (e) {
      console.error('Error loading sessions:', e);
    }
  }

  function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  async function switchSession(sessionId) {
    try {
      var r = await fetch('/api/chat/session/' + sessionId);
      if (!r.ok) return;
      var data = await r.json();
      currentSessionId = sessionId;
      clearMessages();
      (data.messages || []).forEach(function (m) { addMessage(m.role, m.content); });
      loadSessionList();
      sessionDropdown.hidden = true;
    } catch (e) {
      console.error('Error switching session:', e);
    }
  }

  function clearMessages() {
    while (chatMessages.firstChild) {
      chatMessages.removeChild(chatMessages.firstChild);
    }
    chatWelcome.style.display = 'flex';
  }

  btnNewSession.addEventListener('click', async function () {
    currentSessionId = null;
    clearMessages();
    chatInput.value = '';
    autoResize();
    chatInput.focus();
    sessionDropdown.hidden = true;
    loadSessionList();
  });

  btnSessionHistory.addEventListener('click', function (e) {
    e.stopPropagation();
    sessionDropdown.hidden = !sessionDropdown.hidden;
    if (!sessionDropdown.hidden) {
      loadSessionList();
    }
  });

  document.addEventListener('click', function (e) {
    var container = document.getElementById('session-popup-container');
    if (container && !container.contains(e.target) && !sessionDropdown.hidden) {
      sessionDropdown.hidden = true;
    }
  });

  loadSessionList();
  chatInput.focus();

})();
