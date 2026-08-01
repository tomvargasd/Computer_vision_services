(function () {
  'use strict';

  var currentSessionId = null;
  var isLoading = false;
  var cooldownUntil = 0;

  var chatMessages = document.getElementById('chat-messages');
  var chatWelcome = document.getElementById('chat-welcome');
  var chatInput = document.getElementById('chat-input');
  var chatSendBtn = document.getElementById('chat-send-btn');
  var suggestions = document.getElementById('chat-suggestions');
  var sessionDropdown = document.getElementById('session-dropdown');
  var sessionList = document.getElementById('session-list');
  var sessionCountBadge = document.getElementById('session-count-badge');
  var btnSessionHistory = document.getElementById('btn-session-history');
  var btnNewSession = document.getElementById('btn-new-session');

  var typingBubble = null;
  var cooldownTimer = null;

  function autoResize() {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
    updateSendButton();
  }
  chatInput.addEventListener('input', autoResize);
  chatInput.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  function updateSendButton() {
    var now = Date.now();
    if (cooldownUntil > now) {
      var secs = Math.ceil((cooldownUntil - now) / 1000);
      chatSendBtn.disabled = true;
      chatSendBtn.title = 'Wait ' + secs + 's';
    } else if (isLoading) {
      chatSendBtn.disabled = true;
      chatSendBtn.title = '';
    } else {
      chatSendBtn.disabled = !chatInput.value.trim();
      chatSendBtn.title = '';
    }
  }

  function startCooldown(seconds) {
    cooldownUntil = Date.now() + (seconds * 1000);
    if (cooldownTimer) clearInterval(cooldownTimer);
    cooldownTimer = setInterval(function () {
      var now = Date.now();
      if (now >= cooldownUntil) {
        clearInterval(cooldownTimer);
        cooldownTimer = null;
        cooldownUntil = 0;
        updateSendButton();
        return;
      }
      updateSendButton();
    }, 500);
    updateSendButton();
  }

  function renderMarkdown(text) {
    if (typeof marked !== 'undefined') {
      return marked.parse(text, { breaks: true, gfm: true });
    }
    return text.replace(/\n/g, '<br>');
  }

  function addMessage(role, content) {
    chatWelcome.style.display = 'none';

    var displayRole = role === 'model' ? 'assistant' : role;
    var div = document.createElement('div');
    div.className = 'message ' + displayRole;

    var avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'U' : '\u2728';
    div.appendChild(avatar);

    var bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    if (displayRole === 'assistant') {
      bubble.innerHTML = renderMarkdown(content);
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
      var dot = document.createElement('span');
      dot.className = 'typing-dot';
      dots.appendChild(dot);
    }
    bubble.appendChild(dots);

    var label = document.createElement('span');
    label.className = 'typing-label';
    label.textContent = 'Processing';
    bubble.appendChild(label);

    typingBubble.appendChild(bubble);
    chatMessages.appendChild(typingBubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function updateTypingLabel(text) {
    if (typingBubble) {
      var label = typingBubble.querySelector('.typing-label');
      if (label) label.textContent = text;
    }
  }

  function removeTypingBubble() {
    if (typingBubble && typingBubble.parentNode) {
      typingBubble.parentNode.removeChild(typingBubble);
      typingBubble = null;
    }
  }

  function typeMessage(content, callback) {
    chatWelcome.style.display = 'none';

    var div = document.createElement('div');
    div.className = 'message assistant';

    var avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '\u2728';
    div.appendChild(avatar);

    var bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = '';
    div.appendChild(bubble);

    chatMessages.appendChild(div);

    var fullHTML = renderMarkdown(content);
    var tempDiv = document.createElement('div');
    tempDiv.innerHTML = fullHTML;
    var fullText = tempDiv.textContent || tempDiv.innerText || '';
    var totalChars = fullText.length;

    if (totalChars < 20) {
      bubble.innerHTML = fullHTML;
      chatMessages.scrollTop = chatMessages.scrollHeight;
      if (callback) callback();
      return;
    }

    var speed = Math.max(5, Math.min(20, Math.floor(2000 / totalChars)));
    speed = Math.min(speed, 25);
    var chunkSize = Math.max(1, Math.floor(totalChars / 60));

    var pos = 0;
    function typeChunk() {
      if (pos >= totalChars) {
        bubble.innerHTML = fullHTML;
        chatMessages.scrollTop = chatMessages.scrollHeight;
        if (callback) callback();
        return;
      }
      pos = Math.min(pos + chunkSize, totalChars);
      var currentText = fullText.slice(0, pos);
      bubble.innerHTML = renderMarkdown(currentText);
      chatMessages.scrollTop = chatMessages.scrollHeight;
      setTimeout(typeChunk, speed);
    }
    typeChunk();
  }

  async function sendMessage() {
    var prompt = chatInput.value.trim();
    if (!prompt || isLoading) return;

    var now = Date.now();
    if (cooldownUntil > now) {
      var secs = Math.ceil((cooldownUntil - now) / 1000);
      addMessage('assistant', '\u23F1\uFE0F You must wait ' + secs + ' seconds before sending another message. Gemini has a per-minute request limit.');
      return;
    }

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
    chatSendBtn.disabled = true;
    showTypingBubble();
    startCooldown(3);

    var timedOut = false;
    var timeoutId = setTimeout(function () {
      timedOut = true;
      removeTypingBubble();
      addMessage('assistant', '\u23F1\uFE0F The request is taking longer than expected. Gemini may be processing your request. Wait a moment before trying again.');
      isLoading = false;
      updateSendButton();
    }, 45000);

    try {
      var r = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt, session_id: currentSessionId }),
      });
      clearTimeout(timeoutId);

      if (timedOut) return;

      var data = await r.json();

      removeTypingBubble();

      if (!r.ok) {
        if (r.status === 429 && data.retry_after) {
          addMessage('assistant', '\u23F1\uFE0F ' + (data.error || 'Request limit exceeded.'));
          startCooldown(data.retry_after);
        } else {
          addMessage('assistant', '\u26A0\uFE0F ' + (data.error || 'Unknown error'));
        }
        return;
      }

      typeMessage(data.reply, function () {
        if (data.session_id) currentSessionId = data.session_id;
        loadSessionList();
      });
    } catch (e) {
      clearTimeout(timeoutId);
      if (!timedOut) {
        removeTypingBubble();
        addMessage('assistant', '\u26A0\uFE0F Server connection error.');
      }
    } finally {
      isLoading = false;
      updateSendButton();
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
        sessionList.innerHTML = '<div style="padding:20px;text-align:center;color:var(--muted);font-size:0.78rem">No sessions yet</div>';
        return;
      }

      sessionList.innerHTML = sessions.map(function (s) {
        var active = s.id === currentSessionId ? ' active' : '';
        var date = s.updated_at || s.created_at || '';
        var dateStr = date.slice(0, 16).replace('T', ' ');
        return '<div class="session-item' + active + '" data-session-id="' + s.id + '">' +
          '<div class="session-item-icon">\uD83D\uDCAC</div>' +
          '<div class="session-item-body">' +
          '<div class="session-item-title">' + escapeHtml(s.title || 'New session') + '</div>' +
          '<div class="session-item-date">' + dateStr + '</div>' +
          '</div>' +
          '<button class="session-item-delete" data-session-id="' + s.id + '" title="Delete session">\u2715</button>' +
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
