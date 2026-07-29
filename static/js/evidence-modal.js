async function openEvidence(moduleId, eventId) {
  const res = await fetch(`/api/evidences/${moduleId}/${eventId}`);
  if (!res.ok) return;
  const data = await res.json();

  document.getElementById('ev-image').src = data.capture_url || '';
  document.getElementById('ev-module').textContent = data.event.module_id || '';
  document.getElementById('ev-source').textContent = data.event.source_id || '';
  const dt = data.event.created_at || '';
  if (dt) {
    const parts = dt.split(' ');
    document.getElementById('ev-date').textContent = parts[0] || '';
    document.getElementById('ev-time').textContent = parts[1] || '';
  }
  document.getElementById('ev-type').textContent = data.event.event_type || '';
  document.getElementById('ev-description').textContent = data.event.description || '';

  const checklist = data.event.event_data?.checklist;
  const clEl = document.getElementById('ev-checklist');
  const clList = document.getElementById('ev-checklist-list');
  if (checklist && Array.isArray(checklist)) {
    clList.innerHTML = checklist.map(item =>
      `<li class="ev-checklist-${item.present ? 'ok' : 'fail'}">${item.present ? '✓' : '✗'} ${item.label}</li>`
    ).join('');
    clEl.hidden = false;
  } else {
    clEl.hidden = true;
  }

  const extraEl = document.getElementById('ev-extra');
  const extraGrid = document.getElementById('ev-extra-grid');
  if (data.extra_urls && data.extra_urls.length > 0) {
    extraGrid.innerHTML = data.extra_urls.map(url =>
      `<img src="${url}" class="ev-extra-thumb" onclick="window.open('${url}')">`
    ).join('');
    extraEl.hidden = false;
  } else {
    extraEl.hidden = true;
  }

  document.getElementById('ev-modal').hidden = false;
}

document.addEventListener('DOMContentLoaded', () => {
  const modal = document.getElementById('ev-modal');
  const closeBtn = document.getElementById('ev-close');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => { modal.hidden = true; });
  }
  if (modal) {
    modal.addEventListener('click', (e) => {
      if (e.target === modal) modal.hidden = true;
    });
  }
});

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    const modal = document.getElementById('ev-modal');
    if (modal) modal.hidden = true;
  }
});