// ── Analytics shared JS (v3.0) ─────────────────────────────────────────────

async function pollActivePipelines() {
  const r = await fetch('/api/status');
  if (!r.ok) return;
  const data = await r.json();
  let active = 0;
  Object.entries(data).forEach(([modId, mod]) => {
    const kpiEl = document.getElementById(`kpi-${modId}`);
    const statusEl = document.getElementById(`status-${modId}`);
    if (kpiEl) {
      const srcCount = mod.sources || 0;
      kpiEl.textContent = `${srcCount} sources`;
    }
    if (statusEl) {
      const dot = statusEl.querySelector('.status-dot');
      if (dot) {
        dot.classList.toggle('on', mod.enabled);
      }
    }
    if (mod.enabled && mod.sources > 0) active++;
  });
  const el = document.getElementById('active-pipelines');
  if (el) el.textContent = active;
}

document.addEventListener('DOMContentLoaded', () => {
  pollActivePipelines();
  setInterval(pollActivePipelines, 3000);
});