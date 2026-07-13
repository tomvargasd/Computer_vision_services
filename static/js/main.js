/* ══════════════════════════════════════════
   CVVision - Main JavaScript
   Maneja toggles de módulos y funciones
   ══════════════════════════════════════════ */

const API = "/api";

// ── Helpers ─────────────────────────────────
async function apiPost(url) {
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

function setDisabledFuncs(moduleId, disabled) {
  document.querySelectorAll(`.toggle-func[data-module="${moduleId}"]`).forEach(el => {
    el.disabled = disabled;
  });
}

// ── Toggle MÓDULO ────────────────────────────
document.addEventListener("change", async (e) => {
  const el = e.target;

  if (el.classList.contains("toggle-module")) {
    const moduleId = el.dataset.module;
    try {
      const data = await apiPost(`${API}/modules/${moduleId}/toggle`);
      const enabled = data.enabled;

      // Actualizar estado visual — sidebar dot
      const dot = document.querySelector(`.nav-item[href*="${moduleId}"] .status-dot`);
      if (dot) dot.classList.toggle("on", enabled);

      // Dashboard: card activa
      const card = document.getElementById(`card-${moduleId}`);
      if (card) card.classList.toggle("active", enabled);

      // Módulo page: status bar
      const bar = document.getElementById("status-bar");
      if (bar) {
        bar.classList.toggle("status-on", enabled);
        const circle = document.getElementById("status-circle");
        const text   = document.getElementById("status-text");
        if (circle) circle.classList.toggle("on", enabled);
        if (text)   text.textContent = enabled ? "Módulo activo" : "Módulo inactivo";
      }

      // Deshabilitar/habilitar funciones según estado del módulo
      setDisabledFuncs(moduleId, !enabled);

      // Sync todos los checkboxes del mismo módulo
      document.querySelectorAll(`.toggle-module[data-module="${moduleId}"]`).forEach(cb => {
        cb.checked = enabled;
      });

    } catch (err) {
      console.error("Error toggling module:", err);
      el.checked = !el.checked; // revertir UI
    }
  }

  // ── Toggle FUNCIÓN ─────────────────────────
  if (el.classList.contains("toggle-func")) {
    const moduleId = el.dataset.module;
    const funcId   = el.dataset.func;
    try {
      const data = await apiPost(`${API}/modules/${moduleId}/functions/${funcId}/toggle`);
      const enabled = data.enabled;

      // Dashboard: func-item
      const item = document.getElementById(`func-${moduleId}-${funcId}`);
      if (item) item.classList.toggle("active", enabled);

      // Módulo page: func-card + badge
      const fcard = document.getElementById(`fcard-${moduleId}-${funcId}`);
      if (fcard) fcard.classList.toggle("active", enabled);

      const badge = document.getElementById(`badge-${moduleId}-${funcId}`);
      if (badge) {
        badge.classList.toggle("badge-on", enabled);
        badge.textContent = enabled ? "Activa" : "Inactiva";
      }

    } catch (err) {
      console.error("Error toggling function:", err);
      el.checked = !el.checked;
    }
  }
});
