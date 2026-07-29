# Fix Analytics Icons — Plan de ejecución

## 1. CSS: Reemplazar `.icon-inline` por `.analytics-kpi-icon`

**Archivo:** `static/css/style.css`

**Eliminar** (línea ~1753):
```css
.icon-inline { width: 14px; height: 14px; stroke: currentColor; stroke-width: 2; fill: none; stroke-linecap: round; stroke-linejoin: round; vertical-align: middle; margin-right: 4px; flex-shrink: 0; }
```

**Agregar al final del archivo:**
```css
.analytics-kpi-icon {
  width: 16px; height: 16px;
  stroke: currentColor;
  fill: none;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
  margin-right: 6px;
  flex-shrink: 0;
  vertical-align: middle;
}
```

## 2. Template: `analytics.html`

| Buscar | Reemplazar con |
|---|---|
| `<svg class="icon-inline" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> En vivo ahora:` | `<svg class="analytics-kpi-icon" viewBox="0 0 24 24"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg> En vivo ahora:` |

## 3. Template: `analytics_personas.html`

Cada `<svg class="icon-inline"` → `<svg class="analytics-kpi-icon"` con estos paths:

| Label actual | Path actual | Reemplazar path con |
|---|---|---|
| Personas en el área | (inventado) | `<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>` |
| Entradas | (inventado) | `<path d="M15 3h4a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-4"/><polyline points="10 17 15 12 10 7"/><line x1="15" y1="12" x2="3" y2="12"/>` |
| Salidas | (inventado) | `<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>` |
| Permanencia máx | (inventado) | `<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>` |

## 4. Template: `analytics_armas.html`

| Label | Path Lucide |
|---|---|
| Alertas de arma | `<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>` (shield — mantener ese, es correcto) |
| Armas blancas | `<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>` (alert-triangle) |
| Armas de fuego | `<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>` (flame) |
| Capturas rostro | `<path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/>` (camera) |

## 5. Template: `analytics_acciones.html`

Reemplazar TODOS los SVGs con `class="analytics-kpi-icon"` manteniendo el mismo path de alert-triangle (actualmente tienen paths extraños), o mejor mantener el path actual de alert-triangle si es correcto pero solo cambiar la clase.

## 6. Template: `analytics_troncos.html`

| Label | Path Lucide |
|---|---|
| Conteo total | `<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>` (layers) |
| Hoy | `<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>` (clock) |
| Promedio diario | `<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>` (trending-up) |

## 7. Template: `analytics_carga_descarga.html`

| Label | Path |
|---|---|
| Entradas | log-in (igual que personas) |
| Salidas | log-out (igual que personas) |
| Total | `<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>` (activity) |

## 8. Template: `analytics_epp.html`

| Label | Path |
|---|---|
| Detecciones totales | shield |
| Protegidos | `<path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z"/>` (heart) |
| Sin EPP | `<path d="M19.69 14a6.9 6.9 0 0 0 .31-2V5l-8-3-3.16 1.18"/><path d="M4.73 4.73L4 5v7c0 6 8 10 8 10a20.29 20.29 0 0 0 5.62-4.38"/><line x1="1" y1="1" x2="23" y2="23"/>` (shield-off) |
| Personas activas | activity |

## 9. Template: `analytics_vehiculos.html`

| Label | Path |
|---|---|
| Entradas | log-in |
| Salidas | log-out |
| Placas detectadas | `<rect x="1" y="4" width="22" height="16" rx="2" ry="2"/><line x1="1" y1="10" x2="23" y2="10"/>` (credit-card) |

## 10. Template: `analytics_reglamento.html`

| Label | Path |
|---|---|
| Detecciones totales | shield |
| Elementos faltantes | ban |
| Elementos presentes | check |

## 11. Template: `analytics_pallets.html`, `analytics_cajas.html`, `analytics_smoke.html`, `analytics_tanques_gas.html`

Reemplazar todos los paths de iconos inventados por el package/layers/alert-triangle Lucide según corresponda.

## 12. Template: `module.html`

En la línea ~330 (JS template literal), reemplazar:
```js
    ? '📁 ' + src.path.split("/").pop()
```
por:
```js
    ? '<svg class="analytics-kpi-icon" viewBox="0 0 24 24"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/></svg> ' + src.path.split("/").pop()
```

## 13. Live templates

En `armas_live.html`, `acciones_live.html`, `tanques_gas_live.html`:
- Reemplazar cualquier `<svg class="icon-inline"` → `<svg class="analytics-kpi-icon"`
- Reemplazar emojis `⚠️` `✅` `❌` `⏳` en JS template strings con SVGs inline

## Resumen

Son ~13 templates + 1 CSS. El cambio es mecánico: `icon-inline` → `analytics-kpi-icon` y paths de iconos inventados → paths Lucide/Feather reales de 16×16.
