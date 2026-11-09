// Small helpers shared by the dashboard pages (no build step, plain JS).

// Same output as the risk_badge / alert_badge macros in templates/_macros.html.
function riskBadge(level, score) {
  if (!level) return '<span class="badge text-bg-light border">no data</span>';
  // Score capped at 0.99: it is a model output, not a probability
  const s = score == null ? "" : ` <span class="risk-score">${Math.min(Number(score), 0.99).toFixed(2)}</span>`;
  return `<span class="text-nowrap" title="Risk score (model output, not a probability)">` +
         `<span class="badge risk risk-${level}">${level.toUpperCase()}</span>${s}</span>`;
}

function alertBadge(state) {
  if (state === "open_alert") return '<a href="/alerts" class="badge alert-open text-decoration-none">open alert</a>';
  if (state === "recovering") return '<a href="/alerts" class="badge alert-recovering text-decoration-none">recovering</a>';
  if (state === "treated") return '<span class="badge alert-closed">treated</span>';
  if (state === "false_alarm") return '<span class="badge alert-closed">false alarm</span>';
  if (state === "watch") return '<span class="badge alert-watch">watch – 1 high reading</span>';
  return "";
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text == null ? "" : String(text);
  return div.innerHTML;
}

// Shared Chart.js defaults: thin lines, small points, hover tooltip on all series at once.
if (window.Chart) {
  Chart.defaults.font.size = 13;
  Chart.defaults.color = "#52514e";
  Chart.defaults.borderColor = "#e8e7e3";
  Chart.defaults.elements.line.borderWidth = 2;
  Chart.defaults.elements.point.radius = 0;
  Chart.defaults.elements.point.hoverRadius = 5;
  Chart.defaults.interaction = { mode: "index", intersect: false };
  Chart.defaults.maintainAspectRatio = false;
}
