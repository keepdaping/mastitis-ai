// Small helpers shared by the dashboard pages (no build step, plain JS).

function riskBadge(level, score) {
  if (!level) return '<span class="badge text-bg-light border">no data</span>';
  const s = score == null ? "" : " " + Number(score).toFixed(2);
  return `<span class="badge risk risk-${level}">${level.toUpperCase()}${s}</span>`;
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
