/**
 * 0–100 scoring (internal):
 *   100 = best deal  →  5 stars
 *   50  = fair price →  2-3 stars
 *   0   = worst deal →  1 star
 */
export function scoreToStars(score) {
  const s = score ?? 50;
  if (s >= 90) return 5;
  if (s >= 75) return 4;
  if (s >= 60) return 3;
  if (s >= 45) return 2;
  return 1;
}

export function starsDisplay(score) {
  const n = scoreToStars(score);
  return "★".repeat(n) + "☆".repeat(5 - n);
}

export function getDealColor(score) {
  const s = score ?? 50;
  if (s >= 90) return "#22c55e"; // 5 stars
  if (s >= 75) return "#84cc16"; // 4 stars
  if (s >= 60) return "#f59e0b"; // 3 stars
  if (s >= 45) return "#94a3b8"; // 2 stars
  return "#ef4444";              // 1 star
}
