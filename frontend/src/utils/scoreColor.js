/**
 * 0–100 scoring:
 *   100 = best deal (most under market)
 *   50  = fair price
 *   0   = worst deal (most over market)
 */
export function getDealColor(score) {
  const s = score ?? 50;
  if (s >= 90) return "#22c55e"; // Hidden Gem
  if (s >= 75) return "#84cc16"; // Great Deal
  if (s >= 60) return "#f59e0b"; // Good Deal
  if (s >= 45) return "#94a3b8"; // Fair Price
  return "#ef4444"; // Overpriced
}