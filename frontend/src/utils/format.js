export const fmt$ = (n) => (n == null ? "—" : `$${Math.round(n).toLocaleString()}`);
export const fmtN = (n) => (n == null ? "—" : Number(n).toLocaleString());

export function normalizeBodyType(v) {
  const s = (v ?? "").toString().trim();
  if (!s || s.toLowerCase() === "nan") return "Unknown";
  return s;
}