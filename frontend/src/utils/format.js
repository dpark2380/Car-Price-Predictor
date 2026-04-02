export const fmt$ = (n) => (n == null ? "—" : `$${Math.round(n).toLocaleString()}`);
export const fmtN = (n) => (n == null ? "—" : Number(n).toLocaleString());

const BODY_ALIASES = {
  "car van": "Van",
  "minivan": "Van",
  "mini van": "Van",
  "cargo van": "Van",
  "passenger van": "Van",
  "pickup": "Truck",
  "pickup truck": "Truck",
  "sport utility": "SUV",
  "sport utility vehicle": "SUV",
  "crossover": "SUV",
  "mini mpv": "Van",
  "combi": "Van",
  "cutaway": "Truck",
  "chassis cab": "Truck",
  "targa": "Convertible",
};

export function normalizeBodyType(v) {
  const s = (v ?? "").toString().trim();
  if (!s || s.toLowerCase() === "nan") return "Unknown";
  return BODY_ALIASES[s.toLowerCase()] ?? s;
}