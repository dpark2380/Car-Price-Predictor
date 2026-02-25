import { useState, useEffect, useMemo } from "react";
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, ScatterChart, Scatter, ZAxis
} from "recharts";

const API = "http://localhost:5001/api";
const fmt$ = (n) => n == null ? "—" : `$${Math.round(n).toLocaleString()}`;
const fmtN = (n) => n == null ? "—" : n.toLocaleString();

/**
 * 0–100 scoring:
 *   100 = best deal (most under market)
 *   50  = fair price
 *   0   = worst deal (most over market)
 */
function getDealColor(score) {
  const s = score ?? 50;
  if (s >= 90) return "#22c55e"; // Hidden Gem
  if (s >= 75) return "#84cc16"; // Great Deal
  if (s >= 60) return "#f59e0b"; // Good Deal
  if (s >= 45) return "#94a3b8"; // Fair Price
  return "#ef4444";             // Overpriced
}

function DealBadge({ label, score }) {
  const color = getDealColor(score);
  return (
    <span style={{
      background: color + "22", color, border: `1px solid ${color}44`,
      borderRadius: 3, fontSize: 10, fontFamily: "'JetBrains Mono', monospace",
      padding: "2px 6px", letterSpacing: "0.05em", textTransform: "uppercase", whiteSpace: "nowrap",
    }}>{label}</span>
  );
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{ background: "#0f1923", border: "1px solid #1e3a4a", padding: "10px 14px", borderRadius: 6 }}>
      <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 6, fontFamily: "monospace" }}>{label}</div>
      {payload.map((p) => (
        <div key={p.name} style={{ color: p.color, fontSize: 12, fontFamily: "monospace" }}>
          {p.name}: {fmt$(p.value)}
        </div>
      ))}
    </div>
  );
}

function Spinner() {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: 60, gap: 12 }}>
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#ff6b2b", animation: "pulse 1s infinite" }} />
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#ff6b2b", animation: "pulse 1s 0.2s infinite" }} />
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: "#ff6b2b", animation: "pulse 1s 0.4s infinite" }} />
    </div>
  );
}

function normalizeBodyType(v) {
  const s = (v ?? "").toString().trim();
  if (!s || s.toLowerCase() === "nan") return "Unknown";
  return s;
}

export default function CarIntelDashboard() {
  const [activeTab, setActiveTab] = useState("deals");
  const [searchQuery, setSearchQuery] = useState("");

  // 0–100 slider (min score cutoff)
  const [minScore, setMinScore] = useState(0);

  // NEW: body type filter ("" = all)
  const [bodyFilter, setBodyFilter] = useState("");

  const [selectedDeal, setSelectedDeal] = useState(null);
  const [tickerIdx, setTickerIdx] = useState(0);

  const [deals, setDeals] = useState([]);
  const [popular, setPopular] = useState([]);
  const [stats, setStats] = useState({});
  const [trends, setTrends] = useState({ data: [], makes: [] });
  const [listings, setListings] = useState([]);
  const [loading, setLoading] = useState({ deals: true, popular: true, stats: true, trends: true, listings: true });
  const [benchmark, setBenchmark] = useState(null);
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [salesStats, setSalesStats] = useState(null);
  const [salesLoading, setSalesLoading] = useState(false);
  const [apiError, setApiError] = useState(false);

  useEffect(() => {
    const load = async (key, url, setter) => {
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`${res.status}`);
        const data = await res.json();
        setter(data);
      } catch (e) {
        console.error(`Failed to load ${key}:`, e);
        if (key === "stats") setApiError(true);
      } finally {
        setLoading((l) => ({ ...l, [key]: false }));
      }
    };

    // Updated scoring endpoint param: min_score (0–100)
    load("deals", `${API}/deals?limit=500&min_score=0`, setDeals);
    load("popular", `${API}/popular?limit=20`, setPopular);
    load("stats", `${API}/stats`, setStats);
    load("trends", `${API}/trends`, setTrends);
    load("listings", `${API}/listings`, setListings);
  }, []);

  // NEW: dynamic body type options from deals
  const bodyOptions = useMemo(() => {
    const set = new Set();
    (deals || []).forEach((d) => set.add(normalizeBodyType(d.body_type)));
    const opts = Array.from(set).sort((a, b) => a.localeCompare(b));
    // Put Unknown last to keep UX clean
    const unknownIdx = opts.indexOf("Unknown");
    if (unknownIdx >= 0) {
      opts.splice(unknownIdx, 1);
      opts.push("Unknown");
    }
    return opts;
  }, [deals]);

  // Ticker: show top deals (highest score first), rotate top 10
  const topTickerDeals = useMemo(() => {
    return (deals || [])
      .filter((d) => d.deal_score != null)
      .slice()
      .sort((a, b) => (b.deal_score ?? 0) - (a.deal_score ?? 0));
  }, [deals]);

  useEffect(() => {
    if (!topTickerDeals.length) return;
    const id = setInterval(() => setTickerIdx((i) => (i + 1) % Math.min(topTickerDeals.length, 10)), 3000);
    return () => clearInterval(id);
  }, [topTickerDeals]);

  const filteredDeals = useMemo(
    () =>
      (deals || [])
        .filter((d) => d.deal_score != null)
        .filter((d) => (d.deal_score ?? 0) >= minScore)
        .filter(
          (d) =>
            !searchQuery ||
            d.make?.toLowerCase().includes(searchQuery.toLowerCase()) ||
            d.model?.toLowerCase().includes(searchQuery.toLowerCase())
        )
        // NEW: body type filter
        .filter((d) => {
          if (!bodyFilter) return true;
          const bt = normalizeBodyType(d.body_type);
          return bt.toLowerCase() === bodyFilter.toLowerCase();
        })
        // Best first
        .sort((a, b) => (b.deal_score ?? 0) - (a.deal_score ?? 0)),
    [deals, searchQuery, minScore, bodyFilter]
  );

  const makeDistribution = useMemo(() => {
    const counts = {};
    filteredDeals.forEach((d) => {
      if (d.make) counts[d.make] = (counts[d.make] || 0) + 1;
    });
    return Object.entries(counts)
      .map(([make, count]) => ({ make, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);
  }, [filteredDeals]);

  const currentTicker = topTickerDeals[tickerIdx] || {};
  const trendColors = ["#ff6b2b", "#f59e0b", "#3b82f6", "#22c55e", "#a855f7"];
  const isLoading = Object.values(loading).some(Boolean);

  const handleSelectDeal = (deal) => {
    const next = selectedDeal?.listing_id === deal.listing_id ? null : deal;
    setSelectedDeal(next);
    if (!next) {
      setBenchmark(null);
      setSalesStats(null);
      return;
    }

    setBenchmark(null);
    setBenchmarkLoading(true);
    fetch(
      `${API}/benchmark?year=${next.year}&make=${encodeURIComponent(next.make)}&model=${encodeURIComponent(next.model)}&mileage=${next.mileage}`
    )
      .then((r) => r.json())
      .then(setBenchmark)
      .catch(() => setBenchmark(null))
      .finally(() => setBenchmarkLoading(false));

    setSalesStats(null);
    setSalesLoading(true);
    fetch(`${API}/sales-stats?make=${encodeURIComponent(next.make)}&model=${encodeURIComponent(next.model)}`)
      .then((r) => r.json())
      .then(setSalesStats)
      .catch(() => setSalesStats(null))
      .finally(() => setSalesLoading(false));
  };

  return (
    <div style={{ minHeight: "100vh", background: "#080e14", color: "#e2e8f0", fontFamily: "'DM Sans', sans-serif", position: "relative" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500;700&family=Bebas+Neue&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; background: #0f1923; }
        ::-webkit-scrollbar-thumb { background: #1e3a4a; border-radius: 2px; }
        .tab-btn { transition: all 0.2s; cursor: pointer; }
        .tab-btn:hover { color: #ff6b2b !important; }
        .deal-row { transition: background 0.15s; cursor: pointer; }
        .deal-row:hover { background: #0f1923 !important; }
        .pop-row { transition: background 0.15s; }
        .pop-row:hover { background: #0a1520 !important; }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
        @keyframes fadeUp { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
        @keyframes slideIn { from { transform:translateY(8px); opacity:0; } to { transform:translateY(0); opacity:1; } }
        .stat-card { animation: fadeUp 0.5s ease both; }
      `}</style>

      <div
        style={{
          position: "fixed",
          inset: 0,
          pointerEvents: "none",
          backgroundImage:
            "linear-gradient(#1e3a4a11 1px,transparent 1px),linear-gradient(90deg,#1e3a4a11 1px,transparent 1px)",
          backgroundSize: "40px 40px",
          zIndex: 0,
        }}
      />

      {apiError && (
        <div
          style={{
            background: "#ef444422",
            border: "1px solid #ef4444",
            color: "#fca5a5",
            padding: "10px 24px",
            fontFamily: "monospace",
            fontSize: 12,
            textAlign: "center",
            position: "relative",
            zIndex: 20,
          }}
        >
          ⚠ Failed to connect to API at {API}. Make sure api.py is running.
        </div>
      )}

      {/* Header */}
      <header
        style={{
          position: "relative",
          zIndex: 10,
          borderBottom: "1px solid #1e3a4a",
          background: "#080e14ee",
          backdropFilter: "blur(12px)",
          padding: "0 32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          height: 60,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 28, letterSpacing: "0.1em", color: "#ff6b2b", lineHeight: 1 }}>
            CARINTEL
          </div>
          <div style={{ width: 1, height: 24, background: "#1e3a4a" }} />
          <div style={{ fontFamily: "monospace", fontSize: 11, color: "#475569", letterSpacing: "0.1em" }}>MARKET INTELLIGENCE</div>
        </div>

        {currentTicker.make && (
          <div style={{ display: "flex", alignItems: "center", gap: 10, background: "#0f1923", border: "1px solid #1e3a4a", borderRadius: 4, padding: "6px 14px" }}>
            <div style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e", animation: "pulse 2s infinite" }} />
            <span style={{ fontFamily: "monospace", fontSize: 11, color: "#64748b" }}>BEST DEAL:</span>
            <span style={{ fontFamily: "monospace", fontSize: 11, color: "#e2e8f0" }}>{currentTicker.year} {currentTicker.make} {currentTicker.model}</span>
            <span style={{ fontFamily: "monospace", fontSize: 11, color: "#ff6b2b", fontWeight: 700 }}>{fmt$(currentTicker.price)}</span>
            {currentTicker.savings > 0 && (
              <span style={{ fontFamily: "monospace", fontSize: 10, color: "#22c55e" }}>↓{fmt$(currentTicker.savings)} below market</span>
            )}
          </div>
        )}

        <div style={{ display: "flex", gap: 24, alignItems: "center" }}>
          {[
            { label: "LISTINGS", value: isLoading ? "—" : fmtN(stats.active_listings) },
            { label: "MAKES", value: isLoading ? "—" : stats.makes },
            { label: "AVG PRICE", value: isLoading ? "—" : fmt$(stats.avg_price) },
          ].map((s) => (
            <div key={s.label} style={{ textAlign: "right" }}>
              <div style={{ fontFamily: "monospace", fontSize: 9, color: "#475569", letterSpacing: "0.1em" }}>{s.label}</div>
              <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 13, color: "#e2e8f0", fontWeight: 700 }}>{s.value}</div>
            </div>
          ))}
        </div>
      </header>

      {/* Nav */}
      <nav style={{ position: "relative", zIndex: 10, borderBottom: "1px solid #1e3a4a", padding: "0 32px", display: "flex", background: "#080e14cc" }}>
        {[
          { id: "deals", label: "DEAL FINDER" },
          { id: "popular", label: "MARKET MOVERS" },
          { id: "trends", label: "PRICE TRENDS" },
          { id: "scatter", label: "PRICE vs MILEAGE" },
        ].map((tab) => (
          <button
            key={tab.id}
            className="tab-btn"
            onClick={() => setActiveTab(tab.id)}
            style={{
              background: "none",
              border: "none",
              borderBottom: activeTab === tab.id ? "2px solid #ff6b2b" : "2px solid transparent",
              color: activeTab === tab.id ? "#ff6b2b" : "#475569",
              fontFamily: "monospace",
              fontSize: 11,
              letterSpacing: "0.12em",
              padding: "14px 20px",
              marginBottom: -1,
              fontWeight: activeTab === tab.id ? 700 : 400,
            }}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main style={{ position: "relative", zIndex: 1, padding: "28px 32px", maxWidth: 1400, margin: "0 auto" }}>
        {/* DEALS */}
        {activeTab === "deals" && (
          <div style={{ animation: "fadeUp 0.3s ease" }}>
            <div style={{ display: "flex", gap: 16, marginBottom: 24, alignItems: "center" }}>
              <div style={{ flex: 1, position: "relative" }}>
                <input
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Filter by make or model..."
                  style={{
                    width: "100%",
                    background: "#0f1923",
                    border: "1px solid #1e3a4a",
                    borderRadius: 4,
                    color: "#e2e8f0",
                    padding: "9px 14px 9px 36px",
                    fontFamily: "monospace",
                    fontSize: 12,
                    outline: "none",
                  }}
                />
                <span style={{ position: "absolute", left: 12, top: "50%", transform: "translateY(-50%)", color: "#475569", fontSize: 14 }}>⌕</span>
              </div>

              {/* NEW: Body Type dropdown */}
              <div style={{ display: "flex", alignItems: "center", gap: 10, background: "#0f1923", border: "1px solid #1e3a4a", borderRadius: 4, padding: "9px 12px" }}>
                <span style={{ fontFamily: "monospace", fontSize: 11, color: "#64748b" }}>BODY</span>
                <select
                  value={bodyFilter}
                  onChange={(e) => setBodyFilter(e.target.value)}
                  style={{
                    background: "transparent",
                    border: "none",
                    color: "#e2e8f0",
                    fontFamily: "monospace",
                    fontSize: 12,
                    outline: "none",
                    minWidth: 140,
                    cursor: "pointer",
                  }}
                >
                  <option value="">All</option>
                  {bodyOptions.map((bt) => (
                    <option key={bt} value={bt}>{bt}</option>
                  ))}
                </select>
              </div>

              <div style={{ display: "flex", alignItems: "center", gap: 12, background: "#0f1923", border: "1px solid #1e3a4a", borderRadius: 4, padding: "9px 16px" }}>
                <span style={{ fontFamily: "monospace", fontSize: 11, color: "#64748b" }}>MIN SCORE</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={minScore}
                  onChange={(e) => setMinScore(+e.target.value)}
                  style={{ accentColor: "#ff6b2b", width: 110 }}
                />
                <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 13, color: "#ff6b2b", fontWeight: 700, minWidth: 28 }}>{minScore}</span>
              </div>

              <div style={{ fontFamily: "monospace", fontSize: 11, color: "#475569" }}>{filteredDeals.length} RESULTS</div>
            </div>

            <div style={{ background: "#0a1218", border: "1px solid #1e3a4a", borderRadius: 6, overflow: "hidden" }}>
              <div style={{ display: "grid", gridTemplateColumns: "44px 1fr 90px 110px 110px 100px 80px 90px", padding: "10px 20px", borderBottom: "1px solid #1e3a4a", background: "#0f1923" }}>
                {["SCORE", "VEHICLE", "PRICE", "MARKET VALUE", "SAVINGS", "MILEAGE", "STATE", "DEAL"].map((h) => (
                  <div key={h} style={{ fontFamily: "monospace", fontSize: 9, color: "#475569", letterSpacing: "0.1em" }}>{h}</div>
                ))}
              </div>

              {loading.deals ? (
                <Spinner />
              ) : filteredDeals.length === 0 ? (
                <div style={{ padding: 40, textAlign: "center", color: "#475569", fontFamily: "monospace", fontSize: 12 }}>
                  No deals match your filters
                </div>
              ) : (
                filteredDeals.map((deal, i) => (
                  <div
                    key={i}
                    className="deal-row"
                    onClick={() => handleSelectDeal(deal)}
                    style={{
                      display: "grid",
                      gridTemplateColumns: "44px 1fr 90px 110px 110px 100px 80px 90px",
                      padding: "13px 20px",
                      borderBottom: "1px solid #0f1923",
                      background: selectedDeal?.listing_id === deal.listing_id ? "#0f1923" : "transparent",
                      alignItems: "center",
                    }}
                  >
                    <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 13, fontWeight: 700, color: getDealColor(deal.deal_score) }}>
                      {deal.deal_score?.toFixed(1)}
                    </div>

                    <div>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        {deal.url ? (
                          <a
                            href={deal.url}
                            target="_blank"
                            rel="noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            style={{ fontSize: 13, fontWeight: 500, color: "#e2e8f0", textDecoration: "none" }}
                            onMouseEnter={(e) => (e.currentTarget.style.textDecoration = "underline")}
                            onMouseLeave={(e) => (e.currentTarget.style.textDecoration = "none")}
                          >
                            {deal.year} {deal.make} {deal.model}
                          </a>
                        ) : (
                          <div style={{ fontSize: 13, fontWeight: 500, color: "#e2e8f0" }}>
                            {deal.year} {deal.make} {deal.model}
                          </div>
                        )}

                        {deal.url && (
                          <a
                            href={deal.url}
                            target="_blank"
                            rel="noreferrer"
                            onClick={(e) => e.stopPropagation()}
                            title="Open original listing"
                            style={{
                              fontFamily: "monospace",
                              fontSize: 12,
                              color: "#64748b",
                              textDecoration: "none",
                              border: "1px solid #1e3a4a",
                              padding: "1px 6px",
                              borderRadius: 4,
                            }}
                          >
                            ↗
                          </a>
                        )}
                      </div>

                      {/* NEW: show body type */}
                      <div style={{ fontSize: 11, color: "#475569", marginTop: 2 }}>
                        {normalizeBodyType(deal.body_type)} · {deal.trim}
                      </div>
                    </div>

                    <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 13, fontWeight: 700, color: "#e2e8f0" }}>{fmt$(deal.price)}</div>
                    <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12, color: "#64748b" }}>{fmt$(deal.predicted_price)}</div>
                    <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12, color: deal.savings >= 0 ? "#22c55e" : "#ef4444", fontWeight: 600 }}>
                      {deal.savings >= 0 ? "↓ " : "↑ "}
                      {fmt$(Math.abs(deal.savings))}
                    </div>
                    <div style={{ fontFamily: "monospace", fontSize: 11, color: "#64748b" }}>{fmtN(deal.mileage)} mi</div>
                    <div style={{ fontFamily: "monospace", fontSize: 12, color: "#94a3b8" }}>{deal.location_state}</div>
                    <DealBadge label={deal.deal_label} score={deal.deal_score} />
                  </div>
                ))
              )}
            </div>

            {/* Selected deal modal unchanged below... */}
            {selectedDeal && (
              <div style={{ marginTop: 16, background: "#0a1218", border: "1px solid #ff6b2b44", borderRadius: 6, padding: "20px 24px", animation: "slideIn 0.2s ease" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                  <div>
                    <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 24, color: "#ff6b2b", letterSpacing: "0.05em" }}>
                      {selectedDeal.year} {selectedDeal.make} {selectedDeal.model} — {selectedDeal.trim}
                    </div>
                    <div style={{ display: "flex", gap: 32, marginTop: 14, flexWrap: "wrap" }}>
                      {[
                        { label: "Listed Price", value: fmt$(selectedDeal.price), color: "#e2e8f0" },
                        { label: "Market Value", value: fmt$(selectedDeal.predicted_price), color: "#64748b" },
                        { label: "You Save", value: fmt$(selectedDeal.savings), color: selectedDeal.savings >= 0 ? "#22c55e" : "#ef4444" },
                        { label: "Mileage", value: `${fmtN(selectedDeal.mileage)} mi`, color: "#94a3b8" },
                        { label: "Deal Score", value: selectedDeal.deal_score?.toFixed(1), color: getDealColor(selectedDeal.deal_score) },
                        { label: "Location", value: `${selectedDeal.location_city || ""} ${selectedDeal.location_state || ""}`.trim(), color: "#94a3b8" },
                        { label: "Accidents", value: selectedDeal.accident_count ?? 0, color: selectedDeal.accident_count > 0 ? "#ef4444" : "#22c55e" },
                        { label: "Days Listed", value: `${selectedDeal.days_listed}d`, color: "#94a3b8" },
                      ].map(({ label, value, color }) => (
                        <div key={label}>
                          <div style={{ fontFamily: "monospace", fontSize: 9, color: "#475569", letterSpacing: "0.1em", marginBottom: 4 }}>{label}</div>
                          <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 16, fontWeight: 700, color }}>{value}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-end" }}>
                    {selectedDeal.url && !selectedDeal.url.includes("mock") && (
                      <a
                        href={selectedDeal.url}
                        target="_blank"
                        rel="noreferrer"
                        style={{ background: "#ff6b2b", color: "#000", borderRadius: 4, padding: "6px 14px", fontFamily: "monospace", fontSize: 11, fontWeight: 700, textDecoration: "none", letterSpacing: "0.05em" }}
                      >
                        VIEW LISTING →
                      </a>
                    )}
                    <button
                      onClick={() => {
                        setSelectedDeal(null);
                        setBenchmark(null);
                        setSalesStats(null);
                      }}
                      style={{ background: "none", border: "1px solid #1e3a4a", color: "#475569", borderRadius: 4, padding: "6px 12px", cursor: "pointer", fontFamily: "monospace", fontSize: 11 }}
                    >
                      CLOSE ✕
                    </button>
                  </div>
                </div>

                <div style={{ marginTop: 20 }}>
                  <div style={{ fontFamily: "monospace", fontSize: 9, color: "#475569", letterSpacing: "0.1em", marginBottom: 8 }}>PRICE vs MARKET VALUE</div>
                  <div style={{ position: "relative", height: 8, background: "#1e3a4a", borderRadius: 4 }}>
                    <div
                      style={{
                        position: "absolute",
                        left: 0,
                        top: 0,
                        height: "100%",
                        borderRadius: 4,
                        background: getDealColor(selectedDeal.deal_score),
                        width: `${Math.min(100, (selectedDeal.price / selectedDeal.predicted_price) * 100)}%`,
                      }}
                    />
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6 }}>
                    <span style={{ fontFamily: "monospace", fontSize: 10, color: "#94a3b8" }}>Listed {fmt$(selectedDeal.price)}</span>
                    <span style={{ fontFamily: "monospace", fontSize: 10, color: "#64748b" }}>Market {fmt$(selectedDeal.predicted_price)}</span>
                  </div>
                </div>

                <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
                  <div style={{ background: "#080e14", border: "1px solid #1e3a4a", borderRadius: 4, padding: "14px 16px" }}>
                    <div style={{ fontFamily: "monospace", fontSize: 9, color: "#475569", letterSpacing: "0.1em", marginBottom: 10 }}>MARKETCHECK PRICE BENCHMARK</div>
                    {benchmarkLoading ? (
                      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#475569" }}>Fetching benchmark...</div>
                    ) : benchmark && !benchmark.error ? (
                      <div style={{ display: "flex", gap: 24 }}>
                        {[
                          { label: "MC Predicted", value: fmt$(benchmark.mc_predicted_price), color: "#3b82f6" },
                          { label: "Our Model", value: fmt$(selectedDeal.predicted_price), color: "#ff6b2b" },
                          { label: "Listed Price", value: fmt$(selectedDeal.price), color: "#e2e8f0" },
                        ].map(({ label, value, color }) => (
                          <div key={label}>
                            <div style={{ fontFamily: "monospace", fontSize: 9, color: "#475569", letterSpacing: "0.08em", marginBottom: 4 }}>{label}</div>
                            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 14, fontWeight: 700, color }}>{value}</div>
                          </div>
                        ))}
                        {benchmark.mc_price_low && benchmark.mc_price_high && (
                          <div>
                            <div style={{ fontFamily: "monospace", fontSize: 9, color: "#475569", letterSpacing: "0.08em", marginBottom: 4 }}>MC RANGE</div>
                            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12, color: "#64748b" }}>
                              {fmt$(benchmark.mc_price_low)} – {fmt$(benchmark.mc_price_high)}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#475569" }}>No benchmark available</div>
                    )}
                  </div>

                  <div style={{ background: "#080e14", border: "1px solid #1e3a4a", borderRadius: 4, padding: "14px 16px" }}>
                    <div style={{ fontFamily: "monospace", fontSize: 9, color: "#475569", letterSpacing: "0.1em", marginBottom: 10 }}>MARKET SALES DATA (LAST 90 DAYS)</div>
                    {salesLoading ? (
                      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#475569" }}>Fetching sales data...</div>
                    ) : salesStats && !salesStats.error ? (
                      <div style={{ display: "flex", gap: 24 }}>
                        {[
                          { label: "Total Sales", value: fmtN(salesStats.total_sales), color: "#e2e8f0" },
                          { label: "Median DOM", value: salesStats.dom_median ? `${salesStats.dom_median}d` : "—", color: "#f59e0b" },
                          { label: "CPO Sales", value: fmtN(salesStats.cpo_sales), color: "#94a3b8" },
                        ].map(({ label, value, color }) => (
                          <div key={label}>
                            <div style={{ fontFamily: "monospace", fontSize: 9, color: "#475569", letterSpacing: "0.08em", marginBottom: 4 }}>{label}</div>
                            <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 14, fontWeight: 700, color }}>{value}</div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div style={{ fontFamily: "monospace", fontSize: 11, color: "#475569" }}>No sales data available</div>
                    )}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* POPULAR / TRENDS / SCATTER remain the same as your current file */}
        {/* You can keep the rest of your file unchanged below this point. */}
      </main>

      <footer style={{ position: "relative", zIndex: 1, borderTop: "1px solid #1e3a4a", padding: "16px 32px", display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: 40 }}>
        <div style={{ fontFamily: "monospace", fontSize: 10, color: "#1e3a4a" }}>CARINTEL v2.0 — XGBOOST PRICE MODEL</div>
        <div style={{ fontFamily: "monospace", fontSize: 10, color: "#1e3a4a" }}>
          {stats.last_updated ? `LAST UPDATED: ${new Date(stats.last_updated).toLocaleTimeString()}` : ""}
        </div>
      </footer>
    </div>
  );
}