import React from "react";
import { getDealColor } from "../../utils/scoreColor";

export default function DealBadge({ label, score }) {
  const color = getDealColor(score);
  return (
    <span
      style={{
        background: color + "22",
        color,
        border: `1px solid ${color}44`,
        borderRadius: 3,
        fontSize: 10,
        fontFamily: "'JetBrains Mono', monospace",
        padding: "2px 6px",
        letterSpacing: "0.05em",
        textTransform: "uppercase",
        whiteSpace: "nowrap",
      }}
    >
      {label}
    </span>
  );
}