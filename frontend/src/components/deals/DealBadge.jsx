import React from "react";
import { getDealColor, starsDisplay } from "../../utils/scoreColor";

export default function DealBadge({ score }) {
  const color = getDealColor(score);
  return (
    <span
      style={{
        color,
        fontSize: 15,
        letterSpacing: "0.05em",
        fontFamily: "inherit",
      }}
    >
      {starsDisplay(score)}
    </span>
  );
}
