// Trend and summary helpers used by the dashboard, reports, and insights.
// They describe patterns only. They make no causal claims.

import { CalculationInputError } from "./bmi";

export type Point = { x: number; y: number };

/**
 * Least-squares slope (y change per unit x). Returns null when there are fewer
 * than 2 points or x has no variance. A null is shown as "not enough data".
 * For weight, x is in days, so slope × 7 gives the weekly change.
 */
export function linearSlope(points: Point[]): number | null {
  const n = points.length;
  if (n < 2) return null;
  const meanX = points.reduce((s, p) => s + p.x, 0) / n;
  const meanY = points.reduce((s, p) => s + p.y, 0) / n;
  let num = 0;
  let den = 0;
  for (const p of points) {
    num += (p.x - meanX) * (p.y - meanY);
    den += (p.x - meanX) ** 2;
  }
  if (den === 0) return null;
  return num / den;
}

/** Trailing moving average. Returns one value per input index from the window size onward. */
export function movingAverage(values: number[], window: number): Array<number | null> {
  if (!Number.isInteger(window) || window < 1) throw new CalculationInputError("window must be a positive integer");
  return values.map((_, i) => {
    if (i + 1 < window) return null;
    const slice = values.slice(i + 1 - window, i + 1);
    return slice.reduce((s, v) => s + v, 0) / window;
  });
}

/** Completion rate in percent for planned versus completed items. Null when nothing was planned. */
export function completionRate(completed: number, planned: number): number | null {
  if (planned < 0 || completed < 0) throw new CalculationInputError("counts must be >= 0");
  if (planned === 0) return null;
  return Math.round((Math.min(completed, planned) / planned) * 100);
}

/** Weekly rate of change from daily-indexed points. Null when slope is unavailable. */
export function weeklyChange(points: Point[]): number | null {
  const slope = linearSlope(points);
  return slope === null ? null : Math.round(slope * 7 * 100) / 100;
}
