// Module F: sleep duration, consistency, and recommended-range check.
// Registry: SLEEP-AASM-SRS-2015 (see sources.ts).
// Does not diagnose sleep disorders.

import { CalculationInputError } from "./bmi";

const MINUTES_PER_DAY = 24 * 60;

/** Parses "HH:mm" (24-hour) into minutes after midnight. */
export function parseClockMinutes(clock: string): number {
  const match = /^([01]\d|2[0-3]):([0-5]\d)$/.exec(clock);
  if (!match) throw new CalculationInputError(`time must be HH:mm (24-hour), received "${clock}"`);
  return Number(match[1]) * 60 + Number(match[2]);
}

/**
 * Sleep duration in minutes. Handles sessions that cross midnight.
 * Identical bedtime and wake time is rejected because it is ambiguous (0 h or 24 h).
 */
export function sleepDurationMinutes(bedtime: string, wakeTime: string): number {
  const bed = parseClockMinutes(bedtime);
  const wake = parseClockMinutes(wakeTime);
  if (bed === wake) throw new CalculationInputError("bedtime and wake time cannot be identical");
  return wake > bed ? wake - bed : wake + MINUTES_PER_DAY - bed;
}

/**
 * Consistency = circular standard deviation of bedtimes, in minutes.
 * Bedtimes are circular: 23:50 and 00:10 are 20 minutes apart, not 1,420.
 * Lower is more consistent. Needs at least 2 nights.
 */
export function bedtimeConsistencyMinutes(bedtimes: string[]): number | null {
  if (bedtimes.length < 2) return null;
  let sumSin = 0;
  let sumCos = 0;
  for (const t of bedtimes) {
    const angle = (parseClockMinutes(t) / MINUTES_PER_DAY) * 2 * Math.PI;
    sumSin += Math.sin(angle);
    sumCos += Math.cos(angle);
  }
  const n = bedtimes.length;
  const r = Math.sqrt(sumSin * sumSin + sumCos * sumCos) / n;
  // r ≈ 0 means the bedtimes are spread evenly around the clock: no meaningful SD.
  // The epsilon absorbs floating-point noise (exact zero is rare in practice).
  if (r < 1e-9) return null;
  const sdRadians = Math.sqrt(-2 * Math.log(Math.min(1, r)));
  // Math.max guards against -0 when all bedtimes are identical (ln 1 = 0).
  return Math.max(0, Math.round((sdRadians * MINUTES_PER_DAY) / (2 * Math.PI)));
}

export type SleepRangeStatus = "below_recommended" | "within_recommended" | "above_recommended";

/**
 * Recommended ranges (AASM/SRS 2015 consensus, as summarised in sources.ts):
 *   18–64 years: 7 hours or more (upper reference 9 hours)
 *   65+ years:   7–8 hours.
 * The statement describes habitual sleep, so a single night is not assessed.
 */
export function sleepRangeStatus(hours: number, ageYears: number): SleepRangeStatus {
  if (!Number.isFinite(hours) || hours < 0) throw new CalculationInputError("hours must be >= 0");
  const [min, max] = ageYears >= 65 ? [7, 8] : [7, 9];
  if (hours < min) return "below_recommended";
  if (hours > max) return "above_recommended";
  return "within_recommended";
}
