// Module G: hydration reference and progress.
// Registry: WATER-AI-2004 (see sources.ts).
// Total water needs (all sources, including food) are separate from beverages consumed.
// There is no single universal target. The reference is shown with its limits, and
// people with fluid restrictions are routed to clinician review.

import { CalculationInputError } from "./bmi";

/**
 * Adequate Intake for total water from all sources (IOM 2004), adults.
 * It is a population reference, not a personal target. It assumes moderate conditions.
 */
export const TOTAL_WATER_AI_ML = { male: 3700, female: 2700 } as const;

export type HydrationReference = {
  totalWaterReferenceMl: number;
  kind: "population_reference_total_water";
  limits: string[];
  requiresClinicianReview: boolean;
};

export function hydrationReference(input: {
  sex: "male" | "female" | "not_specified";
  fluidRestriction?: boolean;
}): HydrationReference {
  const sexKey = input.sex === "not_specified" ? "female" : input.sex;
  const limits = [
    "This is a population reference for total water from food and drink. It is not a personal target.",
    "Needs vary with climate, activity, body size, and illness.",
  ];
  if (input.sex === "not_specified") {
    limits.push("Sex not specified. The lower reference is shown. Your own needs may be higher.");
  }
  return {
    totalWaterReferenceMl: TOTAL_WATER_AI_ML[sexKey],
    kind: "population_reference_total_water",
    limits,
    requiresClinicianReview: Boolean(input.fluidRestriction),
  };
}

/** Beverage progress against the user's own goal. The goal is the user's setting, not a reference. */
export function beverageProgress(input: { loggedMl: number; goalMl: number }): {
  percent: number;
  remainingMl: number;
  goalReached: boolean;
} {
  if (!Number.isFinite(input.goalMl) || input.goalMl <= 0) throw new CalculationInputError("goalMl must be positive");
  if (!Number.isFinite(input.loggedMl) || input.loggedMl < 0) throw new CalculationInputError("loggedMl must be >= 0");
  const percent = Math.round((input.loggedMl / input.goalMl) * 100);
  const remainingMl = Math.max(0, input.goalMl - input.loggedMl);
  return { percent, remainingMl, goalReached: remainingMl === 0 };
}

/**
 * Logging guard. Rejects non-physiological single entries.
 * Anything above 1500 ml in one entry is refused and the user is asked to check it.
 */
export function validateSingleIntakeMl(ml: number): void {
  if (!Number.isFinite(ml) || ml <= 0) throw new CalculationInputError("intake must be a positive amount");
  if (ml > 1500) throw new CalculationInputError("single entry above 1500 ml. Please check the amount");
}
