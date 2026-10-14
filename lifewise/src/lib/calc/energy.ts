// Module C: resting metabolic rate (Mifflin–St Jeor), total daily energy expenditure,
// and goal-based calorie estimation with safety floors.
// Registry: RMR-MIFFLIN, TDEE-MULTIPLIERS, CAL-FLOOR, CAL-GOAL-ADJ (see sources.ts).
//
// All outputs are ESTIMATES. They are not measurements of energy expenditure.
// Results carry the assumptions and the uncertainty so the UI can show them.

import { CalculationInputError } from "./bmi";

export const CALC_VERSION = "1.0.0";

export type BiologicalSex = "male" | "female";

export const ACTIVITY_LEVELS = [
  "sedentary",
  "light",
  "moderate",
  "very_active",
  "extra_active",
] as const;
export type ActivityLevel = (typeof ACTIVITY_LEVELS)[number];

/**
 * Conventional activity multipliers applied to RMR. They are coarse
 * approximations. Real daily expenditure varies widely between people.
 */
export const ACTIVITY_MULTIPLIERS: Record<ActivityLevel, number> = {
  sedentary: 1.2,
  light: 1.375,
  moderate: 1.55,
  very_active: 1.725,
  extra_active: 1.9,
};

export type EnergyGoal = "maintain" | "lose" | "gain" | "nutritional_improvement";

/** Conservative goal adjustments in kcal/day. Ranges are deliberately modest. */
export const GOAL_ADJUSTMENT_KCAL = {
  lose: { low: -500, high: -250 },
  gain: { low: 250, high: 500 },
} as const;

/**
 * Minimum daily intake used as a safety floor. Intake below this is not suggested
 * without clinical supervision. These are conventional floors, not validated
 * individual thresholds, so they are labelled as such in the source registry.
 */
export const CALORIE_FLOOR_KCAL: Record<BiologicalSex, number> = {
  female: 1200,
  male: 1500,
};

export type EnergyInput = {
  /** "not_specified" triggers the non-calculated pathway (see below). */
  sex: BiologicalSex | "not_specified";
  ageYears: number;
  weightKg: number;
  heightCm: number;
  activityLevel: ActivityLevel;
  goal: EnergyGoal;
  /** Current BMI category, used for the underweight safety rule. */
  bmiCategory?: "underweight" | "healthy" | "overweight" | "obesity_class_1" | "obesity_class_2" | "obesity_class_3";
  pregnantOrLactating?: boolean;
  /** Free-text flags such as "medication:..." or "condition:...". Any entry forces clinician review. */
  clinicalFlags?: string[];
};

export type EnergyResult =
  | {
      status: "estimated";
      calcVersion: typeof CALC_VERSION;
      rmrKcal: number;
      maintenanceKcal: number;
      goal: EnergyGoal;
      /** Goal-adjusted target. Negative values are not allowed. */
      targetKcal: number;
      /** Suggested range around the target. */
      suggestedRangeKcal: { minKcal: number; maxKcal: number };
      goalAdjustmentKcal: number;
      floorApplied: boolean;
      requiresClinicianReview: boolean;
      warnings: string[];
      assumptions: string[];
    }
  | {
      status: "unavailable";
      reason: "sex_not_specified" | "age_below_18" | "pregnancy_or_lactation" | "underweight_weight_loss_requested";
      /** Shown to the user. Points to a non-calculated pathway or a professional. */
      pathway: "non_calculated_guidance" | "clinician_review";
      message: string;
    };

/**
 * Mifflin–St Jeor (1990). Age in years, weight in kg, height in cm.
 * The equation was derived in a population of healthy adults, so its accuracy
 * for people with other body compositions or conditions is uncertain.
 */
export function mifflinStJeorRmr(input: {
  sex: BiologicalSex;
  ageYears: number;
  weightKg: number;
  heightCm: number;
}): number {
  if (!Number.isFinite(input.weightKg) || input.weightKg <= 0) throw new CalculationInputError("weightKg must be positive");
  if (!Number.isFinite(input.heightCm) || input.heightCm <= 0) throw new CalculationInputError("heightCm must be positive");
  if (!Number.isFinite(input.ageYears) || input.ageYears <= 0) throw new CalculationInputError("ageYears must be positive");

  const base = 10 * input.weightKg + 6.25 * input.heightCm - 5 * input.ageYears;
  return Math.round(input.sex === "male" ? base + 5 : base - 161);
}

/** Estimated maintenance energy = RMR × activity multiplier (rounded to nearest kcal). */
export function estimateTdee(rmrKcal: number, activityLevel: ActivityLevel): number {
  return Math.round(rmrKcal * ACTIVITY_MULTIPLIERS[activityLevel]);
}

/**
 * Full energy estimate with safety rules. Order of checks:
 *  1. Non-adult or pregnancy/lactation: no estimate. Return guidance pathway.
 *  2. Sex not specified: the prediction equation cannot be used. Return guidance pathway.
 *  3. Underweight + weight-loss goal: refuse and refer to a professional.
 *  4. Otherwise compute RMR, TDEE, goal target, floor, and warnings.
 */
export function estimateEnergy(input: EnergyInput): EnergyResult {
  if (input.ageYears < 18) {
    return {
      status: "unavailable",
      reason: "age_below_18",
      pathway: "clinician_review",
      message: "Adult energy equations do not apply under 18. Please consult a qualified professional.",
    };
  }
  if (input.pregnantOrLactating) {
    return {
      status: "unavailable",
      reason: "pregnancy_or_lactation",
      pathway: "clinician_review",
      message: "Energy needs change during pregnancy and lactation. Please consult a qualified professional.",
    };
  }
  if (input.sex === "not_specified") {
    return {
      status: "unavailable",
      reason: "sex_not_specified",
      pathway: "non_calculated_guidance",
      message:
        "The prediction equation uses a sex-based term. Without it we cannot estimate. Use the guidance on non-calculated targets, or add the value later.",
    };
  }
  if (input.bmiCategory === "underweight" && input.goal === "lose") {
    return {
      status: "unavailable",
      reason: "underweight_weight_loss_requested",
      pathway: "clinician_review",
      message: "Weight-loss targets are not suggested when BMI is below the healthy range. Please speak with a qualified professional.",
    };
  }

  const rmrKcal = mifflinStJeorRmr({
    sex: input.sex,
    ageYears: input.ageYears,
    weightKg: input.weightKg,
    heightCm: input.heightCm,
  });
  const maintenanceKcal = estimateTdee(rmrKcal, input.activityLevel);

  const warnings: string[] = [];
  let goalAdjustmentKcal = 0;
  let suggestedRange = { minKcal: maintenanceKcal, maxKcal: maintenanceKcal };

  if (input.goal === "lose" || input.goal === "gain") {
    const adj = GOAL_ADJUSTMENT_KCAL[input.goal];
    const midAdj = Math.round((adj.low + adj.high) / 2);
    goalAdjustmentKcal = midAdj;
    suggestedRange = {
      minKcal: maintenanceKcal + adj.low,
      maxKcal: maintenanceKcal + adj.high,
    };
  }

  let targetKcal = maintenanceKcal + goalAdjustmentKcal;
  let floorApplied = false;

  if (input.goal === "lose") {
    const floor = CALORIE_FLOOR_KCAL[input.sex];
    if (targetKcal < floor) {
      targetKcal = floor;
      floorApplied = true;
      warnings.push(
        `The target was raised to the safety floor of ${floor} kcal/day. Going lower needs clinical supervision.`,
      );
    }
    suggestedRange = {
      minKcal: Math.max(suggestedRange.minKcal, floor),
      maxKcal: Math.max(suggestedRange.maxKcal, floor),
    };
    if (suggestedRange.maxKcal < suggestedRange.minKcal) {
      suggestedRange.maxKcal = suggestedRange.minKcal;
    }
  }

  const requiresClinicianReview = (input.clinicalFlags?.length ?? 0) > 0;
  if (requiresClinicianReview) {
    warnings.push("Health conditions or medications were reported. Please have this plan reviewed by a clinician.");
  }
  if (input.bmiCategory === "underweight") {
    warnings.push("BMI is below the healthy range. Focus on gradual gain or maintenance with professional guidance.");
  }

  return {
    status: "estimated",
    calcVersion: CALC_VERSION,
    rmrKcal,
    maintenanceKcal,
    goal: input.goal,
    targetKcal,
    suggestedRangeKcal: suggestedRange,
    goalAdjustmentKcal,
    floorApplied,
    requiresClinicianReview,
    warnings,
    assumptions: [
      "RMR estimated with Mifflin–St Jeor (1990), developed in healthy adults; accuracy varies between individuals.",
      `Activity multiplier ${ACTIVITY_MULTIPLIERS[input.activityLevel]} for "${input.activityLevel}" is a coarse approximation.`,
      "Goal adjustments are modest estimates. Weight change depends on adherence and other factors; it is not guaranteed.",
      "This is an estimate, not a measurement, and not medical advice.",
    ],
  };
}
