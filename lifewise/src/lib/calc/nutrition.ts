// Macronutrient targets and nutritional aggregation.
// Registry: ATWATER-FACTORS, AMDR-MACROS, PROTEIN-RDA, FIBER-AI (see sources.ts).
// Nutrition totals are computed here from structured data, never from LLM output.

import { CalculationInputError } from "./bmi";

export const KCAL_PER_GRAM = { protein: 4, carbs: 4, fat: 9 } as const;

/**
 * Acceptable Macronutrient Distribution Ranges (% of energy), adults.
 * The values are the IOM (2005) AMDRs. Targets outside them are flagged.
 */
export const AMDR_PERCENT = {
  carbs: { min: 45, max: 65 },
  fat: { min: 20, max: 35 },
  protein: { min: 10, max: 35 },
} as const;

/** Dietary protein RDA (0.8 g/kg/day) is the minimum. It is not an optimal target. */
export const PROTEIN_RDA_G_PER_KG = 0.8;

export type NutrientTotals = {
  kcal: number;
  proteinG: number;
  carbsG: number;
  fatG: number;
  fiberG: number;
};

export type MacroTargets = {
  kcal: number;
  proteinG: number;
  carbsG: number;
  fatG: number;
  fiberG: number;
  percentOfEnergy: { protein: number; carbs: number; fat: number };
  outsideAmdr: Array<"protein" | "carbs" | "fat">;
  notes: string[];
};

/** Atwater general factors. Energy from macronutrients (kcal per gram). */
export function kcalFromMacros(proteinG: number, carbsG: number, fatG: number): number {
  return Math.round(
    proteinG * KCAL_PER_GRAM.protein + carbsG * KCAL_PER_GRAM.carbs + fatG * KCAL_PER_GRAM.fat,
  );
}

/** Fiber adequate intake reference: 14 g per 1000 kcal (IOM). */
export function fiberTargetG(kcal: number): number {
  if (!Number.isFinite(kcal) || kcal <= 0) throw new CalculationInputError("kcal must be positive");
  return Math.round((14 * kcal) / 1000);
}

/**
 * Converts an energy target into a macro split.
 * Protein is set by body weight (min 0.8 g/kg from the RDA). Fat is 30% of energy.
 * Carbohydrate takes the remainder. Targets outside the AMDR are flagged, not hidden.
 */
export function macroTargets(input: {
  kcal: number;
  weightKg: number;
  proteinGPerKg?: number;
  fatPercent?: number;
}): MacroTargets {
  if (!Number.isFinite(input.kcal) || input.kcal <= 0) throw new CalculationInputError("kcal must be positive");
  if (!Number.isFinite(input.weightKg) || input.weightKg <= 0) throw new CalculationInputError("weightKg must be positive");

  const proteinPerKg = input.proteinGPerKg ?? PROTEIN_RDA_G_PER_KG;
  const fatPercent = input.fatPercent ?? 30;
  const proteinG = Math.round(input.weightKg * proteinPerKg);
  const fatG = Math.round((input.kcal * (fatPercent / 100)) / KCAL_PER_GRAM.fat);
  const carbsKcal = input.kcal - proteinG * KCAL_PER_GRAM.protein - fatG * KCAL_PER_GRAM.fat;
  const carbsG = Math.max(0, Math.round(carbsKcal / KCAL_PER_GRAM.carbs));

  const pct = (grams: number, gramKcal: number) => Math.round(((grams * gramKcal) / input.kcal) * 100);
  const percentOfEnergy = {
    protein: pct(proteinG, KCAL_PER_GRAM.protein),
    carbs: pct(carbsG, KCAL_PER_GRAM.carbs),
    fat: pct(fatG, KCAL_PER_GRAM.fat),
  };

  const outsideAmdr = (["protein", "carbs", "fat"] as const).filter((m) => {
    const range = AMDR_PERCENT[m];
    return percentOfEnergy[m] < range.min || percentOfEnergy[m] > range.max;
  });

  const notes: string[] = [];
  if (carbsKcal < 0) {
    notes.push("Protein target alone exceeds the energy target. Review the protein setting.");
  }
  if (outsideAmdr.length > 0) {
    notes.push(`Split falls outside the reference range for: ${outsideAmdr.join(", ")}.`);
  }

  return {
    kcal: input.kcal,
    proteinG,
    carbsG,
    fatG,
    fiberG: fiberTargetG(input.kcal),
    percentOfEnergy,
    outsideAmdr,
    notes,
  };
}

/** Sums nutrition across logged food items. Each item is already scaled to the serving eaten. */
export function aggregateNutrition(items: NutrientTotals[]): NutrientTotals {
  return items.reduce<NutrientTotals>(
    (acc, item) => ({
      kcal: acc.kcal + item.kcal,
      proteinG: acc.proteinG + item.proteinG,
      carbsG: acc.carbsG + item.carbsG,
      fatG: acc.fatG + item.fatG,
      fiberG: acc.fiberG + item.fiberG,
    }),
    { kcal: 0, proteinG: 0, carbsG: 0, fatG: 0, fiberG: 0 },
  );
}

/** Progress of a consumed value against a goal. Returned as a percent, uncapped. */
export function percentOfGoal(consumed: number, goal: number): number {
  if (!Number.isFinite(goal) || goal <= 0) throw new CalculationInputError("goal must be positive");
  return Math.round((consumed / goal) * 100);
}
