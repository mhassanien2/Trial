// Module B: BMI, classification, healthy-weight reference range, and waist-to-height ratio.
// Registry: BMI-FORMULA, BMI-WHO-CLASS, BMI-HEALTHY-RANGE, WHTR-SCREEN (see sources.ts).
// BMI is a screening measure. It does not measure body fat and is not a diagnosis.

export type BmiCategory =
  | "underweight"
  | "healthy"
  | "overweight"
  | "obesity_class_1"
  | "obesity_class_2"
  | "obesity_class_3";

export type BmiResult =
  | {
      applicable: true;
      bmi: number;
      category: BmiCategory;
      healthyWeightRangeKg: { minKg: number; maxKg: number };
    }
  | { applicable: false; reason: "age_below_18" | "pregnancy" };

export class CalculationInputError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CalculationInputError";
  }
}

function assertPositiveFinite(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new CalculationInputError(`${label} must be a positive, finite number`);
  }
}

/** BMI = weight (kg) / height (m)^2 */
export function calculateBmi(weightKg: number, heightCm: number): number {
  assertPositiveFinite(weightKg, "weightKg");
  assertPositiveFinite(heightCm, "heightCm");
  const heightM = heightCm / 100;
  return weightKg / (heightM * heightM);
}

/** WHO adult classification thresholds (kg/m^2). Lower bound inclusive. */
export function classifyBmi(bmi: number): BmiCategory {
  assertPositiveFinite(bmi, "bmi");
  if (bmi < 18.5) return "underweight";
  if (bmi < 25) return "healthy";
  if (bmi < 30) return "overweight";
  if (bmi < 35) return "obesity_class_1";
  if (bmi < 40) return "obesity_class_2";
  return "obesity_class_3";
}

/** Weight range (kg) that corresponds to BMI 18.5–24.9 at the given height. */
export function healthyWeightRangeKg(heightCm: number): { minKg: number; maxKg: number } {
  assertPositiveFinite(heightCm, "heightCm");
  const heightM = heightCm / 100;
  const round1 = (n: number) => Math.round(n * 10) / 10;
  return {
    minKg: round1(18.5 * heightM * heightM),
    maxKg: round1(24.9 * heightM * heightM),
  };
}

/**
 * Full BMI assessment. BMI adult categories are not applied to people under 18
 * or to pregnancy, where they are not valid. Those cases return applicable: false
 * so the UI can show an appropriate message instead of a category.
 */
export function assessBmi(input: {
  weightKg: number;
  heightCm: number;
  ageYears: number;
  pregnant?: boolean;
}): BmiResult {
  if (input.ageYears < 18) return { applicable: false, reason: "age_below_18" };
  if (input.pregnant) return { applicable: false, reason: "pregnancy" };

  const bmi = calculateBmi(input.weightKg, input.heightCm);
  return {
    applicable: true,
    bmi: Math.round(bmi * 10) / 10,
    category: classifyBmi(bmi),
    healthyWeightRangeKg: healthyWeightRangeKg(input.heightCm),
  };
}

/** Waist-to-height ratio = waist circumference / height (same units). */
export function waistToHeightRatio(waistCm: number, heightCm: number): number {
  assertPositiveFinite(waistCm, "waistCm");
  assertPositiveFinite(heightCm, "heightCm");
  return Math.round((waistCm / heightCm) * 100) / 100;
}

/**
 * Screening reference only. A ratio at or above 0.5 is commonly used as a
 * flag for further review. The cut-off varies by population, so this is
 * not a diagnosis.
 */
export function waistToHeightScreen(ratio: number): "below_screening_reference" | "at_or_above_screening_reference" {
  return ratio < 0.5 ? "below_screening_reference" : "at_or_above_screening_reference";
}
