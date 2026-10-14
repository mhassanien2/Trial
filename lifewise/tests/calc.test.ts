// Calculation accuracy tests. Reference values were computed by hand from the published
// formulas and are written out in comments so a reviewer can re-derive them.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  aggregateNutrition,
  assessBmi,
  beverageProgress,
  bedtimeConsistencyMinutes,
  calculateBmi,
  classifyBmi,
  cmToFtIn,
  completionRate,
  CalculationInputError,
  estimateEnergy,
  estimateTdee,
  fiberTargetG,
  ftInToCm,
  hydrationReference,
  kcalFromMacros,
  kgToLb,
  lbToKg,
  linearSlope,
  macroTargets,
  mifflinStJeorRmr,
  movingAverage,
  parseClockMinutes,
  percentOfGoal,
  sleepDurationMinutes,
  sleepRangeStatus,
  validateSingleIntakeMl,
  waistToHeightRatio,
  weeklyChange,
  healthyWeightRangeKg,
} from "../src/lib/calc";

// ---- Units -----------------------------------------------------------------

test("unit conversions round-trip", () => {
  assert.ok(Math.abs(lbToKg(220.462) - 100) < 0.01);
  assert.ok(Math.abs(kgToLb(100) - 220.462) < 0.01);
  assert.ok(Math.abs(ftInToCm(5, 10) - 177.8) < 1e-9); // 70 in × 2.54
  assert.deepEqual(cmToFtIn(177.8), { feet: 5, inches: 10 });
});

// ---- BMI (Module B) --------------------------------------------------------

test("BMI reference: 80 kg, 180 cm = 24.69 (healthy)", () => {
  // 80 / 1.8^2 = 80 / 3.24 = 24.691...
  assert.ok(Math.abs(calculateBmi(80, 180) - 24.691) < 0.001);
  const r = assessBmi({ weightKg: 80, heightCm: 180, ageYears: 35 });
  assert.equal(r.applicable, true);
  if (r.applicable) {
    assert.equal(r.bmi, 24.7);
    assert.equal(r.category, "healthy");
    assert.deepEqual(r.healthyWeightRangeKg, { minKg: 59.9, maxKg: 80.7 });
  }
});

test("WHO category boundaries (lower bound inclusive)", () => {
  assert.equal(classifyBmi(18.49), "underweight");
  assert.equal(classifyBmi(18.5), "healthy");
  assert.equal(classifyBmi(24.9), "healthy");
  assert.equal(classifyBmi(25), "overweight");
  assert.equal(classifyBmi(29.99), "overweight");
  assert.equal(classifyBmi(30), "obesity_class_1");
  assert.equal(classifyBmi(35), "obesity_class_2");
  assert.equal(classifyBmi(40), "obesity_class_3");
});

test("BMI not applied to under-18s or pregnancy", () => {
  assert.deepEqual(assessBmi({ weightKg: 60, heightCm: 170, ageYears: 16 }), {
    applicable: false,
    reason: "age_below_18",
  });
  assert.deepEqual(assessBmi({ weightKg: 60, heightCm: 170, ageYears: 30, pregnant: true }), {
    applicable: false,
    reason: "pregnancy",
  });
});

test("healthy weight range at 160 cm", () => {
  // 18.5 × 1.6² = 47.36 → 47.4 ; 24.9 × 1.6² = 63.744 → 63.7
  assert.deepEqual(healthyWeightRangeKg(160), { minKg: 47.4, maxKg: 63.7 });
});

test("waist-to-height ratio: 90 cm waist, 180 cm height = 0.5", () => {
  assert.equal(waistToHeightRatio(90, 180), 0.5);
});

test("BMI rejects invalid height or weight", () => {
  assert.throws(() => calculateBmi(80, 0), CalculationInputError);
  assert.throws(() => calculateBmi(-1, 170), CalculationInputError);
  assert.throws(() => calculateBmi(Number.NaN, 170), CalculationInputError);
});

// ---- Energy (Module C) -----------------------------------------------------

test("Mifflin–St Jeor male reference: 30 y, 80 kg, 180 cm = 1780 kcal", () => {
  // 10×80 + 6.25×180 − 5×30 + 5 = 800 + 1125 − 150 + 5 = 1780
  assert.equal(mifflinStJeorRmr({ sex: "male", ageYears: 30, weightKg: 80, heightCm: 180 }), 1780);
});

test("Mifflin–St Jeor female reference: 30 y, 65 kg, 165 cm = 1370 kcal (1370.25 rounded)", () => {
  // 650 + 1031.25 − 150 − 161 = 1370.25
  assert.equal(mifflinStJeorRmr({ sex: "female", ageYears: 30, weightKg: 65, heightCm: 165 }), 1370);
});

test("Mifflin–St Jeor male reference: 45 y, 100 kg, 175 cm = 1874 kcal (1873.75 rounded)", () => {
  // 1000 + 1093.75 − 225 + 5 = 1873.75
  assert.equal(mifflinStJeorRmr({ sex: "male", ageYears: 45, weightKg: 100, heightCm: 175 }), 1874);
});

test("TDEE: male 30 y 80 kg 180 cm moderate = 1780 × 1.55 = 2759", () => {
  assert.equal(estimateTdee(1780, "moderate"), 2759);
});

test("estimateEnergy maintain: full result for the male reference case", () => {
  const r = estimateEnergy({
    sex: "male",
    ageYears: 30,
    weightKg: 80,
    heightCm: 180,
    activityLevel: "moderate",
    goal: "maintain",
  });
  assert.equal(r.status, "estimated");
  if (r.status !== "estimated") return;
  assert.equal(r.rmrKcal, 1780);
  assert.equal(r.maintenanceKcal, 2759);
  assert.equal(r.targetKcal, 2759);
  assert.deepEqual(r.suggestedRangeKcal, { minKcal: 2759, maxKcal: 2759 });
  assert.equal(r.floorApplied, false);
  assert.equal(r.requiresClinicianReview, false);
  assert.ok(r.assumptions.length >= 3, "assumptions and uncertainty must be disclosed");
});

test("gradual weight loss: female 30 y 65 kg 165 cm sedentary", () => {
  // RMR 1370 → maintenance round(1370 × 1.2) = 1644 → mid adjustment −375 → target 1269 (above floor 1200)
  const r = estimateEnergy({
    sex: "female",
    ageYears: 30,
    weightKg: 65,
    heightCm: 165,
    activityLevel: "sedentary",
    goal: "lose",
  });
  assert.equal(r.status, "estimated");
  if (r.status !== "estimated") return;
  assert.equal(r.maintenanceKcal, 1644);
  assert.equal(r.goalAdjustmentKcal, -375);
  assert.equal(r.targetKcal, 1269);
  assert.deepEqual(r.suggestedRangeKcal, { minKcal: 1200, maxKcal: 1394 });
  assert.equal(r.floorApplied, false);
});

test("weight loss never drops below the female safety floor (1200)", () => {
  // RMR 1158 → maintenance 1390 → target 1015 → floor 1200
  const r = estimateEnergy({
    sex: "female",
    ageYears: 30,
    weightKg: 50,
    heightCm: 155,
    activityLevel: "sedentary",
    goal: "lose",
  });
  assert.equal(r.status, "estimated");
  if (r.status !== "estimated") return;
  assert.equal(r.targetKcal, 1200);
  assert.equal(r.floorApplied, true);
  assert.ok(r.warnings.some((w) => w.includes("safety floor")));
});

test("weight loss never drops below the male safety floor (1500)", () => {
  // RMR 1518 → maintenance 1822 → target 1447 → floor 1500
  const r = estimateEnergy({
    sex: "male",
    ageYears: 30,
    weightKg: 60,
    heightCm: 170,
    activityLevel: "sedentary",
    goal: "lose",
  });
  assert.equal(r.status, "estimated");
  if (r.status !== "estimated") return;
  assert.equal(r.targetKcal, 1500);
  assert.equal(r.floorApplied, true);
});

test("gradual gain: male 30 y 80 kg 180 cm moderate = 2759 + 375 = 3134", () => {
  const r = estimateEnergy({
    sex: "male",
    ageYears: 30,
    weightKg: 80,
    heightCm: 180,
    activityLevel: "moderate",
    goal: "gain",
  });
  assert.equal(r.status, "estimated");
  if (r.status !== "estimated") return;
  assert.equal(r.targetKcal, 3134);
  assert.deepEqual(r.suggestedRangeKcal, { minKcal: 3009, maxKcal: 3259 });
});

test("sex not specified returns the non-calculated guidance pathway, not an estimate", () => {
  const r = estimateEnergy({
    sex: "not_specified",
    ageYears: 30,
    weightKg: 80,
    heightCm: 180,
    activityLevel: "moderate",
    goal: "maintain",
  });
  assert.equal(r.status, "unavailable");
  if (r.status === "unavailable") {
    assert.equal(r.reason, "sex_not_specified");
    assert.equal(r.pathway, "non_calculated_guidance");
  }
});

test("under-18, pregnancy, and underweight with a weight-loss goal are refused", () => {
  const base = { sex: "female" as const, weightKg: 50, heightCm: 165, activityLevel: "light" as const };
  const minor = estimateEnergy({ ...base, ageYears: 15, goal: "maintain" });
  const pregnant = estimateEnergy({ ...base, ageYears: 30, goal: "maintain", pregnantOrLactating: true });
  const underweightLoss = estimateEnergy({ ...base, ageYears: 30, goal: "lose", bmiCategory: "underweight" });
  assert.equal(minor.status, "unavailable");
  assert.equal(pregnant.status, "unavailable");
  assert.equal(underweightLoss.status, "unavailable");
  if (underweightLoss.status === "unavailable") assert.equal(underweightLoss.pathway, "clinician_review");
});

test("clinical flags force clinician review while still returning an estimate", () => {
  const r = estimateEnergy({
    sex: "male",
    ageYears: 50,
    weightKg: 90,
    heightCm: 178,
    activityLevel: "light",
    goal: "maintain",
    clinicalFlags: ["medication:antihypertensive"],
  });
  assert.equal(r.status, "estimated");
  if (r.status === "estimated") assert.equal(r.requiresClinicianReview, true);
});

// ---- Nutrition (macros) ------------------------------------------------------

test("macro targets: 2000 kcal, 80 kg reference split", () => {
  // protein 80×0.8 = 64 g (256 kcal); fat 30% = 600 kcal → 67 g (603 kcal);
  // carbs = 2000 − 256 − 603 = 1141 kcal → 285 g; fibre 14 g/1000 kcal → 28 g
  const t = macroTargets({ kcal: 2000, weightKg: 80 });
  assert.deepEqual(
    { p: t.proteinG, c: t.carbsG, f: t.fatG, fibre: t.fiberG },
    { p: 64, c: 285, f: 67, fibre: 28 },
  );
  assert.deepEqual(t.outsideAmdr, []);
});

test("macro split flags AMDR violations rather than hiding them", () => {
  // Very high protein per kg pushes protein above 35% of energy
  const t = macroTargets({ kcal: 1500, weightKg: 100, proteinGPerKg: 2.2 });
  assert.ok(t.outsideAmdr.includes("protein"));
});

test("kcal from macros uses Atwater factors (4/4/9)", () => {
  assert.equal(kcalFromMacros(64, 285, 67), 1999); // 256 + 1140 + 603
});

test("fibre reference scales with energy", () => {
  assert.equal(fiberTargetG(2000), 28);
  assert.equal(fiberTargetG(1500), 21);
});

test("nutrition aggregation sums logged items and goal progress is computed", () => {
  const total = aggregateNutrition([
    { kcal: 300, proteinG: 20, carbsG: 30, fatG: 10, fiberG: 5 },
    { kcal: 500, proteinG: 25, carbsG: 60, fatG: 15, fiberG: 8 },
  ]);
  assert.deepEqual(total, { kcal: 800, proteinG: 45, carbsG: 90, fatG: 25, fiberG: 13 });
  assert.equal(percentOfGoal(1500, 2000), 75);
  assert.throws(() => percentOfGoal(100, 0), CalculationInputError);
});

// ---- Hydration (Module G) ----------------------------------------------------

test("hydration reference is a population figure, with clinician routing for restrictions", () => {
  assert.equal(hydrationReference({ sex: "male" }).totalWaterReferenceMl, 3700);
  assert.equal(hydrationReference({ sex: "female" }).totalWaterReferenceMl, 2700);
  const unspecified = hydrationReference({ sex: "not_specified" });
  assert.equal(unspecified.totalWaterReferenceMl, 2700);
  assert.ok(unspecified.limits.some((l) => l.includes("Sex not specified")));
  assert.equal(hydrationReference({ sex: "male", fluidRestriction: true }).requiresClinicianReview, true);
});

test("beverage progress: 750 of 2000 ml = 38%, 1250 ml remaining", () => {
  const p = beverageProgress({ loggedMl: 750, goalMl: 2000 });
  assert.deepEqual(p, { percent: 38, remainingMl: 1250, goalReached: false });
  assert.equal(beverageProgress({ loggedMl: 2100, goalMl: 2000 }).goalReached, true);
});

test("single hydration entry above 1500 ml is refused", () => {
  assert.doesNotThrow(() => validateSingleIntakeMl(1500));
  assert.throws(() => validateSingleIntakeMl(1501), CalculationInputError);
  assert.throws(() => validateSingleIntakeMl(0), CalculationInputError);
});

// ---- Sleep (Module F) ----------------------------------------------------------

test("sleep duration across midnight and within the same day", () => {
  assert.equal(sleepDurationMinutes("23:00", "07:00"), 480);
  assert.equal(sleepDurationMinutes("00:30", "06:30"), 360);
  assert.throws(() => sleepDurationMinutes("22:00", "22:00"), CalculationInputError);
});

test("clock parsing rejects invalid times", () => {
  assert.equal(parseClockMinutes("09:15"), 555);
  assert.throws(() => parseClockMinutes("24:00"), CalculationInputError);
  assert.throws(() => parseClockMinutes("9:15"), CalculationInputError);
});

test("bedtime consistency is circular: 23:50 and 00:10 are about 10 min apart, not 1420", () => {
  const sd = bedtimeConsistencyMinutes(["23:50", "00:10"]);
  assert.ok(sd !== null && Math.abs(sd - 10) <= 1, `expected ~10 minutes, got ${sd}`);
  assert.equal(bedtimeConsistencyMinutes(["23:00", "23:00", "23:00"]), 0);
  assert.equal(bedtimeConsistencyMinutes(["23:00"]), null);
  assert.equal(bedtimeConsistencyMinutes(["00:00", "08:00", "16:00"]), null);
});

test("sleep range status by age group", () => {
  assert.equal(sleepRangeStatus(6.5, 30), "below_recommended");
  assert.equal(sleepRangeStatus(8, 30), "within_recommended");
  assert.equal(sleepRangeStatus(9.5, 30), "above_recommended");
  assert.equal(sleepRangeStatus(8.5, 70), "above_recommended");
});

// ---- Trends and summaries (Section 7) ----------------------------------------------

test("linear slope and weekly change for a steady loss of 1 kg per week", () => {
  const points = [
    { x: 0, y: 80 },
    { x: 7, y: 79 },
    { x: 14, y: 78 },
  ];
  assert.ok(Math.abs((linearSlope(points) ?? 0) - -1 / 7) < 1e-9);
  assert.equal(weeklyChange(points), -1);
  assert.equal(linearSlope([{ x: 0, y: 80 }]), null);
  assert.equal(linearSlope([{ x: 1, y: 80 }, { x: 1, y: 79 }]), null);
});

test("moving average, completion rate, and single-value edge cases", () => {
  assert.deepEqual(movingAverage([1, 2, 3, 4], 2), [null, 1.5, 2.5, 3.5]);
  assert.equal(completionRate(3, 4), 75);
  assert.equal(completionRate(5, 4), 100);
  assert.equal(completionRate(0, 0), null);
});

test("energy-to-TDEE helpers do not produce negative totals for valid inputs", () => {
  assert.ok(estimateTdee(1200, "sedentary") > 0);
});
