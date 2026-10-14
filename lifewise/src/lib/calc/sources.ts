// Source registry: the scientific basis for each health-related rule in the calculation engine.
//
// Verification status:
//   "pending_primary_source_check": the citation is recorded from the engine author's knowledge.
//      It has NOT yet been checked against the primary publication. Do not present it to
//      users as verified until the status is changed by a reviewer who has read the source.
//
// No value or reference here may be used in user-facing copy as a clinical claim until its
// status is "verified_against_primary_source".

export type VerificationStatus = "pending_primary_source_check" | "verified_against_primary_source";

export type SourceRule = {
  id: string;
  rule: string;
  citation: string;
  year: number;
  population: string;
  usedIn: string;
  uncertainty: string;
  verification: VerificationStatus;
};

export const SOURCE_REGISTRY: readonly SourceRule[] = [
  {
    id: "BMI-FORMULA",
    rule: "BMI = weight (kg) / height (m)^2",
    citation: "Quetelet A. (1835) and WHO Technical Report Series 854 (1995), Physical status: the use and interpretation of anthropometry.",
    year: 1995,
    population: "Adults",
    usedIn: "calc/bmi.ts calculateBmi",
    uncertainty: "BMI does not measure body fat and is not a diagnosis. It can misclassify muscular or older adults.",
    verification: "pending_primary_source_check",
  },
  {
    id: "BMI-WHO-CLASS",
    rule: "Adult BMI categories: <18.5 underweight; 18.5–24.9 healthy; 25–29.9 overweight; 30–34.9 class 1 obesity; 35–39.9 class 2; ≥40 class 3",
    citation: "WHO Technical Report Series 894 (2000), Obesity: preventing and managing the global epidemic.",
    year: 2000,
    population: "Adults aged 18 and over (not children, pregnancy, or some ethnic groups where other cut-offs apply)",
    usedIn: "calc/bmi.ts classifyBmi",
    uncertainty: "Risk associated with a given BMI varies between populations.",
    verification: "pending_primary_source_check",
  },
  {
    id: "BMI-HEALTHY-RANGE",
    rule: "Healthy weight range = BMI 18.5–24.9 at the given height",
    citation: "Derived from BMI-WHO-CLASS and BMI-FORMULA.",
    year: 2000,
    population: "Adults",
    usedIn: "calc/bmi.ts healthyWeightRangeKg",
    uncertainty: "A range is a screening reference only. It does not describe an ideal weight for an individual.",
    verification: "pending_primary_source_check",
  },
  {
    id: "WHTR-SCREEN",
    rule: "Waist-to-height ratio ≥0.5 is commonly used as a screening flag",
    citation: "Ashwell M, Gunn P, Gibson S. Waist-to-height ratio is a better screening tool than waist circumference and BMI. Obes Rev. 2012;13(3):275–286.",
    year: 2012,
    population: "Adults (cut-off varies by population)",
    usedIn: "calc/bmi.ts waistToHeightScreen",
    uncertainty: "Cut-off differs between populations and age groups. The flag is not a diagnosis.",
    verification: "pending_primary_source_check",
  },
  {
    id: "RMR-MIFFLIN",
    rule: "RMR (kcal/day) = 10×kg + 6.25×cm − 5×age ± (5 male / −161 female)",
    citation: "Mifflin MD, St Jeor ST, Hill LA, Scott BJ, Daugherty SA, Koh YO. A new predictive equation for resting energy expenditure in healthy individuals. Am J Clin Nutr. 1990;51(2):241–247.",
    year: 1990,
    population: "Healthy adults. Developed from a cohort of non-obese and obese adults.",
    usedIn: "calc/energy.ts mifflinStJeorRmr",
    uncertainty: "Typical error versus measured RMR is about ±10% for individuals. Less accurate for atypical body composition or conditions.",
    verification: "pending_primary_source_check",
  },
  {
    id: "TDEE-MULTIPLIERS",
    rule: "Activity multipliers 1.2 / 1.375 / 1.55 / 1.725 / 1.9 applied to RMR",
    citation: "Conventional activity factors used in the Harris–Benedict and FAO/WHO/UNU (2004) energy-requirement approaches. Exact values are a convention, not a measurement.",
    year: 2004,
    population: "Adults",
    usedIn: "calc/energy.ts ACTIVITY_MULTIPLIERS",
    uncertainty: "Coarse approximation. Individual daily energy expenditure can differ substantially.",
    verification: "pending_primary_source_check",
  },
  {
    id: "CAL-GOAL-ADJ",
    rule: "Gradual loss: −250 to −500 kcal/day. Gradual gain: +250 to +500 kcal/day. Conservative pace, not a guarantee of weight change.",
    citation: "Engine convention. Conservative pacing is consistent with commonly used guidance on gradual weight change. A specific primary source has not yet been identified.",
    year: 2026,
    population: "Adults without medical contraindications",
    usedIn: "calc/energy.ts GOAL_ADJUSTMENT_KCAL",
    uncertainty: "Weight response varies with adherence, metabolism, medication, and other factors. Energy-balance models are approximations.",
    verification: "pending_primary_source_check",
  },
  {
    id: "CAL-FLOOR",
    rule: "Safety floor: 1200 kcal/day (female), 1500 kcal/day (male) for weight-loss targets",
    citation: "Conventional minimum intake guidance used in general-wellness settings. Not a validated individual threshold.",
    year: 2026,
    population: "Adults without clinical supervision",
    usedIn: "calc/energy.ts CALORIE_FLOOR_KCAL",
    uncertainty: "Appropriate minimums vary with body size, activity, and health status. Lower intakes require clinical supervision.",
    verification: "pending_primary_source_check",
  },
  {
    id: "ATWATER-FACTORS",
    rule: "Energy: protein 4 kcal/g, carbohydrate 4 kcal/g, fat 9 kcal/g",
    citation: "FAO/WHO. Food energy: methods of analysis and conversion factors. FAO Food and Nutrition Paper 77, 2003.",
    year: 2003,
    population: "General",
    usedIn: "calc/nutrition.ts",
    uncertainty: "General factors. Some foods (e.g., high-fibre foods, polyols) have different actual energy values.",
    verification: "pending_primary_source_check",
  },
  {
    id: "AMDR-MACROS",
    rule: "AMDR (% of energy): carbohydrate 45–65; fat 20–35; protein 10–35",
    citation: "Institute of Medicine. Dietary Reference Intakes: Energy, Carbohydrate, Fiber, Fat, Fatty Acids, Cholesterol, Protein, and Amino Acids. National Academies Press; 2005.",
    year: 2005,
    population: "Adults",
    usedIn: "calc/nutrition.ts AMDR_PERCENT",
    uncertainty: "Ranges describe intake patterns associated with adequacy. They are not targets for a specific person.",
    verification: "pending_primary_source_check",
  },
  {
    id: "PROTEIN-RDA",
    rule: "Protein RDA 0.8 g/kg/day for adults (minimum, not an optimal target)",
    citation: "Institute of Medicine (2005), as above.",
    year: 2005,
    population: "Healthy adults",
    usedIn: "calc/nutrition.ts PROTEIN_RDA_G_PER_KG",
    uncertainty: "Higher intakes are used in some contexts (e.g., resistance training, older adults). Evidence for specific higher values is not yet recorded in this registry.",
    verification: "pending_primary_source_check",
  },
  {
    id: "FIBER-AI",
    rule: "Fibre adequate intake 14 g per 1000 kcal/day",
    citation: "Institute of Medicine (2005), as above.",
    year: 2005,
    population: "Adults",
    usedIn: "calc/nutrition.ts fiberTargetG",
    uncertainty: "An adequate-intake reference, not a precise requirement.",
    verification: "pending_primary_source_check",
  },
  {
    id: "WATER-AI-2004",
    rule: "Total water adequate intake (all sources): 3.7 L/day men; 2.7 L/day women",
    citation: "Institute of Medicine. Dietary Reference Intakes for Water, Potassium, Sodium, Chloride, and Sulfate. National Academies Press; 2004.",
    year: 2004,
    population: "Adults in moderate conditions",
    usedIn: "calc/hydration.ts TOTAL_WATER_AI_ML",
    uncertainty: "Population reference only. Needs vary. Restricted-fluid conditions require clinician guidance.",
    verification: "pending_primary_source_check",
  },
  {
    id: "SLEEP-AASM-SRS-2015",
    rule: "Adults 18–64: at least 7 h habitual sleep; 65+: 7–8 h",
    citation: "Watson NF, Badr MS, Belenky G, et al. Recommended amount of sleep for a healthy adult: a joint consensus statement of the AASM and SRS. Sleep. 2015;38(6):843–844.",
    year: 2015,
    population: "Healthy adults",
    usedIn: "calc/sleep.ts sleepRangeStatus",
    uncertainty: "Sleep needs vary between individuals. Consensus ranges are not diagnostic.",
    verification: "pending_primary_source_check",
  },
  {
    id: "PA-WHO-2020",
    rule: "Adults: 150–300 min moderate-intensity aerobic activity per week (or equivalent), plus muscle-strengthening on 2 or more days",
    citation: "WHO (2020). WHO guidelines on physical activity and sedentary behaviour.",
    year: 2020,
    population: "Adults 18–64 (separate guidance for older adults)",
    usedIn: "Exercise planner (planned, not yet implemented)",
    uncertainty: "Guidance is for general health. Plans must be adapted to individual capacity and limitations.",
    verification: "pending_primary_source_check",
  },
];

export function findSourceRule(id: string): SourceRule | undefined {
  return SOURCE_REGISTRY.find((r) => r.id === id);
}
