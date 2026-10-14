// Server-side input validation (Zod). Every calculation or write endpoint must parse
// its input through these schemas. Frontend validation is a UX aid only.

import { z } from "zod";
import { ACTIVITY_LEVELS } from "../calc/energy";

const finite = z.number().finite();
const clock = z.string().regex(/^([01]\d|2[0-3]):[0-5]\d$/, "Use 24-hour HH:mm");

export const bmiInputSchema = z.object({
  weightKg: finite.min(20, "Weight must be at least 20 kg").max(400, "Weight must be at most 400 kg"),
  heightCm: finite.min(50, "Height must be at least 50 cm").max(260, "Height must be at most 260 cm"),
  ageYears: z.number().int().min(1, "Age must be at least 1").max(120, "Age must be at most 120"),
  pregnant: z.boolean().optional(),
});

export const energyInputSchema = z.object({
  sex: z.enum(["male", "female", "not_specified"]),
  ageYears: z.number().int().min(1).max(120),
  weightKg: finite.min(20).max(400),
  heightCm: finite.min(50).max(260),
  activityLevel: z.enum(ACTIVITY_LEVELS),
  goal: z.enum(["maintain", "lose", "gain", "nutritional_improvement"]),
  bmiCategory: z
    .enum(["underweight", "healthy", "overweight", "obesity_class_1", "obesity_class_2", "obesity_class_3"])
    .optional(),
  pregnantOrLactating: z.boolean().optional(),
  clinicalFlags: z.array(z.string().max(120)).max(20).optional(),
});

export const hydrationLogSchema = z.object({
  amountMl: z.number().int().min(1, "Amount must be at least 1 ml").max(1500, "A single entry above 1500 ml needs checking"),
  loggedAt: z.iso.datetime(),
});

export const sleepLogSchema = z.object({
  bedtime: clock,
  wakeTime: clock,
  quality: z.number().int().min(1).max(5).optional(),
  logDate: z.iso.date(),
});

export type BmiInput = z.infer<typeof bmiInputSchema>;
export type EnergyInputDto = z.infer<typeof energyInputSchema>;
export type HydrationLogInput = z.infer<typeof hydrationLogSchema>;
export type SleepLogInput = z.infer<typeof sleepLogSchema>;
