// Account, consent, and profile input schemas (server-side authority).

import { z } from "zod";
import { ACTIVITY_LEVELS } from "../calc/energy";

export const CONSENT_TYPES = [
  "terms",
  "privacy",
  "health_data_processing",
  "health_conditions_and_medications",
  "ai_processing",
  "email_notifications",
] as const;

export const REQUIRED_AT_REGISTRATION = ["terms", "privacy"] as const;

/** Fields that reveal health information. They need an active health consent to be stored. */
export const SENSITIVE_PROFILE_FIELDS = ["healthConditions", "medications", "pregnantOrLactating"] as const;

const isoDate = z.iso.date();

export const registerSchema = z.object({
  email: z.email().max(254).transform((e) => e.trim().toLowerCase()),
  password: z
    .string()
    .min(12, "Password must be at least 12 characters")
    .max(128, "Password must be at most 128 characters"),
  displayName: z.string().trim().min(1).max(80).optional(),
  locale: z.enum(["en", "ar"]).default("en"),
  timezone: z.string().min(1).max(64).default("UTC"),
  consents: z.array(z.object({ type: z.enum(CONSENT_TYPES), granted: z.boolean() })),
  policyVersion: z.string().min(1).max(32),
});

export const loginSchema = z.object({
  email: z.email().max(254).transform((e) => e.trim().toLowerCase()),
  password: z.string().min(1).max(128),
});

export const consentUpdateSchema = z.object({
  type: z.enum(CONSENT_TYPES),
  granted: z.boolean(),
  policyVersion: z.string().min(1).max(32),
});

export const profileUpdateSchema = z
  .object({
    dateOfBirth: isoDate.optional(),
    sex: z.enum(["male", "female", "not_specified"]).optional(),
    heightCm: z.number().min(50).max(260).optional(),
    unitPreference: z.enum(["metric", "imperial"]).optional(),
    activityLevel: z.enum(ACTIVITY_LEVELS).optional(),
    occupationPattern: z.string().trim().max(200).optional(),
    typicalSleepHours: z.number().min(0).max(24).optional(),
    dietaryPreferences: z.array(z.string().trim().min(1).max(60)).max(20).optional(),
    allergies: z.array(z.string().trim().min(1).max(60)).max(50).optional(),
    fitnessExperience: z.enum(["none", "beginner", "intermediate", "advanced"]).optional(),
    availableExerciseMin: z.number().int().min(0).max(600).optional(),
    // Sensitive. Accepted only with an active health consent (checked in the service).
    healthConditions: z.array(z.string().trim().min(1).max(120)).max(30).optional(),
    medications: z.array(z.string().trim().min(1).max(120)).max(30).optional(),
    pregnantOrLactating: z.boolean().optional(),
  })
  .strict();

export type RegisterInput = z.infer<typeof registerSchema>;
export type LoginInput = z.infer<typeof loginSchema>;
export type ConsentUpdateInput = z.infer<typeof consentUpdateSchema>;
export type ProfileUpdateInput = z.infer<typeof profileUpdateSchema>;
