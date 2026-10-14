// Profile read/update. Every function takes the authenticated userId; callers never pass a
// userId taken from the request body. Onboarding is progressive: every field is optional.

import type { Db } from "../db/client";
import { ageYears } from "./age";
import { profileUpdateSchema, SENSITIVE_PROFILE_FIELDS } from "../validation/account";
import { AppError } from "./errors";
import { writeAudit } from "./audit";
import { hasActiveConsent } from "./consent";

export type ProfileDto = {
  dateOfBirth: string | null;
  ageYears: number | null;
  sex: string | null;
  heightCm: number | null;
  unitPreference: string;
  activityLevel: string | null;
  occupationPattern: string | null;
  typicalSleepHours: number | null;
  dietaryPreferences: string[];
  allergies: string[];
  fitnessExperience: string | null;
  availableExerciseMin: number | null;
  healthConditions: string[];
  medications: string[];
  pregnantOrLactating: boolean;
  /** true when the sensitive fields are readable (health consent is active). */
  sensitiveFieldsVisible: boolean;
};

export async function getProfile(db: Db, userId: string, now: Date = new Date()): Promise<ProfileDto | null> {
  const profile = await db.userProfile.findUnique({ where: { userId } });
  if (!profile) return null;

  const consented = await hasActiveConsent(db, userId, "health_conditions_and_medications");
  const dob = profile.dateOfBirth;
  return {
    dateOfBirth: dob ? dob.toISOString().slice(0, 10) : null,
    ageYears: dob ? ageYears(dob, now) : null,
    sex: profile.sex,
    heightCm: profile.heightCm === null ? null : Number(profile.heightCm),
    unitPreference: profile.unitPreference,
    activityLevel: profile.activityLevel,
    occupationPattern: profile.occupationPattern,
    typicalSleepHours: profile.typicalSleepHours === null ? null : Number(profile.typicalSleepHours),
    dietaryPreferences: profile.dietaryPreferences,
    allergies: profile.allergies,
    fitnessExperience: profile.fitnessExperience,
    availableExerciseMin: profile.availableExerciseMin,
    healthConditions: consented ? profile.healthConditions : [],
    medications: consented ? profile.medications : [],
    pregnantOrLactating: consented ? profile.pregnantOrLactating : false,
    sensitiveFieldsVisible: consented,
  };
}

export async function updateProfile(db: Db, userId: string, raw: unknown): Promise<ProfileDto> {
  const parsed = profileUpdateSchema.safeParse(raw);
  if (!parsed.success) {
    throw new AppError("VALIDATION_FAILED", "Invalid profile details", parsed.error.flatten());
  }
  const input = parsed.data;

  const touchesSensitive = SENSITIVE_PROFILE_FIELDS.some((f) => input[f] !== undefined);
  if (touchesSensitive && !(await hasActiveConsent(db, userId, "health_conditions_and_medications"))) {
    throw new AppError(
      "CONSENT_REQUIRED",
      "Health conditions and medications are stored only with your explicit consent",
      { consentType: "health_conditions_and_medications" },
    );
  }

  const data: Record<string, unknown> = {};
  const { dateOfBirth, sex, heightCm, typicalSleepHours, ...rest } = input;
  Object.assign(data, rest);
  if (dateOfBirth !== undefined) data.dateOfBirth = new Date(`${dateOfBirth}T00:00:00.000Z`);
  if (sex !== undefined) data.sex = sex === "not_specified" ? null : sex;
  if (heightCm !== undefined) data.heightCm = heightCm;
  if (typicalSleepHours !== undefined) data.typicalSleepHours = typicalSleepHours;

  if (dateOfBirth !== undefined) {
    const dob = new Date(`${dateOfBirth}T00:00:00.000Z`);
    if (dob > new Date()) {
      throw new AppError("VALIDATION_FAILED", "Date of birth cannot be in the future");
    }
  }

  await db.userProfile.upsert({
    where: { userId },
    create: { userId, ...data },
    update: data,
  });
  await writeAudit(db, { userId, action: "profile.update", entity: "UserProfile", entityId: userId });

  const dto = await getProfile(db, userId);
  if (!dto) throw new AppError("NOT_FOUND", "Profile not found");
  return dto;
}
