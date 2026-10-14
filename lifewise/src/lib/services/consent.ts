// Consent management. Consent is append-only: each change is a new ConsentRecord, so the
// history is kept. The current state of a type is its most recent record.

import type { ConsentType } from "@prisma/client";

import type { Db } from "../db/client";
import { consentUpdateSchema } from "../validation/account";
import { AppError } from "./errors";
import { writeAudit } from "./audit";

export async function currentConsents(db: Db, userId: string): Promise<Record<string, boolean>> {
  const records = await db.consentRecord.findMany({
    where: { userId },
    orderBy: { recordedAt: "asc" },
  });
  const state: Record<string, boolean> = {};
  for (const r of records) state[r.type] = r.granted;
  return state;
}

export async function hasActiveConsent(db: Db, userId: string, type: ConsentType): Promise<boolean> {
  const latest = await db.consentRecord.findFirst({
    where: { userId, type },
    orderBy: { recordedAt: "desc" },
  });
  return latest?.granted === true;
}

export async function recordConsent(db: Db, userId: string, raw: unknown): Promise<Record<string, boolean>> {
  const parsed = consentUpdateSchema.safeParse(raw);
  if (!parsed.success) {
    throw new AppError("VALIDATION_FAILED", "Invalid consent update", parsed.error.flatten());
  }
  const { type, granted, policyVersion } = parsed.data;

  await db.$transaction(async (tx) => {
    await tx.consentRecord.create({ data: { userId, type, granted, policyVersion } });

    // Withdrawing health-conditions consent erases the data that consent covered.
    if (type === "health_conditions_and_medications" && !granted) {
      await tx.userProfile.updateMany({
        where: { userId },
        data: { healthConditions: [], medications: [], pregnantOrLactating: false },
      });
    }
  });

  await writeAudit(db, { userId, action: "consent.change", entity: "ConsentRecord", entityId: type });
  return currentConsents(db, userId);
}
