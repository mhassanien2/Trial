// Data export (portability) and account deletion. Both require the authenticated userId.
// Deletion requires the current password, so a stolen session alone cannot erase an account.

import type { Db } from "../db/client";
import { verifyPassword } from "../security/password";
import { AppError } from "./errors";
import { writeAudit } from "./audit";

export async function exportAccount(db: Db, userId: string, now: Date = new Date()) {
  const user = await db.user.findUnique({
    where: { id: userId },
    include: {
      profile: true,
      consents: { orderBy: { recordedAt: "asc" } },
      goals: true,
      bodyMeasurements: { orderBy: { measuredAt: "asc" } },
      energyEstimates: { orderBy: { createdAt: "asc" } },
      foodLogs: true,
      exerciseLogs: true,
      sleepLogs: true,
      hydrationLogs: true,
      habits: true,
      notifications: true,
      progressReports: true,
    },
  });
  if (!user) throw new AppError("NOT_FOUND", "Account not found");

  await writeAudit(db, { userId, action: "account.export" });

  // Credentials and session material are deliberately excluded from the export.
  const { passwordHash: _omitHash, ...account } = user;
  return { exportedAt: now.toISOString(), formatVersion: "1", account };
}

export async function deleteAccount(db: Db, userId: string, password: string): Promise<void> {
  const user = await db.user.findUnique({ where: { id: userId } });
  if (!user || !user.passwordHash) throw new AppError("NOT_FOUND", "Account not found");
  if (!(await verifyPassword(password, user.passwordHash))) {
    throw new AppError("CONFIRMATION_FAILED", "Password confirmation failed");
  }

  // Cascades remove sessions, profile, consents, and all logs. The audit row survives with userId = null.
  await db.user.delete({ where: { id: userId } });
  await writeAudit(db, { userId: null, action: "account.delete", entity: "User", entityId: userId });
}
