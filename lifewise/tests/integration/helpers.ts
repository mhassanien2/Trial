// Integration test helpers. Refuses to run against any database that is not clearly a test database,
// so the development database is never wiped by accident.

import { PrismaClient } from "@prisma/client";

const url = process.env.DATABASE_URL ?? "";
if (!/lifewise_test/.test(url)) {
  throw new Error("Integration tests require DATABASE_URL to point at a database named lifewise_test");
}

export const db = new PrismaClient({ datasources: { db: { url } } });

/** Removes all users (cascades to every user-owned table) and audit rows. */
export async function resetDatabase(): Promise<void> {
  await db.$executeRawUnsafe('TRUNCATE TABLE "AuditEvent", "User" CASCADE');
}

export async function closeDatabase(): Promise<void> {
  await db.$disconnect();
}

export const VALID_CONSENTS = [
  { type: "terms" as const, granted: true },
  { type: "privacy" as const, granted: true },
];

export function registrationFor(email: string, password = "a-long-enough-password") {
  return {
    email,
    password,
    displayName: "Demo Person",
    locale: "en" as const,
    timezone: "Asia/Riyadh",
    consents: VALID_CONSENTS,
    policyVersion: "2026-10-01",
  };
}
