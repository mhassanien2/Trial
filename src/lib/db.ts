import { PrismaClient } from "@prisma/client";

// Prisma's driver rejects the `channel_binding` query param that Neon adds
// to its connection strings; strip it (TLS stays enforced via sslmode).
if (process.env.DATABASE_URL) {
  try {
    const u = new URL(process.env.DATABASE_URL);
    if (u.searchParams.has("channel_binding")) {
      u.searchParams.delete("channel_binding");
      process.env.DATABASE_URL = u.toString();
    }
  } catch {
    // leave as-is if it isn't a parseable URL
  }
}

const globalForPrisma = globalThis as unknown as { prisma?: PrismaClient };

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log: process.env.NODE_ENV === "development" ? ["warn", "error"] : ["error"],
  });

if (process.env.NODE_ENV !== "production") globalForPrisma.prisma = prisma;
