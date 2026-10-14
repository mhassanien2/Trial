// Prisma client singleton. Services receive a Db handle, which keeps them testable.

import { PrismaClient } from "@prisma/client";

export type Db = PrismaClient;

const globalForPrisma = globalThis as unknown as { lifewisePrisma?: PrismaClient };

export function getDb(): Db {
  if (!globalForPrisma.lifewisePrisma) {
    globalForPrisma.lifewisePrisma = new PrismaClient();
  }
  return globalForPrisma.lifewisePrisma;
}
