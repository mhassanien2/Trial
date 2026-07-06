import { prisma } from "@/lib/db";
import type { Prisma } from "@prisma/client";

/**
 * Append-only audit trail. Accreditation bodies require traceability,
 * so failures to write audit entries are logged loudly but never
 * swallow the primary operation's result silently.
 */
export async function logAudit(entry: {
  institutionId: string;
  userId?: string | null;
  action: string;
  entityType: string;
  entityId?: string | null;
  metadata?: Prisma.InputJsonValue;
}): Promise<void> {
  try {
    await prisma.auditLog.create({
      data: {
        institutionId: entry.institutionId,
        userId: entry.userId ?? null,
        action: entry.action,
        entityType: entry.entityType,
        entityId: entry.entityId ?? null,
        metadata: entry.metadata ?? {},
      },
    });
  } catch (err) {
    console.error("[audit] failed to write audit log entry", entry.action, err);
  }
}
