// Audit trail. Records who did what, and when. Never records field values or health data.

import type { Db } from "../db/client";

export async function writeAudit(
  db: Db,
  event: { userId: string | null; action: string; entity?: string; entityId?: string },
): Promise<void> {
  await db.auditEvent.create({
    data: {
      userId: event.userId,
      action: event.action,
      entity: event.entity,
      entityId: event.entityId,
    },
  });
}
