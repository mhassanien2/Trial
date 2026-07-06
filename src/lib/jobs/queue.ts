import crypto from "node:crypto";

import { JobStatus, Prisma } from "@prisma/client";

import { prisma } from "@/lib/db";
import { ingestDocumentHandler } from "./handlers/ingest-document";
import { parseTemplateHandler } from "./handlers/parse-template";
import { runReviewHandler } from "./handlers/run-review";

export type JobHandler = (payload: unknown, jobId: string) => Promise<void>;

/** Registry of job types → handlers. Extend in later phases. */
const HANDLERS: Record<string, JobHandler> = {
  ingest_document: ingestDocumentHandler,
  parse_template: parseTemplateHandler,
  run_review: runReviewHandler,
};

export async function enqueueJob(
  type: keyof typeof HANDLERS,
  payload: Prisma.InputJsonValue,
  institutionId?: string
): Promise<string> {
  const job = await prisma.job.create({
    data: { type, payload, institutionId: institutionId ?? null },
  });
  return job.id;
}

/**
 * Claim and run up to `max` pending jobs. Claiming uses
 * FOR UPDATE SKIP LOCKED so concurrent runners never double-process.
 * Called after uploads (fire-and-forget) and from POST /api/jobs/run.
 */
export async function runPendingJobs(max = 5): Promise<number> {
  const workerId = `worker-${crypto.randomUUID().slice(0, 8)}`;
  let processed = 0;

  for (let i = 0; i < max; i++) {
    const claimed = await claimNextJob(workerId);
    if (!claimed) break;
    processed++;

    const handler = HANDLERS[claimed.type];
    try {
      if (!handler) throw new Error(`No handler for job type: ${claimed.type}`);
      await handler(claimed.payload, claimed.id);
      await prisma.job.update({
        where: { id: claimed.id },
        data: { status: JobStatus.COMPLETED, lockedAt: null, lockedBy: null },
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const exhausted = claimed.attempts + 1 >= claimed.maxAttempts;
      await prisma.job.update({
        where: { id: claimed.id },
        data: {
          status: exhausted ? JobStatus.DEAD : JobStatus.PENDING,
          attempts: { increment: 1 },
          lastError: message.slice(0, 2000),
          lockedAt: null,
          lockedBy: null,
          // simple backoff: 30s * attempts
          runAfter: exhausted
            ? undefined
            : new Date(Date.now() + 30_000 * (claimed.attempts + 1)),
        },
      });
      console.error(`[jobs] ${claimed.type} ${claimed.id} failed:`, message);
    }
  }
  return processed;
}

/** Fire-and-forget kick, e.g. right after enqueuing from a request. */
export function kickJobRunner(): void {
  void runPendingJobs().catch((err) =>
    console.error("[jobs] background run failed:", err)
  );
}

interface ClaimedJob {
  id: string;
  type: string;
  payload: unknown;
  attempts: number;
  maxAttempts: number;
}

async function claimNextJob(workerId: string): Promise<ClaimedJob | null> {
  return prisma.$transaction(async (tx) => {
    const rows = await tx.$queryRaw<
      Array<{ id: string; type: string; payload: unknown; attempts: number; maxAttempts: number }>
    >`
      SELECT id, type, payload, attempts, "maxAttempts"
      FROM "Job"
      WHERE status = 'PENDING' AND "runAfter" <= NOW()
      ORDER BY "createdAt" ASC
      LIMIT 1
      FOR UPDATE SKIP LOCKED
    `;
    const row = rows[0];
    if (!row) return null;

    await tx.job.update({
      where: { id: row.id },
      data: {
        status: JobStatus.RUNNING,
        lockedAt: new Date(),
        lockedBy: workerId,
      },
    });
    return row;
  });
}
