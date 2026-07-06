import { NextResponse } from "next/server";
import { z } from "zod";

import { logAudit } from "@/lib/audit";
import { prisma } from "@/lib/db";
import { enqueueJob, kickJobRunner } from "@/lib/jobs/queue";
import { LIMITS, rateLimit, tooManyRequests } from "@/lib/rate-limit";
import { ForbiddenError } from "@/lib/rbac";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

const createSchema = z.object({
  documentId: z.string(),
  programId: z.string(),
});

export async function POST(req: Request) {
  try {
    const tenant = await requireTenantWith("reviews.run");

    const rl = rateLimit(tenant.userId, "ai", LIMITS.ai);
    if (!rl.ok) return tooManyRequests(rl);

    const body = createSchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }

    const [doc, program] = await Promise.all([
      prisma.document.findFirst({
        where: {
          id: body.data.documentId,
          institutionId: tenant.institutionId,
          ingestStatus: "READY",
        },
        select: { id: true },
      }),
      prisma.program.findFirst({
        where: { id: body.data.programId, institutionId: tenant.institutionId },
        select: { id: true },
      }),
    ]);
    if (!doc) return NextResponse.json({ error: "Document not ready" }, { status: 400 });
    if (!program) return NextResponse.json({ error: "Unknown program" }, { status: 400 });

    const run = await prisma.mockPanelRun.create({
      data: {
        institutionId: tenant.institutionId,
        programId: body.data.programId,
        documentId: body.data.documentId,
      },
    });
    await enqueueJob("run_mock_panel", { runId: run.id }, tenant.institutionId);
    kickJobRunner();
    await logAudit({
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      action: "panel.create",
      entityType: "MockPanelRun",
      entityId: run.id,
    });
    return NextResponse.json({ id: run.id }, { status: 201 });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[panel:POST]", err);
    return NextResponse.json({ error: "Create failed" }, { status: 500 });
  }
}
