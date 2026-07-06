import { NextResponse } from "next/server";
import { z } from "zod";

import { logAudit } from "@/lib/audit";
import { prisma } from "@/lib/db";
import { enqueueJob, kickJobRunner } from "@/lib/jobs/queue";
import { ForbiddenError } from "@/lib/rbac";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

const createSchema = z.object({
  documentId: z.string(),
  packId: z.string(),
  programId: z.string().optional(),
});

export async function POST(req: Request) {
  try {
    const tenant = await requireTenantWith("reviews.run");
    const body = createSchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json(
        { error: "Invalid request", details: body.error.flatten() },
        { status: 400 }
      );
    }

    // The document must be a READY, institution-owned upload.
    const doc = await prisma.document.findFirst({
      where: { id: body.data.documentId, institutionId: tenant.institutionId },
      select: { id: true, ingestStatus: true },
    });
    if (!doc) return NextResponse.json({ error: "Unknown document" }, { status: 400 });
    if (doc.ingestStatus !== "READY") {
      return NextResponse.json(
        { error: "Document is not ingested yet — wait for ingestion to complete." },
        { status: 409 }
      );
    }

    const pack = await prisma.standardsPack.findFirst({
      where: {
        id: body.data.packId,
        OR: [{ institutionId: tenant.institutionId }, { institutionId: null }],
      },
      select: { id: true },
    });
    if (!pack) return NextResponse.json({ error: "Unknown pack" }, { status: 400 });

    if (body.data.programId) {
      const program = await prisma.program.findFirst({
        where: { id: body.data.programId, institutionId: tenant.institutionId },
        select: { id: true },
      });
      if (!program) return NextResponse.json({ error: "Unknown program" }, { status: 400 });
    }

    const review = await prisma.review.create({
      data: {
        institutionId: tenant.institutionId,
        documentId: body.data.documentId,
        packId: body.data.packId,
        programId: body.data.programId ?? null,
      },
    });

    await enqueueJob("run_review", { reviewId: review.id }, tenant.institutionId);
    kickJobRunner();

    await logAudit({
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      action: "review.create",
      entityType: "Review",
      entityId: review.id,
      metadata: { documentId: body.data.documentId, packId: body.data.packId },
    });

    return NextResponse.json({ id: review.id }, { status: 201 });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[reviews:POST]", err);
    return NextResponse.json({ error: "Create failed" }, { status: 500 });
  }
}

export async function GET() {
  try {
    const tenant = await requireTenantWith("reviews.view");
    const reviews = await prisma.review.findMany({
      where: { institutionId: tenant.institutionId },
      orderBy: { createdAt: "desc" },
      take: 100,
      select: {
        id: true,
        status: true,
        readinessScore: true,
        createdAt: true,
        completedAt: true,
        document: { select: { title: true } },
        pack: { select: { nameEn: true, code: true } },
        program: { select: { code: true } },
      },
    });
    return NextResponse.json({ reviews });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[reviews:GET]", err);
    return NextResponse.json({ error: "Failed to list reviews" }, { status: 500 });
  }
}
