import { NextResponse } from "next/server";
import { z } from "zod";

import { logAudit } from "@/lib/audit";
import { prisma } from "@/lib/db";
import { ForbiddenError } from "@/lib/rbac";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

const linkSchema = z.object({
  documentId: z.string(),
  criterionId: z.string(),
  note: z.string().max(1000).optional(),
});

/** Tag an uploaded evidence document to a criterion for this program. */
export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: programId } = await params;
    const tenant = await requireTenantWith("evidence.manage");

    const program = await prisma.program.findFirst({
      where: { id: programId, institutionId: tenant.institutionId },
      select: { id: true },
    });
    if (!program) return NextResponse.json({ error: "Unknown program" }, { status: 400 });

    const body = linkSchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }

    // Document must belong to the institution; criterion must exist in a
    // pack visible to the institution (official or its own custom pack).
    const [doc, criterion] = await Promise.all([
      prisma.document.findFirst({
        where: { id: body.data.documentId, institutionId: tenant.institutionId },
        select: { id: true },
      }),
      prisma.criterion.findFirst({
        where: {
          id: body.data.criterionId,
          standard: {
            pack: {
              OR: [{ institutionId: null }, { institutionId: tenant.institutionId }],
            },
          },
        },
        select: { id: true },
      }),
    ]);
    if (!doc) return NextResponse.json({ error: "Unknown document" }, { status: 400 });
    if (!criterion) return NextResponse.json({ error: "Unknown criterion" }, { status: 400 });

    const link = await prisma.evidenceLink.upsert({
      where: {
        documentId_criterionId_programId: {
          documentId: body.data.documentId,
          criterionId: body.data.criterionId,
          programId,
        },
      },
      update: { note: body.data.note },
      create: {
        institutionId: tenant.institutionId,
        programId,
        documentId: body.data.documentId,
        criterionId: body.data.criterionId,
        note: body.data.note,
      },
    });
    await logAudit({
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      action: "evidence.link",
      entityType: "EvidenceLink",
      entityId: link.id,
      metadata: { criterionId: body.data.criterionId },
    });
    return NextResponse.json({ id: link.id }, { status: 201 });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[evidence:POST]", err);
    return NextResponse.json({ error: "Link failed" }, { status: 500 });
  }
}

const deleteSchema = z.object({ linkId: z.string() });

export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: programId } = await params;
    const tenant = await requireTenantWith("evidence.manage");
    const body = deleteSchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }
    const existing = await prisma.evidenceLink.findFirst({
      where: { id: body.data.linkId, programId, institutionId: tenant.institutionId },
      select: { id: true },
    });
    if (!existing) return NextResponse.json({ error: "Not found" }, { status: 404 });
    await prisma.evidenceLink.delete({ where: { id: body.data.linkId } });
    return NextResponse.json({ ok: true });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[evidence:DELETE]", err);
    return NextResponse.json({ error: "Delete failed" }, { status: 500 });
  }
}
