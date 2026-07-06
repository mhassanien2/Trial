import { NextResponse } from "next/server";
import { z } from "zod";

import { logAudit } from "@/lib/audit";
import { prisma } from "@/lib/db";
import { ForbiddenError } from "@/lib/rbac";
import { generatedContentSchema } from "@/lib/templates/schema";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

const updateSchema = z.object({
  content: generatedContentSchema,
  status: z.enum(["DRAFT", "IN_REVIEW", "FINAL"]).optional(),
});

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const tenant = await requireTenantWith("documents.view");
    const doc = await prisma.generatedDocument.findFirst({
      where: { id, institutionId: tenant.institutionId },
      include: {
        template: { select: { id: true, name: true, formCode: true, schemaJson: true } },
        program: { select: { id: true, nameEn: true, nameAr: true, code: true } },
      },
    });
    if (!doc) return NextResponse.json({ error: "Not found" }, { status: 404 });
    return NextResponse.json({ document: doc });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[generated:GET]", err);
    return NextResponse.json({ error: "Load failed" }, { status: 500 });
  }
}

export async function PUT(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const tenant = await requireTenantWith("generator.use");
    const body = updateSchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json(
        { error: "Invalid content", details: body.error.flatten() },
        { status: 400 }
      );
    }

    const existing = await prisma.generatedDocument.findFirst({
      where: { id, institutionId: tenant.institutionId },
      select: { id: true },
    });
    if (!existing) return NextResponse.json({ error: "Not found" }, { status: 404 });

    await prisma.generatedDocument.update({
      where: { id },
      data: {
        contentJson: body.data.content,
        ...(body.data.status ? { status: body.data.status } : {}),
      },
    });

    await logAudit({
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      action: "generated.update",
      entityType: "GeneratedDocument",
      entityId: id,
    });

    return NextResponse.json({ ok: true });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[generated:PUT]", err);
    return NextResponse.json({ error: "Save failed" }, { status: 500 });
  }
}
