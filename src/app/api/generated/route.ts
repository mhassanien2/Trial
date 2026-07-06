import { NextResponse } from "next/server";
import { z } from "zod";

import { logAudit } from "@/lib/audit";
import { prisma } from "@/lib/db";
import { ForbiddenError } from "@/lib/rbac";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

const createSchema = z.object({
  templateId: z.string(),
  programId: z.string(),
  title: z.string().min(1).max(300),
  language: z.enum(["EN", "AR"]).default("EN"),
});

export async function POST(req: Request) {
  try {
    const tenant = await requireTenantWith("generator.use");
    const body = createSchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json(
        { error: "Invalid request", details: body.error.flatten() },
        { status: 400 }
      );
    }

    // Tenant scoping on both references.
    const [template, program] = await Promise.all([
      prisma.template.findFirst({
        where: { id: body.data.templateId, institutionId: tenant.institutionId },
        select: { id: true },
      }),
      prisma.program.findFirst({
        where: { id: body.data.programId, institutionId: tenant.institutionId },
        select: { id: true },
      }),
    ]);
    if (!template || !program) {
      return NextResponse.json({ error: "Unknown template or program" }, { status: 400 });
    }

    const doc = await prisma.generatedDocument.create({
      data: {
        institutionId: tenant.institutionId,
        programId: body.data.programId,
        templateId: body.data.templateId,
        title: body.data.title,
        language: body.data.language,
      },
    });

    await logAudit({
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      action: "generated.create",
      entityType: "GeneratedDocument",
      entityId: doc.id,
      metadata: { templateId: body.data.templateId, programId: body.data.programId },
    });

    return NextResponse.json({ id: doc.id }, { status: 201 });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[generated:POST]", err);
    return NextResponse.json({ error: "Create failed" }, { status: 500 });
  }
}
