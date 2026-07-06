import { NextResponse } from "next/server";
import { PdcaPhase, PlanItemStatus } from "@prisma/client";
import { z } from "zod";

import { logAudit } from "@/lib/audit";
import { prisma } from "@/lib/db";
import { ForbiddenError } from "@/lib/rbac";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

const createSchema = z.object({
  phase: z.nativeEnum(PdcaPhase),
  title: z.string().min(1).max(300),
  description: z.string().max(2000).optional(),
  dueDate: z.string().datetime().optional(),
});

async function assertProgram(institutionId: string, programId: string) {
  const program = await prisma.program.findFirst({
    where: { id: programId, institutionId },
    select: { id: true },
  });
  if (!program) throw new ForbiddenError("Unknown program");
}

export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: programId } = await params;
    const tenant = await requireTenantWith("actions.manage");
    await assertProgram(tenant.institutionId, programId);

    const body = createSchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }

    const item = await prisma.planItem.create({
      data: {
        institutionId: tenant.institutionId,
        programId,
        phase: body.data.phase,
        title: body.data.title,
        description: body.data.description,
        dueDate: body.data.dueDate ? new Date(body.data.dueDate) : null,
      },
    });
    await logAudit({
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      action: "planItem.create",
      entityType: "PlanItem",
      entityId: item.id,
    });
    return NextResponse.json({ id: item.id }, { status: 201 });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[planItems:POST]", err);
    return NextResponse.json({ error: "Create failed" }, { status: 500 });
  }
}

const patchSchema = z.object({
  itemId: z.string(),
  status: z.nativeEnum(PlanItemStatus),
});

export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: programId } = await params;
    const tenant = await requireTenantWith("actions.manage");
    const body = patchSchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }

    // Scope: item must belong to this program + institution.
    const existing = await prisma.planItem.findFirst({
      where: { id: body.data.itemId, programId, institutionId: tenant.institutionId },
      select: { id: true },
    });
    if (!existing) return NextResponse.json({ error: "Not found" }, { status: 404 });

    await prisma.planItem.update({
      where: { id: body.data.itemId },
      data: { status: body.data.status },
    });
    return NextResponse.json({ ok: true });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[planItems:PATCH]", err);
    return NextResponse.json({ error: "Update failed" }, { status: 500 });
  }
}
