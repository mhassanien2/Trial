import { NextResponse } from "next/server";
import { ActionStatus } from "@prisma/client";
import { z } from "zod";

import { logAudit } from "@/lib/audit";
import { prisma } from "@/lib/db";
import { ForbiddenError } from "@/lib/rbac";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

const createSchema = z.object({
  title: z.string().min(1).max(300),
  description: z.string().max(2000).optional(),
  ownerId: z.string().optional(),
  dueDate: z.string().datetime().optional(),
  findingId: z.string().optional(),
});

export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id: programId } = await params;
    const tenant = await requireTenantWith("actions.manage");

    const program = await prisma.program.findFirst({
      where: { id: programId, institutionId: tenant.institutionId },
      select: { id: true },
    });
    if (!program) return NextResponse.json({ error: "Unknown program" }, { status: 400 });

    const body = createSchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }

    // Owner (if any) must be in the same institution.
    if (body.data.ownerId) {
      const owner = await prisma.user.findFirst({
        where: { id: body.data.ownerId, institutionId: tenant.institutionId },
        select: { id: true },
      });
      if (!owner) return NextResponse.json({ error: "Unknown owner" }, { status: 400 });
    }

    const action = await prisma.improvementAction.create({
      data: {
        institutionId: tenant.institutionId,
        programId,
        title: body.data.title,
        description: body.data.description,
        ownerId: body.data.ownerId ?? null,
        dueDate: body.data.dueDate ? new Date(body.data.dueDate) : null,
        findingId: body.data.findingId ?? null,
      },
    });
    await logAudit({
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      action: "action.create",
      entityType: "ImprovementAction",
      entityId: action.id,
    });
    return NextResponse.json({ id: action.id }, { status: 201 });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[actions:POST]", err);
    return NextResponse.json({ error: "Create failed" }, { status: 500 });
  }
}

const patchSchema = z.object({
  actionId: z.string(),
  status: z.nativeEnum(ActionStatus),
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

    const existing = await prisma.improvementAction.findFirst({
      where: { id: body.data.actionId, programId, institutionId: tenant.institutionId },
      select: { id: true },
    });
    if (!existing) return NextResponse.json({ error: "Not found" }, { status: 404 });

    await prisma.improvementAction.update({
      where: { id: body.data.actionId },
      data: {
        status: body.data.status,
        completedAt: body.data.status === ActionStatus.COMPLETED ? new Date() : null,
      },
    });
    return NextResponse.json({ ok: true });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[actions:PATCH]", err);
    return NextResponse.json({ error: "Update failed" }, { status: 500 });
  }
}
