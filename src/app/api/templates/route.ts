import { NextResponse } from "next/server";

import { prisma } from "@/lib/db";
import { ForbiddenError } from "@/lib/rbac";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

export async function GET() {
  try {
    const tenant = await requireTenantWith("documents.view");
    const templates = await prisma.template.findMany({
      where: { institutionId: tenant.institutionId, isActive: true },
      orderBy: { createdAt: "desc" },
      select: {
        id: true,
        name: true,
        formCode: true,
        language: true,
        version: true,
        createdAt: true,
        _count: { select: { generatedDocuments: true } },
      },
    });
    return NextResponse.json({ templates });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[templates:GET]", err);
    return NextResponse.json({ error: "Failed to list templates" }, { status: 500 });
  }
}
