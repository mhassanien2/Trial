import { NextResponse } from "next/server";

import { prisma } from "@/lib/db";
import { ForbiddenError } from "@/lib/rbac";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> }
) {
  try {
    const { id } = await params;
    const tenant = await requireTenantWith("reviews.view");
    const run = await prisma.mockPanelRun.findFirst({
      where: { id, institutionId: tenant.institutionId },
      include: {
        document: { select: { title: true } },
        program: { select: { code: true, nameEn: true } },
      },
    });
    if (!run) return NextResponse.json({ error: "Not found" }, { status: 404 });
    return NextResponse.json({ run });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[panel:id:GET]", err);
    return NextResponse.json({ error: "Load failed" }, { status: 500 });
  }
}
