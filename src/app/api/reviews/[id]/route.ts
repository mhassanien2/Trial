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

    const review = await prisma.review.findFirst({
      where: { id, institutionId: tenant.institutionId },
      include: {
        document: { select: { title: true } },
        pack: { select: { nameEn: true, code: true } },
        program: { select: { code: true, nameEn: true } },
        findings: {
          include: {
            criterion: {
              select: {
                code: true,
                titleEn: true,
                standard: { select: { code: true, titleEn: true, sortOrder: true } },
              },
            },
          },
        },
      },
    });
    if (!review) return NextResponse.json({ error: "Not found" }, { status: 404 });

    return NextResponse.json({ review });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[reviews:id:GET]", err);
    return NextResponse.json({ error: "Load failed" }, { status: 500 });
  }
}
