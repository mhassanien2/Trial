import { NextResponse } from "next/server";

import { logAudit } from "@/lib/audit";
import { prisma } from "@/lib/db";
import { ForbiddenError } from "@/lib/rbac";
import { exportReviewReport } from "@/lib/review/report-docx";
import { isPlaceholder } from "@/lib/standards/schema";
import { getStorage } from "@/lib/storage";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";

interface Citation {
  page: number | null;
  quote: string;
}

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
        document: { select: { title: true, language: true } },
        pack: { select: { nameEn: true } },
        program: { select: { code: true, nameEn: true } },
        findings: {
          include: {
            criterion: {
              select: {
                code: true,
                titleEn: true,
                standard: { select: { titleEn: true, code: true, sortOrder: true } },
              },
            },
          },
        },
      },
    });
    if (!review) return NextResponse.json({ error: "Not found" }, { status: 404 });
    if (review.status !== "COMPLETED") {
      return NextResponse.json({ error: "Review not completed yet" }, { status: 409 });
    }

    const findings = [...review.findings]
      .sort(
        (a, b) =>
          a.criterion.standard.sortOrder - b.criterion.standard.sortOrder ||
          a.criterion.code.localeCompare(b.criterion.code)
      )
      .map((f) => ({
        standardTitle: isPlaceholder(f.criterion.standard.titleEn)
          ? f.criterion.standard.code
          : f.criterion.standard.titleEn,
        criterionCode: f.criterion.code,
        criterionTitle: isPlaceholder(f.criterion.titleEn)
          ? f.criterion.code
          : f.criterion.titleEn,
        verdict: f.verdict,
        score: f.score,
        findingText: f.findingText,
        citations: (f.citations as unknown as Citation[]) ?? [],
        recommendations: (f.recommendations as unknown as string[]) ?? [],
      }));

    const buffer = await exportReviewReport({
      documentTitle: review.document.title,
      packName: review.pack.nameEn,
      programLabel: review.program
        ? `${review.program.code} — ${review.program.nameEn}`
        : null,
      readinessScore: review.readinessScore ?? 0,
      summary: review.summary ?? "",
      findings,
      rtl: review.document.language === "AR",
    });

    const reportKey = `${tenant.institutionId}/reviews/${review.id}.docx`;
    await getStorage().put(
      reportKey,
      buffer,
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    );
    await prisma.review.update({
      where: { id: review.id },
      data: { reportDocxKey: reportKey },
    });
    await logAudit({
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      action: "review.export.docx",
      entityType: "Review",
      entityId: review.id,
    });

    const filename = `review-${review.document.title.replace(/[^\w.-]+/g, "_")}.docx`;
    return new NextResponse(new Uint8Array(buffer), {
      headers: {
        "Content-Type":
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "Content-Disposition": `attachment; filename="${filename}"`,
      },
    });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[reviews:report]", err);
    return NextResponse.json({ error: "Report export failed" }, { status: 500 });
  }
}
