import { getTranslations } from "next-intl/server";

import { ReviewsView } from "@/components/reviews/reviews-view";
import { prisma } from "@/lib/db";
import { can } from "@/lib/rbac";
import { requireTenant } from "@/lib/tenant";

export default async function ReviewsPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const tenant = await requireTenant(locale);
  const t = await getTranslations("reviews");

  const [docs, packs, programs] = await Promise.all([
    prisma.document.findMany({
      where: {
        institutionId: tenant.institutionId,
        ingestStatus: "READY",
        kind: { in: ["REVIEW_SUBJECT", "OTHER", "STANDARDS_SOURCE"] },
      },
      select: { id: true, title: true },
      orderBy: { createdAt: "desc" },
    }),
    prisma.standardsPack.findMany({
      where: {
        isActive: true,
        OR: [{ institutionId: null }, { institutionId: tenant.institutionId }],
      },
      select: { id: true, nameEn: true, nameAr: true, code: true },
      orderBy: [{ origin: "asc" }, { code: "asc" }],
    }),
    prisma.program.findMany({
      where: { institutionId: tenant.institutionId, isActive: true },
      select: { id: true, code: true, nameEn: true, nameAr: true },
      orderBy: { code: "asc" },
    }),
  ]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>
        <p className="text-muted-foreground">{t("subtitle")}</p>
      </div>
      <ReviewsView
        canRun={can(tenant.role, "reviews.run")}
        documents={docs}
        packs={packs.map((p) => ({
          id: p.id,
          name: locale === "ar" && p.nameAr ? p.nameAr : p.nameEn,
          code: p.code,
        }))}
        programs={programs.map((p) => ({
          id: p.id,
          label: `${p.code} — ${locale === "ar" ? p.nameAr : p.nameEn}`,
        }))}
      />
    </div>
  );
}
