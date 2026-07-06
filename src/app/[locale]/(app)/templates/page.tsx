import { getTranslations } from "next-intl/server";

import { TemplatesView } from "@/components/templates/templates-view";
import { prisma } from "@/lib/db";
import { can } from "@/lib/rbac";
import { requireTenant } from "@/lib/tenant";

export default async function TemplatesPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const tenant = await requireTenant(locale);
  const t = await getTranslations("templates");

  const programs = await prisma.program.findMany({
    where: { institutionId: tenant.institutionId, isActive: true },
    select: { id: true, code: true, nameEn: true, nameAr: true },
    orderBy: { code: "asc" },
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>
        <p className="text-muted-foreground">{t("subtitle")}</p>
      </div>
      <TemplatesView
        canManage={can(tenant.role, "templates.manage")}
        canGenerate={can(tenant.role, "generator.use")}
        programs={programs.map((p) => ({
          id: p.id,
          code: p.code,
          name: locale === "ar" ? p.nameAr : p.nameEn,
        }))}
      />
    </div>
  );
}
