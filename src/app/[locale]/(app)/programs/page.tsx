import { getTranslations } from "next-intl/server";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import { Link } from "@/i18n/navigation";
import { prisma } from "@/lib/db";
import { requireTenant } from "@/lib/tenant";

export default async function ProgramsPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const tenant = await requireTenant(locale);
  const t = await getTranslations("programs");
  const ar = locale === "ar";

  const programs = await prisma.program.findMany({
    where: { institutionId: tenant.institutionId, isActive: true },
    orderBy: { code: "asc" },
    include: {
      readinessSnapshots: { orderBy: { createdAt: "desc" }, take: 1 },
    },
  });

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>
        <p className="text-muted-foreground">{t("subtitle")}</p>
      </div>

      {programs.length === 0 ? (
        <Card>
          <CardContent className="p-6 text-sm text-muted-foreground">
            {t("empty")}
          </CardContent>
        </Card>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {programs.map((p) => {
            const latest = p.readinessSnapshots[0];
            return (
              <Link key={p.id} href={`/programs/${p.id}`}>
                <Card className="h-full transition-colors hover:border-primary">
                  <CardContent className="space-y-2 p-5">
                    <div className="flex items-center justify-between">
                      <Badge variant="outline">{p.code}</Badge>
                      <span className="text-xs text-muted-foreground">
                        {p.degreeLevel}
                      </span>
                    </div>
                    <h2 className="font-semibold leading-snug">
                      {ar ? p.nameAr : p.nameEn}
                    </h2>
                    {p.department ? (
                      <p className="text-sm text-muted-foreground">{p.department}</p>
                    ) : null}
                    <div className="pt-2 text-sm">
                      <span className="text-muted-foreground">{t("readiness")}: </span>
                      {latest ? (
                        <span className="font-semibold">{latest.score}/100</span>
                      ) : (
                        <span className="italic text-muted-foreground">
                          {t("notReviewed")}
                        </span>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
