import { notFound } from "next/navigation";
import { getTranslations } from "next-intl/server";
import { ArrowLeft } from "lucide-react";

import { QaChat } from "@/components/standards/qa-chat";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Link } from "@/i18n/navigation";
import { prisma } from "@/lib/db";
import { isPlaceholder } from "@/lib/standards/schema";
import { requireTenant } from "@/lib/tenant";

export default async function PackPage({
  params,
}: {
  params: Promise<{ locale: string; packId: string }>;
}) {
  const { locale, packId } = await params;
  const tenant = await requireTenant(locale);
  const t = await getTranslations("standards");
  const ar = locale === "ar";

  const pack = await prisma.standardsPack.findFirst({
    where: {
      id: packId,
      isActive: true,
      OR: [{ institutionId: null }, { institutionId: tenant.institutionId }],
    },
    include: {
      standards: {
        orderBy: { sortOrder: "asc" },
        include: {
          criteria: {
            orderBy: { sortOrder: "asc" },
            include: {
              indicators: { orderBy: { sortOrder: "asc" } },
              evidenceRequirements: { orderBy: { sortOrder: "asc" } },
            },
          },
        },
      },
    },
  });
  if (!pack) notFound();

  const text = (en: string | null | undefined, arText: string | null | undefined) => {
    const v = ar && arText ? arText : en;
    if (isPlaceholder(v)) {
      return (
        <span className="italic text-muted-foreground">
          {t("officialTextPending")}
        </span>
      );
    }
    return <>{v}</>;
  };

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <Link
          href="/standards"
          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="h-4 w-4 rtl:rotate-180" />
          {t("backToPacks")}
        </Link>
        <div className="flex flex-wrap items-center gap-2">
          <h1 className="text-2xl font-bold tracking-tight">
            {ar && pack.nameAr ? pack.nameAr : pack.nameEn}
          </h1>
          <Badge variant="outline">
            {pack.code} v{pack.version}
          </Badge>
        </div>
        {pack.description ? (
          <p className="max-w-3xl text-sm text-muted-foreground">{pack.description}</p>
        ) : null}
      </div>

      <div className="space-y-4">
        {pack.standards.map((std) => (
          <Card key={std.id}>
            <CardHeader>
              <CardTitle className="flex items-baseline gap-2 text-base">
                <Badge>{std.code}</Badge>
                {text(std.titleEn, std.titleAr)}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {std.criteria.length === 0 ? (
                <p className="text-sm text-muted-foreground">{t("noCriteria")}</p>
              ) : (
                std.criteria.map((c) => (
                  <div key={c.id} className="rounded-md border p-3">
                    <p className="text-sm font-medium">
                      <span className="me-2 text-muted-foreground">{c.code}</span>
                      {text(c.titleEn, c.titleAr)}
                    </p>
                    {c.indicators.length > 0 ? (
                      <div className="mt-2">
                        <p className="text-xs font-semibold uppercase text-muted-foreground">
                          {t("indicators")}
                        </p>
                        <ul className="ms-4 list-disc text-sm">
                          {c.indicators.map((ind) => (
                            <li key={ind.id}>
                              <span className="me-1 text-muted-foreground">{ind.code}</span>
                              {text(ind.textEn, ind.textAr)}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                    {c.evidenceRequirements.length > 0 ? (
                      <div className="mt-2">
                        <p className="text-xs font-semibold uppercase text-muted-foreground">
                          {t("evidenceRequirements")}
                        </p>
                        <ul className="ms-4 list-disc text-sm">
                          {c.evidenceRequirements.map((e) => (
                            <li key={e.id}>{text(e.textEn, e.textAr)}</li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                ))
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      <QaChat locale={locale} packId={pack.id} />
    </div>
  );
}
