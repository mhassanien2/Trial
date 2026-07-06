import { getTranslations } from "next-intl/server";

import { QaChat } from "@/components/standards/qa-chat";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Link } from "@/i18n/navigation";
import { prisma } from "@/lib/db";
import { requireTenant } from "@/lib/tenant";

export default async function StandardsPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const tenant = await requireTenant(locale);
  const t = await getTranslations("standards");

  const packs = await prisma.standardsPack.findMany({
    where: {
      isActive: true,
      OR: [{ institutionId: null }, { institutionId: tenant.institutionId }],
    },
    orderBy: [{ origin: "asc" }, { country: "asc" }, { code: "asc" }],
    select: {
      id: true,
      code: true,
      nameEn: true,
      nameAr: true,
      country: true,
      origin: true,
      version: true,
      _count: { select: { standards: true } },
    },
  });

  const official = packs.filter((p) => p.origin === "OFFICIAL");
  const custom = packs.filter((p) => p.origin === "CUSTOM");

  const countryLabel = (c: string) =>
    c === "SA" ? t("countrySA") : c === "EG" ? t("countryEG") : t("countryOther");

  const packCard = (pack: (typeof packs)[number]) => (
    <Link key={pack.id} href={`/standards/${pack.id}`}>
      <Card className="h-full transition-colors hover:border-primary">
        <CardHeader>
          <div className="flex items-center justify-between gap-2">
            <Badge variant="outline">{countryLabel(pack.country)}</Badge>
            <span className="text-xs text-muted-foreground">
              {pack.code} v{pack.version}
            </span>
          </div>
          <CardTitle className="text-base leading-snug">
            {locale === "ar" && pack.nameAr ? pack.nameAr : pack.nameEn}
          </CardTitle>
          <CardDescription>
            {t("standardsCount", { count: pack._count.standards })}
          </CardDescription>
        </CardHeader>
      </Card>
    </Link>
  );

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>
        <p className="text-muted-foreground">{t("subtitle")}</p>
      </div>

      <section className="space-y-3">
        <h2 className="text-lg font-semibold">{t("officialPacks")}</h2>
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {official.map(packCard)}
        </div>
      </section>

      {custom.length > 0 ? (
        <section className="space-y-3">
          <h2 className="text-lg font-semibold">{t("customPacks")}</h2>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {custom.map(packCard)}
          </div>
        </section>
      ) : null}

      <QaChat locale={locale} />
    </div>
  );
}
