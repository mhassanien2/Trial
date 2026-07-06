import { getTranslations } from "next-intl/server";
import { FileText, GraduationCap, BookOpenCheck } from "lucide-react";

import { prisma } from "@/lib/db";
import { requireTenant } from "@/lib/tenant";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export default async function DashboardPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const tenant = await requireTenant(locale);
  const t = await getTranslations();

  const [user, programCount, packCount, documentCount] = await Promise.all([
    prisma.user.findUniqueOrThrow({
      where: { id: tenant.userId },
      select: { name: true, email: true },
    }),
    prisma.program.count({
      where: { institutionId: tenant.institutionId, isActive: true },
    }),
    prisma.standardsPack.count({
      where: {
        isActive: true,
        OR: [{ institutionId: tenant.institutionId }, { institutionId: null }],
      },
    }),
    prisma.document.count({ where: { institutionId: tenant.institutionId } }),
  ]);

  const stats = [
    {
      key: "programs",
      value: programCount,
      icon: GraduationCap,
      description: t("dashboard.programsDescription"),
      label: t("dashboard.programs"),
    },
    {
      key: "packs",
      value: packCount,
      icon: BookOpenCheck,
      description: t("dashboard.standardsPacksDescription"),
      label: t("dashboard.standardsPacks"),
    },
    {
      key: "documents",
      value: documentCount,
      icon: FileText,
      description: t("dashboard.documentsDescription"),
      label: t("dashboard.documents"),
    },
  ];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">
          {t("dashboard.title")}
        </h1>
        <p className="text-muted-foreground">
          {t("dashboard.welcome", { name: user.name ?? user.email })}
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {stats.map(({ key, value, icon: Icon, label, description }) => (
          <Card key={key}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{label}</CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold">{value}</div>
              <CardDescription>{description}</CardDescription>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
