import { getTranslations } from "next-intl/server";

import { DocumentsView } from "@/components/documents/documents-view";
import { requireTenant } from "@/lib/tenant";
import { can } from "@/lib/rbac";

export default async function DocumentsPage({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const tenant = await requireTenant(locale);
  const t = await getTranslations("documents");

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">{t("title")}</h1>
        <p className="text-muted-foreground">{t("subtitle")}</p>
      </div>
      <DocumentsView canUpload={can(tenant.role, "documents.upload")} />
    </div>
  );
}
