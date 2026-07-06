import { redirect } from "@/i18n/navigation";
import { getTenant } from "@/lib/tenant";

export default async function Home({
  params,
}: {
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const tenant = await getTenant();
  redirect({ href: tenant ? "/dashboard" : "/login", locale });
}
