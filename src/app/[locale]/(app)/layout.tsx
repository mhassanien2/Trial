import { requireTenant } from "@/lib/tenant";
import { prisma } from "@/lib/db";
import { AppSidebar } from "@/components/layout/app-sidebar";
import { AppTopbar } from "@/components/layout/app-topbar";

export default async function AppLayout({
  children,
  params,
}: {
  children: React.ReactNode;
  params: Promise<{ locale: string }>;
}) {
  const { locale } = await params;
  const tenant = await requireTenant(locale);

  const [user, institution] = await Promise.all([
    prisma.user.findUniqueOrThrow({
      where: { id: tenant.userId },
      select: { name: true, email: true, image: true, role: true },
    }),
    prisma.institution.findUniqueOrThrow({
      where: { id: tenant.institutionId },
      select: { nameEn: true, nameAr: true },
    }),
  ]);

  return (
    <div className="flex min-h-screen">
      <AppSidebar />
      <div className="flex flex-1 flex-col">
        <AppTopbar
          user={{
            name: user.name ?? user.email,
            email: user.email,
            image: user.image,
            role: user.role,
          }}
          institutionName={locale === "ar" ? institution.nameAr : institution.nameEn}
        />
        <main className="flex-1 bg-muted/30 p-6">{children}</main>
      </div>
    </div>
  );
}
