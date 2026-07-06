"use client";

import {
  LayoutDashboard,
  GraduationCap,
  BookOpenCheck,
  FileText,
  LayoutTemplate,
  ClipboardCheck,
  FolderCheck,
} from "lucide-react";
import { useTranslations } from "next-intl";

import { Link, usePathname } from "@/i18n/navigation";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { href: "/dashboard", key: "dashboard", icon: LayoutDashboard, enabled: true },
  { href: "/programs", key: "programs", icon: GraduationCap, enabled: false },
  { href: "/standards", key: "standards", icon: BookOpenCheck, enabled: true },
  { href: "/documents", key: "documents", icon: FileText, enabled: true },
  { href: "/templates", key: "templates", icon: LayoutTemplate, enabled: true },
  { href: "/reviews", key: "reviews", icon: ClipboardCheck, enabled: true },
  { href: "/evidence", key: "evidence", icon: FolderCheck, enabled: false },
] as const;

export function AppSidebar() {
  const t = useTranslations();
  const pathname = usePathname();

  return (
    <aside className="hidden w-60 shrink-0 flex-col border-e bg-sidebar text-sidebar-foreground md:flex">
      <div className="flex h-14 items-center border-b px-4">
        <Link href="/dashboard" className="font-bold tracking-tight text-primary">
          {t("app.name")}
        </Link>
      </div>
      <nav className="flex-1 space-y-1 p-3">
        {NAV_ITEMS.map(({ href, key, icon: Icon, enabled }) => {
          const active = pathname === href || pathname.startsWith(`${href}/`);
          const cls = cn(
            "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
            active
              ? "bg-primary text-primary-foreground"
              : "text-muted-foreground hover:bg-accent hover:text-accent-foreground",
            !enabled && "cursor-not-allowed opacity-50"
          );
          return enabled ? (
            <Link key={key} href={href} className={cls}>
              <Icon className="h-4 w-4" />
              {t(`nav.${key}`)}
            </Link>
          ) : (
            <span key={key} className={cls} title={t("dashboard.comingSoon")}>
              <Icon className="h-4 w-4" />
              {t(`nav.${key}`)}
            </span>
          );
        })}
      </nav>
    </aside>
  );
}
