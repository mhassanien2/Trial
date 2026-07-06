"use client";

import { useTranslations } from "next-intl";

import { Badge } from "@/components/ui/badge";
import { LocaleSwitcher } from "@/components/layout/locale-switcher";
import { UserMenu } from "@/components/layout/user-menu";
import type { Role } from "@prisma/client";

export function AppTopbar({
  user,
  institutionName,
}: {
  user: { name: string; email: string; image: string | null; role: Role };
  institutionName: string;
}) {
  const t = useTranslations();

  return (
    <header className="flex h-14 items-center justify-between border-b bg-background px-4">
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium">{institutionName}</span>
        <Badge variant="secondary">{t(`roles.${user.role}`)}</Badge>
      </div>
      <div className="flex items-center gap-2">
        <LocaleSwitcher />
        <UserMenu user={user} />
      </div>
    </header>
  );
}
