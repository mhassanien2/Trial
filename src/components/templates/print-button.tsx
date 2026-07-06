"use client";

import { Printer } from "lucide-react";
import { useTranslations } from "next-intl";

import { Button } from "@/components/ui/button";

export function PrintButton() {
  const t = useTranslations("wizard");
  return (
    <Button onClick={() => window.print()}>
      <Printer className="h-4 w-4" />
      {t("printView")}
    </Button>
  );
}
