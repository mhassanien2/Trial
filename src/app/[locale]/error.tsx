"use client";

import { useEffect } from "react";
import { useTranslations } from "next-intl";

import { Button } from "@/components/ui/button";

export default function LocaleError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  const t = useTranslations("common");

  useEffect(() => {
    console.error("[error-boundary]", error);
  }, [error]);

  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 p-6 text-center">
      <div>
        <h2 className="text-xl font-semibold">Something went wrong</h2>
        <p className="mt-1 text-sm text-muted-foreground">
          An unexpected error occurred. You can try again.
        </p>
        {error.digest ? (
          <p className="mt-2 text-xs text-muted-foreground">Ref: {error.digest}</p>
        ) : null}
      </div>
      <Button onClick={reset}>{t("loading") === "Loading…" ? "Try again" : "إعادة المحاولة"}</Button>
    </div>
  );
}
