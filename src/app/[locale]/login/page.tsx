import { getTranslations } from "next-intl/server";

import { LoginForm } from "@/components/auth/login-form";

export default async function LoginPage() {
  const t = await getTranslations();

  return (
    <main className="flex min-h-screen items-center justify-center bg-muted/40 p-4">
      <div className="w-full max-w-sm space-y-6">
        <div className="space-y-1 text-center">
          <h1 className="text-2xl font-bold tracking-tight text-primary">
            {t("app.name")}
          </h1>
          <p className="text-sm text-muted-foreground">{t("app.tagline")}</p>
        </div>
        <LoginForm
          googleEnabled={Boolean(
            process.env.AUTH_GOOGLE_ID && process.env.AUTH_GOOGLE_SECRET
          )}
        />
      </div>
    </main>
  );
}
