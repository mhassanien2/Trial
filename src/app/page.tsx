import { redirect } from "next/navigation";

import { routing } from "@/i18n/routing";

// Root entry: send visitors to the default-locale home, which then routes
// to the dashboard or login. Replaces the locale redirect the Node proxy
// used to do, so the app needs no middleware (Cloudflare-compatible).
export default function RootPage() {
  redirect(`/${routing.defaultLocale}`);
}
