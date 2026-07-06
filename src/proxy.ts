import createIntlMiddleware from "next-intl/middleware";
import { NextRequest } from "next/server";

import { routing } from "@/i18n/routing";

const intlMiddleware = createIntlMiddleware(routing);

// Paths reachable without a session (locale-stripped).
const PUBLIC_PATHS = ["/login"];

/**
 * Edge gate: locale routing + a fast session-cookie check that redirects
 * anonymous visitors to the login page. This is UX-level protection only —
 * authoritative auth + tenant scoping is enforced server-side in
 * `requireTenant`/`requireTenantWith` on every page, action and API route.
 */
export default function proxy(req: NextRequest) {
  const { pathname } = req.nextUrl;

  const localeMatch = pathname.match(/^\/(en|ar)(\/.*)?$/);
  const locale = localeMatch?.[1] ?? routing.defaultLocale;
  const path = localeMatch ? (localeMatch[2] ?? "/") : pathname;

  const isPublic =
    path === "/" || PUBLIC_PATHS.some((p) => path === p || path.startsWith(`${p}/`));

  const hasSessionCookie =
    req.cookies.has("authjs.session-token") ||
    req.cookies.has("__Secure-authjs.session-token");

  if (!isPublic && !hasSessionCookie) {
    const loginUrl = new URL(`/${locale}/login`, req.url);
    loginUrl.searchParams.set("callbackUrl", pathname);
    return Response.redirect(loginUrl);
  }

  return intlMiddleware(req);
}

export const config = {
  matcher: ["/((?!api|_next|_vercel|.*\\..*).*)"],
};
