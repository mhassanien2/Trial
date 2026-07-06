import { redirect } from "next/navigation";

import { auth } from "@/lib/auth";
import { can, ForbiddenError, type Permission } from "@/lib/rbac";
import type { Role } from "@prisma/client";

export interface TenantContext {
  userId: string;
  institutionId: string;
  role: Role;
}

/**
 * Resolve the current tenant context from the session.
 * Every server action / route handler MUST go through this (or
 * `requireTenant`) so that institutionId scoping is never skipped.
 */
export async function getTenant(): Promise<TenantContext | null> {
  const session = await auth();
  if (!session?.user?.id || !session.user.institutionId) return null;
  return {
    userId: session.user.id,
    institutionId: session.user.institutionId,
    role: session.user.role,
  };
}

/** For server components/pages: redirects to login when unauthenticated. */
export async function requireTenant(locale?: string): Promise<TenantContext> {
  const tenant = await getTenant();
  if (!tenant) redirect(locale ? `/${locale}/login` : "/login");
  return tenant;
}

/** For API routes / server actions: throws instead of redirecting. */
export async function requireTenantWith(
  permission: Permission
): Promise<TenantContext> {
  const tenant = await getTenant();
  if (!tenant) throw new ForbiddenError("Not authenticated");
  if (!can(tenant.role, permission)) {
    throw new ForbiddenError(`Missing permission: ${permission}`);
  }
  return tenant;
}
