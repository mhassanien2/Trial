import { Role } from "@prisma/client";

/**
 * Role capability matrix. REVIEWER is strictly read-only.
 * Higher roles inherit nothing implicitly — permissions are explicit
 * so the matrix stays auditable for accreditation traceability.
 */
export type Permission =
  | "institution.manage"
  | "users.manage"
  | "programs.manage"
  | "programs.view"
  | "standards.manage"
  | "standards.view"
  | "documents.upload"
  | "documents.view"
  | "templates.manage"
  | "generator.use"
  | "reviews.run"
  | "reviews.view"
  | "evidence.manage"
  | "evidence.view"
  | "actions.manage"
  | "actions.view"
  | "audit.view";

const MATRIX: Record<Role, ReadonlySet<Permission>> = {
  [Role.ADMIN]: new Set<Permission>([
    "institution.manage",
    "users.manage",
    "programs.manage",
    "programs.view",
    "standards.manage",
    "standards.view",
    "documents.upload",
    "documents.view",
    "templates.manage",
    "generator.use",
    "reviews.run",
    "reviews.view",
    "evidence.manage",
    "evidence.view",
    "actions.manage",
    "actions.view",
    "audit.view",
  ]),
  [Role.QA_DIRECTOR]: new Set<Permission>([
    "programs.manage",
    "programs.view",
    "standards.manage",
    "standards.view",
    "documents.upload",
    "documents.view",
    "templates.manage",
    "generator.use",
    "reviews.run",
    "reviews.view",
    "evidence.manage",
    "evidence.view",
    "actions.manage",
    "actions.view",
    "audit.view",
  ]),
  [Role.PROGRAM_COORDINATOR]: new Set<Permission>([
    "programs.view",
    "standards.view",
    "documents.upload",
    "documents.view",
    "generator.use",
    "reviews.run",
    "reviews.view",
    "evidence.manage",
    "evidence.view",
    "actions.manage",
    "actions.view",
  ]),
  [Role.FACULTY]: new Set<Permission>([
    "programs.view",
    "standards.view",
    "documents.upload",
    "documents.view",
    "generator.use",
    "reviews.view",
    "evidence.view",
    "actions.view",
  ]),
  [Role.REVIEWER]: new Set<Permission>([
    "programs.view",
    "standards.view",
    "documents.view",
    "reviews.view",
    "evidence.view",
    "actions.view",
  ]),
};

export function can(role: Role, permission: Permission): boolean {
  return MATRIX[role]?.has(permission) ?? false;
}

export function assertCan(role: Role, permission: Permission): void {
  if (!can(role, permission)) {
    throw new ForbiddenError(`Role ${role} lacks permission ${permission}`);
  }
}

export class ForbiddenError extends Error {
  readonly status = 403;
}
