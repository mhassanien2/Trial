// Authentication: registration, login, session resolution, logout.
// Session tokens are opaque; only their hash is stored. Failed logins return one generic error.

import { Prisma } from "@prisma/client";

import type { Db } from "../db/client";
import { createRateLimiter } from "../security/rate-limit";
import { hashPassword, verifyPassword } from "../security/password";
import { generateSessionToken, hashToken } from "../security/tokens";
import { loginSchema, REQUIRED_AT_REGISTRATION, registerSchema, type RegisterInput } from "../validation/account";
import { AppError } from "./errors";
import { writeAudit } from "./audit";

export const SESSION_TTL_MS = 14 * 24 * 60 * 60 * 1000;

export const loginLimiter = createRateLimiter({ limit: 5, windowMs: 15 * 60 * 1000 });

export type SessionInfo = { token: string; expiresAt: Date; userId: string };

let timingDummyHash: Promise<string> | undefined;
function getTimingDummyHash(): Promise<string> {
  timingDummyHash ??= hashPassword("timing-equaliser-not-a-real-password");
  return timingDummyHash;
}

async function createSession(db: Db, userId: string, now: Date = new Date()): Promise<SessionInfo> {
  const token = generateSessionToken();
  const expiresAt = new Date(now.getTime() + SESSION_TTL_MS);
  await db.session.create({ data: { userId, tokenHash: hashToken(token), expiresAt } });
  return { token, expiresAt, userId };
}

export async function register(db: Db, raw: unknown): Promise<SessionInfo> {
  const parsed = registerSchema.safeParse(raw);
  if (!parsed.success) {
    throw new AppError("VALIDATION_FAILED", "Invalid registration details", parsed.error.flatten());
  }
  const input: RegisterInput = parsed.data;

  const granted = new Set(input.consents.filter((c) => c.granted).map((c) => c.type));
  const missing = REQUIRED_AT_REGISTRATION.filter((t) => !granted.has(t));
  if (missing.length > 0) {
    throw new AppError("CONSENT_REQUIRED", "Terms and privacy consent are required", { missing });
  }

  const passwordHash = await hashPassword(input.password);
  const now = new Date();
  const token = generateSessionToken();

  try {
    const user = await db.$transaction(async (tx) => {
      const created = await tx.user.create({
        data: {
          email: input.email,
          passwordHash,
          displayName: input.displayName,
          locale: input.locale,
          timezone: input.timezone,
          profile: { create: {} },
          consents: {
            create: input.consents.map((c) => ({
              type: c.type,
              granted: c.granted,
              policyVersion: input.policyVersion,
            })),
          },
          sessions: {
            create: { tokenHash: hashToken(token), expiresAt: new Date(now.getTime() + SESSION_TTL_MS) },
          },
        },
      });
      await tx.auditEvent.create({ data: { userId: created.id, action: "auth.register" } });
      return created;
    });
    return { token, expiresAt: new Date(now.getTime() + SESSION_TTL_MS), userId: user.id };
  } catch (err) {
    if (err instanceof Prisma.PrismaClientKnownRequestError && err.code === "P2002") {
      // Note: this reveals that the email is registered. See docs/ARCHITECTURE.md (known limitations).
      throw new AppError("EMAIL_TAKEN", "An account with this email already exists");
    }
    throw err;
  }
}

export async function login(db: Db, raw: unknown, now: Date = new Date()): Promise<SessionInfo> {
  const parsed = loginSchema.safeParse(raw);
  if (!parsed.success) {
    throw new AppError("INVALID_CREDENTIALS", "Email or password is incorrect");
  }
  const { email, password } = parsed.data;

  const limit = loginLimiter.hit(email, now.getTime());
  if (!limit.ok) {
    throw new AppError("RATE_LIMITED", "Too many sign-in attempts. Try again later.", {
      retryAfterMs: limit.retryAfterMs,
    });
  }

  const user = await db.user.findFirst({ where: { email, deletedAt: null } });
  // Verify against a dummy hash when the user is unknown, so response time does not reveal existence.
  const ok = await verifyPassword(password, user?.passwordHash ?? (await getTimingDummyHash()));
  if (!user || !user.passwordHash || !ok) {
    await writeAudit(db, { userId: user?.id ?? null, action: "auth.login_failed" });
    throw new AppError("INVALID_CREDENTIALS", "Email or password is incorrect");
  }

  loginLimiter.reset(email);
  const session = await createSession(db, user.id, now);
  await writeAudit(db, { userId: user.id, action: "auth.login" });
  return session;
}

/** Returns the owning user for a live session token, or null. Expired sessions are deleted. */
export async function resolveSession(
  db: Db,
  token: string | undefined | null,
  now: Date = new Date(),
): Promise<{ userId: string } | null> {
  if (!token) return null;
  const session = await db.session.findUnique({
    where: { tokenHash: hashToken(token) },
    include: { user: { select: { deletedAt: true } } },
  });
  if (!session || session.user.deletedAt) return null;
  if (session.expiresAt <= now) {
    await db.session.delete({ where: { id: session.id } }).catch(() => undefined);
    return null;
  }
  return { userId: session.userId };
}

export async function logout(db: Db, token: string | undefined | null): Promise<void> {
  if (!token) return;
  const existing = await db.session.findUnique({ where: { tokenHash: hashToken(token) } });
  if (!existing) return;
  await db.session.delete({ where: { id: existing.id } });
  await writeAudit(db, { userId: existing.userId, action: "auth.logout" });
}
