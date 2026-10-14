// Database-backed tests for Phase 3b: auth, sessions, consent, profile, isolation, export, deletion.
// All users here are fictional. Emails use the example.test domain.

import assert from "node:assert/strict";
import { after, before, beforeEach, test } from "node:test";

import { AppError } from "../../src/lib/services/errors";
import { login, loginLimiter, logout, register, resolveSession } from "../../src/lib/services/auth";
import { recordConsent, currentConsents } from "../../src/lib/services/consent";
import { getProfile, updateProfile } from "../../src/lib/services/profile";
import { exportAccount, deleteAccount } from "../../src/lib/services/account";
import { hashToken } from "../../src/lib/security/tokens";
import { closeDatabase, db, registrationFor, resetDatabase } from "./helpers";

async function expectAppError(promise: Promise<unknown>, code: AppError["code"]): Promise<void> {
  await assert.rejects(promise, (err: unknown) => err instanceof AppError && err.code === code, `expected ${code}`);
}

before(async () => {
  await resetDatabase();
});

beforeEach(async () => {
  await resetDatabase();
  loginLimiter.reset("amal@example.test");
  loginLimiter.reset("nour@example.test");
});

after(async () => {
  await closeDatabase();
});

test("registration creates user, empty profile, consent history and a session; token is stored only as a hash", async () => {
  const session = await register(db, registrationFor("amal@example.test"));

  const user = await db.user.findUniqueOrThrow({ where: { email: "amal@example.test" } });
  assert.notEqual(user.passwordHash, "a-long-enough-password", "password is hashed");
  assert.ok(await db.userProfile.findUnique({ where: { userId: user.id } }), "profile created");
  assert.equal((await db.consentRecord.count({ where: { userId: user.id } })), 2);

  const stored = await db.session.findFirstOrThrow({ where: { userId: user.id } });
  assert.equal(stored.tokenHash, hashToken(session.token));
  assert.notEqual(stored.tokenHash, session.token, "raw token is never persisted");

  assert.deepEqual(await resolveSession(db, session.token), { userId: user.id });
});

test("registration requires terms and privacy consent and a strong password", async () => {
  await expectAppError(
    register(db, { ...registrationFor("nour@example.test"), consents: [{ type: "terms", granted: true }] }),
    "CONSENT_REQUIRED",
  );
  await expectAppError(
    register(db, { ...registrationFor("nour@example.test"), password: "short" }),
    "VALIDATION_FAILED",
  );
  assert.equal(await db.user.count(), 0, "no partial account is left behind");
});

test("duplicate email is rejected", async () => {
  await register(db, registrationFor("amal@example.test"));
  await expectAppError(register(db, registrationFor("AMAL@example.test")), "EMAIL_TAKEN");
});

test("login: wrong password and unknown email give the same error; correct credentials succeed", async () => {
  await register(db, registrationFor("amal@example.test"));
  await expectAppError(login(db, { email: "amal@example.test", password: "wrong-password-here" }), "INVALID_CREDENTIALS");
  await expectAppError(login(db, { email: "ghost@example.test", password: "wrong-password-here" }), "INVALID_CREDENTIALS");

  const session = await login(db, { email: "amal@example.test", password: "a-long-enough-password" });
  assert.ok((await resolveSession(db, session.token))?.userId);
});

test("repeated failed logins are rate limited", async () => {
  await register(db, registrationFor("amal@example.test"));
  for (let i = 0; i < 5; i++) {
    await expectAppError(login(db, { email: "amal@example.test", password: "wrong-password-here" }), "INVALID_CREDENTIALS");
  }
  await expectAppError(login(db, { email: "amal@example.test", password: "a-long-enough-password" }), "RATE_LIMITED");
});

test("logout revokes the session; expired and deleted-user sessions are rejected", async () => {
  const session = await register(db, registrationFor("amal@example.test"));
  await logout(db, session.token);
  assert.equal(await resolveSession(db, session.token), null);

  const second = await login(db, { email: "amal@example.test", password: "a-long-enough-password" });
  await db.session.updateMany({ data: { expiresAt: new Date(Date.now() - 1000) } });
  assert.equal(await resolveSession(db, second.token), null, "expired session rejected");
  assert.equal(await db.session.count(), 0, "expired session is cleaned up");
});

test("progressive onboarding: non-sensitive fields save without any health consent; unknown fields are refused", async () => {
  const session = await register(db, registrationFor("amal@example.test"));
  const { userId } = (await resolveSession(db, session.token))!;

  const dto = await updateProfile(db, userId, { heightCm: 165, activityLevel: "light", dateOfBirth: "1990-06-15" });
  assert.equal(dto.heightCm, 165);
  assert.equal(dto.activityLevel, "light");
  assert.equal(dto.dateOfBirth, "1990-06-15");

  await expectAppError(updateProfile(db, userId, { heightCm: 165, favouriteColour: "teal" }), "VALIDATION_FAILED");
  await expectAppError(updateProfile(db, userId, { dateOfBirth: "2999-01-01" }), "VALIDATION_FAILED");
});

test("sensitive health fields need explicit consent; withdrawing consent erases them", async () => {
  const session = await register(db, registrationFor("amal@example.test"));
  const { userId } = (await resolveSession(db, session.token))!;

  await expectAppError(
    updateProfile(db, userId, { medications: ["example-medication"] }),
    "CONSENT_REQUIRED",
  );

  await recordConsent(db, userId, {
    type: "health_conditions_and_medications",
    granted: true,
    policyVersion: "2026-10-01",
  });
  const saved = await updateProfile(db, userId, { healthConditions: ["example-condition"], medications: ["example-medication"] });
  assert.deepEqual(saved.medications, ["example-medication"]);
  assert.equal(saved.sensitiveFieldsVisible, true);

  const consents = await currentConsents(db, userId);
  assert.equal(consents.health_conditions_and_medications, true);

  await recordConsent(db, userId, {
    type: "health_conditions_and_medications",
    granted: false,
    policyVersion: "2026-10-01",
  });
  const after = await getProfile(db, userId);
  assert.deepEqual(after?.medications, []);
  assert.deepEqual(after?.healthConditions, []);
  const raw = await db.userProfile.findUniqueOrThrow({ where: { userId } });
  assert.deepEqual(raw.medications, [], "erased from storage, not only hidden");
  assert.equal(
    await db.consentRecord.count({ where: { userId, type: "health_conditions_and_medications" } }),
    2,
    "consent history is kept",
  );
});

test("isolation: a second user cannot see the first user's profile, consents, or export", async () => {
  const a = await register(db, registrationFor("amal@example.test"));
  const b = await register(db, registrationFor("nour@example.test"));
  const aId = (await resolveSession(db, a.token))!.userId;
  const bId = (await resolveSession(db, b.token))!.userId;

  await updateProfile(db, aId, { heightCm: 171, occupationPattern: "desk-based" });

  const bProfile = await getProfile(db, bId);
  assert.equal(bProfile?.heightCm, null);
  assert.equal(bProfile?.occupationPattern, null);

  const bExport = await exportAccount(db, bId);
  const serialised = JSON.stringify(bExport);
  assert.ok(!serialised.includes("amal@example.test"));
  assert.ok(!serialised.includes("desk-based"));
});

test("export contains the user's own data and never credential material", async () => {
  const session = await register(db, registrationFor("amal@example.test"));
  const { userId } = (await resolveSession(db, session.token))!;
  await updateProfile(db, userId, { heightCm: 165 });

  const exported = await exportAccount(db, userId);
  assert.equal(exported.formatVersion, "1");
  assert.equal(exported.account.email, "amal@example.test");
  assert.equal(exported.account.consents.length, 2);
  assert.ok(!("passwordHash" in exported.account), "password hash excluded");
  assert.equal(
    await db.auditEvent.count({ where: { userId, action: "account.export" } }),
    1,
    "export is audited",
  );
});

test("account deletion requires the current password and removes all user data", async () => {
  const session = await register(db, registrationFor("amal@example.test"));
  const { userId } = (await resolveSession(db, session.token))!;

  await expectAppError(deleteAccount(db, userId, "not-the-password"), "CONFIRMATION_FAILED");
  assert.ok(await db.user.findUnique({ where: { id: userId } }), "still exists after failed confirmation");

  await deleteAccount(db, userId, "a-long-enough-password");
  assert.equal(await db.user.findUnique({ where: { id: userId } }), null);
  assert.equal(await db.userProfile.count({ where: { userId } }), 0);
  assert.equal(await db.consentRecord.count({ where: { userId } }), 0);
  assert.equal(await db.session.count({ where: { userId } }), 0);
  assert.equal(await resolveSession(db, session.token), null, "old session no longer works");

  const trail = await db.auditEvent.findFirst({ where: { action: "account.delete" } });
  assert.ok(trail, "deletion is audited");
  assert.equal(trail.userId, null, "audit row keeps no link to the deleted user");
});
