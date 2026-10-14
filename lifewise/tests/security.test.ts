import assert from "node:assert/strict";
import { test } from "node:test";

import { ageYears } from "../src/lib/services/age";
import { createRateLimiter } from "../src/lib/security/rate-limit";
import { hashPassword, verifyPassword } from "../src/lib/security/password";
import { generateSessionToken, hashToken } from "../src/lib/security/tokens";

test("password hash verifies the right password only", async () => {
  const stored = await hashPassword("correct horse battery");
  assert.equal(stored.split("$").length, 6, "stores algorithm and parameters with the hash");
  assert.equal(await verifyPassword("correct horse battery", stored), true);
  assert.equal(await verifyPassword("correct horse batterY", stored), false);
});

test("password hashes are salted (same password, different output)", async () => {
  const a = await hashPassword("same-password-here");
  const b = await hashPassword("same-password-here");
  assert.notEqual(a, b);
});

test("malformed stored hashes never verify", async () => {
  assert.equal(await verifyPassword("anything", "not-a-hash"), false);
  assert.equal(await verifyPassword("anything", "scrypt$16384$8$1$AAAA$"), false);
});

test("session tokens are random, URL-safe, and hashed deterministically", () => {
  const t1 = generateSessionToken();
  const t2 = generateSessionToken();
  assert.notEqual(t1, t2);
  assert.match(t1, /^[A-Za-z0-9_-]{43}$/, "256-bit base64url");
  assert.equal(hashToken(t1), hashToken(t1));
  assert.notEqual(hashToken(t1), t1, "the stored value is not the token itself");
  assert.match(hashToken(t1), /^[0-9a-f]{64}$/);
});

test("rate limiter blocks after the limit and reports retry time", () => {
  const limiter = createRateLimiter({ limit: 2, windowMs: 1000 });
  assert.equal(limiter.hit("k", 0).ok, true);
  assert.equal(limiter.hit("k", 0).ok, true);
  const blocked = limiter.hit("k", 100);
  assert.equal(blocked.ok, false);
  if (!blocked.ok) assert.equal(blocked.retryAfterMs, 900);
  assert.equal(limiter.hit("k", 1000).ok, true, "window resets");
});

test("age is computed on calendar dates, not by simple year subtraction", () => {
  const dob = new Date("1990-06-15T00:00:00.000Z");
  assert.equal(ageYears(dob, new Date("2026-06-14T12:00:00.000Z")), 35, "day before birthday");
  assert.equal(ageYears(dob, new Date("2026-06-15T00:00:00.000Z")), 36, "on birthday");
});
