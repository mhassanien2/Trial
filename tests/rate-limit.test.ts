import assert from "node:assert/strict";
import { test } from "node:test";

import { rateLimit } from "../src/lib/rate-limit";

test("allows up to the limit then blocks within the window", () => {
  const opts = { limit: 3, windowMs: 10_000 };
  const t0 = 1_000_000;

  assert.equal(rateLimit("u1", "ai", opts, t0).ok, true);
  assert.equal(rateLimit("u1", "ai", opts, t0).ok, true);
  const third = rateLimit("u1", "ai", opts, t0);
  assert.equal(third.ok, true);
  assert.equal(third.remaining, 0);

  const fourth = rateLimit("u1", "ai", opts, t0);
  assert.equal(fourth.ok, false);
});

test("resets after the window elapses", () => {
  const opts = { limit: 1, windowMs: 5_000 };
  const t0 = 2_000_000;
  assert.equal(rateLimit("u2", "ai", opts, t0).ok, true);
  assert.equal(rateLimit("u2", "ai", opts, t0).ok, false);
  // after the window
  assert.equal(rateLimit("u2", "ai", opts, t0 + 5_001).ok, true);
});

test("buckets and identities are isolated", () => {
  const opts = { limit: 1, windowMs: 10_000 };
  const t0 = 3_000_000;
  assert.equal(rateLimit("a", "ai", opts, t0).ok, true);
  assert.equal(rateLimit("a", "ai", opts, t0).ok, false);
  // different identity
  assert.equal(rateLimit("b", "ai", opts, t0).ok, true);
  // different bucket, same identity
  assert.equal(rateLimit("a", "upload", opts, t0).ok, true);
});
