import { NextResponse } from "next/server";

/**
 * In-memory sliding-window rate limiter (v1, no Redis per the stack).
 * Suitable for a single instance; swap for a shared store when scaling
 * horizontally. Keys are per-identity+bucket so AI endpoints can be
 * limited independently from uploads.
 */
interface Entry {
  count: number;
  resetAt: number;
}

const buckets = new Map<string, Entry>();

// Periodic cleanup so the map does not grow unbounded.
let lastSweep = 0;
function sweep(now: number) {
  if (now - lastSweep < 60_000) return;
  lastSweep = now;
  for (const [key, e] of buckets) {
    if (e.resetAt <= now) buckets.delete(key);
  }
}

export interface RateLimitResult {
  ok: boolean;
  remaining: number;
  resetAt: number;
  limit: number;
}

export function rateLimit(
  identity: string,
  bucket: string,
  opts: { limit: number; windowMs: number },
  now = Date.now()
): RateLimitResult {
  sweep(now);
  const key = `${bucket}:${identity}`;
  const existing = buckets.get(key);

  if (!existing || existing.resetAt <= now) {
    const entry = { count: 1, resetAt: now + opts.windowMs };
    buckets.set(key, entry);
    return { ok: true, remaining: opts.limit - 1, resetAt: entry.resetAt, limit: opts.limit };
  }

  existing.count += 1;
  const ok = existing.count <= opts.limit;
  return {
    ok,
    remaining: Math.max(0, opts.limit - existing.count),
    resetAt: existing.resetAt,
    limit: opts.limit,
  };
}

/** Standard 429 response with rate-limit headers. */
export function tooManyRequests(result: RateLimitResult): NextResponse {
  const retryAfter = Math.max(1, Math.ceil((result.resetAt - Date.now()) / 1000));
  return NextResponse.json(
    { error: "Rate limit exceeded. Please slow down." },
    {
      status: 429,
      headers: {
        "Retry-After": String(retryAfter),
        "X-RateLimit-Limit": String(result.limit),
        "X-RateLimit-Remaining": String(result.remaining),
        "X-RateLimit-Reset": String(Math.ceil(result.resetAt / 1000)),
      },
    }
  );
}

/** Preset windows used across the app. */
export const LIMITS = {
  ai: { limit: 20, windowMs: 60_000 }, // AI calls: 20/min/user
  upload: { limit: 30, windowMs: 60_000 }, // uploads: 30/min/user
  mutation: { limit: 120, windowMs: 60_000 }, // general writes
} as const;
