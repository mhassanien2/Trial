// Fixed-window in-memory rate limiter for auth endpoints.
// Limits are per process. A multi-instance deployment needs a shared store (e.g. Redis or the
// database). That is recorded as a known limitation in docs/ARCHITECTURE.md.

export type RateLimitResult = { ok: true; remaining: number } | { ok: false; retryAfterMs: number };

export function createRateLimiter(opts: { limit: number; windowMs: number }) {
  const buckets = new Map<string, { count: number; resetAt: number }>();

  return {
    hit(key: string, now: number = Date.now()): RateLimitResult {
      const current = buckets.get(key);
      if (!current || now >= current.resetAt) {
        buckets.set(key, { count: 1, resetAt: now + opts.windowMs });
        return { ok: true, remaining: opts.limit - 1 };
      }
      if (current.count >= opts.limit) {
        return { ok: false, retryAfterMs: current.resetAt - now };
      }
      current.count += 1;
      return { ok: true, remaining: opts.limit - current.count };
    },
    reset(key: string): void {
      buckets.delete(key);
    },
  };
}
