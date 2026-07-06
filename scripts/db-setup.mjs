// Database setup for deploy: apply migrations (and optionally seed) using a
// connection that works with Neon.
//
// Two Neon/Prisma specifics handled here:
//  1. Migrations need a DIRECT (non-pooled) connection — the pooler
//     (PgBouncer, transaction mode) doesn't support the advisory locks /
//     session features Prisma Migrate uses. We derive the direct host by
//     dropping "-pooler" from the hostname.
//  2. Prisma's driver rejects the `channel_binding` parameter, so we strip
//     it (TLS is still enforced by sslmode=require).
import { execSync } from "node:child_process";

function tidy(raw) {
  const u = new URL(raw);
  u.searchParams.delete("channel_binding");
  return u;
}

const raw = process.env.DATABASE_URL;
if (!raw) {
  console.error("[db:setup] DATABASE_URL is not set");
  process.exit(1);
}

const direct = tidy(raw);
direct.hostname = direct.hostname.replace("-pooler", ""); // Neon direct host
const env = { ...process.env, DATABASE_URL: direct.toString() };

console.log("[db:setup] applying migrations via direct connection…");
execSync("pnpm prisma migrate deploy", { stdio: "inherit", env });

if (process.env.SEED_DEMO_DATA === "true") {
  console.log("[db:setup] SEED_DEMO_DATA=true → seeding demo data…");
  execSync("pnpm db:seed", { stdio: "inherit", env });
} else {
  console.log("[db:setup] SEED_DEMO_DATA not 'true' → skipping seed");
}
console.log("[db:setup] done");
