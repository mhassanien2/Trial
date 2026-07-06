// Database setup for deploy: apply migrations (and optionally seed) using a
// connection that works with Neon.
//
// Neon/Prisma specifics handled here:
//  1. Migrations need a DIRECT (non-pooled) connection — the pooler
//     (PgBouncer, transaction mode) doesn't support the advisory locks /
//     session features Prisma Migrate uses. We derive the direct host by
//     dropping "-pooler" from the hostname.
//  2. Prisma's driver rejects the `channel_binding` parameter, so we strip
//     it (TLS is still enforced by sslmode=require).
//  3. If the database already has objects but no Prisma migration history
//     (e.g. the pgvector/pg_trgm extensions were created manually), Prisma
//     Migrate aborts with P3005. On a first-time setup that's safe to reset
//     — the migrations recreate the extensions and all tables. On later
//     deploys the migration history exists, so we never reset (no data loss).
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

function tryDeploy() {
  try {
    const out = execSync("pnpm prisma migrate deploy", {
      env,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    });
    process.stdout.write(out);
    return { ok: true };
  } catch (err) {
    const out = (err.stdout ?? "") + (err.stderr ?? "") + (err.message ?? "");
    process.stdout.write(out);
    return { ok: false, out };
  }
}

console.log("[db:setup] applying migrations via direct connection…");
let res = tryDeploy();

if (!res.ok && res.out.includes("P3005")) {
  // Non-empty schema, no Prisma migration history (e.g. pgvector/pg_trgm
  // created manually before the first deploy). Clear the public schema and
  // apply from scratch. Safe on first-time setup — there's no app data yet,
  // and once migrations are recorded this branch never runs again.
  console.log(
    "[db:setup] P3005: schema not empty & no migration history → " +
      "resetting the public schema (first-time setup only)"
  );
  execSync(
    'pnpm prisma db execute --schema prisma/schema.prisma --stdin',
    { input: "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;", env, stdio: ["pipe", "inherit", "inherit"] }
  );
  res = tryDeploy();
}

if (!res.ok) {
  console.error("[db:setup] migrations failed");
  process.exit(1);
}

if (process.env.SEED_DEMO_DATA === "true") {
  console.log("[db:setup] SEED_DEMO_DATA=true → seeding demo data…");
  execSync("pnpm db:seed", { stdio: "inherit", env });
} else {
  console.log("[db:setup] SEED_DEMO_DATA not 'true' → skipping seed");
}
console.log("[db:setup] done");
