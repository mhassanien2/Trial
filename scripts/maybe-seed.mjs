// Seeds demo data only when SEED_DEMO_DATA=true. Safe to run on every
// deploy: the seed is idempotent (upserts). Used by `pnpm db:setup` so a
// hosting platform can set up the database with zero manual steps.
import { execSync } from "node:child_process";

if (process.env.SEED_DEMO_DATA === "true") {
  console.log("[db:setup] SEED_DEMO_DATA=true → seeding demo data");
  execSync("pnpm db:seed", { stdio: "inherit" });
} else {
  console.log("[db:setup] SEED_DEMO_DATA not 'true' → skipping seed");
}
