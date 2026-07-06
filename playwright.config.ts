import { defineConfig, devices } from "@playwright/test";

/**
 * E2E config for the three critical flows (Phase 7):
 *   ingest → ask, template → export, upload → review.
 *
 * Requires a seeded database (pnpm db:seed) and a running app. By default
 * it reuses an app already on :3000; set PW_START=1 to have Playwright
 * build+start it. Chromium is the pre-installed browser in this env.
 */
export default defineConfig({
  testDir: "./tests/e2e",
  globalSetup: "./tests/e2e/global-setup.ts",
  timeout: 120_000,
  expect: { timeout: 30_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: process.env.BASE_URL ?? "http://localhost:3000",
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        launchOptions: {
          executablePath:
            process.env.PLAYWRIGHT_CHROMIUM_PATH ?? "/opt/pw-browsers/chromium",
        },
      },
    },
  ],
  webServer: process.env.PW_START
    ? {
        command: "pnpm build && pnpm start",
        url: "http://localhost:3000/en/login",
        timeout: 300_000,
        reuseExistingServer: true,
      }
    : undefined,
});
