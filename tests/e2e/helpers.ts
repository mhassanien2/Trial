import path from "node:path";
import { expect, type Page } from "@playwright/test";

export const DEMO = { email: "qa@demo.edu", password: "Demo1234!" };
export const ARTIFACT_DIR = path.join(__dirname, ".artifacts");

export async function login(page: Page) {
  await page.goto("/en/login");
  await page.fill("#email", DEMO.email);
  await page.fill("#password", DEMO.password);
  await page.click('button[type="submit"]');
  await page.waitForURL("**/en/dashboard", { timeout: 30_000 });
}

/** Nudge the in-process job runner and wait for a predicate to hold. */
export async function waitFor(
  page: Page,
  predicate: () => Promise<boolean>,
  { timeout = 90_000, interval = 2_000 } = {}
) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    await page.request.post("/api/jobs/run");
    if (await predicate()) return;
    await page.waitForTimeout(interval);
  }
  throw new Error("waitFor: timed out");
}

export { expect };
