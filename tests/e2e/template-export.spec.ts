import { test } from "@playwright/test";

import { expect, login } from "./helpers";

// Critical flow 2: template → guided wizard → DOCX export.
test("create a document from a template, fill it, and export DOCX", async ({ page }) => {
  await login(page);

  // The seeded TP-153 Course Specification template drives the wizard.
  await page.goto("/en/templates");
  await page.waitForSelector("text=Course Specification (TP-153)", { timeout: 30_000 });

  await page.fill("#doctitle", `E2E Course Spec ${Date.now()}`);
  await page.click('button:has-text("Create")');
  await page.waitForURL("**/en/generated/**", { timeout: 30_000 });

  // Wizard renders sections from the parsed schema (heading appears in
  // both the section nav and the card title, so match the first).
  await expect(page.locator("text=A. Course Identification").first()).toBeVisible();

  // Fill a field, then export.
  await page
    .locator('td:has(span:text-is("Course Title")) input')
    .first()
    .fill("Clinical Pharmacokinetics");
  await page.click('button:has-text("Save draft")');
  await page.waitForSelector("text=Saved", { timeout: 15_000 });

  const docId = page.url().split("/generated/")[1].split(/[/?#]/)[0];
  const res = await page.request.get(`/api/generated/${docId}/export`);
  expect(res.ok()).toBeTruthy();
  expect(res.headers()["content-type"]).toContain("wordprocessingml");
  const body = await res.body();
  expect(body.length).toBeGreaterThan(1000);
});
