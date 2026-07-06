import { test } from "@playwright/test";

import { expect, login, waitFor } from "./helpers";

// Critical flow 3: review a completed document → readiness score + findings.
test("run an AI review of the seeded SSR and get a scored, cited result", async ({
  page,
}) => {
  await login(page);

  await page.goto("/en/reviews");
  await page.waitForSelector("#doc", { timeout: 30_000 });
  await page.selectOption("#doc", { label: "PharmD Self-Study Report (SSR)" });

  const packOptions = await page.locator("#pack option").allTextContents();
  const ncaaa = packOptions.find((o) => o.includes("NCAAA-PROG-2022"));
  await page.selectOption("#pack", { label: ncaaa });

  await page.click('button:has-text("Run review")');
  await page.waitForURL("**/en/reviews/**", { timeout: 30_000 });
  const reviewId = page.url().split("/reviews/")[1].split(/[/?#]/)[0];

  // Drive the job queue until the review completes.
  await waitFor(page, async () => {
    const res = await page.request.get(`/api/reviews/${reviewId}`);
    const { review } = (await res.json()) as { review: { status: string } };
    return review.status === "COMPLETED";
  });

  await page.reload();
  await expect(page.locator("text=Readiness score")).toBeVisible();

  // Findings rendered with verdicts.
  const verdicts = page.locator("text=/Met|Partially Met|Not Met/");
  expect(await verdicts.count()).toBeGreaterThan(0);

  // Report export works.
  const res = await page.request.get(`/api/reviews/${reviewId}/report`);
  expect(res.ok()).toBeTruthy();
});
