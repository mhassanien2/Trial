import path from "node:path";
import { test } from "@playwright/test";

import { ARTIFACT_DIR, expect, login, waitFor } from "./helpers";

// Critical flow 1: upload a document → ingest → cited Q&A.
test("ingest a document then get a cited answer", async ({ page }) => {
  await login(page);

  // Upload a DOCX as a standards source (goes through the ingestion pipeline).
  await page.goto("/en/documents");
  await page.click("text=Upload document");
  await page.setInputFiles("#file", path.join(ARTIFACT_DIR, "standards-source.docx"));
  await page.fill("#title", `E2E Ingest ${Date.now()}`);
  await page.fill("#country", "SA");
  await page.click('button:has-text("Upload"):not(:has-text("document"))');

  // Wait for ingestion to reach READY.
  await waitFor(page, async () => {
    const res = await page.request.get("/api/documents");
    const { documents } = (await res.json()) as {
      documents: Array<{ title: string; ingestStatus: string }>;
    };
    return documents.some(
      (d) => d.title.startsWith("E2E Ingest") && d.ingestStatus === "READY"
    );
  });

  // Ask a question that the document supports → expect cited sources.
  await page.goto("/en/standards");
  await page.fill(
    "input[maxlength='2000']",
    "What does the report say about teaching and learning outcomes?"
  );
  await page.click('button:has-text("Ask")');
  await page.waitForSelector('[data-testid="qa-result"]', { timeout: 30_000 });

  const sources = page.locator('[data-testid="qa-source"]');
  await expect(sources.first()).toBeVisible();
  expect(await sources.count()).toBeGreaterThan(0);
});
