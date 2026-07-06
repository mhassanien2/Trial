import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import { buildCourseSpecTemplate } from "../fixtures/course-spec-template";
import { buildSampleSSR } from "../fixtures/sample-ssr";

export const ARTIFACT_DIR = path.join(__dirname, ".artifacts");

/** Materialise DOCX fixtures the upload specs need. */
export default async function globalSetup() {
  mkdirSync(ARTIFACT_DIR, { recursive: true });
  writeFileSync(
    path.join(ARTIFACT_DIR, "standards-source.docx"),
    await buildSampleSSR()
  );
  writeFileSync(
    path.join(ARTIFACT_DIR, "course-template.docx"),
    await buildCourseSpecTemplate()
  );
}
