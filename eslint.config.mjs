import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    // Generated deploy output (Cloudflare / OpenNext):
    ".open-next/**",
    ".wrangler/**",
    // LifeWise AI is a separate sub-project with its own toolchain (see lifewise/README.md):
    "lifewise/**",
  ]),
]);

export default eslintConfig;
