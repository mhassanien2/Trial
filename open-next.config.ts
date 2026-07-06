// OpenNext → Cloudflare adapter config.
// Requires: pnpm add -D @opennextjs/cloudflare wrangler
//
// This file is intentionally minimal (defaults are good for this app).
// See https://opennext.js.org/cloudflare for cache/queue overrides.
import { defineCloudflareConfig } from "@opennextjs/cloudflare";

export default defineCloudflareConfig();
