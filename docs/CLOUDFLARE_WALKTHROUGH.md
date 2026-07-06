# Publishing AccreditGenius on Cloudflare — step by step

This is a follow-along guide. It assumes the repo is pushed to GitHub
(done) and you have a Cloudflare account.

The app needs three things Cloudflare doesn't provide natively:
**Postgres with pgvector**, **object storage**, and a **Node-capable
runtime** for PDF/DOCX processing. There are two proven ways to satisfy
them. Pick one.

| Path | Effort | Runtime risk | Recommended for |
| --- | --- | --- | --- |
| **A. Node host behind Cloudflare** | Low | None (runs exactly as tested) | Getting live fast, reliably |
| **B. Cloudflare Workers (OpenNext)** | Medium | Some (Prisma driver, PDF on workerd) | All-in on Cloudflare's edge |

> **Recommendation: start with Path A.** It serves the app from *your*
> Cloudflare domain (Cloudflare DNS + CDN + SSL in front) while the Next.js
> server runs on a Node host that executes the exact code the tests cover.
> You can migrate to Path B later without changing the product.

Both paths share the same **database** and **storage** setup, so do that
part once.

---

## Shared setup

### 1. Database — Neon (Postgres + pgvector)

1. Sign up at **neon.tech** → **Create project** → name it
   `accreditgenius`, pick the region closest to your users.
2. Open **SQL Editor** and run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS pg_trgm;
   ```
3. **Connection Details** → copy the **pooled** connection string. It looks
   like:
   ```
   postgresql://USER:PASSWORD@ep-xxx-pooler.REGION.aws.neon.tech/neondb?sslmode=require
   ```
   Keep both the **pooled** and the **direct** (non-pooler) strings — Path B
   needs the direct one for Hyperdrive.
4. From your machine, apply the schema and (optionally) demo data:
   ```bash
   DATABASE_URL="<pooled-string>" pnpm db:deploy
   DATABASE_URL="<pooled-string>" pnpm db:seed     # demo institution + data
   ```

### 2. Storage — Cloudflare R2

1. Cloudflare dashboard → **R2** → **Create bucket** →
   `accreditgenius-uploads`.
2. **R2 → Manage R2 API Tokens → Create API Token** (Object Read & Write).
   Note the **Access Key ID**, **Secret Access Key**, and your
   **account ID**.
3. Your S3 endpoint is:
   `https://<ACCOUNT_ID>.r2.cloudflarestorage.com`

You now have every value the app needs. The environment variables are:

```
DATABASE_URL          = <Neon pooled string>   (Path A)  /  Hyperdrive (Path B)
AUTH_SECRET           = openssl rand -base64 32
AUTH_URL              = https://<your production URL>
STORAGE_DRIVER        = s3
S3_ENDPOINT           = https://<ACCOUNT_ID>.r2.cloudflarestorage.com
S3_BUCKET             = accreditgenius-uploads
S3_ACCESS_KEY_ID      = <R2 access key id>
S3_SECRET_ACCESS_KEY  = <R2 secret>
S3_REGION             = auto
JOBS_RUN_SECRET       = <any long random string>
ANTHROPIC_API_KEY     = <optional — enables AI generation/review>
# optional embeddings: VOYAGE_API_KEY or OPENAI_API_KEY
```

---

## Path A — Node host behind Cloudflare (recommended)

Runs the Next.js server on a Node host; Cloudflare fronts it for
DNS/CDN/SSL. Any Node host works — steps below use **Render** (free tier).

### A1. Deploy the app on Render
1. render.com → **New → Web Service** → connect your GitHub repo, branch
   `claude/accreditgenius-saas-build-y2kut8` (or `main` after you merge).
2. Settings:
   - **Build command**: `pnpm install && pnpm build`
   - **Start command**: `pnpm start`
   - **Environment**: add every variable from the list above
     (`AUTH_URL` = your Render URL for now, e.g.
     `https://accreditgenius.onrender.com`).
3. Create the service and wait for the first deploy to go green.

### A2. Drain the job queue on a schedule
Render → **New → Cron Job** (or use the service's built-in cron):
- Schedule `* * * * *` (every minute)
- Command:
  ```bash
  curl -fsS -X POST "$APP_URL/api/jobs/run" \
    -H "Authorization: Bearer $JOBS_RUN_SECRET"
  ```
  (set `APP_URL` and `JOBS_RUN_SECRET` as cron env vars)

### A3. Put Cloudflare in front
1. Cloudflare dashboard → **Add a site** → your domain → follow the
   nameserver steps (or use an existing zone).
2. **DNS → Add record**: `CNAME  app  accreditgenius.onrender.com`
   (Proxy status **Proxied / orange cloud**).
3. **SSL/TLS → Overview → Full (strict)**.
4. Update `AUTH_URL` on Render to `https://app.yourdomain.com` and add the
   same host to Google OAuth redirect URIs if you enabled Google login.

Done — the app is live on your Cloudflare-managed domain.

---

## Path B — Cloudflare Workers (OpenNext)

The Worker bundle already builds (`pnpm cf:build` → `.open-next/worker.js`).
Two runtime specifics to wire up:

### B1. Postgres over Workers — Cloudflare Hyperdrive
Workers can't open arbitrary TCP to Postgres; Hyperdrive gives a pooled
endpoint they can reach.
```bash
pnpm exec wrangler hyperdrive create accreditgenius-db \
  --connection-string="<Neon DIRECT (non-pooler) string>"
```
Add the returned id to `wrangler.jsonc`:
```jsonc
"hyperdrive": [
  { "binding": "HYPERDRIVE", "id": "<hyperdrive-id>" }
]
```
At runtime set `DATABASE_URL` to the Hyperdrive binding's
`connectionString` (see the Prisma × Hyperdrive guide:
https://developers.cloudflare.com/hyperdrive/examples/prisma/). This keeps
Prisma unchanged. *If you prefer no Hyperdrive, add the Neon serverless
driver adapter (`@prisma/adapter-neon`) in `src/lib/db.ts` and enable
`driverAdapters` in the Prisma schema instead.*

### B2. R2 binding
`wrangler.jsonc` already declares the `UPLOADS` R2 binding. The app's S3
driver also works against R2 over HTTPS using the `S3_*` secrets, so either
approach is fine.

### B3. Secrets + deploy
```bash
# set each secret (repeat per variable in the shared list)
pnpm exec wrangler secret put AUTH_SECRET
pnpm exec wrangler secret put S3_SECRET_ACCESS_KEY
# …etc…

pnpm cf:deploy      # opennextjs-cloudflare build && wrangler deploy
```
Or connect the repo in **Workers & Pages → Create → Connect to Git** with
build `pnpm cf:build` and deploy `pnpm exec wrangler deploy`.

### B4. Cron for jobs
`wrangler.jsonc` already has a `* * * * *` cron trigger; point a scheduled
handler (or external cron) at `POST /api/jobs/run` with the
`JOBS_RUN_SECRET` bearer token.

### Path B caveats (be aware)
- **PDF ingestion on workerd**: `pdf-parse`/`pdfjs` use heavy Node APIs. If
  ingestion misbehaves on Workers, run *only* the ingestion/export jobs in a
  small companion Node service (Fly/Render/Cloudflare Container) — the
  DB-backed job queue already decouples this, so the web app stays on
  Workers. Everything else (auth, RAG query, review, exports of already-
  parsed content) is fine.
- Verify with `pnpm cf:preview` (local workerd) before shipping.

---

## Post-deploy checklist
- [ ] Neon has `vector` + `pg_trgm`; `pnpm db:deploy` applied
- [ ] `AUTH_SECRET` strong; `AUTH_URL` = the real HTTPS URL
- [ ] `STORAGE_DRIVER=s3` with working R2 keys (upload a doc → it appears in R2)
- [ ] A scheduled call hits `/api/jobs/run` every minute
- [ ] Sign in with a seeded demo account (`qa@demo.edu` / `Demo1234!`)
- [ ] Google OAuth redirect URIs include the prod domain (if enabled)
```
