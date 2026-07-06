import { NextResponse } from "next/server";

import { runPendingJobs } from "@/lib/jobs/queue";
import { ForbiddenError } from "@/lib/rbac";
import { getTenant } from "@/lib/tenant";

export const runtime = "nodejs";

/**
 * Drains pending jobs. Callable by any signed-in user (uploads already
 * kick the runner in-process) or by an external scheduler using
 * the JOBS_RUN_SECRET bearer token.
 */
export async function POST(req: Request) {
  try {
    const secret = process.env.JOBS_RUN_SECRET;
    const bearer = req.headers.get("authorization")?.replace(/^Bearer\s+/i, "");
    const bySecret = Boolean(secret && bearer && bearer === secret);

    if (!bySecret) {
      const tenant = await getTenant();
      if (!tenant) throw new ForbiddenError("Not authenticated");
    }

    const processed = await runPendingJobs(10);
    return NextResponse.json({ processed });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[jobs:run]", err);
    return NextResponse.json({ error: "Job run failed" }, { status: 500 });
  }
}
