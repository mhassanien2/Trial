import { NextResponse } from "next/server";
import { z } from "zod";

import { logAudit } from "@/lib/audit";
import { computeMapping } from "@/lib/mapper/engine";
import { LIMITS, rateLimit, tooManyRequests } from "@/lib/rate-limit";
import { ForbiddenError } from "@/lib/rbac";
import { requireTenantWith } from "@/lib/tenant";

export const runtime = "nodejs";
export const maxDuration = 60;

const bodySchema = z.object({
  programId: z.string(),
  fromPackId: z.string(),
  toPackId: z.string(),
  locale: z.enum(["en", "ar"]).default("en"),
});

/** Computes a cross-standard mapping matrix for a program (synchronous). */
export async function POST(req: Request) {
  try {
    const tenant = await requireTenantWith("standards.view");

    const rl = rateLimit(tenant.userId, "ai", LIMITS.ai);
    if (!rl.ok) return tooManyRequests(rl);

    const body = bodySchema.safeParse(await req.json());
    if (!body.success) {
      return NextResponse.json({ error: "Invalid request" }, { status: 400 });
    }
    if (body.data.fromPackId === body.data.toPackId) {
      return NextResponse.json(
        { error: "Choose two different standards packs" },
        { status: 400 }
      );
    }

    const result = await computeMapping({
      institutionId: tenant.institutionId,
      programId: body.data.programId,
      fromPackId: body.data.fromPackId,
      toPackId: body.data.toPackId,
      locale: body.data.locale,
    });

    await logAudit({
      institutionId: tenant.institutionId,
      userId: tenant.userId,
      action: "mapper.compute",
      entityType: "Program",
      entityId: body.data.programId,
      metadata: { fromPackId: body.data.fromPackId, toPackId: body.data.toPackId },
    });

    return NextResponse.json({ result });
  } catch (err) {
    if (err instanceof ForbiddenError) {
      return NextResponse.json({ error: err.message }, { status: 403 });
    }
    console.error("[mapper:POST]", err);
    return NextResponse.json({ error: "Mapping failed" }, { status: 500 });
  }
}
