import { z } from "zod";

import { runReview } from "@/lib/review/engine";

const payloadSchema = z.object({ reviewId: z.string() });

export async function runReviewHandler(rawPayload: unknown): Promise<void> {
  const { reviewId } = payloadSchema.parse(rawPayload);
  await runReview(reviewId);
}
