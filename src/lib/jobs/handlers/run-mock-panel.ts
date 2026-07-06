import { z } from "zod";

import { runMockPanel } from "@/lib/panel/engine";

const payloadSchema = z.object({ runId: z.string() });

export async function runMockPanelHandler(rawPayload: unknown): Promise<void> {
  const { runId } = payloadSchema.parse(rawPayload);
  await runMockPanel(runId);
}
