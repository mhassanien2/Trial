import Anthropic from "@anthropic-ai/sdk";

/** Model is pinned per spec; override only via env for testing. */
export const AI_MODEL = process.env.AI_MODEL ?? "claude-sonnet-4-6";

export function isAiEnabled(): boolean {
  return Boolean(process.env.ANTHROPIC_API_KEY);
}

let client: Anthropic | null = null;

export function getAnthropic(): Anthropic {
  if (!isAiEnabled()) {
    throw new Error(
      "ANTHROPIC_API_KEY is not configured — AI features are disabled"
    );
  }
  client ??= new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
  return client;
}
