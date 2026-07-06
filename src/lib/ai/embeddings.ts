import crypto from "node:crypto";

/**
 * Single embeddings abstraction — the ONLY module allowed to talk to an
 * embeddings provider, so the provider can be swapped via env config.
 *
 * Providers:
 *  - "voyage": Voyage AI (VOYAGE_API_KEY), recommended for production.
 *  - "openai": OpenAI text-embedding-3-small (OPENAI_API_KEY).
 *  - "local":  deterministic hashed bag-of-words — NO API needed.
 *              Dev/demo only; retrieval quality is limited. Selected
 *              automatically when no provider key is configured.
 *
 * All providers are normalized to EMBEDDING_DIM (zero-padded when the
 * native dimension is smaller — cosine distance is unaffected).
 */
export const EMBEDDING_DIM = 1536;

type ProviderName = "voyage" | "openai" | "local";

function resolveProvider(): ProviderName {
  const explicit = process.env.EMBEDDINGS_PROVIDER as ProviderName | undefined;
  if (explicit) return explicit;
  if (process.env.VOYAGE_API_KEY) return "voyage";
  if (process.env.OPENAI_API_KEY) return "openai";
  return "local";
}

export function embeddingsProviderName(): ProviderName {
  return resolveProvider();
}

export async function embedTexts(texts: string[]): Promise<number[][]> {
  if (texts.length === 0) return [];
  const provider = resolveProvider();
  switch (provider) {
    case "voyage":
      return pad(await voyageEmbed(texts));
    case "openai":
      return pad(await openaiEmbed(texts));
    case "local":
      return texts.map(localEmbed);
  }
}

export async function embedQuery(text: string): Promise<number[]> {
  return (await embedTexts([text]))[0];
}

function pad(vectors: number[][]): number[][] {
  return vectors.map((v) => {
    if (v.length === EMBEDDING_DIM) return v;
    if (v.length > EMBEDDING_DIM) return v.slice(0, EMBEDDING_DIM);
    return [...v, ...new Array(EMBEDDING_DIM - v.length).fill(0)];
  });
}

async function voyageEmbed(texts: string[]): Promise<number[][]> {
  const res = await fetch("https://api.voyageai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.VOYAGE_API_KEY}`,
    },
    body: JSON.stringify({
      model: process.env.VOYAGE_MODEL ?? "voyage-3",
      input: texts,
    }),
  });
  if (!res.ok) throw new Error(`Voyage embeddings failed: ${res.status} ${await res.text()}`);
  const json = (await res.json()) as { data: Array<{ embedding: number[] }> };
  return json.data.map((d) => d.embedding);
}

async function openaiEmbed(texts: string[]): Promise<number[][]> {
  const res = await fetch("https://api.openai.com/v1/embeddings", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
    },
    body: JSON.stringify({ model: "text-embedding-3-small", input: texts }),
  });
  if (!res.ok) throw new Error(`OpenAI embeddings failed: ${res.status} ${await res.text()}`);
  const json = (await res.json()) as { data: Array<{ embedding: number[] }> };
  return json.data.map((d) => d.embedding);
}

/**
 * Deterministic local embedding: hashed word + character-trigram features,
 * L2-normalized. Language-agnostic (works for Arabic), zero-dependency,
 * and stable across runs — but NOT semantically strong. Dev/demo only.
 */
function localEmbed(text: string): number[] {
  const vec = new Array<number>(EMBEDDING_DIM).fill(0);
  const normalized = text.toLowerCase().normalize("NFKC");

  const add = (feature: string, weight: number) => {
    const h = crypto.createHash("md5").update(feature).digest();
    const idx = h.readUInt32BE(0) % EMBEDDING_DIM;
    const sign = h[4] % 2 === 0 ? 1 : -1;
    vec[idx] += sign * weight;
  };

  const words = normalized.split(/[^\p{L}\p{N}]+/u).filter((w) => w.length > 1);
  for (const w of words) {
    add(`w:${w}`, 1.0);
    for (let i = 0; i + 3 <= w.length; i++) add(`t:${w.slice(i, i + 3)}`, 0.35);
  }

  const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0)) || 1;
  return vec.map((x) => x / norm);
}
