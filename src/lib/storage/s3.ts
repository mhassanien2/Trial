import crypto from "node:crypto";

import type { StorageProvider } from "./index";

/**
 * S3-compatible storage (AWS S3 / Cloudflare R2) using AWS Signature v4,
 * signed with node:crypto — no SDK dependency. Configure via env:
 *   S3_ENDPOINT           e.g. https://<accountid>.r2.cloudflarestorage.com
 *   S3_BUCKET             bucket name
 *   S3_ACCESS_KEY_ID
 *   S3_SECRET_ACCESS_KEY
 *   S3_REGION             default "auto" (R2) / your AWS region
 */
export class S3StorageProvider implements StorageProvider {
  private endpoint: string;
  private bucket: string;
  private accessKeyId: string;
  private secretAccessKey: string;
  private region: string;

  constructor() {
    this.endpoint = required("S3_ENDPOINT").replace(/\/+$/, "");
    this.bucket = required("S3_BUCKET");
    this.accessKeyId = required("S3_ACCESS_KEY_ID");
    this.secretAccessKey = required("S3_SECRET_ACCESS_KEY");
    this.region = process.env.S3_REGION ?? "auto";
  }

  async put(key: string, data: Buffer, contentType: string): Promise<void> {
    const res = await this.send("PUT", key, data, contentType);
    if (!res.ok) throw new Error(`S3 put failed: ${res.status} ${await res.text()}`);
  }

  async get(key: string): Promise<Buffer> {
    const res = await this.send("GET", key);
    if (!res.ok) throw new Error(`S3 get failed: ${res.status}`);
    return Buffer.from(await res.arrayBuffer());
  }

  async delete(key: string): Promise<void> {
    const res = await this.send("DELETE", key);
    if (!res.ok && res.status !== 404) {
      throw new Error(`S3 delete failed: ${res.status}`);
    }
  }

  async exists(key: string): Promise<boolean> {
    const res = await this.send("HEAD", key);
    return res.ok;
  }

  private async send(
    method: string,
    key: string,
    body?: Buffer,
    contentType?: string
  ): Promise<Response> {
    const url = new URL(`${this.endpoint}/${this.bucket}/${encodeKey(key)}`);
    const host = url.host;
    const now = new Date();
    const amzDate = now.toISOString().replace(/[:-]|\.\d{3}/g, "");
    const dateStamp = amzDate.slice(0, 8);

    const payloadHash = sha256Hex(body ?? Buffer.alloc(0));
    const headers: Record<string, string> = {
      host,
      "x-amz-content-sha256": payloadHash,
      "x-amz-date": amzDate,
    };
    if (contentType) headers["content-type"] = contentType;

    const signedHeaders = Object.keys(headers).sort().join(";");
    const canonicalHeaders =
      Object.keys(headers)
        .sort()
        .map((h) => `${h}:${headers[h]}\n`)
        .join("") ;

    const canonicalRequest = [
      method,
      url.pathname,
      "", // no query
      canonicalHeaders,
      signedHeaders,
      payloadHash,
    ].join("\n");

    const scope = `${dateStamp}/${this.region}/s3/aws4_request`;
    const stringToSign = [
      "AWS4-HMAC-SHA256",
      amzDate,
      scope,
      sha256Hex(Buffer.from(canonicalRequest)),
    ].join("\n");

    const signingKey = this.signingKey(dateStamp);
    const signature = hmac(signingKey, stringToSign).toString("hex");

    const authorization =
      `AWS4-HMAC-SHA256 Credential=${this.accessKeyId}/${scope}, ` +
      `SignedHeaders=${signedHeaders}, Signature=${signature}`;

    const sendBody =
      method === "GET" || method === "HEAD" || !body
        ? undefined
        : new Uint8Array(body);
    return fetch(url, {
      method,
      headers: { ...headers, Authorization: authorization },
      body: sendBody,
    });
  }

  private signingKey(dateStamp: string): Buffer {
    const kDate = hmac(Buffer.from(`AWS4${this.secretAccessKey}`), dateStamp);
    const kRegion = hmac(kDate, this.region);
    const kService = hmac(kRegion, "s3");
    return hmac(kService, "aws4_request");
  }
}

function required(name: string): string {
  const v = process.env[name];
  if (!v) throw new Error(`Missing required storage env var: ${name}`);
  return v;
}

function encodeKey(key: string): string {
  return key
    .split("/")
    .map((seg) => encodeURIComponent(seg))
    .join("/");
}

function sha256Hex(data: Buffer): string {
  return crypto.createHash("sha256").update(data).digest("hex");
}

function hmac(key: Buffer, data: string): Buffer {
  return crypto.createHmac("sha256", key).update(data, "utf8").digest();
}
