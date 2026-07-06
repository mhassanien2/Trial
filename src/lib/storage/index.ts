import { LocalStorageProvider } from "./local";

/**
 * Storage abstraction. Dev uses the local filesystem under UPLOADS_DIR;
 * an S3-compatible provider can be added later without touching callers.
 */
export interface StorageProvider {
  put(key: string, data: Buffer, contentType: string): Promise<void>;
  get(key: string): Promise<Buffer>;
  delete(key: string): Promise<void>;
  exists(key: string): Promise<boolean>;
}

let provider: StorageProvider | null = null;

export function getStorage(): StorageProvider {
  if (provider) return provider;
  const driver = process.env.STORAGE_DRIVER ?? "local";
  switch (driver) {
    case "local":
      provider = new LocalStorageProvider(process.env.UPLOADS_DIR ?? "./uploads");
      return provider;
    default:
      throw new Error(`Unknown STORAGE_DRIVER: ${driver}`);
  }
}
