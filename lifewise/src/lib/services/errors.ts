// Domain errors. Each carries a stable code the API layer maps to an HTTP status and a
// localisable message key. Messages are never built from user input.

export type AppErrorCode =
  | "VALIDATION_FAILED"
  | "EMAIL_TAKEN"
  | "INVALID_CREDENTIALS"
  | "RATE_LIMITED"
  | "UNAUTHENTICATED"
  | "CONSENT_REQUIRED"
  | "NOT_FOUND"
  | "CONFIRMATION_FAILED";

export class AppError extends Error {
  readonly code: AppErrorCode;
  readonly details?: unknown;

  constructor(code: AppErrorCode, message: string, details?: unknown) {
    super(message);
    this.name = "AppError";
    this.code = code;
    this.details = details;
  }
}
