import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 p-6 text-center">
      <h2 className="text-2xl font-bold">404</h2>
      <p className="text-sm text-muted-foreground">This page could not be found.</p>
      <Link href="/" className="text-primary hover:underline">
        Go home
      </Link>
    </div>
  );
}
