// Passthrough root layout. The real <html>/<body> live in the locale
// layout (src/app/[locale]/layout.tsx) so lang/dir are locale-aware.
// This split lets the app run without Node-runtime middleware (required
// for Cloudflare Workers), while "/" is handled by src/app/page.tsx.
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return children;
}
