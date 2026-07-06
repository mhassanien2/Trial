import type { NextConfig } from "next";
import createNextIntlPlugin from "next-intl/plugin";

const withNextIntl = createNextIntlPlugin();

const nextConfig: NextConfig = {
  // pdf-parse/pdfjs-dist must stay external: bundling breaks the pdf.js
  // worker resolution at runtime.
  serverExternalPackages: ["@prisma/client", "bcryptjs", "pdf-parse", "pdfjs-dist"],
};

export default withNextIntl(nextConfig);
