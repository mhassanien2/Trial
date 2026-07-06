import NextAuth from "next-auth";
import type { NextAuthConfig } from "next-auth";
import Credentials from "next-auth/providers/credentials";
import Google from "next-auth/providers/google";
import { PrismaAdapter } from "@auth/prisma-adapter";
import bcrypt from "bcryptjs";
import { z } from "zod";

import { prisma } from "@/lib/db";
import type { Role } from "@prisma/client";

const credentialsSchema = z.object({
  email: z.string().email(),
  password: z.string().min(1),
});

const providers: NextAuthConfig["providers"] = [
  Credentials({
    name: "Credentials",
    credentials: {
      email: { label: "Email", type: "email" },
      password: { label: "Password", type: "password" },
    },
    async authorize(credentials) {
      const parsed = credentialsSchema.safeParse(credentials);
      if (!parsed.success) return null;

      const user = await prisma.user.findUnique({
        where: { email: parsed.data.email.toLowerCase() },
      });
      if (!user || !user.passwordHash || !user.isActive) return null;

      const valid = await bcrypt.compare(parsed.data.password, user.passwordHash);
      if (!valid) return null;

      return {
        id: user.id,
        email: user.email,
        name: user.name,
        image: user.image,
      };
    },
  }),
];

// Google is optional in dev; enabled when env vars are present.
if (process.env.AUTH_GOOGLE_ID && process.env.AUTH_GOOGLE_SECRET) {
  providers.push(
    Google({ allowDangerousEmailAccountLinking: false })
  );
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  adapter: PrismaAdapter(prisma),
  session: { strategy: "jwt" },
  pages: { signIn: "/login" },
  // Trust the deployment host header so AUTH_URL need not be set manually
  // on the hosting platform (Render/Cloudflare fill the host at runtime).
  trustHost: true,
  providers,
  callbacks: {
    // Multi-tenant: no self-signup. Google sign-in is only allowed for
    // users that already exist (provisioned by an institution admin).
    async signIn({ user, account }) {
      if (account?.provider === "credentials") return true;
      if (!user.email) return false;
      const existing = await prisma.user.findUnique({
        where: { email: user.email.toLowerCase() },
      });
      return !!existing && existing.isActive;
    },
    async jwt({ token, user }) {
      // On first sign-in, enrich the JWT with tenant + RBAC claims.
      if (user?.id || !token.institutionId) {
        const dbUser = await prisma.user.findUnique({
          where: user?.id ? { id: user.id } : { email: token.email ?? "" },
          select: { id: true, role: true, institutionId: true, locale: true, isActive: true },
        });
        if (dbUser) {
          token.sub = dbUser.id;
          token.role = dbUser.role;
          token.institutionId = dbUser.institutionId;
          token.locale = dbUser.locale;
        }
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user) {
        session.user.id = token.sub as string;
        session.user.role = token.role as Role;
        session.user.institutionId = token.institutionId as string;
      }
      return session;
    },
  },
});
