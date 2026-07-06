import { notFound } from "next/navigation";

import { ReviewDetail } from "@/components/reviews/review-detail";
import { prisma } from "@/lib/db";
import { requireTenant } from "@/lib/tenant";

export default async function ReviewPage({
  params,
}: {
  params: Promise<{ locale: string; id: string }>;
}) {
  const { locale, id } = await params;
  const tenant = await requireTenant(locale);

  const review = await prisma.review.findFirst({
    where: { id, institutionId: tenant.institutionId },
    include: {
      document: { select: { title: true } },
      pack: { select: { nameEn: true, nameAr: true, code: true } },
      program: { select: { code: true, nameEn: true, nameAr: true } },
    },
  });
  if (!review) notFound();

  return (
    <ReviewDetail
      reviewId={review.id}
      documentTitle={review.document.title}
      packName={locale === "ar" && review.pack.nameAr ? review.pack.nameAr : review.pack.nameEn}
      programLabel={
        review.program
          ? `${review.program.code} — ${locale === "ar" ? review.program.nameAr : review.program.nameEn}`
          : null
      }
      initialStatus={review.status}
    />
  );
}

export const dynamic = "force-dynamic";
