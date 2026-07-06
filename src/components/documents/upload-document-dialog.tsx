"use client";

import { useRef, useState } from "react";
import { Upload } from "lucide-react";
import { useTranslations } from "next-intl";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const KINDS = [
  "STANDARDS_SOURCE",
  "TEMPLATE_SOURCE",
  "REVIEW_SUBJECT",
  "EVIDENCE",
  "OTHER",
] as const;

export function UploadDocumentDialog({
  onUploaded,
  defaultKind = "STANDARDS_SOURCE",
  compact = false,
}: {
  onUploaded: () => void;
  defaultKind?: (typeof KINDS)[number];
  compact?: boolean;
}) {
  const t = useTranslations("documents");
  const [open, setOpen] = useState(false);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const formRef = useRef<HTMLFormElement>(null);

  async function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setPending(true);
    setError(null);

    const formData = new FormData(e.currentTarget);
    const res = await fetch("/api/documents", { method: "POST", body: formData });

    if (!res.ok) {
      const json = (await res.json().catch(() => null)) as { error?: string } | null;
      setError(json?.error ?? t("uploadFailed"));
      setPending(false);
      return;
    }
    formRef.current?.reset();
    setPending(false);
    setOpen(false);
    onUploaded();
  }

  if (!open) {
    return (
      <Button onClick={() => setOpen(true)} size={compact ? "sm" : "default"}>
        <Upload />
        {t("upload")}
      </Button>
    );
  }

  return (
    <Card>
      <CardContent className="pt-6">
        <form ref={formRef} onSubmit={onSubmit} className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2 sm:col-span-2">
            <Label htmlFor="file">{t("file")}</Label>
            <Input id="file" name="file" type="file" required accept=".pdf,.docx,.xlsx" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="title">{t("titleField")}</Label>
            <Input id="title" name="title" placeholder="NCAAA Accreditation Documents" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="kind">{t("kindField")}</Label>
            <select
              id="kind"
              name="kind"
              defaultValue={defaultKind}
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
            >
              {KINDS.map((k) => (
                <option key={k} value={k}>
                  {t(`kinds.${k}`)}
                </option>
              ))}
            </select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="language">{t("languageField")}</Label>
            <select
              id="language"
              name="language"
              defaultValue="EN"
              className="flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm"
            >
              <option value="EN">English</option>
              <option value="AR">العربية</option>
            </select>
          </div>
          <div className="space-y-2">
            <Label htmlFor="country">{t("countryField")}</Label>
            <Input id="country" name="country" maxLength={2} placeholder="SA" />
          </div>
          <div className="space-y-2">
            <Label htmlFor="formCode">{t("formCodeField")}</Label>
            <Input id="formCode" name="formCode" maxLength={30} placeholder="TP-153" />
          </div>
          {error ? (
            <p role="alert" className="text-sm text-destructive sm:col-span-2">
              {error}
            </p>
          ) : null}
          <div className="flex gap-2 sm:col-span-2">
            <Button type="submit" disabled={pending}>
              {pending ? t("uploading") : t("submit")}
            </Button>
            <Button type="button" variant="ghost" onClick={() => setOpen(false)}>
              ✕
            </Button>
          </div>
        </form>
      </CardContent>
    </Card>
  );
}
