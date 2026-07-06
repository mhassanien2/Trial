"use client";

import { useState } from "react";
import { MessageSquareQuote } from "lucide-react";
import { useTranslations } from "next-intl";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";

interface Source {
  index: number;
  documentTitle: string;
  page: number | null;
  headingPath: string | null;
  excerpt: string;
}

interface AskResponse {
  answer: string | null;
  lowConfidence?: boolean;
  aiDisabled?: boolean;
  sources: Source[];
  error?: string;
}

export function QaChat({ locale, packId }: { locale: string; packId?: string }) {
  const t = useTranslations("chat");
  const [question, setQuestion] = useState("");
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [asked, setAsked] = useState<string | null>(null);

  async function ask(e: React.FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;
    setPending(true);
    setResult(null);
    setAsked(question);

    const res = await fetch("/api/standards/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, packId, locale }),
    });
    setResult((await res.json()) as AskResponse);
    setPending(false);
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <MessageSquareQuote className="h-4 w-4" />
          {t("title")}
        </CardTitle>
        <CardDescription>{t("subtitle")}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <form onSubmit={ask} className="flex gap-2">
          <Input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={t("placeholder")}
            maxLength={2000}
          />
          <Button type="submit" disabled={pending || question.trim().length < 3}>
            {pending ? t("asking") : t("ask")}
          </Button>
        </form>

        {result ? (
          <div className="space-y-4" data-testid="qa-result">
            {asked ? <p className="text-sm font-medium">{asked}</p> : null}

            {result.lowConfidence ? (
              <p className="rounded-md border border-dashed p-3 text-sm text-muted-foreground">
                {t("lowConfidence")}
              </p>
            ) : null}

            {result.aiDisabled ? (
              <p className="rounded-md bg-secondary p-3 text-sm text-secondary-foreground">
                {t("aiDisabled")}
              </p>
            ) : null}

            {result.answer ? (
              <div className="whitespace-pre-wrap rounded-md bg-muted p-4 text-sm leading-relaxed">
                {result.answer}
              </div>
            ) : null}

            {result.sources.length > 0 ? (
              <div className="space-y-2">
                <p className="text-xs font-semibold uppercase text-muted-foreground">
                  {t("sources")}
                </p>
                {result.sources.map((s) => (
                  <div
                    key={s.index}
                    className="rounded-md border p-3 text-sm"
                    data-testid="qa-source"
                  >
                    <div className="mb-1 flex flex-wrap items-center gap-2">
                      <Badge variant="secondary">[{s.index}]</Badge>
                      <span className="font-medium">{s.documentTitle}</span>
                      {s.page != null ? (
                        <span className="text-xs text-muted-foreground">
                          {t("page", { page: s.page })}
                        </span>
                      ) : null}
                      {s.headingPath ? (
                        <span className="truncate text-xs text-muted-foreground">
                          {s.headingPath}
                        </span>
                      ) : null}
                    </div>
                    <p className="line-clamp-3 text-muted-foreground">{s.excerpt}</p>
                  </div>
                ))}
              </div>
            ) : null}
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}
