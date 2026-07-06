"use client";

import { useMemo, useState } from "react";
import { Download, Plus, Printer, Sparkles, Trash2 } from "lucide-react";
import { useLocale, useTranslations } from "next-intl";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type {
  GeneratedContent,
  TemplateField,
  TemplateSchema,
  TemplateTable,
} from "@/lib/templates/schema";

type FieldLike = Omit<TemplateField, "kind">;

export function DocumentWizard({
  docId,
  title,
  templateName,
  formCode,
  programLabel,
  schema,
  initialContent,
  aiEnabled,
}: {
  docId: string;
  title: string;
  templateName: string;
  formCode: string | null;
  programLabel: string;
  schema: TemplateSchema;
  initialContent: GeneratedContent;
  aiEnabled: boolean;
}) {
  const t = useTranslations("wizard");
  const locale = useLocale();
  const [content, setContent] = useState<GeneratedContent>(initialContent);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<number | null>(null);
  const [suggesting, setSuggesting] = useState<string | null>(null);
  const [fieldNotes, setFieldNotes] = useState<Record<string, string>>({});

  const setValue = (fieldId: string, value: string) =>
    setContent((c) => ({ ...c, [fieldId]: value }));

  const getValue = (fieldId: string): string => {
    const v = content[fieldId];
    return typeof v === "string" ? v : "";
  };

  const tableRows = (tableId: string): Array<Record<string, string>> => {
    const v = content[tableId];
    return v && typeof v === "object" && "rows" in v ? v.rows : [];
  };

  async function save() {
    setSaving(true);
    await fetch(`/api/generated/${docId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content }),
    });
    setSaving(false);
    setSavedAt(Date.now());
  }

  async function suggest(field: FieldLike) {
    setSuggesting(field.id);
    setFieldNotes((n) => ({ ...n, [field.id]: "" }));
    const res = await fetch(`/api/generated/${docId}/suggest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fieldId: field.id, locale }),
    });
    const json = (await res.json()) as {
      suggestion: string | null;
      notEnoughInfo?: boolean;
      aiDisabled?: boolean;
    };
    if (json.suggestion) {
      setValue(field.id, json.suggestion);
    } else if (json.notEnoughInfo) {
      setFieldNotes((n) => ({ ...n, [field.id]: t("notEnoughInfo") }));
    }
    setSuggesting(null);
  }

  const fieldInput = (field: FieldLike) => {
    const common = {
      id: field.id,
      value: getValue(field.id),
      onChange: (
        e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>
      ) => setValue(field.id, e.target.value),
      placeholder: field.placeholder ?? "",
    };
    return field.fieldType === "textarea" ? (
      <textarea
        {...common}
        rows={3}
        className="flex w-full rounded-md border border-input bg-transparent px-3 py-2 text-sm shadow-sm placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
      />
    ) : (
      <Input {...common} type={field.fieldType === "date" ? "date" : "text"} />
    );
  };

  const suggestButton = (field: FieldLike) =>
    aiEnabled ? (
      <Button
        type="button"
        variant="ghost"
        size="sm"
        disabled={suggesting !== null}
        onClick={() => void suggest(field)}
      >
        <Sparkles className="h-3.5 w-3.5" />
        {suggesting === field.id ? t("suggesting") : t("suggest")}
      </Button>
    ) : null;

  const sectionsNav = useMemo(
    () =>
      schema.sections.map((s) => (
        <a
          key={s.id}
          href={`#${s.id}`}
          className="block rounded px-2 py-1 text-sm text-muted-foreground hover:bg-accent hover:text-accent-foreground"
        >
          {s.heading}
        </a>
      )),
    [schema.sections]
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
          <p className="text-sm text-muted-foreground">
            {t("template")}: {formCode ? `${formCode} — ` : ""}
            {templateName} · {t("program")}: {programLabel}
          </p>
          <p className="text-xs text-muted-foreground">{t("requiresInputNote")}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button onClick={() => void save()} disabled={saving}>
            {saving ? t("saving") : t("save")}
          </Button>
          {savedAt ? <Badge variant="secondary">{t("saved")}</Badge> : null}
          <Button asChild variant="outline">
            <a href={`/api/generated/${docId}/export`}>
              <Download />
              {t("exportDocx")}
            </a>
          </Button>
          <Button asChild variant="outline">
            <a href={`/${locale}/generated/${docId}/print`} target="_blank" rel="noopener">
              <Printer />
              {t("printView")}
            </a>
          </Button>
        </div>
      </div>

      <div className="flex gap-6">
        <nav className="hidden w-56 shrink-0 space-y-1 lg:block">
          <p className="px-2 text-xs font-semibold uppercase text-muted-foreground">
            {t("sections")}
          </p>
          {sectionsNav}
        </nav>

        <div className="min-w-0 flex-1 space-y-6">
          {schema.sections.map((section) => (
            <Card key={section.id} id={section.id}>
              <CardHeader>
                <CardTitle className="text-base">{section.heading}</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {section.blocks.map((block) => {
                  if (block.kind === "static") {
                    return (
                      <p
                        key={block.id}
                        className={
                          block.bold
                            ? "text-sm font-semibold"
                            : "text-sm text-muted-foreground"
                        }
                      >
                        {block.text}
                      </p>
                    );
                  }
                  if (block.kind === "field") {
                    return (
                      <div key={block.id} className="space-y-1.5">
                        <div className="flex items-center justify-between">
                          <Label htmlFor={block.id}>{block.label}</Label>
                          {suggestButton(block)}
                        </div>
                        {fieldInput(block)}
                        {fieldNotes[block.id] ? (
                          <p className="text-xs text-muted-foreground">
                            {fieldNotes[block.id]}
                          </p>
                        ) : null}
                      </div>
                    );
                  }
                  return (
                    <TableEditor
                      key={block.id}
                      table={block}
                      getValue={getValue}
                      setValue={setValue}
                      rows={tableRows(block.id)}
                      setRows={(rows) =>
                        setContent((c) => ({ ...c, [block.id]: { rows } }))
                      }
                      suggestButton={suggestButton}
                      fieldNotes={fieldNotes}
                    />
                  );
                })}
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}

function TableEditor({
  table,
  getValue,
  setValue,
  rows,
  setRows,
  suggestButton,
  fieldNotes,
}: {
  table: TemplateTable;
  getValue: (id: string) => string;
  setValue: (id: string, v: string) => void;
  rows: Array<Record<string, string>>;
  setRows: (rows: Array<Record<string, string>>) => void;
  suggestButton: (field: FieldLike) => React.ReactNode;
  fieldNotes: Record<string, string>;
}) {
  const t = useTranslations("wizard");

  return (
    <div className="overflow-x-auto rounded-md border">
      <table className="w-full text-sm">
        <tbody>
          {table.rows.map((row, ri) => (
            <tr key={ri} className="border-b last:border-0">
              {row.cells.map((cell, ci) => (
                <td
                  key={ci}
                  colSpan={cell.colSpan > 1 ? cell.colSpan : undefined}
                  className={`p-2 align-top ${cell.header ? "bg-muted font-semibold" : ""}`}
                >
                  {cell.field ? (
                    <div className="space-y-1">
                      <div className="flex items-center justify-between gap-1">
                        <span className="text-xs text-muted-foreground">
                          {cell.field.label}
                        </span>
                        {suggestButton(cell.field)}
                      </div>
                      <Input
                        value={getValue(cell.field.id)}
                        onChange={(e) => setValue(cell.field!.id, e.target.value)}
                      />
                      {fieldNotes[cell.field.id] ? (
                        <p className="text-xs text-muted-foreground">
                          {fieldNotes[cell.field.id]}
                        </p>
                      ) : null}
                    </div>
                  ) : (
                    cell.static
                  )}
                </td>
              ))}
            </tr>
          ))}

          {table.repeatingRow
            ? rows.map((rowValues, ri) => (
                <tr key={`extra-${ri}`} className="border-b last:border-0">
                  {table.repeatingRow!.cells.map((cell, ci) => (
                    <td key={ci} className="p-2 align-top">
                      {cell.field ? (
                        <Input
                          value={rowValues[cell.field.id] ?? ""}
                          onChange={(e) => {
                            const next = rows.map((r, i) =>
                              i === ri ? { ...r, [cell.field!.id]: e.target.value } : r
                            );
                            setRows(next);
                          }}
                        />
                      ) : (
                        cell.static
                      )}
                    </td>
                  ))}
                  <td className="w-10 p-2">
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      aria-label={t("removeRow")}
                      onClick={() => setRows(rows.filter((_, i) => i !== ri))}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </td>
                </tr>
              ))
            : null}
        </tbody>
      </table>
      {table.repeatingRow ? (
        <div className="border-t p-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => setRows([...rows, {}])}
          >
            <Plus className="h-4 w-4" />
            {t("addRow")}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
