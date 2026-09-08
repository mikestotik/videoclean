export type TargetKind = "watermark" | "text_overlay" | "object"

export type TargetRow = { kind: TargetKind; query: string; where: string | null }

export function parseTargetsJson(text: string): TargetRow[] {
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch {
    return []
  }
  if (!Array.isArray(data)) return []
  return data
    .map((item): TargetRow | null => {
      const r = (item ?? {}) as Record<string, unknown>
      const query = String(r.query ?? "").trim()
      if (!query) return null
      const kind = ["watermark", "text_overlay", "object"].includes(String(r.kind))
        ? (r.kind as TargetKind)
        : "object"
      const where = r.where ? String(r.where) : null
      return { kind, query, where }
    })
}

export function targetsToJson(rows: TargetRow[]): string {
  return JSON.stringify(
    rows.map((r) => ({ kind: r.kind, query: r.query, where: r.where })),
    null,
    2,
  )
}
