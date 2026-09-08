import type { TargetKind, TargetRow } from "@/entities/targets"
import { Button } from "@/shared/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card"
import { Input } from "@/shared/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/shared/ui/select"

const KINDS: TargetKind[] = ["text_overlay", "watermark", "object"]
const WHERES = [
  "", "top", "bottom", "left", "right", "center",
  "top-left", "top-right", "bottom-left", "bottom-right",
]

type Props = { rows: TargetRow[]; onChange: (rows: TargetRow[]) => void }

export function TargetTable({ rows, onChange }: Props) {
  const update = (i: number, patch: Partial<TargetRow>) =>
    onChange(rows.map((r, j) => (j === i ? { ...r, ...patch } : r)))

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Таргеты (режим detect)</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-2">
        {rows.map((r, i) => (
          <div key={i} className="flex items-center gap-2">
            <Select value={r.kind} onValueChange={(v) => update(i, { kind: v as TargetKind })}>
              <SelectTrigger className="w-40">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {KINDS.map((k) => (
                  <SelectItem key={k} value={k}>{k}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Input
              value={r.query}
              onChange={(e) => update(i, { query: e.target.value })}
              className="flex-1"
              placeholder="query"
            />
            <Select value={r.where ?? ""} onValueChange={(v) => update(i, { where: v || null })}>
              <SelectTrigger className="w-36">
                <SelectValue placeholder="—" />
              </SelectTrigger>
              <SelectContent>
                {WHERES.map((w) => (
                  <SelectItem key={w} value={w}>{w || "—"}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button size="icon" variant="destructive" onClick={() => onChange(rows.filter((_, j) => j !== i))}>
              ×
            </Button>
          </div>
        ))}
        <Button
          size="sm"
          variant="outline"
          className="self-start"
          onClick={() => onChange([...rows, { kind: "object", query: "", where: null }])}
        >
          + таргет
        </Button>
      </CardContent>
    </Card>
  )
}
