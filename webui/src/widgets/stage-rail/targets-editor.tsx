import type { TargetKind } from "@/entities/targets"
import type { StageTarget } from "./params"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import { Input } from "@/shared/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select"
import { Switch } from "@/shared/ui/switch"

const KINDS: TargetKind[] = ["text_overlay", "watermark", "object"]
const WHERES = [
  "", "top", "bottom", "left", "right", "center",
  "top-left", "top-right", "bottom-left", "bottom-right",
]

type Props = {
  targets: StageTarget[]
  onChange: (targets: StageTarget[]) => void
  disabled?: boolean
}

export function TargetsEditor({ targets, onChange, disabled }: Props) {
  const update = (i: number, patch: Partial<StageTarget>) =>
    onChange(targets.map((t, j) => (j === i ? { ...t, ...patch } : t)))

  const remove = (i: number) => onChange(targets.filter((_, j) => j !== i))

  const add = () =>
    onChange([
      ...targets,
      { kind: "object", query: "", where: null, enabled: true, source: "manual" },
    ])

  return (
    <div className="flex flex-col gap-2">
      {targets.map((t, i) => (
        <div key={i} className="flex flex-col gap-1 rounded-md border p-2">
          <div className="flex items-center gap-1.5">
            <Select
              value={t.kind}
              onValueChange={(v) => update(i, { kind: v as TargetKind })}
              disabled={disabled}
            >
              <SelectTrigger size="sm" className="w-28 shrink-0">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {KINDS.map((k) => (
                  <SelectItem key={k} value={k}>{k}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Input
              value={t.query}
              onChange={(e) => update(i, { query: e.target.value })}
              placeholder="Что удалить"
              disabled={disabled}
              className="h-6 flex-1 text-xs"
            />
            <Switch
              size="sm"
              checked={t.enabled}
              onCheckedChange={(checked) => update(i, { enabled: checked })}
              aria-label="Использовать таргет"
              disabled={disabled}
            />
            <Button
              size="icon-xs"
              variant="ghost"
              aria-label="Удалить таргет"
              disabled={disabled}
              onClick={() => remove(i)}
            >
              ×
            </Button>
          </div>
          <div className="flex items-center gap-1.5">
            <Select
              value={t.where ?? ""}
              onValueChange={(v) => update(i, { where: v || null })}
              disabled={disabled}
            >
              <SelectTrigger size="sm" className="w-28 shrink-0">
                <SelectValue placeholder="где угодно" />
              </SelectTrigger>
              <SelectContent>
                {WHERES.map((w) => (
                  <SelectItem key={w} value={w}>{w || "где угодно"}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Badge variant={t.source === "auto" ? "secondary" : "outline"} className="text-[10px]">
              {t.source === "auto" ? "авто" : "вручную"}
            </Badge>
          </div>
        </div>
      ))}
      <Button size="sm" variant="outline" className="self-start" disabled={disabled} onClick={add}>
        + Таргет
      </Button>
    </div>
  )
}
