import type { TargetKind } from "@/entities/targets"
import type { StageTarget } from "./params"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import { Input } from "@/shared/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select"
import { Switch } from "@/shared/ui/switch"

const KINDS: { value: TargetKind; label: string }[] = [
  { value: "text_overlay", label: "Текст" },
  { value: "watermark", label: "Вотермарк" },
  { value: "object", label: "Объект" },
]

const WHERES: { value: string; label: string }[] = [
  { value: "", label: "Где угодно" },
  { value: "top", label: "Сверху" },
  { value: "bottom", label: "Снизу" },
  { value: "left", label: "Слева" },
  { value: "right", label: "Справа" },
  { value: "center", label: "Центр" },
  { value: "top-left", label: "Верх-лево" },
  { value: "top-right", label: "Верх-право" },
  { value: "bottom-left", label: "Низ-лево" },
  { value: "bottom-right", label: "Низ-право" },
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

  if (targets.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border/80 px-3 py-4 text-center">
        <p className="text-xs text-muted-foreground">
          Целей пока нет. Добавьте вручную или нажмите «Интерпретировать» выше.
        </p>
        <Button size="sm" variant="outline" className="mt-2" disabled={disabled} onClick={add}>
          Добавить цель
        </Button>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-2">
      {targets.map((t, i) => (
        <div key={i} className="flex flex-col gap-1.5 rounded-md border border-border/70 bg-background/40 p-2">
          <div className="flex items-center gap-1.5">
            <Select
              value={t.kind}
              onValueChange={(v) => update(i, { kind: v as TargetKind })}
              disabled={disabled}
            >
              <SelectTrigger size="sm" className="w-[7.5rem] shrink-0">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {KINDS.map((k) => (
                  <SelectItem key={k.value} value={k.value}>
                    {k.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Input
              value={t.query}
              onChange={(e) => update(i, { query: e.target.value })}
              placeholder="Что удалить"
              disabled={disabled}
              className="h-7 flex-1 text-xs"
            />
            <Switch
              size="sm"
              checked={t.enabled}
              onCheckedChange={(checked) => update(i, { enabled: checked })}
              aria-label="Использовать цель"
              disabled={disabled}
            />
            <Button
              size="icon-xs"
              variant="ghost"
              aria-label="Удалить цель"
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
              <SelectTrigger size="sm" className="w-[7.5rem] shrink-0">
                <SelectValue placeholder="Где угодно" />
              </SelectTrigger>
              <SelectContent>
                {WHERES.map((w) => (
                  <SelectItem key={w.value || "any"} value={w.value}>
                    {w.label}
                  </SelectItem>
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
        Добавить цель
      </Button>
    </div>
  )
}
