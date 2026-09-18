import { useCallback, useEffect, useState } from "react"
import { deletePreset, listPresets, savePreset, type Preset } from "@/entities/preset"
import { usePoll } from "@/shared/hooks/usePoll"
import { api } from "@/shared/api/client"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import { Input } from "@/shared/ui/input"
import { Label } from "@/shared/ui/label"
import { Popover, PopoverContent, PopoverTrigger } from "@/shared/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select"
import { Slider } from "@/shared/ui/slider"
import { cn } from "@/shared/lib/utils"
import { ADVANCED_DEFAULTS, applyPreset, presetSnapshot, type EditorParams } from "./params"

type PollShape = { ollama: { ok: boolean; models: string[] } }

type OptionsShape = {
  detectors: string[]
  segmenters: string[]
}

export function BackendSelectors({
  detector,
  segmenter,
  onChange,
  disabled,
}: {
  detector: string
  segmenter: string
  onChange: (patch: { detector?: string; segmenter?: string }) => void
  disabled?: boolean
}) {
  const [opts, setOpts] = useState<OptionsShape>({ detectors: [], segmenters: [] })
  useEffect(() => {
    api<OptionsShape>("/api/options")
      .then((r) => setOpts({ detectors: r.detectors ?? [], segmenters: r.segmenters ?? [] }))
      .catch(() => setOpts({ detectors: [], segmenters: [] }))
  }, [])

  return (
    <div className="flex flex-col gap-1.5 text-xs">
      <div className="flex items-center gap-2">
        <Label className="w-20 shrink-0">Детектор</Label>
        <Select value={detector} onValueChange={(v) => { if (v) onChange({ detector: v }) }} disabled={disabled}>
          <SelectTrigger size="sm" className="flex-1">
            <SelectValue placeholder="grounding-dino" />
          </SelectTrigger>
          <SelectContent>
            {opts.detectors.map((d) => (
              <SelectItem key={d} value={d}>{d}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex items-center gap-2">
        <Label className="w-20 shrink-0">Сегментер</Label>
        <Select value={segmenter} onValueChange={(v) => { if (v) onChange({ segmenter: v }) }} disabled={disabled}>
          <SelectTrigger size="sm" className="flex-1">
            <SelectValue placeholder="sam2-video" />
          </SelectTrigger>
          <SelectContent>
            {opts.segmenters.map((s) => (
              <SelectItem key={s} value={s}>{s}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}

export function StageSection({
  n,
  title,
  hint,
  active,
  open,
  onOpenChange,
  children,
}: {
  n: number
  title: string
  hint?: string
  active: boolean
  open: boolean
  onOpenChange: (open: boolean) => void
  children: React.ReactNode
}) {
  return (
    <section
      className={cn(
        "border-b border-border/60 last:border-b-0",
        active && "bg-primary/[0.04]",
      )}
    >
      <button
        type="button"
        className="flex w-full items-start gap-2.5 px-3 py-3 text-left hover:bg-muted/30"
        onClick={() => onOpenChange(!open)}
        aria-expanded={open}
      >
        <span
          className={cn(
            "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold tabular-nums",
            active ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground",
          )}
        >
          {n}
        </span>
        <div className="min-w-0 flex-1">
          <h3 className="text-sm font-medium leading-none">{title}</h3>
          {hint && !open && <p className="mt-1 text-[11px] text-muted-foreground">{hint}</p>}
        </div>
        <span className="mt-0.5 text-[10px] text-muted-foreground">{open ? "▾" : "▸"}</span>
      </button>
      {open && (
        <div className="flex flex-col gap-2.5 px-3 pb-3.5 pl-10">
          {hint && <p className="-mt-1 text-[11px] text-muted-foreground">{hint}</p>}
          {children}
        </div>
      )}
    </section>
  )
}

export function LlmChip({
  value,
  onChange,
  onOpenConfig,
  disabled,
}: {
  value: string
  onChange: (model: string) => void
  onOpenConfig?: () => void
  disabled?: boolean
}) {
  const [ok, setOk] = useState<boolean | null>(null)
  const [models, setModels] = useState<string[]>([])
  const poll = useCallback(() => {
    return api<PollShape>("/api/poll")
      .then((r) => {
        setOk(Boolean(r.ollama?.ok))
        setModels(r.ollama?.models ?? [])
      })
      .catch(() => setOk(false))
  }, [])
  usePoll(poll, 5000)

  if (ok === null) return <span className="text-xs text-muted-foreground">Проверяю LLM…</span>
  if (!ok)
    return (
      <span className="flex flex-wrap items-center gap-2 text-xs text-destructive">
        Ollama недоступна
        {onOpenConfig && (
          <Button size="xs" variant="link" className="h-auto p-0" onClick={onOpenConfig}>
            Открыть систему
          </Button>
        )}
      </span>
    )
  return (
    <div className="flex items-center gap-2 text-xs">
      <Badge variant="secondary" className="bg-ok/15 text-ok">
        LLM готов
      </Badge>
      <Select value={value} onValueChange={(v) => { if (v) onChange(v) }} disabled={disabled}>
        <SelectTrigger size="sm" className="flex-1">
          <SelectValue placeholder="Модель" />
        </SelectTrigger>
        <SelectContent>
          {models.map((m) => (
            <SelectItem key={m} value={m}>{m}</SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  )
}

export function ParamSlider({
  label,
  value,
  min,
  max,
  disabled,
  onChange,
}: {
  label: string
  value: number
  min: number
  max: number
  disabled?: boolean
  onChange: (v: number) => void
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <Label className="w-20 shrink-0">{label}</Label>
      <Slider
        className="w-32"
        min={min}
        max={max}
        value={value}
        disabled={disabled}
        onValueChange={(v) => {
          if (typeof v === "number") onChange(v)
        }}
      />
      <span className="text-muted-foreground">{value}</span>
    </div>
  )
}

export function AdvancedFields({
  params,
  onParamsChange,
  disabled,
}: {
  params: EditorParams
  onParamsChange: (p: EditorParams) => void
  disabled?: boolean
}) {
  return (
    <div className="flex flex-col gap-1.5 rounded-md border p-2">
      {Object.keys(ADVANCED_DEFAULTS).map((key) => (
        <div key={key} className="flex items-center gap-2 text-xs">
          <Label className="w-40 shrink-0 truncate" title={key}>{key}</Label>
          <Input
            value={params.advanced[key] ?? ""}
            onChange={(e) =>
              onParamsChange({
                ...params,
                advanced: { ...params.advanced, [key]: e.target.value },
              })
            }
            disabled={disabled}
            placeholder={ADVANCED_DEFAULTS[key] || "—"}
            className="h-6 flex-1 text-xs"
          />
        </div>
      ))}
      <div className="flex items-center gap-2 text-xs">
        <Label className="w-40 shrink-0">detector_model</Label>
        <Input
          value={params.run.detector_model}
          onChange={(e) => onParamsChange({ ...params, run: { ...params.run, detector_model: e.target.value } })}
          disabled={disabled}
          placeholder="IDEA-Research/grounding-dino-tiny"
          className="h-6 flex-1 text-xs"
        />
      </div>
      <div className="flex items-center gap-2 text-xs">
        <Label className="w-40 shrink-0">segmenter_model</Label>
        <Input
          value={params.run.segmenter_model}
          onChange={(e) => onParamsChange({ ...params, run: { ...params.run, segmenter_model: e.target.value } })}
          disabled={disabled}
          placeholder="facebook/sam2-hiera-tiny"
          className="h-6 flex-1 text-xs"
        />
      </div>
      <div className="flex items-center gap-2 text-xs">
        <Label className="w-40 shrink-0">llm_base_url</Label>
        <Input
          value={params.run.llm_base_url}
          onChange={(e) => onParamsChange({ ...params, run: { ...params.run, llm_base_url: e.target.value } })}
          disabled={disabled}
          className="h-6 flex-1 text-xs"
        />
      </div>
      <div className="flex items-center gap-2 text-xs">
        <Label className="w-40 shrink-0">llm_api_key</Label>
        <Input
          type="password"
          value={params.run.llm_api_key}
          onChange={(e) => onParamsChange({ ...params, run: { ...params.run, llm_api_key: e.target.value } })}
          disabled={disabled}
          className="h-6 flex-1 text-xs"
        />
      </div>
    </div>
  )
}

export function PresetsPopover({
  params,
  onParamsChange,
  disabled,
}: {
  params: EditorParams
  onParamsChange: (p: EditorParams) => void
  disabled?: boolean
}) {
  const [open, setOpen] = useState(false)
  const [presets, setPresets] = useState<Preset[]>([])
  const [name, setName] = useState("")
  const [busy, setBusy] = useState(false)

  const refresh = () => {
    listPresets()
      .then(setPresets)
      .catch(() => setPresets([]))
  }

  useEffect(() => {
    if (open) refresh()
  }, [open])

  const save = () => {
    const trimmed = name.trim()
    if (!trimmed || busy) return
    setBusy(true)
    savePreset(trimmed, presetSnapshot(params))
      .then(() => {
        setName("")
        refresh()
      })
      .catch(() => {})
      .finally(() => setBusy(false))
  }

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger
        render={
          <Button size="sm" variant="outline" disabled={disabled}>
            Пресеты
          </Button>
        }
      />
      <PopoverContent align="start" className="w-80">
        <div className="flex flex-col gap-2 text-xs">
          <div className="flex items-center gap-1.5">
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Имя пресета"
              className="h-6 flex-1 text-xs"
            />
            <Button size="sm" disabled={!name.trim() || busy} onClick={save}>
              Сохранить
            </Button>
          </div>
          {presets.length === 0 ? (
            <p className="text-muted-foreground">Пресетов нет — сохраните текущие параметры.</p>
          ) : (
            presets.map((p) => (
              <div key={p.id} className="flex items-center gap-1.5">
                <Button
                  size="sm"
                  variant="secondary"
                  className="flex-1 justify-start"
                  onClick={() => {
                    onParamsChange(applyPreset(params, p.payload))
                    setOpen(false)
                  }}
                >
                  {p.name}
                </Button>
                <Button
                  size="icon-sm"
                  variant="ghost"
                  aria-label={`Удалить ${p.name}`}
                  onClick={() => deletePreset(p.id).then(refresh).catch(() => {})}
                >
                  ×
                </Button>
              </div>
            ))
          )}
        </div>
      </PopoverContent>
    </Popover>
  )
}
