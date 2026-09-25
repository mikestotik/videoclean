import { CircleHelp } from "lucide-react"
import { useEventsOptional } from "@/shared/events"
import { Button } from "@/shared/ui/button"
import { Input } from "@/shared/ui/input"
import { Label } from "@/shared/ui/label"
import { Switch } from "@/shared/ui/switch"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select"
import { Slider } from "@/shared/ui/slider"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/shared/ui/tooltip"
import { cn } from "@/shared/lib/utils"

type SegmenterModelOpt = {
  id: string
  title: string
  model_ref: string
  size_hint?: string
  ready?: boolean
  backend?: string
}

type DetectorModelOpt = {
  id: string
  title: string
  model_ref: string
  size_hint?: string
  ready?: boolean
  backend?: string
}

type InpainterModelOpt = {
  id: string
  title: string
  backend: string
  model_ref: string
  size_hint?: string
  state?: string
  message?: string
}

type LlmModelOpt = {
  id: string
  title: string
  model: string
  provider_id: string
  base_url?: string
  ready?: boolean
}

export type OptionsShape = {
  detectors: string[]
  segmenters: string[]
  inpainters?: string[]
  detector_models?: DetectorModelOpt[]
  segmenter_models?: SegmenterModelOpt[]
  default_detector_model?: string
  default_segmenter_model?: string
  llm_model_options?: LlmModelOpt[]
  models?: { inpainter?: InpainterModelOpt[] }
}

export function BackendSelectors({
  detector,
  detectorModel,
  segmenter,
  segmenterModel,
  onChange,
  disabled,
  detectorOnly = false,
  segmenterOnly = false,
}: {
  detector: string
  detectorModel?: string
  segmenter: string
  segmenterModel: string
  onChange: (patch: {
    detector?: string
    detector_model?: string
    segmenter?: string
    segmenter_model?: string
  }) => void
  disabled?: boolean
  /**
   * Detect/preview: mode is forced to sam2 server-side — show detector + SAM weights,
   * hide mode switch.
   */
  detectorOnly?: boolean
  /** Inpaint by tracks/masks: only segmenter mode + SAM weights. */
  segmenterOnly?: boolean
}) {
  const events = useEventsOptional()
  const opts = (events?.snapshot?.options as OptionsShape | undefined) ?? {
    detectors: [],
    segmenters: [],
  }

  const showDetector = !segmenterOnly
  const showSegmenterMode = !detectorOnly
  const showSegmenterModel = !detectorOnly

  const detectorModelChoices = (opts.detector_models ?? []).filter(
    (m) => !m.backend || m.backend === detector,
  )
  const detectorModelValue = detectorModel || ""

  const segmenterModels = opts.segmenter_models ?? []
  const modelValue = segmenterModel || ""
  const selectedSeg =
    segmenterModels.find((m) => m.model_ref === modelValue) ||
    segmenterModels.find((m) => m.model_ref === segmenterModel)

  return (
    <div className="flex flex-col gap-1.5 text-xs">
      {showDetector && (
        <>
          <StackedField label="Детектор" hint="Какая нейросеть ищет объекты по тексту цели." param="detector">
            <Select value={detector} onValueChange={(v) => { if (v) onChange({ detector: v }) }} disabled={disabled}>
              <SelectTrigger size="sm" className="w-full">
                <SelectValue placeholder="Детектор" />
              </SelectTrigger>
              <SelectContent>
                {opts.detectors.map((d) => (
                  <SelectItem key={d} value={d}>{d}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </StackedField>
          {detectorModelChoices.length > 0 && (
            <StackedField label="Модель детектора" hint="Веса детектора из настроек." param="detector_model">
              <Select
                value={detectorModelValue}
                onValueChange={(v) => { if (v) onChange({ detector_model: v }) }}
                disabled={disabled}
              >
                <SelectTrigger size="sm" className="w-full">
                  <SelectValue placeholder="DINO">
                    {(value: string | null) => {
                      const picked = detectorModelChoices.find((m) => m.model_ref === value)
                      if (!picked) return value
                      return `${picked.title}${picked.size_hint ? ` · ${picked.size_hint}` : ""}`
                    }}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {detectorModelChoices.map((m) => (
                    <SelectItem key={m.id} value={m.model_ref}>
                      {m.title}
                      {m.size_hint ? ` · ${m.size_hint}` : ""}
                      {m.ready === false ? " · не скачана" : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </StackedField>
          )}
        </>
      )}
      {showSegmenterMode && (
        <StackedField
          label="Режим сегментации"
          hint="Покадрово (sam2) или с протяжкой маски по клипу (sam2-video)."
          param="segmenter"
        >
          <Select value={segmenter} onValueChange={(v) => { if (v) onChange({ segmenter: v }) }} disabled={disabled}>
            <SelectTrigger size="sm" className="w-full">
              <SelectValue placeholder="Покадрово">
                {(value: string | null) =>
                  value === "sam2-video" ? "С протяжкой по клипу" : value === "sam2" ? "Покадрово" : value
                }
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {(opts.segmenters.length ? opts.segmenters : ["sam2", "sam2-video"]).map((s) => (
                <SelectItem key={s} value={s}>
                  {s === "sam2-video" ? "С протяжкой по клипу" : "Покадрово"}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </StackedField>
      )}
      {showSegmenterModel && (
        <StackedField
          label="Модель сегментации"
          hint="Веса SAM2.1. Крупнее — точнее и тяжелее. Скачать можно в Системе."
          param="segmenter_model"
        >
          <Select
            value={modelValue}
            onValueChange={(v) => { if (v) onChange({ segmenter_model: v }) }}
            disabled={disabled || segmenterModels.length === 0}
          >
            <SelectTrigger size="sm" className="w-full">
              <SelectValue placeholder="Модель сегментации">
                {selectedSeg
                  ? `${selectedSeg.title}${selectedSeg.size_hint ? ` · ${selectedSeg.size_hint}` : ""}`
                  : null}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {segmenterModels.map((m) => (
                <SelectItem key={m.id} value={m.model_ref}>
                  {m.title}
                  {m.size_hint ? ` · ${m.size_hint}` : ""}
                  {m.ready === false ? " · не скачана" : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </StackedField>
      )}
    </div>
  )
}

export function FieldLabel({
  children,
  hint,
  className,
}: {
  children: React.ReactNode
  hint?: string
  className?: string
}) {
  return (
    <div className={cn("flex min-w-0 items-start gap-1", className)}>
      <Label className="min-w-0 flex-1 items-start font-medium whitespace-normal leading-snug">
        <span className="min-w-0 whitespace-normal">{children}</span>
      </Label>
      {hint ? <ParamHint text={hint} /> : null}
    </div>
  )
}

export function StackedField({
  label,
  hint,
  param,
  children,
}: {
  label: React.ReactNode
  hint?: string
  param?: string
  children: React.ReactNode
}) {
  return (
    <div data-param={param} className="flex min-w-0 flex-col gap-1 text-xs">
      <FieldLabel hint={hint}>{label}</FieldLabel>
      {children}
    </div>
  )
}

export function SwitchRow({
  label,
  hint,
  param,
  checked,
  disabled,
  onCheckedChange,
}: {
  label: React.ReactNode
  hint?: string
  param?: string
  checked: boolean
  disabled?: boolean
  onCheckedChange: (checked: boolean) => void
}) {
  return (
    <div data-param={param} className="flex items-start justify-between gap-3">
      <FieldLabel hint={hint} className="min-w-0 flex-1">
        {label}
      </FieldLabel>
      <Switch checked={checked} disabled={disabled} onCheckedChange={onCheckedChange} />
    </div>
  )
}

export function ParamHint({ text }: { text: string }) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger
          render={
            <button
              type="button"
              className="inline-flex size-4 shrink-0 items-center justify-center rounded-full text-muted-foreground hover:text-foreground"
              aria-label="Подсказка"
            >
              <CircleHelp className="size-3.5" />
            </button>
          }
        />
        <TooltipContent side="left" className="max-w-[240px] text-left leading-snug">
          {text}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

export function LlmChip({
  value,
  onChange,
  onOpenConfig,
  disabled,
}: {
  value: string
  onChange: (patch: { llm_model: string; llm_base_url?: string }) => void
  onOpenConfig?: () => void
  disabled?: boolean
}) {
  const events = useEventsOptional()
  const snap = events?.snapshot
  const fromOptions = snap?.options?.llm_model_options ?? []
  const options: LlmModelOpt[] = fromOptions.length
    ? fromOptions
    : (snap?.ollama?.models ?? []).map((m) => ({
        id: m,
        title: m,
        model: m,
        provider_id: "ollama",
        ready: true,
      }))
  if (snap == null) return <span className="text-xs text-muted-foreground">Проверяю модели…</span>
  const ready = options.filter((m) => m.ready !== false && m.model)
  if (ready.length === 0) {
    return (
      <p className="text-xs leading-snug text-destructive">
        Модели для разбора фразы пока нет. Установите её, чтобы разобрать фразу на цели.
        {onOpenConfig && (
          <Button size="xs" variant="link" className="h-auto px-1 text-destructive" onClick={onOpenConfig}>
            Открыть систему
          </Button>
        )}
      </p>
    )
  }
  return (
    <StackedField label="Модель разбора" hint="Какая модель разбирает фразу на цели." param="llm_model">
      <Select
        value={value}
        onValueChange={(v) => {
          if (!v) return
          const hit = ready.find((m) => m.model === v || m.id === v)
          onChange({
            llm_model: hit?.model ?? v,
            llm_base_url: hit?.base_url || "",
          })
        }}
        disabled={disabled}
      >
        <SelectTrigger size="sm" className="w-full">
          <SelectValue placeholder="Модель">
            {(shown: string | null) => {
              if (!shown) return "Модель"
              const hit = ready.find((m) => m.model === shown || m.id === shown)
              return hit?.title ?? shown
            }}
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {ready.map((m) => (
            <SelectItem key={m.id} value={m.model}>
              {m.title}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </StackedField>
  )
}

export function ParamSlider({
  label,
  hint,
  value,
  min,
  max,
  step = 1,
  disabled,
  onChange,
  formatValue,
  param,
}: {
  label: string
  hint?: string
  value: number
  min: number
  max: number
  step?: number
  disabled?: boolean
  onChange: (v: number) => void
  formatValue?: (v: number) => string
  param?: string
}) {
  const shown =
    formatValue?.(value) ??
    (step < 0.001 ? value.toFixed(4) : step < 0.01 ? value.toFixed(3) : step < 1 ? value.toFixed(2) : String(value))
  return (
    <div data-param={param} className="flex min-w-0 flex-col gap-1 text-xs">
      <div className="flex items-start justify-between gap-2">
        <FieldLabel hint={hint} className="min-w-0 flex-1">
          {label}
        </FieldLabel>
        <span className="shrink-0 pt-0.5 tabular-nums text-muted-foreground">{shown}</span>
      </div>
      <Slider
        className="w-full"
        min={min}
        max={max}
        step={step}
        value={[value]}
        disabled={disabled}
        onValueChange={(v) => {
          const n = Array.isArray(v) ? v[0] : v
          if (typeof n === "number" && Number.isFinite(n)) onChange(n)
        }}
      />
    </div>
  )
}

export function ParamDecimal({
  label,
  hint,
  value,
  min,
  max,
  step,
  disabled,
  onChange,
  param,
  readout,
}: {
  label: string
  hint?: string
  value: number
  min: number
  max: number
  step: number
  disabled?: boolean
  onChange: (v: number) => void
  param?: string
  readout?: string
}) {
  return (
    <div data-param={param} className="flex min-w-0 flex-col gap-1 text-xs">
      <div className="flex items-start justify-between gap-2">
        <FieldLabel hint={hint} className="min-w-0 flex-1">
          {label}
        </FieldLabel>
        {readout ? <span className="shrink-0 pt-0.5 tabular-nums text-muted-foreground">{readout}</span> : null}
      </div>
      <Input
        type="number"
        inputMode="decimal"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        className="h-7 text-xs"
        onChange={(e) => {
          const n = Number(e.target.value)
          if (!Number.isFinite(n)) return
          onChange(Math.min(max, Math.max(min, n)))
        }}
      />
    </div>
  )
}
