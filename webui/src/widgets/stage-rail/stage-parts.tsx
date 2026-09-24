import { CircleHelp } from "lucide-react"
import { useEventsOptional } from "@/shared/events"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import { Label } from "@/shared/ui/label"
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
  const detectorModelValue =
    detectorModel ||
    opts.default_detector_model ||
    detectorModelChoices[0]?.model_ref ||
    "IDEA-Research/grounding-dino-tiny"

  const segmenterModels = opts.segmenter_models ?? []
  const modelValue =
    segmenterModel ||
    opts.default_segmenter_model ||
    segmenterModels[0]?.model_ref ||
    "facebook/sam2-hiera-tiny"
  const selectedSeg =
    segmenterModels.find((m) => m.model_ref === modelValue) ||
    segmenterModels.find((m) => m.model_ref === segmenterModel)

  return (
    <div className="flex flex-col gap-1.5 text-xs">
      {showDetector && (
        <>
          <div className="flex items-center gap-2">
            <FieldLabel hint="Какая нейросеть ищет объекты по тексту цели.">Детектор</FieldLabel>
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
          {detectorModelChoices.length > 0 && (
            <div className="flex items-center gap-2">
              <FieldLabel hint="Веса детектора из настроек.">Модель</FieldLabel>
              <Select
                value={detectorModelValue}
                onValueChange={(v) => { if (v) onChange({ detector_model: v }) }}
                disabled={disabled}
              >
                <SelectTrigger size="sm" className="flex-1">
                  <SelectValue placeholder="DINO" />
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
            </div>
          )}
        </>
      )}
      {showSegmenterMode && (
        <div className="flex items-center gap-2">
          <FieldLabel hint="Как строить маску: покадрово (sam2) или с пропагацией по клипу (sam2-video).">Режим SAM</FieldLabel>
          <Select value={segmenter} onValueChange={(v) => { if (v) onChange({ segmenter: v }) }} disabled={disabled}>
            <SelectTrigger size="sm" className="flex-1">
              <SelectValue placeholder="sam2" />
            </SelectTrigger>
            <SelectContent>
              {(opts.segmenters.length ? opts.segmenters : ["sam2", "sam2-video"]).map((s) => (
                <SelectItem key={s} value={s}>
                  {s === "sam2-video" ? "sam2-video (пропагация)" : "sam2 (покадрово)"}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}
      {showSegmenterModel && (
        <div className="flex items-center gap-2">
          <FieldLabel hint="Веса SAM2/SAM2.1. Крупнее — точнее и тяжелее. Скачать можно в Системе.">
            Модель SAM
          </FieldLabel>
          <Select
            value={modelValue}
            onValueChange={(v) => { if (v) onChange({ segmenter_model: v }) }}
            disabled={disabled || segmenterModels.length === 0}
          >
            <SelectTrigger size="sm" className="flex-1">
              <SelectValue placeholder="SAM2 tiny">
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
        </div>
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
    <div className={cn("flex min-w-0 items-center gap-1", className)}>
      <Label className="truncate">{children}</Label>
      {hint ? <ParamHint text={hint} /> : null}
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
        <TooltipContent side="top" className="max-w-[240px] text-left leading-snug">
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
  const ok =
    snap == null
      ? null
      : fromOptions.length
        ? fromOptions.some((m) => m.ready) || Boolean(snap.ollama?.ok) || (snap.providers?.length ?? 0) > 0
        : Boolean(snap.ollama?.ok)

  if (ok === null) return <span className="text-xs text-muted-foreground">Проверяю LLM…</span>
  if (!ok && options.length === 0)
    return (
      <span className="flex flex-wrap items-center gap-2 text-xs text-destructive">
        LLM недоступен
        {onOpenConfig && (
          <Button size="xs" variant="link" className="h-auto p-0" onClick={onOpenConfig}>
            Открыть систему
          </Button>
        )}
      </span>
    )
  const selectValue = value || options.find((m) => m.ready)?.model || options[0]?.model || ""
  return (
    <div className="flex items-center gap-2 text-xs">
      <Badge variant="secondary" className={ok ? "bg-ok/15 text-ok" : "bg-destructive/15 text-destructive"}>
        {ok ? "LLM готов" : "LLM"}
      </Badge>
      <Select
        value={selectValue}
        onValueChange={(v) => {
          if (!v) return
          const hit = options.find((m) => m.model === v || m.id === v)
          onChange({
            llm_model: hit?.model ?? v,
            llm_base_url: hit?.base_url || "",
          })
        }}
        disabled={disabled}
      >
        <SelectTrigger size="sm" className="flex-1">
          <SelectValue placeholder="Модель" />
        </SelectTrigger>
        <SelectContent>
          {options.map((m) => (
            <SelectItem key={m.id} value={m.model}>
              {m.title}
              {m.ready === false ? " · не скачана" : ""}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
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
}) {
  const shown =
    formatValue?.(value) ??
    (step < 0.001 ? value.toFixed(4) : step < 0.01 ? value.toFixed(3) : step < 1 ? value.toFixed(2) : String(value))
  return (
    <div className="flex items-center gap-2 text-xs">
      <FieldLabel hint={hint} className="w-[7.5rem] shrink-0">
        {label}
      </FieldLabel>
      <Slider
        className="min-w-0 flex-1"
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
      <span className="w-12 shrink-0 text-right tabular-nums text-muted-foreground">{shown}</span>
    </div>
  )
}
