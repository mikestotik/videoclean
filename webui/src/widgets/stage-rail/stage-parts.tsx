import { useCallback, useEffect, useState } from "react"
import { CircleHelp } from "lucide-react"
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
import { Switch } from "@/shared/ui/switch"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/shared/ui/tooltip"
import { cn } from "@/shared/lib/utils"
import {
  ADVANCED_META,
  DEVICE_META,
  INPAINTER_META,
  RUN_PARAM_META,
} from "./param-meta"
import {
  ADVANCED_DEFAULTS,
  DEVICE_OPTIONS,
  INPAINTER_OPTIONS,
  applyBuiltInProfile,
  applyPreset,
  presetSnapshot,
  type EditorParams,
  type PipelineProfileId,
} from "./params"

const PROFILE_OPTIONS: { id: PipelineProfileId; label: string; hint: string }[] = [
  { id: "fast", label: "Быстро", hint: "LaMa, без verify — черновик." },
  { id: "balanced", label: "Баланс", hint: "LaMa + один verify-pass." },
  { id: "quality", label: "Качество", hint: "Больше keyframes; на CUDA — sam2-video + ProPainter и до 2 verify-pass." },
  { id: "custom", label: "Свой", hint: "Ручные настройки без пресета." },
]

type PollShape = { ollama: { ok: boolean; models: string[] } }

type SegmenterModelOpt = {
  id: string
  title: string
  model_ref: string
  size_hint?: string
  ready?: boolean
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

type OptionsShape = {
  detectors: string[]
  segmenters: string[]
  inpainters?: string[]
  segmenter_models?: SegmenterModelOpt[]
  default_segmenter_model?: string
  models?: { inpainter?: InpainterModelOpt[] }
}

export function BackendSelectors({
  detector,
  segmenter,
  segmenterModel,
  onChange,
  disabled,
  detectorOnly = false,
}: {
  detector: string
  segmenter: string
  segmenterModel: string
  onChange: (patch: { detector?: string; segmenter?: string; segmenter_model?: string }) => void
  disabled?: boolean
  /** Preview stage: segmenter is forced to sam2 server-side — don't pretend otherwise. */
  detectorOnly?: boolean
}) {
  const [opts, setOpts] = useState<OptionsShape>({ detectors: [], segmenters: [] })
  useEffect(() => {
    api<OptionsShape>("/api/options")
      .then((r) =>
        setOpts({
          detectors: r.detectors ?? [],
          segmenters: r.segmenters ?? [],
          segmenter_models: r.segmenter_models ?? [],
          default_segmenter_model: r.default_segmenter_model,
        }),
      )
      .catch(() => setOpts({ detectors: [], segmenters: [] }))
  }, [])

  const modelValue =
    segmenterModel ||
    opts.default_segmenter_model ||
    opts.segmenter_models?.[0]?.model_ref ||
    "facebook/sam2-hiera-tiny"

  return (
    <div className="flex flex-col gap-1.5 text-xs">
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
      {!detectorOnly && (
        <>
          <div className="flex items-center gap-2">
            <FieldLabel hint="Как строить маску: покадрово (sam2) или с пропагацией по клипу (sam2-video).">Режим</FieldLabel>
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
          <div className="flex items-center gap-2">
            <FieldLabel hint="Веса сегментатора. Крупнее — точнее и тяжелее.">Модель</FieldLabel>
            <Select
              value={modelValue}
              onValueChange={(v) => { if (v) onChange({ segmenter_model: v }) }}
              disabled={disabled}
            >
              <SelectTrigger size="sm" className="flex-1">
                <SelectValue placeholder="SAM2 tiny" />
              </SelectTrigger>
              <SelectContent>
                {(opts.segmenter_models ?? []).map((m) => (
                  <SelectItem key={m.id} value={m.model_ref}>
                    {m.title}
                    {m.size_hint ? ` · ${m.size_hint}` : ""}
                    {m.ready === false ? " · не скачана" : ""}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </>
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

export function StageSection({
  n,
  title,
  hint,
  active,
  done,
  open,
  onOpenChange,
  children,
}: {
  n: number
  title: string
  hint?: string
  active: boolean
  /** Stage already has a result — accent the step number. */
  done?: boolean
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
        className="flex w-full items-baseline gap-2.5 px-3 py-3 text-left hover:bg-muted/30"
        onClick={() => onOpenChange(!open)}
        aria-expanded={open}
      >
        <span
          className={cn(
            "flex size-5 shrink-0 items-center justify-center rounded-full text-[10px] font-semibold tabular-nums relative -top-px",
            done
              ? "bg-primary text-primary-foreground"
              : active
                ? "bg-primary/20 text-primary ring-1 ring-primary/40"
                : "bg-muted text-muted-foreground",
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
        <div className="flex flex-col gap-2.5 px-3 pb-3.5 pl-3">
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

/** API/HF refs for the form; never use raw download URLs (LaMa catalog uses a GitHub URL). */
function modelRefForApi(m: InpainterModelOpt): string {
  if (m.backend === "lama") return "big-lama"
  if (m.model_ref.startsWith("http://") || m.model_ref.startsWith("https://")) {
    return m.title
  }
  return m.model_ref
}

export function InpaintControls({
  params,
  onParamsChange,
  disabled,
}: {
  params: EditorParams
  onParamsChange: (p: EditorParams) => void
  disabled?: boolean
}) {
  const [opts, setOpts] = useState<OptionsShape>({ detectors: [], segmenters: [] })
  useEffect(() => {
    api<OptionsShape>("/api/options")
      .then(setOpts)
      .catch(() => setOpts({ detectors: [], segmenters: [] }))
  }, [])

  const setRun = (patch: Partial<EditorParams["run"]>, markCustom = false) =>
    onParamsChange({
      ...params,
      run: {
        ...params.run,
        ...patch,
        ...(markCustom ? { profile: "custom" as const } : {}),
      },
    })

  const inpainters =
    opts.inpainters?.length ? opts.inpainters : [...INPAINTER_OPTIONS]
  const catalog = opts.models?.inpainter ?? []
  const modelChoices = catalog.filter((m) => m.backend === params.run.inpainter)
  const needsModel = params.run.inpainter === "lama" || params.run.inpainter === "propainter"
  // Select by catalog id — model_ref for LaMa is a GitHub URL and looks broken in the trigger.
  const selectedModel =
    modelChoices.find((m) => m.id === params.run.inpainter_model) ||
    modelChoices.find((m) => m.model_ref === params.run.inpainter_model) ||
    modelChoices.find((m) => modelRefForApi(m) === params.run.inpainter_model) ||
    modelChoices.find((m) => m.state === "ready") ||
    modelChoices[0]
  const modelSelectValue = selectedModel?.id ?? ""
  const profileHint =
    PROFILE_OPTIONS.find((p) => p.id === params.run.profile)?.hint ?? PROFILE_OPTIONS[3].hint

  return (
    <div className="flex flex-col gap-2 text-xs">
      <div className="flex items-center gap-2">
        <FieldLabel
          className="w-[7.5rem] shrink-0"
          hint="Готовый стек под задачу. «Качество» на CUDA включает sam2-video, ProPainter и повторную проверку остатков."
        >
          Профиль
        </FieldLabel>
        <Select
          value={params.run.profile}
          onValueChange={(v) => {
            if (!v) return
            onParamsChange(applyBuiltInProfile(params, v as PipelineProfileId))
          }}
          disabled={disabled}
        >
          <SelectTrigger size="sm" className="flex-1">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PROFILE_OPTIONS.map((p) => (
              <SelectItem key={p.id} value={p.id}>
                {p.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <ParamHint text={profileHint} />
      </div>

      <div className="flex items-center gap-2">
        <FieldLabel
          className="w-[7.5rem] shrink-0"
          hint="Чем заполнять вырезанные области. На CPU — LaMa; для финального качества на GPU — ProPainter."
        >
          Инпейнтер
        </FieldLabel>
        <Select
          value={params.run.inpainter}
          onValueChange={(v) => {
            if (!v) return
            const nextModels = catalog.filter((m) => m.backend === v)
            const next =
              nextModels.find((m) => m.state === "ready") || nextModels[0]
            setRun(
              {
                inpainter: v,
                // Prefer HF id / short ref; never persist raw download URLs in the UI value path.
                inpainter_model: next ? modelRefForApi(next) : "",
              },
              true,
            )
          }}
          disabled={disabled}
        >
          <SelectTrigger size="sm" className="flex-1">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {inpainters.map((o) => (
              <SelectItem key={o} value={o}>
                {INPAINTER_META[o]?.label ?? o}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select
          value={params.run.device || "cpu"}
          onValueChange={(v) => {
            if (!v) return
            // Re-resolve profile knobs for the new device when a built-in profile is active.
            if (params.run.profile !== "custom") {
              onParamsChange(
                applyBuiltInProfile(
                  { ...params, run: { ...params.run, device: v } },
                  params.run.profile,
                ),
              )
              return
            }
            setRun({ device: v })
          }}
          disabled={disabled}
        >
          <SelectTrigger size="sm" className="w-28">
            <SelectValue placeholder="device" />
          </SelectTrigger>
          <SelectContent>
            {DEVICE_OPTIONS.map((d) => (
              <SelectItem key={d} value={d}>
                {DEVICE_META[d]?.label ?? d}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <ParamHint text={DEVICE_META[params.run.device || "cpu"]?.hint ?? "Где считать нейросети."} />
      </div>

      {needsModel && (
        <div className="flex items-center gap-2">
          <FieldLabel
            className="w-[7.5rem] shrink-0"
            hint="Веса выбранного инпейнтера из каталога моделей. Скачать недостающие можно в Настройках."
          >
            Модель
          </FieldLabel>
          <Select
            value={modelSelectValue}
            onValueChange={(id) => {
              if (!id) return
              const m = modelChoices.find((x) => x.id === id)
              if (m) setRun({ inpainter_model: modelRefForApi(m) })
            }}
            disabled={disabled || modelChoices.length === 0}
          >
            <SelectTrigger size="sm" className="flex-1">
              <SelectValue placeholder="Выберите модель">
                {selectedModel
                  ? `${selectedModel.title}${selectedModel.size_hint ? ` · ${selectedModel.size_hint}` : ""}`
                  : null}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {modelChoices.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  {m.title}
                  {m.size_hint ? ` · ${m.size_hint}` : ""}
                  {m.state && m.state !== "ready" ? ` · ${m.state}` : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      )}

      <ParamSlider
        label={RUN_PARAM_META.mask_dilate_px.label}
        hint={RUN_PARAM_META.mask_dilate_px.hint}
        value={params.run.mask_dilate_px}
        min={RUN_PARAM_META.mask_dilate_px.min}
        max={RUN_PARAM_META.mask_dilate_px.max}
        step={RUN_PARAM_META.mask_dilate_px.step}
        disabled={disabled}
        onChange={(v) => setRun({ mask_dilate_px: v })}
      />

      {params.run.inpainter === "propainter" && (
        <>
          {(["propainter_mask_dilation", "propainter_ref_stride", "propainter_neighbor_length", "propainter_subvideo_length", "propainter_raft_iter"] as const).map((key) => {
            const meta = ADVANCED_META[key]
            const raw = Number(params.advanced[key] ?? ADVANCED_DEFAULTS[key] ?? 0)
            return (
              <ParamSlider
                key={key}
                label={meta.label}
                hint={meta.hint}
                value={Number.isFinite(raw) ? raw : meta.min ?? 0}
                min={meta.min ?? 0}
                max={meta.max ?? 100}
                step={meta.step ?? 1}
                disabled={disabled}
                onChange={(v) =>
                  onParamsChange({
                    ...params,
                    advanced: { ...params.advanced, [key]: String(v) },
                  })
                }
              />
            )
          })}
        </>
      )}

      <div className="flex items-center gap-2">
        <FieldLabel className="flex-1" hint={RUN_PARAM_META.verify.hint}>
          {RUN_PARAM_META.verify.label}
        </FieldLabel>
        <Switch
          checked={params.run.verify}
          onCheckedChange={(checked) => setRun({ verify: checked })}
          disabled={disabled}
        />
      </div>
      <div className="flex items-center gap-2">
        <FieldLabel className="flex-1" hint={RUN_PARAM_META.keep_workdir.hint}>
          {RUN_PARAM_META.keep_workdir.label}
        </FieldLabel>
        <Switch
          checked={params.run.keep_workdir}
          onCheckedChange={(checked) => setRun({ keep_workdir: checked })}
          disabled={disabled}
        />
      </div>
    </div>
  )
}

function advancedNumber(params: EditorParams, key: string, fallback: number): number {
  const raw = params.advanced[key]
  if (raw === undefined || raw === "") return fallback
  const n = Number(raw)
  return Number.isFinite(n) ? n : fallback
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
  const setAdvanced = (key: string, value: string) =>
    onParamsChange({
      ...params,
      run: { ...params.run, profile: "custom" },
      advanced: { ...params.advanced, [key]: value },
    })

  const sliderKeys = [
    "detector_threshold",
    "detector_nms_iou",
    "detector_max_box_area",
    "tracker_min_score",
    "tracker_max_template_area",
    "prompt_frame_stride",
    "prompt_frame_max",
    "parse_chunk_frames",
    "vision_batch",
    "verify_max_passes",
    "inpaint_workers",
    "inpaint_chunk_overlap",
    "propainter_mask_dilation",
    "propainter_ref_stride",
    "propainter_neighbor_length",
    "propainter_subvideo_length",
    "propainter_raft_iter",
  ] as const

  return (
    <div className="flex flex-col gap-2 rounded-md border border-border/60 p-2">
      <ParamSlider
        label={ADVANCED_META.detector_keyframes.label}
        hint={`${ADVANCED_META.detector_keyframes.hint} 0 = авто.`}
        value={advancedNumber(params, "detector_keyframes", 0)}
        min={0}
        max={ADVANCED_META.detector_keyframes.max ?? 40}
        step={1}
        disabled={disabled}
        formatValue={(v) => (v <= 0 ? "авто" : String(v))}
        onChange={(v) => setAdvanced("detector_keyframes", v <= 0 ? "" : String(v))}
      />
      {sliderKeys.map((key) => {
        const meta = ADVANCED_META[key]
        const fallback = Number(ADVANCED_DEFAULTS[key] || 0)
        return (
          <ParamSlider
            key={key}
            label={meta.label}
            hint={meta.hint}
            value={advancedNumber(params, key, fallback)}
            min={meta.min ?? 0}
            max={meta.max ?? 100}
            step={meta.step ?? 1}
            disabled={disabled}
            onChange={(v) => setAdvanced(key, String(v))}
          />
        )
      })}
      <ParamSlider
        label={RUN_PARAM_META.min_mask_coverage.label}
        hint={RUN_PARAM_META.min_mask_coverage.hint}
        value={params.run.min_mask_coverage}
        min={RUN_PARAM_META.min_mask_coverage.min}
        max={RUN_PARAM_META.min_mask_coverage.max}
        step={RUN_PARAM_META.min_mask_coverage.step}
        disabled={disabled}
        onChange={(v) => onParamsChange({ ...params, run: { ...params.run, min_mask_coverage: v } })}
      />
      <ParamSlider
        label={RUN_PARAM_META.verify_max_coverage.label}
        hint={RUN_PARAM_META.verify_max_coverage.hint}
        value={params.run.verify_max_coverage}
        min={RUN_PARAM_META.verify_max_coverage.min}
        max={RUN_PARAM_META.verify_max_coverage.max}
        step={RUN_PARAM_META.verify_max_coverage.step}
        disabled={disabled}
        onChange={(v) => onParamsChange({ ...params, run: { ...params.run, verify_max_coverage: v } })}
      />
      <div className="flex items-center gap-2 text-xs">
        <FieldLabel className="w-[7.5rem] shrink-0" hint="Свой HF id детектора, если нужен не дефолтный.">
          Модель детектора
        </FieldLabel>
        <Input
          value={params.run.detector_model}
          onChange={(e) => onParamsChange({ ...params, run: { ...params.run, detector_model: e.target.value } })}
          disabled={disabled}
          placeholder="IDEA-Research/grounding-dino-tiny"
          className="h-7 flex-1 font-mono text-xs"
        />
      </div>
      <div className="flex items-center gap-2 text-xs">
        <FieldLabel className="w-[7.5rem] shrink-0" hint="Свой HF id сегментатора, если нет в списке выше.">
          Модель SAM
        </FieldLabel>
        <Input
          value={params.run.segmenter_model}
          onChange={(e) => onParamsChange({ ...params, run: { ...params.run, segmenter_model: e.target.value } })}
          disabled={disabled}
          placeholder="facebook/sam2.1-hiera-small"
          className="h-7 flex-1 font-mono text-xs"
        />
      </div>
      <div className="flex items-center gap-2 text-xs">
        <FieldLabel className="w-[7.5rem] shrink-0" hint="Свой OpenAI-совместимый URL вместо локальной Ollama.">
          LLM URL
        </FieldLabel>
        <Input
          value={params.run.llm_base_url}
          onChange={(e) => onParamsChange({ ...params, run: { ...params.run, llm_base_url: e.target.value } })}
          disabled={disabled}
          className="h-7 flex-1 text-xs"
        />
      </div>
      <div className="flex items-center gap-2 text-xs">
        <FieldLabel className="w-[7.5rem] shrink-0" hint="Ключ для удалённого LLM API.">
          LLM ключ
        </FieldLabel>
        <Input
          type="password"
          value={params.run.llm_api_key}
          onChange={(e) => onParamsChange({ ...params, run: { ...params.run, llm_api_key: e.target.value } })}
          disabled={disabled}
          className="h-7 flex-1 text-xs"
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
