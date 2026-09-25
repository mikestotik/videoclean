import { useEffect, useState } from "react"
import { cancelJob, type Job } from "@/entities/job"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/shared/ui/accordion"
import { Button } from "@/shared/ui/button"
import { Input } from "@/shared/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select"
import { Switch } from "@/shared/ui/switch"
import { Textarea } from "@/shared/ui/textarea"
import { useEventsOptional } from "@/shared/events"
import {
  ADVANCED_META,
  DEVICE_META,
  INPAINTER_META,
  MODE_HINTS,
  RUN_PARAM_META,
  formatShare,
  stationForParam,
} from "./param-meta"
import {
  DEVICE_OPTIONS,
  INPAINTER_OPTIONS,
  targetsFromParsedPrompt,
  type EditorParams,
} from "./params"
import { TargetsEditor } from "./targets-editor"
import { OutputFormatBlock } from "./output-format"
import {
  BackendSelectors,
  FieldLabel,
  LlmChip,
  ParamDecimal,
  ParamSlider,
  StackedField,
  SwitchRow,
  type OptionsShape,
} from "./stage-parts"

const CEILING_HINT = "Пустое поле — вся машина, без искусственного потолка."

const PROPAINTER_KEYS = [
  "propainter_mask_dilation",
  "propainter_ref_stride",
  "propainter_neighbor_length",
  "propainter_subvideo_length",
  "propainter_raft_iter",
] as const

const PACKAGE_FORMATS = new Set(["hls-fmp4", "hls-ts", "dash"])

type Focus = { key: string; nonce: number }

type Props = {
  params: EditorParams
  onChange: (next: EditorParams) => void
  disabled?: boolean
  findDisabled?: boolean
  onFind: () => void
  detectRunning?: boolean
  detectJobId?: string | null
  detectProgress?: { fraction: number; eta: string }
  detectError?: string
  foundTracks?: number
  tracksFull?: boolean
  useTracks?: boolean
  onUseTracks?: (checked: boolean) => void
  onOpenConfig?: () => void
  focus?: Focus | null
  parseRunning?: boolean
  parseError?: string
  parseDisabled?: boolean
  onParse?: () => Promise<boolean>
  resultJob?: Job | null
  onResultJobChange?: (job: Job) => void
}

function Station({
  id,
  title,
  summary,
  summaryClassName,
  children,
}: {
  id: string
  title: string
  summary: string
  summaryClassName?: string
  children: React.ReactNode
}) {
  return (
    <AccordionItem value={id}>
      <AccordionTrigger>
        <span className="flex min-w-0 flex-1 items-baseline justify-between gap-2 pr-1">
          <span className="shrink-0 font-medium">{title}</span>
          <span className={`truncate font-normal ${summaryClassName ?? "text-muted-foreground"}`}>{summary}</span>
        </span>
      </AccordionTrigger>
      <AccordionContent>{children}</AccordionContent>
    </AccordionItem>
  )
}

function CeilingInput({
  label,
  unit,
  param,
  value,
  disabled,
  onChange,
}: {
  label: string
  unit: string
  param: string
  value: string
  disabled?: boolean
  onChange: (value: string) => void
}) {
  return (
    <StackedField label={label} hint={CEILING_HINT} param={param}>
      <div className="flex items-center gap-2">
        <Input
          value={value}
          disabled={disabled}
          inputMode="numeric"
          className="h-7 flex-1 text-xs"
          onChange={(e) => {
            const next = e.target.value.trim()
            if (next !== "" && !/^\d+$/.test(next)) return
            onChange(next)
          }}
        />
        <span className="w-10 shrink-0 text-muted-foreground">{unit}</span>
      </div>
    </StackedField>
  )
}

function firstReadyLlm(
  listed: { model: string; ready?: boolean; base_url?: string }[] | undefined,
  ollamaModels: string[] | undefined,
): { model: string; base_url: string } | null {
  const options = listed?.length
    ? listed
    : (ollamaModels ?? []).map((model) => ({ model, ready: true, base_url: "" }))
  const picked = options.find((m) => m.ready !== false && m.model)
  if (!picked?.model) return null
  return { model: picked.model, base_url: picked.base_url || "" }
}

function shortName(value: string): string {
  const trimmed = value.trim()
  if (!trimmed) return ""
  const parts = trimmed.split(/[/:]/)
  return parts[parts.length - 1] || trimmed
}

export function ExpertStations({
  params,
  onChange,
  disabled,
  findDisabled,
  onFind,
  detectRunning,
  detectJobId,
  detectProgress,
  detectError,
  foundTracks = 0,
  tracksFull,
  useTracks,
  onUseTracks,
  onOpenConfig,
  focus,
  parseRunning,
  parseError,
  parseDisabled,
  onParse,
  resultJob,
  onResultJobChange,
}: Props) {
  const events = useEventsOptional()
  const [open, setOpen] = useState<string[]>([])
  const [seenNonce, setSeenNonce] = useState<number | null>(null)
  if (focus && focus.nonce !== seenNonce) {
    setSeenNonce(focus.nonce)
    const station = stationForParam(focus.key)
    if (station && station !== "masks") {
      setOpen((prev) => (prev.includes(station) ? prev : [...prev, station]))
    }
  }
  const pickedLlm = firstReadyLlm(events?.snapshot?.options?.llm_model_options, events?.snapshot?.ollama?.models)
  const opts = (events?.snapshot?.options as OptionsShape | undefined) ?? null
  const catalog = opts?.models?.inpainter ?? []
  const inpainters = opts?.inpainters?.length ? opts.inpainters : [...INPAINTER_OPTIONS]
  const modelChoices = catalog.filter((m) => m.backend === params.run.inpainter)
  const selectedModel =
    modelChoices.find((m) => m.id === params.run.inpainter_model) ||
    modelChoices.find((m) => m.model_ref === params.run.inpainter_model) ||
    modelChoices[0]

  const detectorPool = (opts?.detector_models ?? []).filter(
    (m) => !params.run.detector || !m.backend || m.backend === params.run.detector,
  )
  const pickedDetector = detectorPool.find((m) => m.ready !== false && m.model_ref)
  const pickedSegmenter = (opts?.segmenter_models ?? []).find((m) => m.ready && m.model_ref)

  useEffect(() => {
    if (params.run.segmenter_model || !pickedSegmenter) return
    onChange({
      ...params,
      run: { ...params.run, segmenter_model: pickedSegmenter.model_ref },
    })
  }, [onChange, params, pickedSegmenter])

  useEffect(() => {
    if (params.run.detector_model || !pickedDetector) return
    onChange({
      ...params,
      run: {
        ...params.run,
        detector: params.run.detector || pickedDetector.backend || opts?.detectors?.[0] || "",
        detector_model: pickedDetector.model_ref,
      },
    })
  }, [onChange, opts?.detectors, params, pickedDetector])

  useEffect(() => {
    if (params.run.llm_model || !pickedLlm) return
    onChange({
      ...params,
      run: { ...params.run, llm_model: pickedLlm.model, llm_base_url: pickedLlm.base_url },
    })
  }, [onChange, params, pickedLlm])

  useEffect(() => {
    if (params.run.formats.includes("mp4")) return
    onChange({ ...params, run: { ...params.run, formats: ["mp4", ...params.run.formats] } })
  }, [onChange, params])

  const setRun = (patch: Partial<EditorParams["run"]>) =>
    onChange({ ...params, run: { ...params.run, ...patch } })
  const setAdv = (key: string, value: string) =>
    onChange({ ...params, advanced: { ...params.advanced, [key]: value } })

  const llava = /llava/i.test(params.run.llm_model || "")
  const verifyOn = params.run.verify
  const propainter = params.run.inpainter === "propainter"
  const showProPainter = propainter || Boolean(focus?.key.startsWith("propainter_"))
  const showWebm = params.run.formats.includes("webm") || focus?.key === "webm_crf"
  const showSegment =
    params.run.formats.some((f) => PACKAGE_FORMATS.has(f)) || focus?.key === "segment_seconds"
  const namedTargets = params.targets.filter((t) => t.query.trim()).length
  const llmListed = events?.snapshot?.options?.llm_model_options
  const llmReady =
    (llmListed?.some((m) => m.ready && m.model) ?? false) ||
    (events?.snapshot?.ollama?.models?.length ?? 0) > 0
  const llmMissing = events?.snapshot != null && !llmReady && !params.run.llm_model
  const phraseSummary = llmMissing ? "нет модели" : shortName(params.run.llm_model) || "модель не выбрана"
  const targetsSummary = namedTargets > 0 ? `цели: ${namedTargets}` : "нет целей"
  const detectSummary =
    foundTracks > 0 ? `рамки: ${foundTracks}` : `порог ${params.advanced.detector_threshold || "0.15"}`
  const segmentSummary = `${params.run.segmenter === "sam2-video" ? "с протяжкой" : "покадрово"} · ${params.run.mask_dilate_px} px`
  const fillSummary = INPAINTER_META[params.run.inpainter]?.label ?? params.run.inpainter
  const verifySummary = verifyOn ? "включена" : "выключена"
  const deviceSummary = DEVICE_META[params.run.device || "auto"]?.label ?? "Авто"
  const formatSummary = params.run.formats
    .map((f) => (f === "mp4" ? "MP4" : f === "webm" ? "WebM" : f === "mov" ? "MOV" : f === "mkv" ? "MKV" : f))
    .join(", ")

  return (
    <Accordion multiple value={open} onValueChange={setOpen} className="border-t border-border/60">
      <Station
        id="phrase"
        title="Разбор фразы"
        summary={phraseSummary}
        summaryClassName={llmMissing ? "text-destructive" : undefined}
      >
        <LlmChip
          value={params.run.llm_model}
          disabled={disabled}
          onOpenConfig={onOpenConfig}
          onChange={(patch) => setRun(patch)}
        />
        <ParamSlider
          param="prompt_frame_stride"
          label={ADVANCED_META.prompt_frame_stride.label}
          hint={ADVANCED_META.prompt_frame_stride.hint}
          value={Number(params.advanced.prompt_frame_stride || 0)}
          min={ADVANCED_META.prompt_frame_stride.min ?? 0}
          max={ADVANCED_META.prompt_frame_stride.max ?? 20}
          step={1}
          disabled={disabled}
          onChange={(v) => setAdv("prompt_frame_stride", String(v))}
        />
        <ParamSlider
          param="prompt_frame_max"
          label={ADVANCED_META.prompt_frame_max.label}
          hint={ADVANCED_META.prompt_frame_max.hint}
          value={Number(params.advanced.prompt_frame_max || 8)}
          min={ADVANCED_META.prompt_frame_max.min ?? 1}
          max={ADVANCED_META.prompt_frame_max.max ?? 24}
          step={1}
          disabled={disabled}
          onChange={(v) => setAdv("prompt_frame_max", String(v))}
        />
        <ParamSlider
          param="vision_batch"
          label={ADVANCED_META.vision_batch.label}
          hint={ADVANCED_META.vision_batch.hint}
          value={Math.min(Number(params.advanced.vision_batch || 2), llava ? 2 : 8)}
          min={ADVANCED_META.vision_batch.min ?? 1}
          max={llava ? 2 : (ADVANCED_META.vision_batch.max ?? 8)}
          step={1}
          disabled={disabled}
          onChange={(v) => setAdv("vision_batch", String(llava ? Math.min(v, 2) : v))}
        />
        <ParamSlider
          param="parse_chunk_frames"
          label={ADVANCED_META.parse_chunk_frames.label}
          hint={ADVANCED_META.parse_chunk_frames.hint}
          value={Number(params.advanced.parse_chunk_frames || 0)}
          min={ADVANCED_META.parse_chunk_frames.min ?? 0}
          max={ADVANCED_META.parse_chunk_frames.max ?? 300}
          step={ADVANCED_META.parse_chunk_frames.step ?? 10}
          disabled={disabled}
          onChange={(v) => setAdv("parse_chunk_frames", String(v))}
        />
        <Button
          size="sm"
          className="self-start"
          disabled={parseDisabled || parseRunning || !onParse}
          onClick={() => {
            void onParse?.().then((ok) => {
              if (!ok) return
              setOpen((prev) => (prev.includes("targets") ? prev : [...prev, "targets"]))
            })
          }}
        >
          {parseRunning ? "Разбираю фразу…" : "Разобрать фразу"}
        </Button>
        {parseError && <p className="text-xs text-destructive">{parseError}</p>}
        <StackedField
          label="Результат разбора"
          hint="Текст, который модель получила из вашей фразы. Цели собираются из него."
          param="parsed_prompt"
        >
          <Textarea
            value={params.parsedPrompt}
            disabled={disabled}
            placeholder="Появится после «Разобрать фразу»"
            className="min-h-16 text-xs"
            onChange={(e) => {
              const parsedPrompt = e.target.value
              onChange({ ...params, parsedPrompt, targets: targetsFromParsedPrompt(parsedPrompt) })
            }}
          />
        </StackedField>
      </Station>

      <Station id="targets" title="Цели" summary={targetsSummary}>
        <TargetsEditor
          targets={params.targets}
          disabled={disabled}
          onChange={(targets) => onChange({ ...params, targets })}
        />
      </Station>

      <Station id="detect" title="Поиск рамок" summary={detectSummary}>
        <BackendSelectors
          detector={params.run.detector}
          detectorModel={params.run.detector_model}
          segmenter={params.run.segmenter}
          segmenterModel={params.run.segmenter_model}
          onChange={(patch) => setRun(patch)}
          disabled={disabled}
          detectorOnly
        />
        <ParamSlider
          param="detector_threshold"
          label={ADVANCED_META.detector_threshold.label}
          hint={ADVANCED_META.detector_threshold.hint}
          value={Number(params.advanced.detector_threshold || 0.15)}
          min={ADVANCED_META.detector_threshold.min ?? 0.05}
          max={ADVANCED_META.detector_threshold.max ?? 0.5}
          step={ADVANCED_META.detector_threshold.step ?? 0.01}
          disabled={disabled}
          onChange={(v) => setAdv("detector_threshold", String(v))}
        />
        <ParamSlider
          param="detector_keyframes"
          label={ADVANCED_META.detector_keyframes.label}
          hint={ADVANCED_META.detector_keyframes.hint}
          value={Number(params.advanced.detector_keyframes || 0)}
          min={0}
          max={ADVANCED_META.detector_keyframes.max ?? 40}
          step={1}
          disabled={disabled}
          formatValue={(v) => (v <= 0 ? "авто" : String(v))}
          onChange={(v) => setAdv("detector_keyframes", v <= 0 ? "" : String(v))}
        />
        <ParamSlider
          param="detector_nms_iou"
          label={ADVANCED_META.detector_nms_iou.label}
          hint={ADVANCED_META.detector_nms_iou.hint}
          value={Number(params.advanced.detector_nms_iou || 0.3)}
          min={ADVANCED_META.detector_nms_iou.min ?? 0.1}
          max={ADVANCED_META.detector_nms_iou.max ?? 0.7}
          step={ADVANCED_META.detector_nms_iou.step ?? 0.05}
          disabled={disabled}
          onChange={(v) => setAdv("detector_nms_iou", String(v))}
        />
        <ParamSlider
          param="detector_max_box_area"
          label={ADVANCED_META.detector_max_box_area.label}
          hint={ADVANCED_META.detector_max_box_area.hint}
          value={Number(params.advanced.detector_max_box_area || 0.45)}
          min={ADVANCED_META.detector_max_box_area.min ?? 0.05}
          max={ADVANCED_META.detector_max_box_area.max ?? 0.7}
          step={ADVANCED_META.detector_max_box_area.step ?? 0.01}
          disabled={disabled}
          formatValue={formatShare}
          onChange={(v) => setAdv("detector_max_box_area", String(v))}
        />
        <ParamSlider
          param="detect_stride"
          label="Осмотр, каждый N-й"
          hint={MODE_HINTS.detectStride}
          value={params.detect.stride}
          min={1}
          max={30}
          step={1}
          disabled={disabled}
          onChange={(v) => onChange({ ...params, detect: { ...params.detect, stride: v, all: false } })}
        />
        <ParamSlider
          param="tracker_min_score"
          label={ADVANCED_META.tracker_min_score.label}
          hint={ADVANCED_META.tracker_min_score.hint}
          value={Number(params.advanced.tracker_min_score || 0.55)}
          min={ADVANCED_META.tracker_min_score.min ?? 0.2}
          max={ADVANCED_META.tracker_min_score.max ?? 0.9}
          step={ADVANCED_META.tracker_min_score.step ?? 0.05}
          disabled={disabled}
          onChange={(v) => setAdv("tracker_min_score", String(v))}
        />
        <ParamSlider
          param="tracker_max_template_area"
          label={ADVANCED_META.tracker_max_template_area.label}
          hint={ADVANCED_META.tracker_max_template_area.hint}
          value={Number(params.advanced.tracker_max_template_area || 0.12)}
          min={ADVANCED_META.tracker_max_template_area.min ?? 0.02}
          max={ADVANCED_META.tracker_max_template_area.max ?? 0.4}
          step={ADVANCED_META.tracker_max_template_area.step ?? 0.01}
          disabled={disabled}
          formatValue={formatShare}
          onChange={(v) => setAdv("tracker_max_template_area", String(v))}
        />
        <SwitchRow
          param="select_relax"
          label={ADVANCED_META.select_relax.label}
          hint={ADVANCED_META.select_relax.hint}
          checked={(params.advanced.select_relax ?? "1") !== "0"}
          disabled={disabled}
          onCheckedChange={(checked) => setAdv("select_relax", checked ? "1" : "0")}
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" disabled={findDisabled || disabled} onClick={onFind}>
            Найти рамки
          </Button>
          {detectRunning && (
            <>
              <span className="text-xs text-muted-foreground">
                {Math.round((detectProgress?.fraction ?? 0) * 100)}%
                {detectProgress?.eta ? ` · ETA ${detectProgress.eta}` : ""}
              </span>
              <Button
                size="xs"
                variant="outline"
                disabled={!detectJobId}
                onClick={() => detectJobId && void cancelJob(detectJobId)}
              >
                Стоп
              </Button>
            </>
          )}
        </div>
        {detectError && <p className="text-xs text-destructive">{detectError}</p>}
        {foundTracks > 0 && (
          <p className="text-xs text-muted-foreground">
            {foundTracks === 1 ? "Найдена 1 рамка" : `Найдено рамок: ${foundTracks}`}
          </p>
        )}
        {tracksFull && (
          <div className="flex items-center justify-between gap-3">
            <FieldLabel className="min-w-0 flex-1">Удалять по найденным рамкам</FieldLabel>
            <Switch checked={useTracks} disabled={disabled} onCheckedChange={(checked) => onUseTracks?.(checked)} />
          </div>
        )}
      </Station>

      <Station id="segment" title="Сегментация" summary={segmentSummary}>
        <BackendSelectors
          detector={params.run.detector}
          detectorModel={params.run.detector_model}
          segmenter={params.run.segmenter}
          segmenterModel={params.run.segmenter_model}
          onChange={(patch) => setRun(patch)}
          disabled={disabled}
          segmenterOnly
        />
        <ParamSlider
          param="mask_dilate_px"
          label={RUN_PARAM_META.mask_dilate_px.label}
          hint={RUN_PARAM_META.mask_dilate_px.hint}
          value={params.run.mask_dilate_px}
          min={RUN_PARAM_META.mask_dilate_px.min ?? 0}
          max={RUN_PARAM_META.mask_dilate_px.max ?? 15}
          step={1}
          disabled={disabled}
          formatValue={(v) => `${v} px`}
          onChange={(v) => setRun({ mask_dilate_px: v })}
        />
      </Station>

      <Station id="fill" title="Заливка" summary={fillSummary}>
        <StackedField label="Способ заливки" hint="Чем заполнять вырезанные области." param="inpainter">
          <Select
            value={params.run.inpainter}
            disabled={disabled}
            onValueChange={(v) => {
              if (!v) return
              const nextModels = catalog.filter((m) => m.backend === v)
              const next = nextModels.find((m) => m.state === "ready") || nextModels[0]
              setRun({
                inpainter: v,
                inpainter_model: next ? modelRef(next.backend, next.model_ref, next.title) : "",
              })
            }}
          >
            <SelectTrigger size="sm" className="w-full">
              <SelectValue>
                {(value: string | null) => (value && INPAINTER_META[value]?.label) || value}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {inpainters.map((o) => (
                <SelectItem key={o} value={o}>
                  {INPAINTER_META[o]?.label ?? o}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </StackedField>
        <StackedField label="Модель заливки" hint="Веса выбранного способа заливки." param="inpainter_model">
          <Select
            value={selectedModel?.id ?? ""}
            disabled={disabled || modelChoices.length === 0}
            onValueChange={(id) => {
              if (!id) return
              const m = modelChoices.find((x) => x.id === id)
              if (m) setRun({ inpainter_model: modelRef(m.backend, m.model_ref, m.title) })
            }}
          >
            <SelectTrigger size="sm" className="w-full">
              <SelectValue placeholder="Модель">
                {selectedModel ? selectedModel.title : null}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {modelChoices.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  {m.title}
                  {m.state && m.state !== "ready" ? ` · ${m.state}` : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </StackedField>
        {showProPainter &&
          PROPAINTER_KEYS.map((key) => {
            const meta = ADVANCED_META[key]
            return (
              <ParamSlider
                key={key}
                param={key}
                label={meta.label}
                hint={meta.hint}
                value={Number(params.advanced[key] || meta.min || 0)}
                min={meta.min ?? 0}
                max={meta.max ?? 100}
                step={meta.step ?? 1}
                disabled={disabled || !propainter}
                onChange={(v) => setAdv(key, String(v))}
              />
            )
          })}
        <ParamSlider
          param="inpaint_chunk_overlap"
          label={ADVANCED_META.inpaint_chunk_overlap.label}
          hint={ADVANCED_META.inpaint_chunk_overlap.hint}
          value={Number(params.advanced.inpaint_chunk_overlap || 0)}
          min={ADVANCED_META.inpaint_chunk_overlap.min ?? 0}
          max={ADVANCED_META.inpaint_chunk_overlap.max ?? 32}
          step={1}
          disabled={disabled}
          onChange={(v) => setAdv("inpaint_chunk_overlap", String(v))}
        />
        <ParamSlider
          param="inpaint_workers"
          label={ADVANCED_META.inpaint_workers.label}
          hint={ADVANCED_META.inpaint_workers.hint}
          value={Number(params.advanced.inpaint_workers || 0)}
          min={0}
          max={ADVANCED_META.inpaint_workers.max ?? 16}
          step={1}
          disabled={disabled}
          formatValue={(v) => (v <= 0 ? "авто" : String(v))}
          onChange={(v) => setAdv("inpaint_workers", String(v))}
        />
        <CeilingInput
          label="Сторона заливки"
          unit="px"
          param="inpaint_max_side"
          value={params.run.inpaint_max_side}
          disabled={disabled}
          onChange={(v) => setRun({ inpaint_max_side: v })}
        />
      </Station>

      <Station id="verify" title="Проверка" summary={verifySummary}>
        <SwitchRow
          param="verify"
          label={RUN_PARAM_META.verify.label}
          hint={RUN_PARAM_META.verify.hint}
          checked={params.run.verify}
          disabled={disabled}
          onCheckedChange={(checked) => setRun({ verify: checked })}
        />
        <ParamSlider
          param="verify_max_passes"
          label={ADVANCED_META.verify_max_passes.label}
          hint={ADVANCED_META.verify_max_passes.hint}
          value={Number(params.advanced.verify_max_passes || 0)}
          min={ADVANCED_META.verify_max_passes.min ?? 0}
          max={ADVANCED_META.verify_max_passes.max ?? 3}
          step={1}
          disabled={disabled || !verifyOn}
          onChange={(v) => setAdv("verify_max_passes", String(v))}
        />
        <ParamSlider
          param="verify_max_coverage"
          label={RUN_PARAM_META.verify_max_coverage.label}
          hint={RUN_PARAM_META.verify_max_coverage.hint}
          value={params.run.verify_max_coverage}
          min={RUN_PARAM_META.verify_max_coverage.min ?? 0.01}
          max={RUN_PARAM_META.verify_max_coverage.max ?? 0.5}
          step={RUN_PARAM_META.verify_max_coverage.step ?? 0.01}
          disabled={disabled || !verifyOn}
          formatValue={formatShare}
          onChange={(v) => setRun({ verify_max_coverage: v })}
        />
        <ParamDecimal
          param="min_mask_coverage"
          label={RUN_PARAM_META.min_mask_coverage.label}
          hint={RUN_PARAM_META.min_mask_coverage.hint}
          value={params.run.min_mask_coverage}
          min={RUN_PARAM_META.min_mask_coverage.min ?? 0}
          max={RUN_PARAM_META.min_mask_coverage.max ?? 0.02}
          step={RUN_PARAM_META.min_mask_coverage.step ?? 0.0001}
          disabled={disabled}
          readout={formatShare(params.run.min_mask_coverage)}
          onChange={(v) => setRun({ min_mask_coverage: v })}
        />
        <SwitchRow
          param="verify_redetect"
          label="Повторный поиск"
          hint="Второй проход детектора по остатку. Работает только вместе с проверкой остатков."
          checked={params.run.verify_redetect}
          disabled={disabled || !verifyOn}
          onCheckedChange={(checked) => setRun({ verify_redetect: checked })}
        />
      </Station>

      <Station id="device" title="Устройство" summary={deviceSummary}>
        <StackedField label="Устройство" hint={DEVICE_META[params.run.device || "auto"]?.hint} param="device">
          <Select
            value={params.run.device || "auto"}
            disabled={disabled}
            onValueChange={(v) => {
              if (v) setRun({ device: v })
            }}
          >
            <SelectTrigger size="sm" className="w-full">
              <SelectValue>
                {(value: string | null) => DEVICE_META[value || "auto"]?.label ?? value}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {DEVICE_OPTIONS.map((d) => (
                <SelectItem key={d} value={d}>
                  {DEVICE_META[d]?.label ?? d}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </StackedField>
        <CeilingInput
          label="Потолок VRAM"
          unit="МБ"
          param="max_vram_mb"
          value={params.run.max_vram_mb}
          disabled={disabled}
          onChange={(v) => setRun({ max_vram_mb: v })}
        />
        <CeilingInput
          label="Потоки CPU"
          unit="ядра"
          param="cpu_threads"
          value={params.run.cpu_threads}
          disabled={disabled}
          onChange={(v) => setRun({ cpu_threads: v })}
        />
        <SwitchRow
          param="keep_workdir"
          label={RUN_PARAM_META.keep_workdir.label}
          hint={RUN_PARAM_META.keep_workdir.hint}
          checked={params.run.keep_workdir}
          disabled={disabled}
          onCheckedChange={(checked) => setRun({ keep_workdir: checked })}
        />
      </Station>

      <Station id="output" title="Формат" summary={formatSummary || "MP4"}>
        <OutputFormatBlock
          params={params}
          onChange={onChange}
          disabled={disabled}
          resultJob={resultJob}
          onResultJobChange={onResultJobChange}
          showWebm={showWebm}
          showSegment={showSegment}
        />
      </Station>
    </Accordion>
  )
}

function modelRef(backend: string, modelRefValue: string, title: string): string {
  if (backend === "lama") return "big-lama"
  if (modelRefValue.startsWith("http://") || modelRefValue.startsWith("https://")) return title
  return modelRefValue
}
