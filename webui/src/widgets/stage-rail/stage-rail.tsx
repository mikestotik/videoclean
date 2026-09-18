import { useEffect, useMemo, useRef, useState } from "react"
import { useInterpret } from "@features/interpret"
import { tracksAreFullLength, useDetectRun } from "@features/detect-run"
import { useInpaintRun } from "@features/inpaint-run"
import {
  cancelJob,
  downloadJobOutput,
  getJob,
  outputDownloadName,
  packageJob,
  pollJobToCompletion,
  type Job,
} from "@/entities/job"
import type { Source } from "@/entities/source"
import { api } from "@/shared/api/client"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import { Label } from "@/shared/ui/label"
import { Slider } from "@/shared/ui/slider"
import { Switch } from "@/shared/ui/switch"
import { Textarea } from "@/shared/ui/textarea"
import { ToggleGroup, ToggleGroupItem } from "@/shared/ui/toggle-group"
import { TargetsEditor } from "./targets-editor"
import {
  AdvancedFields,
  BackendSelectors,
  FieldLabel,
  InpaintControls,
  LlmChip,
  ParamHint,
  ParamSlider,
  PresetsPopover,
  StageSection,
  type OptionsShape,
} from "./stage-parts"
import { ADVANCED_META, FORMAT_META, MODE_HINTS, RUN_PARAM_META } from "./param-meta"
import {
  OUTPUT_FORMATS,
  autoStride,
  enabledTargets,
  resetParams,
  toDetectParams,
  toRunParams,
  type EditorParams,
  type InpaintMode,
} from "./params"

type Props = {
  source: Source | null
  frameCount: number
  params: EditorParams
  onParamsChange: (p: EditorParams) => void
  interpret: ReturnType<typeof useInterpret>
  detect: ReturnType<typeof useDetectRun>
  inpaint: ReturnType<typeof useInpaintRun>
  resultJob: Job | null
  masks?: number[]
  onRunAll?: () => void
  runAllBusy?: boolean
  runAllError?: string
  onOpenConfig?: () => void
  onOpenResult?: () => void
  onResultJobChange?: (job: Job) => void
}

function ResultPanel({
  job,
  onOpenResult,
  onResultJobChange,
}: {
  job: Job
  onOpenResult?: () => void
  onResultJobChange?: (job: Job) => void
}) {
  const [convertFormats, setConvertFormats] = useState<string[]>(["webm"])
  const [webmCrf, setWebmCrf] = useState(32)
  const [segmentSeconds, setSegmentSeconds] = useState(6)
  const [packBusy, setPackBusy] = useState(false)
  const [packJobId, setPackJobId] = useState<string | null>(null)
  const [packProgress, setPackProgress] = useState({ fraction: 0, detail: "", eta: "" })
  const [packError, setPackError] = useState("")
  const [busyFmt, setBusyFmt] = useState<string | null>(null)
  const [dlError, setDlError] = useState("")

  const entries =
    job.outputs && Object.keys(job.outputs).length > 0
      ? Object.entries(job.outputs)
      : job.output_url
        ? [["default", job.output_url] as const]
        : []
  const built = new Set(entries.map(([fmt]) => fmt))
  const needsWebm = convertFormats.includes("webm")
  const needsSegment = convertFormats.some((f) => f === "hls-fmp4" || f === "hls-ts" || f === "dash")
  const canPackage = job.can_package !== false
  const allReady = convertFormats.length > 0 && convertFormats.every((f) => built.has(f))

  const onDownload = async (fmt: string, url: string) => {
    setDlError("")
    setBusyFmt(fmt)
    try {
      await downloadJobOutput(url, outputDownloadName(fmt, job.id))
    } catch (e) {
      setDlError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyFmt(null)
    }
  }

  const onConvert = async () => {
    if (!canPackage || packBusy || convertFormats.length === 0) return
    setPackError("")
    setPackBusy(true)
    setPackProgress({ fraction: 0, detail: "", eta: "" })
    try {
      const queued = await packageJob(job.id, {
        formats: convertFormats,
        webm_crf: needsWebm ? webmCrf : undefined,
        segment_seconds: needsSegment ? segmentSeconds : undefined,
        overwrite: true,
      })
      setPackJobId(queued.id)
      await pollJobToCompletion(queued.id, (j) => {
        setPackProgress({ fraction: j.fraction, detail: j.detail, eta: j.eta })
      })
      const refreshed = await getJob(job.id)
      onResultJobChange?.(refreshed)
    } catch (e) {
      setPackError(e instanceof Error ? e.message : String(e))
    } finally {
      setPackBusy(false)
      setPackJobId(null)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-ok">Удаление готово.</p>
      {onOpenResult && (
        <Button size="sm" variant="outline" className="w-fit" onClick={onOpenResult}>
          Сравнить до/после
        </Button>
      )}

      <div className="flex flex-col gap-1.5 text-xs">
        <FieldLabel hint="Конвертация из мастер-файла удаления. Можно запускать повторно в разные форматы.">
          Форматы для скачивания
        </FieldLabel>
        <div className="flex flex-col gap-1.5">
          {OUTPUT_FORMATS.map((f) => {
            const meta = FORMAT_META[f]
            const ready = built.has(f)
            return (
              <label key={f} className="flex items-center gap-2">
                <Switch
                  size="sm"
                  checked={convertFormats.includes(f)}
                  disabled={packBusy || !canPackage}
                  onCheckedChange={(checked) => {
                    setConvertFormats((prev) => {
                      const next = checked ? [...prev, f] : prev.filter((x) => x !== f)
                      return next.length ? next : prev
                    })
                  }}
                />
                <span className="min-w-0 flex-1">
                  {meta?.label ?? f}
                  {ready ? <span className="ml-1 text-ok">· есть</span> : null}
                </span>
                {meta?.hint ? <ParamHint text={meta.hint} /> : null}
              </label>
            )
          })}
        </div>
      </div>

      {needsWebm && (
        <ParamSlider
          label="WebM CRF"
          hint="Меньше — лучше качество и больше файл (VP9). Обычно 28–36."
          value={webmCrf}
          min={18}
          max={45}
          step={1}
          disabled={packBusy || !canPackage}
          onChange={setWebmCrf}
        />
      )}
      {needsSegment && (
        <ParamSlider
          label="Сегмент, с"
          hint="Длина сегмента HLS/DASH в секундах."
          value={segmentSeconds}
          min={2}
          max={12}
          step={1}
          disabled={packBusy || !canPackage}
          onChange={setSegmentSeconds}
        />
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button
          size="sm"
          disabled={!canPackage || packBusy || convertFormats.length === 0}
          onClick={() => void onConvert()}
        >
          {packBusy ? "Конвертация…" : allReady ? "Переконвертировать" : "Сконвертировать"}
        </Button>
        {packBusy && (
          <>
            <span className="text-xs text-muted-foreground">
              {Math.round(packProgress.fraction * 100)}%
              {packProgress.detail ? ` · ${packProgress.detail}` : ""}
              {packProgress.eta ? ` · ETA ${packProgress.eta}` : ""}
            </span>
            <Button
              size="xs"
              variant="outline"
              disabled={!packJobId}
              onClick={() => packJobId && void cancelJob(packJobId)}
            >
              Стоп
            </Button>
          </>
        )}
      </div>
      {!canPackage && (
        <p className="text-[11px] text-muted-foreground">
          Мастер-файл недоступен — перезапустите удаление.
        </p>
      )}
      {packError && <p className="text-xs text-destructive">{packError}</p>}

      {entries.length > 0 && (
        <div className="flex flex-col gap-1.5">
          <FieldLabel hint="Готовые файлы этого результата.">Скачать</FieldLabel>
          <div className="flex flex-wrap gap-2">
            {entries.map(([fmt, url]) => (
              <Button
                key={fmt}
                size="sm"
                variant="secondary"
                disabled={busyFmt === fmt}
                onClick={() => void onDownload(fmt, url)}
              >
                {busyFmt === fmt
                  ? "…"
                  : FORMAT_META[fmt]?.label ?? (fmt === "default" ? "файл" : fmt)}
              </Button>
            ))}
          </div>
        </div>
      )}
      {dlError && <p className="text-xs text-destructive">{dlError}</p>}
    </div>
  )
}

export function StageRail({
  source,
  frameCount,
  params,
  onParamsChange,
  interpret,
  detect,
  inpaint,
  resultJob,
  masks,
  onRunAll,
  runAllBusy,
  runAllError,
  onOpenConfig,
  onOpenResult,
  onResultJobChange,
}: Props) {
  const [inpaintMode, setInpaintMode] = useState<InpaintMode>("tracks")
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [openStages, setOpenStages] = useState<Record<number, boolean>>({ 1: true })
  const [pipelineOpts, setPipelineOpts] = useState<OptionsShape | null>(null)
  const noSource = !source
  const set = (patch: Partial<EditorParams>) => onParamsChange({ ...params, ...patch })

  useEffect(() => {
    api<OptionsShape>("/api/options")
      .then(setPipelineOpts)
      .catch(() => setPipelineOpts(null))
  }, [])

  const readiness = useMemo(() => {
    const opts = pipelineOpts
    if (!opts) return { detector: null as string | null, segmenter: null as string | null, inpainter: null as string | null }
    const detModels = opts.detector_models ?? []
    const segModels = opts.segmenter_models ?? []
    const inpModels = opts.models?.inpainter ?? []
    const detId = params.run.detector_model || opts.default_detector_model || ""
    const segId = params.run.segmenter_model || opts.default_segmenter_model || ""
    const det =
      detModels.find((m) => m.id === detId || m.model_ref === detId) ||
      detModels.find((m) => !params.run.detector || m.backend === params.run.detector) ||
      detModels[0]
    const seg =
      segModels.find((m) => m.id === segId || m.model_ref === segId) ||
      segModels.find((m) => m.backend === (params.run.segmenter || "sam2")) ||
      segModels[0]
    const inp =
      inpModels.find((m) => m.id === params.run.inpainter_model || m.model_ref === params.run.inpainter_model) ||
      inpModels.find((m) => m.backend === params.run.inpainter) ||
      inpModels.find((m) => m.state === "ready") ||
      inpModels[0]
    return {
      detector: det && det.ready === false ? det.title || det.id : null,
      segmenter: seg && seg.ready === false ? seg.title || seg.id : null,
      inpainter: inp && inp.state && inp.state !== "ready" ? inp.title || inp.id : null,
    }
  }, [pipelineOpts, params.run.detector, params.run.detector_model, params.run.segmenter, params.run.segmenter_model, params.run.inpainter, params.run.inpainter_model])

  const hasMasks = (masks?.length ?? 0) > 0
  const hasPrompt = params.prompt.trim().length > 0
  const hasTargets = enabledTargets(params).length > 0
  const hasDetectResult = Boolean(detect.manifest) || detect.tracks.length > 0
  const hasResult = resultJob?.state === "COMPLETED"
  const tracksReady = tracksAreFullLength(detect.enabledTracks, frameCount)
  const detectBlocked = Boolean(readiness.detector)
  const inpaintBlocked = Boolean(readiness.segmenter || readiness.inpainter)
  const canFindMasks =
    !noSource &&
    !detect.running &&
    !detectBlocked &&
    (params.detect.mode === "targets" ? hasTargets : hasPrompt)

  const runAllDisabled =
    noSource ||
    runAllBusy ||
    interpret.running ||
    detect.running ||
    inpaint.running ||
    (!hasPrompt && !hasMasks && !hasTargets) ||
    (hasMasks ? inpaintBlocked : detectBlocked || inpaintBlocked)

  // Accent numbers for stages that already produced a result; "active" is the next step.
  const stageDone: Record<number, boolean> = {
    1: hasPrompt || hasMasks,
    2: hasTargets || Boolean(interpret.result),
    3: hasDetectResult,
    4: hasResult,
    5: hasResult,
  }

  const activeStage = hasResult
    ? 5
    : inpaint.running
      ? 4
      : detect.running
        ? 3
        : interpret.running
          ? 1
          : hasDetectResult
            ? 4
            : hasTargets || Boolean(interpret.result)
              ? 3
              : hasPrompt || hasMasks
                ? 2
                : 1

  useEffect(() => {
    setOpenStages((prev) => (prev[activeStage] ? prev : { ...prev, [activeStage]: true }))
  }, [activeStage])

  const stageOpen = (n: number) => openStages[n] ?? false
  const setStageOpen = (n: number, open: boolean) =>
    setOpenStages((prev) => ({ ...prev, [n]: open }))

  const appliedResultRef = useRef<unknown>(null)
  useEffect(() => {
    if (!interpret.result || appliedResultRef.current === interpret.result) return
    appliedResultRef.current = interpret.result
    onParamsChange({
      ...params,
      prompt: interpret.result.prompt || params.prompt,
      targets: interpret.result.targets.map((t) => ({
        kind: t.kind === "watermark" || t.kind === "text_overlay" ? t.kind : "object",
        query: t.query,
        where: t.where,
        enabled: true,
        source: "auto" as const,
      })),
    })
  }, [interpret.result, params, onParamsChange])

  const runDetect = () =>
    detect.run({
      mode: params.detect.mode === "targets" ? "detect" : "parse",
      prompt: params.prompt,
      targets: enabledTargets(params),
      all: params.detect.all,
      stride: params.detect.stride,
      params: toDetectParams(params),
    })

  const runInpaint = () => {
    if (inpaintMode === "tracks" && !tracksReady) return
    const targets = enabledTargets(params)
    const payload =
      inpaintMode === "tracks"
        ? { mode: "tracks" as const, tracks: detect.enabledTracks }
        : inpaintMode === "masks"
          ? {
              mode: "masks" as const,
              masks: masks ?? [],
              targets: targets.length > 0 ? targets : undefined,
              prompt: params.prompt,
            }
          : { mode: "prompt" as const, prompt: params.prompt, targets }
    void inpaint.run(payload, toRunParams(params))
  }

  const setAdvanced = (key: string, value: string) =>
    set({ advanced: { ...params.advanced, [key]: value } })

  return (
    <aside className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 space-y-2 border-b border-border/70 px-3 py-3">
        <div className="flex items-start justify-between gap-2">
          <div>
            <h2 className="text-sm font-semibold">Конвейер</h2>
            <p className="text-[11px] text-muted-foreground">От промпта до готового ролика</p>
          </div>
          {onRunAll && (
            <Button size="sm" disabled={runAllDisabled} onClick={onRunAll}>
              {runAllBusy ? "Идёт…" : "Запустить всё"}
            </Button>
          )}
        </div>
        {runAllError && <p className="text-xs text-destructive">{runAllError}</p>}
        {noSource && (
          <p className="rounded-md bg-muted/50 px-2.5 py-2 text-[11px] text-muted-foreground">
            Выберите видео слева, чтобы настроить и запустить обработку.
          </p>
        )}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
      <StageSection
        n={1}
        title="Промпт"
        hint="Что нужно убрать из ролика"
        active={activeStage === 1}
        done={stageDone[1]}
        open={stageOpen(1)}
        onOpenChange={(o) => setStageOpen(1, o)}
      >
        <Textarea
          value={params.prompt}
          onChange={(e) => set({ prompt: e.target.value })}
          placeholder="Например: логотип в правом верхнем углу"
          className="min-h-20 text-xs"
          disabled={noSource}
        />
        <LlmChip
          value={params.run.llm_model}
          onChange={(patch) => set({ run: { ...params.run, ...patch } })}
          onOpenConfig={onOpenConfig}
          disabled={noSource}
        />
        <Button
          size="sm"
          disabled={noSource || interpret.running || (!params.prompt.trim() && !(masks && masks.length > 0))}
          onClick={() => void interpret.run(params.prompt, params.run.llm_model)}
        >
          Интерпретировать
        </Button>
        {interpret.running && <p className="text-xs text-muted-foreground">Разбираю кадры через LLM…</p>}
        {interpret.error && <p className="text-xs text-destructive">{interpret.error}</p>}
      </StageSection>

      <StageSection
        n={2}
        title="Цели"
        hint="Список объектов для поиска"
        active={activeStage === 2}
        done={stageDone[2]}
        open={stageOpen(2)}
        onOpenChange={(o) => setStageOpen(2, o)}
      >
        <TargetsEditor targets={params.targets} onChange={(targets) => set({ targets })} disabled={noSource} />
        {interpret.result && (interpret.result.parseMode || interpret.result.framesUsed || interpret.result.visionFrameIndices) && (
          <p className="text-[11px] text-muted-foreground">
            parse: {interpret.result.parseMode ?? "—"}
            {interpret.result.defaulted ? " · defaulted" : ""}
            {(() => {
              const frames = interpret.result.visionFrameIndices ?? interpret.result.framesUsed
              return frames?.length ? ` · кадры: ${frames.join(",")}` : ""
            })()}
          </p>
        )}
        {detect.manifest?.selectRelaxed && (
          <p className="text-[11px] text-amber-600 dark:text-amber-400">relaxed match — where/ordinal ослаблены</p>
        )}
      </StageSection>

      <StageSection
        n={3}
        title="Маски"
        hint="Где именно вырезать"
        active={activeStage === 3}
        done={stageDone[3]}
        open={stageOpen(3)}
        onOpenChange={(o) => setStageOpen(3, o)}
      >
        <div className="flex items-center gap-1.5">
          <ToggleGroup
            variant="outline"
            size="sm"
            value={[params.detect.all ? "all" : "stride"]}
            onValueChange={(v) => {
              const next = v.at(-1)
              if (next) set({ detect: { ...params.detect, all: next === "all" } })
            }}
          >
            <ToggleGroupItem value="all">Всё видео</ToggleGroupItem>
            <ToggleGroupItem value="stride">Каждый N-й кадр</ToggleGroupItem>
          </ToggleGroup>
          <ParamHint text={params.detect.all ? MODE_HINTS.detectAll : MODE_HINTS.detectStride} />
        </div>
        {!params.detect.all && (
          <div className="flex items-center gap-2 text-xs">
            <Slider
              className="w-32"
              min={1}
              max={30}
              step={1}
              value={[params.detect.stride]}
              onValueChange={(v) => {
                const n = Array.isArray(v) ? v[0] : v
                if (typeof n === "number") set({ detect: { ...params.detect, stride: n } })
              }}
            />
            <span className="text-muted-foreground">шаг {params.detect.stride}</span>
            <Button
              size="xs"
              variant="ghost"
              disabled={frameCount <= 0}
              onClick={() => set({ detect: { ...params.detect, stride: autoStride(frameCount) } })}
            >
              Авто: {autoStride(frameCount)}
            </Button>
          </div>
        )}
        <ToggleGroup
          variant="outline"
          size="sm"
          value={[params.detect.mode]}
          onValueChange={(v) => {
            const next = v.at(-1)
            if (next === "targets" || next === "prompt") set({ detect: { ...params.detect, mode: next } })
          }}
        >
          <ToggleGroupItem value="targets">По таргетам</ToggleGroupItem>
          <ToggleGroupItem value="prompt">По промпту</ToggleGroupItem>
        </ToggleGroup>
        <BackendSelectors
          detector={params.run.detector}
          detectorModel={params.run.detector_model}
          segmenter={params.run.segmenter}
          segmenterModel={params.run.segmenter_model}
          onChange={(patch) => set({ run: { ...params.run, ...patch } })}
          disabled={noSource}
          detectorOnly
        />
        <Badge variant="secondary" className="w-fit text-[10px] font-normal">
          Превью: sam2 покадрово, без inpaint / sam2-video
        </Badge>
        {readiness.detector && (
          <p className="text-xs text-destructive">Детектор не готов: {readiness.detector} — скачайте в Настройках</p>
        )}
        <ParamSlider
          label={ADVANCED_META.detector_threshold.label}
          hint={ADVANCED_META.detector_threshold.hint}
          value={Number(params.advanced.detector_threshold || 0.15)}
          min={ADVANCED_META.detector_threshold.min ?? 0.05}
          max={ADVANCED_META.detector_threshold.max ?? 0.5}
          step={ADVANCED_META.detector_threshold.step ?? 0.01}
          disabled={noSource}
          onChange={(v) => setAdvanced("detector_threshold", String(v))}
        />
        <ParamSlider
          label={ADVANCED_META.detector_keyframes.label}
          hint={`${ADVANCED_META.detector_keyframes.hint} 0 = авто.`}
          value={Number(params.advanced.detector_keyframes || 0)}
          min={0}
          max={ADVANCED_META.detector_keyframes.max ?? 40}
          step={1}
          disabled={noSource}
          formatValue={(v) => (v <= 0 ? "авто" : String(v))}
          onChange={(v) => setAdvanced("detector_keyframes", v <= 0 ? "" : String(v))}
        />
        <ParamSlider
          label={RUN_PARAM_META.mask_dilate_px.label}
          hint={RUN_PARAM_META.mask_dilate_px.hint}
          value={params.run.mask_dilate_px}
          min={0}
          max={15}
          disabled={noSource}
          onChange={(v) => set({ run: { ...params.run, mask_dilate_px: v } })}
        />
        <Button size="sm" disabled={!canFindMasks} onClick={runDetect}>
          Найти маски
        </Button>
        {detect.running && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">
              {Math.round(detect.progress.fraction * 100)}% {detect.progress.eta && `· ETA ${detect.progress.eta}`}
            </span>
            <Button size="xs" variant="outline" disabled={!detect.jobId} onClick={() => detect.jobId && void cancelJob(detect.jobId)}>
              Стоп
            </Button>
          </div>
        )}
        {detect.error && <p className="text-xs text-destructive">{detect.error}</p>}
        {!detect.running && detect.manifest && (
          <p className="text-xs text-muted-foreground">
            Покрытие: {((detect.manifest.meanMaskCoverage ?? 0) * 100).toFixed(2)}% · треков:{" "}
            {detect.enabledTracks.length}/{detect.tracks.length}
            {detect.tracks[0] ? ` · длина ${detect.tracks[0].boxes.length}/${frameCount}` : ""}
            {tracksReady ? " · готово к удалению" : params.detect.all ? "" : " · только осмотр"}
            {detect.manifest.selectRelaxed ? " · relaxed match" : ""}
            {detect.manifest.parseMode ? ` · parse ${detect.manifest.parseMode}` : ""}
          </p>
        )}
        {hasDetectResult && (
          <div className="space-y-1.5">
            <div className="flex items-center gap-1.5">
              <Label className="text-[11px] text-muted-foreground">Правка рамки</Label>
              <ParamHint text={detect.boxEditMode === "hold" ? MODE_HINTS.boxHold : MODE_HINTS.boxFrame} />
            </div>
            <ToggleGroup
              variant="outline"
              size="sm"
              value={[detect.boxEditMode]}
              onValueChange={(v) => {
                const next = v.at(-1)
                if (next === "frame" || next === "hold") detect.setBoxEditMode(next)
              }}
            >
              <ToggleGroupItem value="hold">Протянуть вперёд</ToggleGroupItem>
              <ToggleGroupItem value="frame">Только кадр</ToggleGroupItem>
            </ToggleGroup>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                variant={detect.tracksDirty ? "default" : "outline"}
                disabled={!detect.lastJobId || detect.saving || detect.tracks.length === 0}
                onClick={() => void detect.saveTracks()}
              >
                {detect.saving ? "Сохраняю…" : detect.tracksDirty ? "Сохранить правки" : "Сохранено"}
              </Button>
              {detect.tracksDirty && (
                <span className="text-[11px] text-muted-foreground">не сохранено</span>
              )}
            </div>
            {detect.saveError && <p className="text-xs text-destructive">{detect.saveError}</p>}
          </div>
        )}
      </StageSection>

      <StageSection
        n={4}
        title="Удаление"
        hint="Заполнение вырезанных областей"
        active={activeStage === 4}
        done={stageDone[4]}
        open={stageOpen(4)}
        onOpenChange={(o) => setStageOpen(4, o)}
      >
        <div className="flex items-center gap-1.5">
          <ToggleGroup
            variant="outline"
            size="sm"
            value={[inpaintMode]}
            onValueChange={(v) => {
              const next = v.at(-1)
              if (next === "tracks" || next === "masks" || next === "prompt") setInpaintMode(next)
            }}
          >
            <ToggleGroupItem value="tracks">По трекам</ToggleGroupItem>
            <ToggleGroupItem value="masks">По маскам</ToggleGroupItem>
            <ToggleGroupItem value="prompt">По промпту</ToggleGroupItem>
          </ToggleGroup>
          <ParamHint
            text={
              inpaintMode === "tracks"
                ? MODE_HINTS.inpaintTracks
                : inpaintMode === "masks"
                  ? MODE_HINTS.inpaintMasks
                  : MODE_HINTS.inpaintPrompt
            }
          />
        </div>
        {inpaintMode === "masks" && (
          <div className="flex items-center gap-1.5">
            <ToggleGroup
              variant="outline"
              size="sm"
              value={[params.maskPolicy]}
              onValueChange={(v) => {
                const next = v.at(-1)
                if (next === "static" || next === "propagate") set({ maskPolicy: next })
              }}
            >
              <ToggleGroupItem value="static">Держать</ToggleGroupItem>
              <ToggleGroupItem value="propagate">Протянуть</ToggleGroupItem>
            </ToggleGroup>
            <ParamHint text={params.maskPolicy === "static" ? MODE_HINTS.maskStatic : MODE_HINTS.maskPropagate} />
          </div>
        )}
        {inpaintMode === "prompt" ? (
          <BackendSelectors
            detector={params.run.detector}
            detectorModel={params.run.detector_model}
            segmenter={params.run.segmenter}
            segmenterModel={params.run.segmenter_model}
            onChange={(patch) => set({ run: { ...params.run, ...patch } })}
            disabled={noSource}
          />
        ) : (
          <BackendSelectors
            detector={params.run.detector}
            detectorModel={params.run.detector_model}
            segmenter={params.run.segmenter}
            segmenterModel={params.run.segmenter_model}
            onChange={(patch) => set({ run: { ...params.run, ...patch } })}
            disabled={noSource}
            segmenterOnly
          />
        )}
        <InpaintControls params={params} onParamsChange={onParamsChange} disabled={noSource} />
        <Button size="sm" variant="ghost" className="self-start" onClick={() => setAdvancedOpen((o) => !o)}>
          {advancedOpen ? "Скрыть тонкие настройки" : "Тонкие настройки"}
        </Button>
        {advancedOpen && (
          <AdvancedFields
            params={params}
            onParamsChange={onParamsChange}
            disabled={noSource}
          />
        )}
        {(readiness.segmenter || readiness.inpainter) && (
          <p className="text-xs text-destructive">
            {readiness.segmenter ? `Сегментер не готов: ${readiness.segmenter}` : null}
            {readiness.segmenter && readiness.inpainter ? " · " : null}
            {readiness.inpainter ? `Инпейнтер не готов: ${readiness.inpainter}` : null}
            {" — скачайте в Настройках"}
          </p>
        )}
        <Button
          size="sm"
          disabled={
            noSource ||
            inpaint.running ||
            inpaintBlocked ||
            (inpaintMode === "tracks" && !tracksReady) ||
            (inpaintMode === "masks" && !(masks && masks.length > 0)) ||
            (inpaintMode === "prompt" && !params.prompt.trim())
          }
          onClick={runInpaint}
        >
          Запустить удаление
        </Button>
        {inpaint.running && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">
              {Math.round(inpaint.progress.fraction * 100)}% {inpaint.progress.eta && `· ETA ${inpaint.progress.eta}`}
            </span>
            <Button size="xs" variant="outline" disabled={!inpaint.jobId} onClick={() => inpaint.jobId && void cancelJob(inpaint.jobId)}>
              Стоп
            </Button>
          </div>
        )}
        {inpaint.error && <p className="text-xs text-destructive">{inpaint.error}</p>}
      </StageSection>

      <StageSection
        n={5}
        title="Результат"
        hint="Конвертация и скачивание из мастер-файла удаления"
        active={activeStage === 5}
        done={stageDone[5]}
        open={stageOpen(5)}
        onOpenChange={(o) => setStageOpen(5, o)}
      >
        {resultJob?.state === "COMPLETED" ? (
          <ResultPanel
            job={resultJob}
            onOpenResult={onOpenResult}
            onResultJobChange={onResultJobChange}
          />
        ) : (
          <p className="text-xs text-muted-foreground">Результата ещё нет. Запустите удаление выше.</p>
        )}
      </StageSection>
      </div>

      <div className="flex shrink-0 items-center gap-2 border-t border-border/70 px-3 py-2.5">
        <PresetsPopover params={params} onParamsChange={onParamsChange} disabled={noSource} />
        <Button
          size="sm"
          variant="ghost"
          className="ml-auto"
          disabled={noSource}
          onClick={() => onParamsChange(resetParams())}
        >
          Сбросить
        </Button>
      </div>
    </aside>
  )
}

