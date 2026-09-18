import { useEffect, useRef, useState } from "react"
import { useInterpret } from "@features/interpret"
import { tracksAreFullLength, useDetectRun } from "@features/detect-run"
import { useInpaintRun } from "@features/inpaint-run"
import { cancelJob, type Job } from "@/entities/job"
import type { Source } from "@/entities/source"
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
}: Props) {
  const [inpaintMode, setInpaintMode] = useState<InpaintMode>("tracks")
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const [openStages, setOpenStages] = useState<Record<number, boolean>>({ 1: true })
  const noSource = !source
  const set = (patch: Partial<EditorParams>) => onParamsChange({ ...params, ...patch })

  const hasMasks = (masks?.length ?? 0) > 0
  const hasPrompt = params.prompt.trim().length > 0
  const hasTargets = enabledTargets(params).length > 0
  const hasDetectResult = Boolean(detect.manifest) || detect.tracks.length > 0
  const hasResult = resultJob?.state === "COMPLETED"
  const tracksReady = tracksAreFullLength(detect.enabledTracks, frameCount)
  const canFindMasks =
    !noSource &&
    !detect.running &&
    (params.detect.mode === "targets" ? hasTargets : hasPrompt)

  const runAllDisabled =
    noSource ||
    runAllBusy ||
    interpret.running ||
    detect.running ||
    inpaint.running ||
    (!hasPrompt && !hasMasks && !hasTargets)

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
    const payload =
      inpaintMode === "tracks"
        ? { mode: "tracks" as const, tracks: detect.enabledTracks }
        : inpaintMode === "masks"
          ? { mode: "masks" as const, masks: masks ?? [] }
          : { mode: "prompt" as const, prompt: params.prompt, targets: enabledTargets(params) }
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
          onChange={(model) => set({ run: { ...params.run, llm_model: model } })}
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
          segmenter={params.run.segmenter}
          segmenterModel={params.run.segmenter_model}
          onChange={(patch) => set({ run: { ...params.run, ...patch } })}
          disabled={noSource}
          detectorOnly
        />
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
        {inpaintMode === "prompt" && (
          <BackendSelectors
            detector={params.run.detector}
            segmenter={params.run.segmenter}
            segmenterModel={params.run.segmenter_model}
            onChange={(patch) => set({ run: { ...params.run, ...patch } })}
            disabled={noSource}
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
        <Button
          size="sm"
          disabled={
            noSource ||
            inpaint.running ||
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
        hint="Форматы, скачивание и сравнение до/после"
        active={activeStage === 5}
        done={stageDone[5]}
        open={stageOpen(5)}
        onOpenChange={(o) => setStageOpen(5, o)}
      >
        <div className="flex flex-col gap-1.5 text-xs">
          <FieldLabel hint="Что собрать после удаления. HLS/DASH — пакеты со плейлистом (скачиваются zip).">
            Форматы вывода
          </FieldLabel>
          <div className="flex flex-col gap-1.5">
            {OUTPUT_FORMATS.map((f) => {
              const meta = FORMAT_META[f]
              return (
                <label key={f} className="flex items-center gap-2">
                  <Switch
                    size="sm"
                    checked={params.run.formats.includes(f)}
                    disabled={noSource}
                    onCheckedChange={(checked) => {
                      const next = checked
                        ? [...params.run.formats, f]
                        : params.run.formats.filter((x) => x !== f)
                      set({
                        run: {
                          ...params.run,
                          formats: next.length ? next : ["mp4"],
                        },
                      })
                    }}
                  />
                  <span className="min-w-0 flex-1">{meta?.label ?? f}</span>
                  {meta?.hint ? <ParamHint text={meta.hint} /> : null}
                </label>
              )
            })}
          </div>
        </div>
        {resultJob?.state === "COMPLETED" ? (
          <>
            <p className="text-xs text-ok">Готово.</p>
            <div className="flex flex-wrap gap-2">
              {onOpenResult && (
                <Button size="sm" variant="outline" onClick={onOpenResult}>
                  Сравнить до/после
                </Button>
              )}
              {(resultJob.outputs && Object.keys(resultJob.outputs).length > 0
                ? Object.entries(resultJob.outputs)
                : resultJob.output_url
                  ? [["default", resultJob.output_url] as const]
                  : []
              ).map(([fmt, url]) => (
                <Button
                  key={fmt}
                  size="sm"
                  nativeButton={false}
                  render={<a href={url} download />}
                >
                  Скачать {FORMAT_META[fmt]?.label ?? (fmt === "default" ? "файл" : fmt)}
                </Button>
              ))}
            </div>
          </>
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

