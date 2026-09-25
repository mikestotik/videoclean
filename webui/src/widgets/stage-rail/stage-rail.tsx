import { useEffect, useRef, useState } from "react"
import { tracksAreFullLength, useDetectRun } from "@features/detect-run"
import { useInterpret } from "@features/interpret"
import { useInpaintRun } from "@features/inpaint-run"
import { deletePreset, listPresets, savePreset, updatePreset, type Preset } from "@/entities/preset"
import {
  cancelJob,
  downloadJobOutput,
  outputDownloadName,
  type Job,
} from "@/entities/job"
import type { Source } from "@/entities/source"
import { useEventsOptional } from "@/shared/events"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/dialog"
import { Input } from "@/shared/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select"
import { Switch } from "@/shared/ui/switch"
import { Textarea } from "@/shared/ui/textarea"
import { ToggleGroup, ToggleGroupItem } from "@/shared/ui/toggle-group"
import { ExpertStations } from "./expert-stations"
import { OutputFormatBlock } from "./output-format"
import { RunReport } from "./run-report"
import { FieldLabel, StackedField } from "./stage-parts"
import { MODE_HINTS, paramAnchor } from "./param-meta"
import {
  applyBuiltInProfile,
  applyPreset,
  cleanBuiltinParams,
  divergentEntries,
  effectiveRecipeDevice,
  enabledTargets,
  explicitFields,
  hasEnabledManualTarget,
  inferScenario,
  pipelineBaselineParams,
  presetPayload,
  samePresetPayload,
  structuralDivergence,
  toDetectParams,
  type BuiltinProfileId,
  type EditorParams,
  type Scenario,
} from "./params"
import { chooseRun, pendingRunStatus } from "./run-choice"

const EXPERT_KEY = "videoclean.expert"

const SCENARIOS: { id: BuiltinProfileId; label: string; hint: string }[] = [
  { id: "fast", label: "Быстро", hint: "LaMa, без проверки, 6 ключевых кадров." },
  { id: "balanced", label: "Баланс", hint: "LaMa, одна проверка остатка, 10 ключевых кадров." },
  { id: "quality", label: "Качество", hint: "Больше ключевых кадров, на CUDA sam2-video и ProPainter. Дырка заливается кропом до потолка памяти, не целым кадром. До 2 проходов проверки остатка." },
]

type Bound =
  | { kind: "builtin"; id: BuiltinProfileId }
  | { kind: "preset"; id: string; name: string; payload: Record<string, unknown> }

type Props = {
  source: Source | null
  frameCount: number
  params: EditorParams
  onParamsChange: (p: EditorParams) => void
  detect: ReturnType<typeof useDetectRun>
  inpaint: ReturnType<typeof useInpaintRun>
  resultJob: Job | null
  masks?: number[]
  failedError?: string
  onClearFailed?: () => void
  onOpenConfig?: () => void
  onOpenResult?: () => void
  onResultJobChange?: (job: Job) => void
  phrase: ReturnType<typeof useInterpret>
}

function readExpert(): boolean {
  try {
    return localStorage.getItem(EXPERT_KEY) === "1"
  } catch {
    return false
  }
}

function resolvedDeviceFrom(events: ReturnType<typeof useEventsOptional>): string {
  const snap = events?.snapshot
  const fromOptions = snap?.options && typeof snap.options.device === "string" ? snap.options.device : ""
  const top = typeof snap?.device === "string" ? snap.device : ""
  const raw = (fromOptions || top).trim().toLowerCase()
  if (raw === "cuda" || raw === "mps" || raw === "cpu") return raw
  return "cpu"
}

function formatJobStatus(job: {
  stageTitle?: string
  stage?: string
  detail?: string
  fraction?: number
  eta?: string
}): string {
  const title = (job.stageTitle ?? "").trim() || (job.stage ?? "").trim()
  const detail = (job.detail ?? "").trim()
  const eta = (job.eta ?? "").trim()
  const fraction = job.fraction ?? 0
  if (!title && !detail && !eta && fraction <= 0) return ""
  return [title, detail, `${Math.round(fraction * 100)}%`, eta ? `ETA ${eta}` : ""].filter(Boolean).join(" · ")
}

function ResultPanel({
  job,
  onOpenResult,
}: {
  job: Job
  onOpenResult?: () => void
}) {
  const [busy, setBusy] = useState(false)
  const [dlError, setDlError] = useState("")
  const mp4Url = job.outputs?.mp4 || job.output_url || ""

  return (
    <div className="flex flex-col gap-2">
      <RunReport key={job.id} jobId={job.id} />
      <div className="flex flex-wrap gap-2">
        {mp4Url && (
          <Button
            size="sm"
            variant="outline"
            disabled={busy}
            onClick={() => {
              setDlError("")
              setBusy(true)
              void downloadJobOutput(mp4Url, outputDownloadName("mp4", job.id))
                .catch((e) => setDlError(e instanceof Error ? e.message : String(e)))
                .finally(() => setBusy(false))
            }}
          >
            {busy ? "…" : "Скачать MP4"}
          </Button>
        )}
        {onOpenResult && (
          <Button size="sm" variant="outline" onClick={onOpenResult}>
            Сравнить до/после
          </Button>
        )}
      </div>
      {dlError && <p className="text-xs text-destructive">{dlError}</p>}
    </div>
  )
}

export function StageRail({
  source,
  frameCount,
  params,
  onParamsChange,
  detect,
  inpaint,
  resultJob,
  masks,
  failedError,
  onClearFailed,
  onOpenConfig,
  onOpenResult,
  onResultJobChange,
  phrase,
}: Props) {
  const events = useEventsOptional()
  const resolvedDevice = resolvedDeviceFrom(events)
  const [expert, setExpert] = useState(readExpert)
  const [focus, setFocus] = useState<{ key: string; nonce: number } | null>(null)
  const railRef = useRef<HTMLElement>(null)
  const [useTracks, setUseTracks] = useState(true)
  const [bound, setBound] = useState<Bound>({ kind: "builtin", id: "balanced" })
  const [presets, setPresets] = useState<Preset[]>([])
  const [presetError, setPresetError] = useState("")
  const [saveName, setSaveName] = useState("")
  const [saveOpen, setSaveOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState(false)

  const [prevSourceId, setPrevSourceId] = useState(source?.id)
  if (prevSourceId !== source?.id) {
    setPrevSourceId(source?.id)
    setBound({ kind: "builtin", id: "balanced" })
    setUseTracks(true)
    setSaveName("")
    setSaveOpen(false)
    setPresetError("")
  }

  const boundId = bound.kind === "preset" ? bound.id : bound.id
  const [prevBound, setPrevBound] = useState(boundId + bound.kind)
  const boundKey = `${bound.kind}:${boundId}`
  if (prevBound !== boundKey) {
    setPrevBound(boundKey)
    setCopied(false)
  }

  const deviceRef = useRef(resolvedDevice)
  useEffect(() => {
    const prev = deviceRef.current
    if (prev === resolvedDevice) return
    deviceRef.current = resolvedDevice
    if (bound.kind !== "builtin") return
    if (structuralDivergence(params, prev)) return
    onParamsChange(applyBuiltInProfile(params, bound.id, resolvedDevice))
  }, [bound, onParamsChange, params, resolvedDevice])

  useEffect(() => {
    let alive = true
    listPresets()
      .then((rows) => {
        if (alive) setPresets(rows)
      })
      .catch(() => {
        if (alive) setPresets([])
      })
    return () => {
      alive = false
    }
  }, [source?.id])

  const noSource = !source
  const recipeDevice = effectiveRecipeDevice(params, resolvedDevice)
  const scenario: Scenario =
    bound.kind === "preset"
      ? { kind: "preset", id: bound.id, payload: bound.payload, resolvedDevice }
      : inferScenario(params, resolvedDevice)
  const unnamed = bound.kind === "builtin" && structuralDivergence(params, resolvedDevice)
  const selectValue = unnamed ? "custom" : bound.kind === "preset" ? bound.id : bound.id
  const badgeBase =
    bound.kind === "preset"
      ? applyPreset(pipelineBaselineParams(), bound.payload, resolvedDevice)
      : params.run.profile === "fast" || params.run.profile === "balanced" || params.run.profile === "quality"
        ? cleanBuiltinParams(params.run.profile, recipeDevice)
        : pipelineBaselineParams()
  const listedLlm = events?.snapshot?.options?.llm_model_options
  const autoLlm =
    listedLlm?.find((m) => m.ready && m.model)?.model || events?.snapshot?.ollama?.models?.[0] || ""
  const detectorModels = events?.snapshot?.options?.detector_models as
    | { model_ref?: string; ready?: boolean; backend?: string }[]
    | undefined
  const autoDetector = detectorModels?.find((m) => m.ready !== false && m.model_ref)
  const segmenterModels = events?.snapshot?.options?.segmenter_models as
    | { model_ref?: string; ready?: boolean }[]
    | undefined
  const autoSegmenter = segmenterModels?.find((m) => m.ready && m.model_ref)
  const badges = divergentEntries(params, badgeBase).filter((row) => {
    if (row.key === "llm_base_url") return false
    if (row.key === "llm_model" && !badgeBase.run.llm_model && params.run.llm_model === autoLlm) return false
    if (
      row.key === "detector_model" &&
      !badgeBase.run.detector_model &&
      params.run.detector_model === autoDetector?.model_ref
    ) {
      return false
    }
    if (
      row.key === "detector" &&
      !badgeBase.run.detector &&
      autoDetector?.backend &&
      params.run.detector === autoDetector.backend
    ) {
      return false
    }
    if (
      row.key === "segmenter_model" &&
      !badgeBase.run.segmenter_model &&
      params.run.segmenter_model === autoSegmenter?.model_ref
    ) {
      return false
    }
    return true
  })
  const presetDirty = bound.kind === "preset" && !samePresetPayload(params, bound.payload, resolvedDevice)
  const scenarioHint = SCENARIOS.find((s) => s.id === (bound.kind === "builtin" ? bound.id : params.run.profile))?.hint

  const liveRun = (events?.jobs ?? []).find(
    (j) =>
      j.kind === "run" &&
      j.source_id === source?.id &&
      (j.state === "QUEUED" || j.state === "RUNNING"),
  )
  const locked = Boolean(liveRun) || inpaint.running
  const strokes = (masks?.length ?? 0) > 0
  const tracksFull = tracksAreFullLength(detect.enabledTracks, frameCount)
  const choice = chooseRun({
    strokes,
    tracksFull,
    useTracks,
    manualTargets: hasEnabledManualTarget(params),
    maskPolicy: params.maskPolicy,
  })
  const liveStatus = liveRun ? formatJobStatus(liveRun) : ""
  const hookStatus = formatJobStatus({
    stage: inpaint.progress.stage,
    stageTitle: inpaint.progress.stageTitle,
    detail: inpaint.progress.detail,
    fraction: inpaint.progress.fraction,
    eta: inpaint.progress.eta,
  })
  const statusText = locked ? liveStatus || hookStatus || pendingRunStatus(choice) : ""
  const stopId = liveRun?.id || inpaint.jobId

  const queries = params.targets
    .map((t, index) => ({ t, index }))
    .filter(({ t }) => t.query.trim().length > 0)

  const canRemove =
    !noSource &&
    !locked &&
    (choice.path === "masks" ||
      choice.path === "tracks" ||
      choice.path === "targets" ||
      params.prompt.trim().length > 0)

  const canFind =
    !noSource &&
    !locked &&
    !detect.running &&
    (enabledTargets(params).length > 0 || params.prompt.trim().length > 0)

  const refreshPresets = () => {
    listPresets()
      .then(setPresets)
      .catch(() => setPresets([]))
  }

  const applyBuiltin = (id: BuiltinProfileId) => {
    setBound({ kind: "builtin", id })
    setSaveName("")
    setSaveOpen(false)
    setPresetError("")
    onParamsChange(applyBuiltInProfile(params, id, recipeDevice))
  }

  const applySaved = (preset: Preset) => {
    setBound({ kind: "preset", id: preset.id, name: preset.name, payload: preset.payload })
    setSaveName("")
    setSaveOpen(false)
    setPresetError("")
    onParamsChange(applyPreset(params, preset.payload, resolvedDevice))
  }

  const runRemove = () => {
    if (!canRemove) return
    onClearFailed?.()
    const fields = explicitFields(params, scenario)
    if (choice.path === "masks" && choice.maskPolicy) fields.mask_policy = choice.maskPolicy
    const targets = enabledTargets(params)
    if (choice.path === "masks") {
      void inpaint.run(
        { mode: "masks", masks: masks ?? [], prompt: params.prompt },
        fields,
      )
      return
    }
    if (choice.path === "tracks") {
      void inpaint.run({ mode: "tracks", tracks: detect.enabledTracks }, fields)
      return
    }
    if (choice.path === "targets") {
      void inpaint.run({ mode: "prompt", prompt: params.prompt, targets }, fields)
      return
    }
    void inpaint.run({ mode: "prompt", prompt: params.prompt }, fields)
  }

  const parsePhrase = async () => {
    const prompt = params.prompt.trim()
    if (!prompt || !params.run.llm_model) return false
    const jobParams: Record<string, string | number | boolean> = { llm_model: params.run.llm_model }
    if (params.run.llm_base_url) jobParams.llm_base_url = params.run.llm_base_url
    for (const key of ["prompt_frame_stride", "prompt_frame_max", "vision_batch", "parse_chunk_frames"] as const) {
      const value = params.advanced[key]
      if (value) jobParams[key] = value
    }
    const result = await phrase.run(prompt, jobParams)
    if (!result) return false
    const kinds = new Set(["watermark", "text_overlay", "object"])
    onParamsChange({
      ...params,
      parsedPrompt: result.prompt,
      targets: result.targets.map((t) => ({
        kind: kinds.has(t.kind) ? (t.kind as "watermark" | "text_overlay" | "object") : "object",
        query: t.query,
        where: t.where,
        enabled: true,
        source: "auto" as const,
      })),
    })
    return true
  }

  const findMasks = () => {
    const targets = enabledTargets(params)
    void detect.run({
      mode: targets.length > 0 ? "detect" : "parse",
      prompt: params.prompt,
      targets,
      stride: params.detect.stride,
      params: toDetectParams(params, scenario),
    })
  }

  const saveNew = (name: string) => {
    const trimmed = name.trim()
    if (!trimmed || busy) return
    setBusy(true)
    setPresetError("")
    savePreset(trimmed, presetPayload(params, resolvedDevice))
      .then((created) => {
        setSaveName("")
        setSaveOpen(false)
        setBound({ kind: "preset", id: created.id, name: created.name, payload: created.payload })
        refreshPresets()
      })
      .catch((e: unknown) => setPresetError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false))
  }

  const updateCurrent = () => {
    if (bound.kind !== "preset" || busy) return
    setBusy(true)
    setPresetError("")
    const payload = presetPayload(params, resolvedDevice)
    updatePreset(bound.id, { payload })
      .then((updated) => {
        setBound({ kind: "preset", id: updated.id, name: updated.name, payload: updated.payload ?? payload })
        refreshPresets()
      })
      .catch((e: unknown) => setPresetError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false))
  }

  const persistExpert = (checked: boolean) => {
    setExpert(checked)
    try {
      localStorage.setItem(EXPERT_KEY, checked ? "1" : "0")
    } catch {
      // ignore private mode
    }
  }

  const revealParam = (key: string) => {
    if (!expert) persistExpert(true)
    setFocus((prev) => ({ key, nonce: (prev?.nonce ?? 0) + 1 }))
  }

  useEffect(() => {
    if (!focus) return
    const anchor = paramAnchor(focus.key)
    let frame = 0
    let tries = 0
    let highlight: HTMLElement | null = null
    const tick = () => {
      const el = railRef.current?.querySelector(`[data-param="${anchor}"]`)
      if (el instanceof HTMLElement) {
        el.scrollIntoView({ block: "nearest" })
        el.classList.add("rounded-md", "ring-2", "ring-ring")
        highlight = el
        return
      }
      if (tries++ < 10) frame = window.requestAnimationFrame(tick)
    }
    frame = window.requestAnimationFrame(tick)
    const clear = window.setTimeout(() => {
      highlight?.classList.remove("rounded-md", "ring-2", "ring-ring")
    }, 1500)
    return () => {
      window.cancelAnimationFrame(frame)
      window.clearTimeout(clear)
      highlight?.classList.remove("rounded-md", "ring-2", "ring-ring")
    }
  }, [expert, focus])

  const removePreset = () => {
    if (bound.kind !== "preset") return
    if (!window.confirm(`Удалить «${bound.name}» и связанные данные?`)) return
    deletePreset(bound.id)
      .then(() => {
        setBound({ kind: "builtin", id: "balanced" })
        onParamsChange(applyBuiltInProfile(params, "balanced", recipeDevice))
        refreshPresets()
      })
      .catch((e: unknown) => setPresetError(e instanceof Error ? e.message : String(e)))
  }

  const statusBlock = statusText || stopId || failedError || inpaint.error || detect.error

  return (
    <aside ref={railRef} className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border/70 px-3 py-3">
        <h2 className="text-sm font-semibold">Конвейер</h2>
        <label className="flex items-center gap-2 text-xs">
          <span>Эксперт</span>
          <Switch checked={expert} onCheckedChange={persistExpert} aria-label="Эксперт" />
        </label>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <fieldset className={`flex flex-col gap-3 border-0 px-3 py-3 ${locked ? "opacity-60" : ""}`}>
          <StackedField label="Сценарий" hint={scenarioHint}>
            <Select
              value={selectValue}
              disabled={locked || noSource}
              onValueChange={(v) => {
                if (!v || v === "custom") return
                if (v === "fast" || v === "balanced" || v === "quality") {
                  applyBuiltin(v)
                  return
                }
                const preset = presets.find((p) => p.id === v)
                if (preset) applySaved(preset)
              }}
            >
              <SelectTrigger size="sm" className="w-full">
                <SelectValue placeholder="Сценарий">
                  {(value: string | null) => {
                    if (!value) return "Сценарий"
                    const scenario = SCENARIOS.find((s) => s.id === value)
                    if (scenario) return scenario.label
                    if (value === "custom") return "Без имени"
                    return presets.find((p) => p.id === value)?.name ?? value
                  }}
                </SelectValue>
              </SelectTrigger>
              <SelectContent>
                {SCENARIOS.map((s) => (
                  <SelectItem key={s.id} value={s.id}>
                    {s.label}
                  </SelectItem>
                ))}
                {presets.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.name}
                  </SelectItem>
                ))}
                {unnamed && <SelectItem value="custom">Без имени</SelectItem>}
              </SelectContent>
            </Select>
          </StackedField>

          {badges.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {badges.map((row) => (
                <Badge
                  key={row.key}
                  variant="secondary"
                  className="cursor-pointer text-[10px] font-normal hover:bg-muted"
                  render={<button type="button" />}
                  onClick={() => revealParam(row.key)}
                >
                  {row.label}
                </Badge>
              ))}
            </div>
          )}

          <div className="flex flex-wrap gap-1.5">
            <Button
              size="sm"
              variant="outline"
              disabled={locked || busy || noSource}
              onClick={() => {
                setSaveName("")
                setSaveOpen(true)
              }}
            >
              Сохранить как пресет
            </Button>
            {bound.kind === "preset" && presetDirty && (
              <Button size="sm" disabled={locked || busy} onClick={updateCurrent}>
                {`Обновить ${bound.name}`}
              </Button>
            )}
            {bound.kind === "preset" && (
              <Button size="sm" variant="ghost" disabled={locked || busy} onClick={removePreset}>
                Удалить
              </Button>
            )}
          </div>
          {presetError && <p className="text-xs text-destructive">{presetError}</p>}

          <Textarea
            value={params.prompt}
            disabled={locked || noSource}
            onChange={(e) => onParamsChange({ ...params, prompt: e.target.value })}
            placeholder="Например: логотип в правом верхнем углу"
            className="min-h-20 text-xs"
          />

          {strokes && (
            <div data-param="mask_policy" className="flex flex-col gap-1.5">
              <FieldLabel hint={params.maskPolicy === "static" ? MODE_HINTS.maskStatic : MODE_HINTS.maskPropagate}>
                Мазки
              </FieldLabel>
              <ToggleGroup
                variant="outline"
                size="sm"
                className="w-full flex-wrap"
                value={[params.maskPolicy]}
                disabled={locked}
                onValueChange={(v) => {
                  const next = v.at(-1)
                  if (next === "static" || next === "propagate") onParamsChange({ ...params, maskPolicy: next })
                }}
              >
                <ToggleGroupItem value="static">на весь ролик</ToggleGroupItem>
                <ToggleGroupItem value="propagate">по движению</ToggleGroupItem>
              </ToggleGroup>
            </div>
          )}

          {!expert && tracksFull && (
            <div className="flex items-center justify-between gap-3">
              <FieldLabel className="min-w-0 flex-1">Удалять по найденным рамкам</FieldLabel>
              <Switch checked={useTracks} disabled={locked} onCheckedChange={setUseTracks} />
            </div>
          )}

          {!expert && queries.length > 0 && (
            <div className="flex flex-col gap-1.5">
              <FieldLabel>Что нашлось</FieldLabel>
              {queries.map(({ t, index }) => (
                <Input
                  key={index}
                  value={t.query}
                  disabled={locked}
                  className="h-7 text-xs"
                  onChange={(e) => {
                    const query = e.target.value
                    onParamsChange({
                      ...params,
                      targets: params.targets.map((row, i) =>
                        i === index ? { ...row, query, source: "manual" } : row,
                      ),
                    })
                  }}
                />
              ))}
            </div>
          )}

          {expert && (
            <ExpertStations
              params={params}
              disabled={locked || noSource}
              onChange={onParamsChange}
              findDisabled={!canFind}
              onFind={findMasks}
              detectRunning={detect.running}
              detectJobId={detect.jobId}
              detectProgress={detect.progress}
              detectError={detect.error}
              foundTracks={detect.tracks.length}
              tracksFull={tracksFull}
              useTracks={useTracks}
              onUseTracks={setUseTracks}
              onOpenConfig={onOpenConfig}
              focus={focus}
              parseRunning={phrase.running}
              parseError={phrase.error}
              parseDisabled={!params.prompt.trim() || !params.run.llm_model}
              onParse={parsePhrase}
              resultJob={resultJob}
              onResultJobChange={onResultJobChange}
            />
          )}

          {!expert && (
            <div className="flex flex-col gap-1.5">
              <FieldLabel>Формат</FieldLabel>
              <OutputFormatBlock
                params={params}
                onChange={onParamsChange}
                disabled={locked || noSource}
                resultJob={resultJob}
                onResultJobChange={onResultJobChange}
                showWebm={params.run.formats.includes("webm")}
                showSegment={params.run.formats.some((f) => f === "hls-fmp4" || f === "hls-ts" || f === "dash")}
              />
            </div>
          )}

          {expert && bound.kind === "preset" && (
            <div className="flex flex-col gap-1.5 border-t border-border/60 pt-3">
              <div className="flex items-center gap-1.5">
                <code className="min-w-0 flex-1 truncate font-mono text-[11px]">{bound.id}</code>
                <Button
                  size="xs"
                  variant="outline"
                  onClick={() => {
                    void navigator.clipboard.writeText(bound.id).then(() => setCopied(true)).catch(() => {})
                  }}
                >
                  {copied ? "Скопировано" : "Копировать"}
                </Button>
              </div>
            </div>
          )}
        </fieldset>
      </div>

      <Dialog open={saveOpen} onOpenChange={setSaveOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Сохранить как пресет</DialogTitle>
          </DialogHeader>
          <Input
            value={saveName}
            disabled={busy}
            autoFocus
            placeholder="Имя пресета"
            onChange={(e) => setSaveName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && saveName.trim()) saveNew(saveName)
            }}
          />
          <DialogFooter>
            <Button size="sm" disabled={busy || !saveName.trim()} onClick={() => saveNew(saveName)}>
              Сохранить
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <div className="shrink-0 border-t border-border/70 bg-card/70">
        {(statusBlock || resultJob?.state === "COMPLETED") && (
          <div className="flex max-h-[40vh] flex-col gap-2 overflow-y-auto px-3 pt-3">
            {statusText && <p className="text-xs text-muted-foreground">{statusText}</p>}
            {!locked && failedError && <p className="text-xs text-destructive">{failedError}</p>}
            {inpaint.error && <p className="text-xs text-destructive">{inpaint.error}</p>}
            {detect.error && <p className="text-xs text-destructive">{detect.error}</p>}
            {resultJob?.state === "COMPLETED" && (
              <ResultPanel job={resultJob} onOpenResult={onOpenResult} />
            )}
          </div>
        )}
        <div className="px-3 py-3">
          <Button
            size="sm"
            className="w-full"
            variant={locked ? "outline" : "default"}
            disabled={locked ? !stopId : !canRemove}
            onClick={() => {
              if (locked && stopId) void cancelJob(stopId)
              else runRemove()
            }}
          >
            {locked ? "Отменить" : "Обработать"}
          </Button>
        </div>
      </div>
    </aside>
  )
}
