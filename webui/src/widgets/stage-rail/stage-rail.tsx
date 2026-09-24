import { useEffect, useRef, useState } from "react"
import { tracksAreFullLength, useDetectRun } from "@features/detect-run"
import { useInpaintRun } from "@features/inpaint-run"
import { deletePreset, listPresets, savePreset, updatePreset, type Preset } from "@/entities/preset"
import {
  cancelJob,
  downloadJobOutput,
  outputDownloadName,
  packageJob,
  waitJobToCompletion,
  type Job,
} from "@/entities/job"
import type { Source } from "@/entities/source"
import { findCachedJob, useEventsOptional } from "@/shared/events"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import { Input } from "@/shared/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select"
import { Switch } from "@/shared/ui/switch"
import { Textarea } from "@/shared/ui/textarea"
import { ToggleGroup, ToggleGroupItem } from "@/shared/ui/toggle-group"
import { ExpertStations } from "./expert-stations"
import { FieldLabel, ParamHint, ParamSlider } from "./stage-parts"
import { FORMAT_META, MODE_HINTS } from "./param-meta"
import {
  applyBuiltInProfile,
  applyPreset,
  cleanBuiltinParams,
  divergentLabels,
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

const OTHER_FORMATS = ["webm", "mov", "mkv", "hls-fmp4", "hls-ts", "dash"] as const

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
  onResultJobChange,
}: {
  job: Job
  onOpenResult?: () => void
  onResultJobChange?: (job: Job) => void
}) {
  const [open, setOpen] = useState(false)
  const [convertFormats, setConvertFormats] = useState<string[]>([])
  const [webmCrf, setWebmCrf] = useState(32)
  const [segmentSeconds, setSegmentSeconds] = useState(6)
  const [packBusy, setPackBusy] = useState(false)
  const [packJobId, setPackJobId] = useState<string | null>(null)
  const [packProgress, setPackProgress] = useState({ fraction: 0, detail: "", eta: "" })
  const [packError, setPackError] = useState("")
  const [busyFmt, setBusyFmt] = useState<string | null>(null)
  const [dlError, setDlError] = useState("")

  const mp4Url = job.outputs?.mp4 || job.output_url || ""
  const needsWebm = convertFormats.includes("webm")
  const needsSegment = convertFormats.some((f) => f === "hls-fmp4" || f === "hls-ts" || f === "dash")
  const canPackage = job.can_package !== false
  const built = new Set(Object.keys(job.outputs ?? {}))

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
      await waitJobToCompletion(
        queued.id,
        (j) => setPackProgress({ fraction: j.fraction, detail: j.detail, eta: j.eta }),
        { seed: queued },
      )
      onResultJobChange?.(findCachedJob(job.id) ?? job)
    } catch (e) {
      setPackError(e instanceof Error ? e.message : String(e))
    } finally {
      setPackBusy(false)
      setPackJobId(null)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      {mp4Url && (
        <Button size="sm" className="w-fit" disabled={busyFmt === "mp4"} onClick={() => void onDownload("mp4", mp4Url)}>
          {busyFmt === "mp4" ? "…" : "Скачать MP4"}
        </Button>
      )}
      {onOpenResult && (
        <Button size="sm" variant="outline" className="w-fit" onClick={onOpenResult}>
          Сравнить до/после
        </Button>
      )}
      <div>
        <Button size="sm" variant="ghost" className="px-0" onClick={() => setOpen((v) => !v)}>
          {open ? "▾" : "▸"} Другие форматы
        </Button>
        {open && (
          <div className="mt-2 flex flex-col gap-2">
            {OTHER_FORMATS.map((f) => (
              <label key={f} className="flex items-center gap-2 text-xs">
                <Switch
                  size="sm"
                  checked={convertFormats.includes(f)}
                  disabled={packBusy || !canPackage}
                  onCheckedChange={(checked) => {
                    setConvertFormats((prev) => (checked ? [...prev, f] : prev.filter((x) => x !== f)))
                  }}
                />
                <span className="min-w-0 flex-1">
                  {FORMAT_META[f]?.label ?? f}
                  {built.has(f) ? <span className="ml-1 text-ok">· есть</span> : null}
                </span>
                {FORMAT_META[f]?.hint ? <ParamHint text={FORMAT_META[f].hint} /> : null}
              </label>
            ))}
            {needsWebm && (
              <ParamSlider
                label="WebM CRF"
                hint="Меньше — лучше качество и больше файл (VP9)."
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
                {packBusy ? "Конвертация…" : "Сконвертировать"}
              </Button>
              {packBusy && (
                <Button size="xs" variant="outline" disabled={!packJobId} onClick={() => packJobId && void cancelJob(packJobId)}>
                  Стоп
                </Button>
              )}
            </div>
            {packBusy && (
              <p className="text-xs text-muted-foreground">
                {[
                  `${Math.round(packProgress.fraction * 100)}%`,
                  packProgress.detail,
                  packProgress.eta ? `ETA ${packProgress.eta}` : "",
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </p>
            )}
            {packError && <p className="text-xs text-destructive">{packError}</p>}
          </div>
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
}: Props) {
  const events = useEventsOptional()
  const resolvedDevice = resolvedDeviceFrom(events)
  const [expert, setExpert] = useState(readExpert)
  const [useTracks, setUseTracks] = useState(true)
  const [bound, setBound] = useState<Bound>({ kind: "builtin", id: "balanced" })
  const [presets, setPresets] = useState<Preset[]>([])
  const [presetError, setPresetError] = useState("")
  const [saveName, setSaveName] = useState("")
  const [saveAs, setSaveAs] = useState(false)
  const [rename, setRename] = useState("")
  const [busy, setBusy] = useState(false)
  const [copied, setCopied] = useState(false)

  const [prevSourceId, setPrevSourceId] = useState(source?.id)
  if (prevSourceId !== source?.id) {
    setPrevSourceId(source?.id)
    setBound({ kind: "builtin", id: "balanced" })
    setUseTracks(true)
    setSaveName("")
    setSaveAs(false)
    setPresetError("")
    setRename("")
  }

  const boundId = bound.kind === "preset" ? bound.id : bound.id
  const [prevBound, setPrevBound] = useState(boundId + bound.kind)
  const boundKey = `${bound.kind}:${boundId}`
  if (prevBound !== boundKey) {
    setPrevBound(boundKey)
    setRename(bound.kind === "preset" ? bound.name : "")
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
  const badges = divergentLabels(params, badgeBase)
  const presetDirty = bound.kind === "preset" && !samePresetPayload(params, bound.payload, resolvedDevice)
  const showSave = bound.kind === "builtin" && badges.length > 0
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
    setSaveAs(false)
    setPresetError("")
    onParamsChange(applyBuiltInProfile(params, id, recipeDevice))
  }

  const applySaved = (preset: Preset) => {
    setBound({ kind: "preset", id: preset.id, name: preset.name, payload: preset.payload })
    setSaveName("")
    setSaveAs(false)
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
        setSaveAs(false)
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

  const renameCurrent = () => {
    if (bound.kind !== "preset" || busy) return
    const trimmed = rename.trim()
    if (!trimmed || trimmed === bound.name) return
    setBusy(true)
    setPresetError("")
    updatePreset(bound.id, { name: trimmed })
      .then((updated) => {
        setBound({
          kind: "preset",
          id: updated.id || bound.id,
          name: updated.name || trimmed,
          payload: updated.payload ?? bound.payload,
        })
        refreshPresets()
      })
      .catch((e: unknown) => setPresetError(e instanceof Error ? e.message : String(e)))
      .finally(() => setBusy(false))
  }

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

  return (
    <aside className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border/70 px-3 py-3">
        <h2 className="text-sm font-semibold">Конвейер</h2>
        <label className="flex items-center gap-2 text-xs">
          <span>Эксперт</span>
          <Switch
            checked={expert}
            onCheckedChange={(checked) => {
              setExpert(checked)
              try {
                localStorage.setItem(EXPERT_KEY, checked ? "1" : "0")
              } catch {
                // ignore private mode
              }
            }}
            aria-label="Эксперт"
          />
        </label>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <fieldset disabled={locked} className="flex flex-col gap-3 border-0 px-3 py-3 disabled:opacity-60">
          <div className="flex items-center gap-2">
            <FieldLabel className="w-[7.5rem] shrink-0" hint={scenarioHint}>
              Сценарий
            </FieldLabel>
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
              <SelectTrigger size="sm" className="flex-1">
                <SelectValue placeholder="Сценарий" />
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
          </div>

          {badges.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {badges.map((label) => (
                <Badge key={label} variant="secondary" className="text-[10px] font-normal">
                  {label}
                </Badge>
              ))}
            </div>
          )}

          {showSave && (
            <div className="flex items-center gap-1.5">
              <Input
                value={saveName}
                disabled={locked || busy}
                onChange={(e) => setSaveName(e.target.value)}
                className="h-7 flex-1 text-xs"
              />
              <Button size="sm" disabled={locked || busy || !saveName.trim()} onClick={() => saveNew(saveName)}>
                Сохранить
              </Button>
            </div>
          )}

          {bound.kind === "preset" && presetDirty && (
            <div className="flex flex-col gap-1.5">
              <div className="flex flex-wrap gap-1.5">
                <Button size="sm" disabled={locked || busy} onClick={updateCurrent}>
                  {`Обновить ${bound.name}`}
                </Button>
                <Button size="sm" variant="outline" disabled={locked || busy} onClick={() => setSaveAs((v) => !v)}>
                  Сохранить как…
                </Button>
              </div>
              {saveAs && (
                <div className="flex items-center gap-1.5">
                  <Input
                    value={saveName}
                    disabled={locked || busy}
                    onChange={(e) => setSaveName(e.target.value)}
                    className="h-7 flex-1 text-xs"
                  />
                  <Button size="sm" disabled={locked || busy || !saveName.trim()} onClick={() => saveNew(saveName)}>
                    Сохранить
                  </Button>
                </div>
              )}
            </div>
          )}

          {bound.kind === "preset" && (
            <div className="flex flex-wrap items-center gap-1.5">
              <Button size="sm" variant="ghost" disabled={locked || busy} onClick={removePreset}>
                Удалить
              </Button>
            </div>
          )}
          {presetError && <p className="text-xs text-destructive">{presetError}</p>}

          <Textarea
            value={params.prompt}
            disabled={locked || noSource}
            onChange={(e) => onParamsChange({ ...params, prompt: e.target.value })}
            placeholder="Например: логотип в правом верхнем углу"
            className="min-h-20 text-xs"
          />

          {strokes && (
            <div className="flex items-center gap-2">
              <FieldLabel hint={params.maskPolicy === "static" ? MODE_HINTS.maskStatic : MODE_HINTS.maskPropagate}>
                Мазки
              </FieldLabel>
              <ToggleGroup
                variant="outline"
                size="sm"
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

          {tracksFull && (
            <div className="flex items-center gap-2">
              <FieldLabel className="flex-1">Удалять по найденным рамкам</FieldLabel>
              <Switch checked={useTracks} disabled={locked} onCheckedChange={setUseTracks} />
            </div>
          )}

          {queries.length > 0 && (
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
              onOpenConfig={onOpenConfig}
            />
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
              <div className="flex items-center gap-1.5">
                <Input
                  value={rename}
                  disabled={locked || busy}
                  onChange={(e) => setRename(e.target.value)}
                  className="h-7 flex-1 text-xs"
                />
                <Button
                  size="sm"
                  variant="outline"
                  disabled={locked || busy || !rename.trim() || rename.trim() === bound.name}
                  onClick={renameCurrent}
                >
                  Переименовать
                </Button>
              </div>
            </div>
          )}

          <Button size="sm" disabled={!canRemove} onClick={runRemove}>
            Убрать
          </Button>
        </fieldset>

        {(statusText || stopId || failedError || inpaint.error || detect.error) && (
          <div className="flex flex-col gap-1.5 px-3 pb-3">
            {statusText && <p className="text-xs text-muted-foreground">{statusText}</p>}
            {!locked && failedError && <p className="text-xs text-destructive">{failedError}</p>}
            {inpaint.error && <p className="text-xs text-destructive">{inpaint.error}</p>}
            {detect.error && <p className="text-xs text-destructive">{detect.error}</p>}
            {stopId && (locked || inpaint.running) && (
              <Button size="xs" variant="outline" className="w-fit" onClick={() => void cancelJob(stopId)}>
                Стоп
              </Button>
            )}
          </div>
        )}

        {resultJob?.state === "COMPLETED" && (
          <div className="border-t border-border/60 px-3 py-3">
            <ResultPanel job={resultJob} onOpenResult={onOpenResult} onResultJobChange={onResultJobChange} />
          </div>
        )}
      </div>
    </aside>
  )
}
