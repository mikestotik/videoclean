import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Film, Upload } from "lucide-react"
import { EditorViewer, type EditorMode } from "@widgets/editor-viewer"
import { Library } from "@widgets/library"
import {
  DEFAULT_PARAMS,
  StageRail,
  enabledTargets,
  toRunParams,
  type EditorParams,
  type StageTarget,
} from "@widgets/stage-rail"
import { Timeline } from "@widgets/timeline"
import { useAnnotate } from "@features/annotate"
import { tracksAreFullLength, useDetectRun } from "@features/detect-run"
import { useInpaintRun } from "@features/inpaint-run"
import { useInterpret, type InterpretTarget } from "@features/interpret"
import { fetchJobReport, getJob, type Job } from "@/entities/job"
import { previewArtifactUrl } from "@/entities/preview"
import type { TargetKind, TargetRow } from "@/entities/targets"
import { getSource, type Source } from "@/entities/source"
import { formatTimecode } from "@/shared/lib/format"
import { Timecode } from "@/shared/ui/timecode"
import { ToggleGroup, ToggleGroupItem } from "@/shared/ui/toggle-group"
import { cn } from "@/shared/lib/utils"

const MODE_LABEL = { annotate: "Разметка", detect: "Маски", result: "Результат" } as const

function toTargetRows(targets: InterpretTarget[]): TargetRow[] {
  return targets
    .filter((t) => t.query.trim())
    .map((t) => ({
      kind: t.kind === "watermark" || t.kind === "text_overlay" ? (t.kind as TargetKind) : "object",
      query: t.query,
      where: t.where,
    }))
}

function stageTargetsFromReport(value: unknown): StageTarget[] {
  if (!Array.isArray(value)) return []
  const rows: StageTarget[] = []
  for (const item of value) {
    const r = (item ?? {}) as Record<string, unknown>
    const query = String(r.query ?? "").trim()
    if (!query) continue
    const kindRaw = String(r.kind ?? "object")
    rows.push({
      kind: kindRaw === "watermark" || kindRaw === "text_overlay" ? kindRaw : "object",
      query,
      where: r.where ? String(r.where) : null,
      enabled: true,
      source: "auto",
    })
  }
  return rows
}

function applyReportToParams(
  prev: EditorParams,
  report: { prompt?: unknown; targets?: unknown },
): EditorParams {
  const targets = stageTargetsFromReport(report.targets)
  const prompt = String(report.prompt ?? "").trim()
  return {
    ...prev,
    prompt: prompt || prev.prompt,
    targets: targets.length > 0 ? targets : prev.targets,
  }
}

type Props = {
  routeSourceId: string | null
  onRouteSourceIdChange: (id: string | null, replace?: boolean) => void
}

export function WorkspacePage({ routeSourceId, onRouteSourceIdChange }: Props) {
  const [source, setSourceState] = useState<Source | null>(null)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [viewerMode, setViewerMode] = useState<EditorMode>("annotate")
  const [params, setParams] = useState<EditorParams>(DEFAULT_PARAMS)
  const [maskOpacity, setMaskOpacity] = useState(0.6)
  const [runAllBusy, setRunAllBusy] = useState(false)
  const [runAllError, setRunAllError] = useState("")
  const [resultJob, setResultJob] = useState<Job | null>(null)
  const [libraryTick, setLibraryTick] = useState(0)
  const [restoreError, setRestoreError] = useState("")

  const sourceId = source?.id ?? null
  const probe = source?.probe
  const videoSize = useMemo(
    () => (source ? { width: source.probe.width, height: source.probe.height } : null),
    [source],
  )
  const annotate = useAnnotate(source, videoSize)
  const interpret = useInterpret(source)
  const detect = useDetectRun(source)
  const inpaint = useInpaintRun(source)

  const loadedSourceIdRef = useRef<string | null>(null)
  loadedSourceIdRef.current = source?.id ?? null

  const setSource = useCallback(
    (next: Source | null) => {
      setSourceState(next)
      onRouteSourceIdChange(next?.id ?? null)
    },
    [onRouteSourceIdChange],
  )

  useEffect(() => {
    let cancelled = false
    if (!routeSourceId) {
      setSourceState(null)
      return
    }
    if (loadedSourceIdRef.current === routeSourceId) return
    void getSource(routeSourceId)
      .then((row) => {
        if (!cancelled) setSourceState(row)
      })
      .catch(() => {
        if (cancelled) return
        setSourceState(null)
        onRouteSourceIdChange(null, true)
      })
    return () => {
      cancelled = true
    }
  }, [routeSourceId, onRouteSourceIdChange])

  const activeJobs = useMemo(
    () => ({
      prompt: interpret.lastJobId,
      preview: detect.lastJobId,
      run: inpaint.lastJobId,
    }),
    [detect.lastJobId, inpaint.lastJobId, interpret.lastJobId],
  )

  const [prevSourceId, setPrevSourceId] = useState<string | null>(sourceId)
  if (prevSourceId !== sourceId) {
    setPrevSourceId(sourceId)
    setCurrentFrame(0)
    setViewerMode("annotate")
    setParams(DEFAULT_PARAMS)
    setMaskOpacity(0.6)
    setResultJob(null)
    setRestoreError("")
    setRunAllError("")
  }

  const selectJob = useCallback(
    async (job: Job) => {
      if (job.state !== "COMPLETED") return
      setRestoreError("")
      try {
        if (job.kind === "prompt") {
          const loaded = await interpret.loadFromJob(job.id)
          if (!loaded) {
            setRestoreError("Не удалось загрузить результат промпта")
            return
          }
          setParams((prev) => applyReportToParams(prev, loaded))
          setViewerMode("annotate")
          return
        }
        if (job.kind === "preview") {
          const report = (await fetchJobReport(job.id).catch(() => null)) as
            | { prompt?: unknown; targets?: unknown }
            | null
          if (report) setParams((prev) => applyReportToParams(prev, report))
          await detect.loadFromJob(job.id)
          setViewerMode("detect")
          return
        }
        if (job.kind === "run") {
          const report = (await fetchJobReport(job.id).catch(() => null)) as
            | { prompt?: unknown; targets?: unknown }
            | null
          if (report) setParams((prev) => applyReportToParams(prev, report))
          inpaint.loadFromJob(job.id)
          const full = await getJob(job.id)
          setResultJob(full)
          setViewerMode("result")
        }
      } catch (e) {
        setRestoreError(e instanceof Error ? e.message : String(e))
      }
    },
    [detect, inpaint, interpret],
  )

  const [prevDetectJob, setPrevDetectJob] = useState<string | null>(null)
  if (detect.lastJobId && detect.lastJobId !== prevDetectJob) {
    setPrevDetectJob(detect.lastJobId)
    if (viewerMode === "annotate") setViewerMode("detect")
  }

  const masks = useMemo(
    () =>
      Object.keys(annotate.strokesByFrame)
        .map(Number)
        .filter((f) => (annotate.strokesByFrame[f] ?? []).length > 0)
        .sort((a, b) => a - b),
    [annotate.strokesByFrame],
  )
  const annotatedFrames = useMemo(
    () => Object.keys(annotate.strokesByFrame).map(Number),
    [annotate.strokesByFrame],
  )
  const maskedFrames = useMemo(
    () => (detect.manifest?.frames ?? []).map((f) => f.index),
    [detect.manifest],
  )

  const detectFrame = useMemo(
    () => detect.manifest?.frames.find((f) => f.index === currentFrame) ?? null,
    [detect.manifest, currentFrame],
  )
  const detectMaskUrl =
    detect.lastJobId && detectFrame
      ? previewArtifactUrl(detect.lastJobId, detectFrame.artifacts.mask)
      : null

  const overlayTracks = useMemo(() => {
    const n = probe?.frame_count ?? 0
    const previewFrames = detect.manifest?.frames ?? []
    return detect.tracks.map((t) => {
      let box = null as (typeof t.boxes)[number]
      if (n > 0 && t.boxes.length === n) box = t.boxes[currentFrame] ?? null
      else {
        const i = previewFrames.findIndex((f) => f.index === currentFrame)
        if (i >= 0) box = t.boxes[i] ?? null
      }
      return {
        id: t.id,
        label: t.label,
        box,
        enabled: !detect.excludedIds.includes(t.id),
      }
    })
  }, [detect.tracks, detect.excludedIds, detect.manifest, currentFrame, probe?.frame_count])

  const selectedTrack = useMemo(
    () => detect.tracks.find((t) => t.id === detect.selectedTrackId) ?? null,
    [detect.tracks, detect.selectedTrackId],
  )

  /** Map box-array key indices → video frame numbers for the timeline. */
  const keyframeFrames = useMemo(() => {
    if (!selectedTrack || !probe) return []
    const n = probe.frame_count
    if (selectedTrack.boxes.length === n) return selectedTrack.keyframes
    const previewFrames = detect.manifest?.frames ?? []
    return selectedTrack.keyframes
      .map((i) => previewFrames[i]?.index)
      .filter((f): f is number => typeof f === "number")
  }, [selectedTrack, probe, detect.manifest])

  const resultJobId = inpaint.lastJobId
  if (resultJob && resultJob.id !== resultJobId) setResultJob(null)
  useEffect(() => {
    if (!resultJobId) return
    let alive = true
    getJob(resultJobId)
      .then((j) => {
        if (!alive) return
        if (j.state === "COMPLETED") {
          setResultJob(j)
          setViewerMode("result")
        }
      })
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [resultJobId])

  const runAll = async () => {
    if (!source || runAllBusy) return
    const prompt = params.prompt.trim()
    setRunAllError("")
    setRunAllBusy(true)
    try {
      if (masks.length > 0 && !prompt && enabledTargets(params).length === 0) {
        await inpaint.run({ mode: "masks", masks }, toRunParams(params))
        return
      }

      let targets = enabledTargets(params)
      let runPrompt = prompt
      if (targets.length === 0 && prompt) {
        const interp = await interpret.run(prompt, params.run.llm_model)
        targets = interp ? toTargetRows(interp.targets) : []
        runPrompt = interp?.prompt || prompt
        if (targets.length > 0) {
          setParams((prev) => ({
            ...prev,
            prompt: runPrompt,
            targets: targets.map((t) => ({ ...t, enabled: true, source: "auto" as const })),
          }))
        }
      }
      if (targets.length === 0) {
        setRunAllError(
          prompt
            ? interpret.error || "Таргеты не найдены — уточните промпт"
            : "Нужен промпт, цели или обводка",
        )
        return
      }

      const found = await detect.run({
        mode: "detect",
        prompt: runPrompt,
        targets,
        all: true,
        params: toRunParams(params),
      })
      const frameCount = source.probe.frame_count
      if (found && tracksAreFullLength(found, frameCount)) {
        setViewerMode("detect")
        await inpaint.run({ mode: "tracks", tracks: found }, toRunParams(params))
      } else {
        setRunAllError("Маски не покрыли весь ролик — проверьте цели и запустите Маски ещё раз")
      }
    } finally {
      setRunAllBusy(false)
    }
  }

  const openConfig = () => {
    window.dispatchEvent(new CustomEvent("videoclean:open-config"))
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-[260px_minmax(0,1fr)_340px] grid-rows-[minmax(0,1fr)_auto] gap-0">
      <aside className="row-span-2 min-h-0 min-w-0 border-r border-border/80 bg-card/40">
        <Library
          selectedId={sourceId}
          onSelect={setSource}
          onClearSelection={() => setSource(null)}
          activeJobs={activeJobs}
          onSelectJob={(job) => void selectJob(job)}
          refreshKey={libraryTick}
          onUploaded={() => setLibraryTick((t) => t + 1)}
          maskTracks={detect.tracks}
          excludedIds={detect.excludedIds}
          selectedTrackId={detect.selectedTrackId}
          onToggleTrack={detect.toggleTrack}
          onSelectTrack={(id) => {
            detect.setSelectedTrackId(id)
            setViewerMode("detect")
          }}
        />
        {restoreError && (
          <p className="shrink-0 border-t border-border/70 px-3 py-2 text-xs text-destructive">
            {restoreError}
          </p>
        )}
      </aside>

      {source && probe ? (
        <>
          <div className="col-start-2 row-start-1 flex min-h-0 min-w-0 flex-col gap-0 overflow-hidden">
            <div className="flex h-11 shrink-0 items-center gap-3 border-b border-border/60 px-4">
              <Timecode>{formatTimecode(currentFrame, probe.fps)}</Timecode>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{source.name}</p>
                <p className="text-[11px] text-muted-foreground">
                  {probe.width}×{probe.height} · {probe.fps.toFixed(2)} fps · кадр {currentFrame + 1} /{" "}
                  {probe.frame_count}
                </p>
              </div>
              <ToggleGroup
                variant="outline"
                size="sm"
                value={[viewerMode]}
                onValueChange={(v) => {
                  const next = v.at(-1)
                  if (next === "annotate" || next === "detect" || next === "result") setViewerMode(next)
                }}
              >
                {(["annotate", "detect", "result"] as const).map((m) => (
                  <ToggleGroupItem key={m} value={m} className={cn(viewerMode === m && "bg-muted")}>
                    {MODE_LABEL[m]}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            </div>
            <div className="flex min-h-0 flex-1 flex-col overflow-hidden p-3">
              <EditorViewer
                source={source}
                currentFrame={currentFrame}
                onFrameChange={setCurrentFrame}
                annotate={annotate}
                mode={viewerMode}
                detectMaskUrl={detectMaskUrl}
                maskOpacity={maskOpacity}
                onMaskOpacityChange={setMaskOpacity}
                detectBoxes={detectFrame?.boxes ?? []}
                detectTracks={overlayTracks}
                selectedTrackId={detect.selectedTrackId}
                onSelectTrack={detect.setSelectedTrackId}
                onResizeTrack={(id, box) => {
                  const n = probe.frame_count
                  const track = detect.tracks.find((t) => t.id === id)
                  if (!track) return
                  if (track.boxes.length === n) {
                    detect.patchBox(id, currentFrame, box)
                    return
                  }
                  const i = (detect.manifest?.frames ?? []).findIndex((f) => f.index === currentFrame)
                  if (i >= 0) detect.patchBox(id, i, box)
                }}
                resultJobId={resultJobId}
              />
            </div>
          </div>
          <div className="col-start-2 row-start-2 min-h-0 min-w-0 border-t border-border/80 bg-card/30 px-3 py-2">
            <Timeline
              src={source.video_url}
              fps={probe.fps}
              frameCount={probe.frame_count}
              currentFrame={currentFrame}
              onFrameChange={setCurrentFrame}
              annotatedFrames={annotatedFrames}
              maskedFrames={maskedFrames}
              keyframeFrames={viewerMode === "detect" ? keyframeFrames : []}
              trackLabel={
                viewerMode === "detect" && selectedTrack
                  ? selectedTrack.label || `трек ${selectedTrack.id}`
                  : null
              }
              onClearKey={
                viewerMode === "detect" && selectedTrack
                  ? (frame) => {
                      const n = probe.frame_count
                      if (selectedTrack.boxes.length === n) {
                        detect.clearKey(selectedTrack.id, frame)
                        return
                      }
                      const i = (detect.manifest?.frames ?? []).findIndex((f) => f.index === frame)
                      if (i >= 0) detect.clearKey(selectedTrack.id, i)
                    }
                  : undefined
              }
            />
          </div>
        </>
      ) : (
        <div className="col-start-2 row-span-2 flex min-h-0 items-center justify-center p-8">
          <div className="flex max-w-sm flex-col items-center text-center">
            <span className="mb-4 flex size-14 items-center justify-center rounded-2xl bg-primary/10 text-primary">
              <Film className="size-6" />
            </span>
            <h2 className="text-lg font-semibold tracking-tight">Выберите видео</h2>
            <p className="mt-2 text-sm text-muted-foreground">
              Загрузите ролик слева или выберите уже загруженный источник. Дальше опишите, что
              удалить, и запустите конвейер.
            </p>
            <p className="mt-4 inline-flex items-center gap-1.5 text-xs text-muted-foreground">
              <Upload className="size-3.5" />
              Поддерживаются обычные видеофайлы
            </p>
          </div>
        </div>
      )}

      <aside className="col-start-3 row-span-2 min-h-0 min-w-0 border-l border-border/80 bg-card/40">
        <StageRail
          source={source}
          frameCount={probe?.frame_count ?? 0}
          params={params}
          onParamsChange={setParams}
          interpret={interpret}
          detect={detect}
          inpaint={inpaint}
          resultJob={resultJob}
          masks={masks}
          onRunAll={() => void runAll()}
          runAllBusy={runAllBusy}
          runAllError={runAllError}
          onOpenConfig={openConfig}
          onOpenResult={() => setViewerMode("result")}
          onResultJobChange={setResultJob}
        />
      </aside>
    </div>
  )
}
