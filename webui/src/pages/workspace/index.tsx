import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Film, Upload } from "lucide-react"
import { EditorViewer, type EditorMode } from "@widgets/editor-viewer"
import { Library } from "@widgets/library"
import {
  DEFAULT_PARAMS,
  StageRail,
  cloneParams,
  type EditorParams,
  type StageTarget,
} from "@widgets/stage-rail"
import { Timeline } from "@widgets/timeline"
import { useAnnotate } from "@features/annotate"
import { useDetectRun } from "@features/detect-run"
import { useInpaintRun } from "@features/inpaint-run"
import { useInterpret } from "@features/interpret"
import { fetchJobReport, type Job } from "@/entities/job"
import { previewArtifactUrl } from "@/entities/preview"

import { getSource, videoUrl, type Source } from "@/entities/source"
import { useEvents } from "@/shared/events"
import { formatTimecode } from "@/shared/lib/format"
import { Timecode } from "@/shared/ui/timecode"
import { ToggleGroup, ToggleGroupItem } from "@/shared/ui/toggle-group"
import { cn } from "@/shared/lib/utils"

const MODE_LABEL = { annotate: "Разметка", detect: "Рамки", result: "Результат" } as const

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
  report: { userPrompt?: unknown; prompt?: unknown; targets?: unknown },
): EditorParams {
  const targets = stageTargetsFromReport(report.targets)
  const userPrompt = typeof report.userPrompt === "string" ? report.userPrompt.trim() : ""
  const parsedPrompt = typeof report.prompt === "string" ? report.prompt.trim() : ""
  return {
    ...prev,
    ...(userPrompt ? { prompt: userPrompt } : {}),
    ...(parsedPrompt ? { parsedPrompt } : {}),
    ...(targets.length > 0 ? { targets } : {}),
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
  const [resultJob, setResultJob] = useState<Job | null>(null)
  const [restoreError, setRestoreError] = useState("")
  const [failedError, setFailedError] = useState("")
  const [holdFailed, setHoldFailed] = useState(false)
  const { jobs: liveJobs } = useEvents()

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

  const setSource = useCallback(
    (next: Source | null) => {
      setSourceState(next)
      onRouteSourceIdChange(next?.id ?? null)
    },
    [onRouteSourceIdChange],
  )

  // Render-phase reset when the route is cleared (no sync setState in effect).
  if (!routeSourceId && sourceId !== null) {
    setSourceState(null)
  }

  useEffect(() => {
    let cancelled = false
    if (!routeSourceId) return
    if (sourceId === routeSourceId) return
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
  }, [routeSourceId, sourceId, onRouteSourceIdChange])

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
    setParams(cloneParams(DEFAULT_PARAMS))
    setMaskOpacity(0.6)
    setResultJob(null)
    setRestoreError("")
    if (holdFailed) setHoldFailed(false)
    else setFailedError("")
  }

  const allowSourceChange = useCallback(() => {
    const dirty = params.prompt.trim().length > 0 || detect.tracksDirty
    if (!dirty) return true
    return window.confirm("Сменить видео? Промпт и несохранённые рамки будут сброшены.")
  }, [detect.tracksDirty, params.prompt])

  const selectJob = useCallback(
    async (job: Job) => {
      if (job.state === "FAILED") {
        setRestoreError("")
        if (job.source_id && job.source_id !== source?.id) {
          if (!allowSourceChange()) return
          try {
            const row = await getSource(job.source_id)
            setHoldFailed(true)
            setSource(row)
          } catch (e) {
            setRestoreError(e instanceof Error ? e.message : String(e))
            return
          }
        }
        setFailedError(job.error || "Ошибка")
        return
      }
      if (job.state !== "COMPLETED") return
      setRestoreError("")
      setFailedError("")
      try {
        if (job.kind === "prompt") {
          const loaded = await interpret.loadFromJob(job.id)
          if (!loaded) {
            setRestoreError("Не удалось загрузить результат промпта")
            return
          }
          setParams((prev) =>
            applyReportToParams(prev, {
              userPrompt: loaded.userPrompt,
              prompt: loaded.prompt,
              targets: loaded.targets,
            }),
          )
          setViewerMode("annotate")
          return
        }
        if (job.kind === "preview") {
          const report = (await fetchJobReport(job.id).catch(() => null)) as
            | { userPrompt?: unknown; prompt?: unknown; targets?: unknown }
            | null
          if (report) setParams((prev) => applyReportToParams(prev, report))
          await detect.loadFromJob(job.id)
          setViewerMode("detect")
          return
        }
        if (job.kind === "run") {
          const report = (await fetchJobReport(job.id).catch(() => null)) as
            | { userPrompt?: unknown; prompt?: unknown; targets?: unknown }
            | null
          if (report) {
            setParams((prev) => {
              const next = applyReportToParams(prev, report)
              if (prev.targets.some((t) => t.source === "manual")) {
                return { ...next, targets: prev.targets, parsedPrompt: prev.parsedPrompt }
              }
              return next
            })
          }
          inpaint.loadFromJob(job.id)
          setResultJob(job)
          setViewerMode("result")
        }
      } catch (e) {
        setRestoreError(e instanceof Error ? e.message : String(e))
      }
    },
    [allowSourceChange, detect, inpaint, interpret, setSource, source?.id],
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
  const liveResult = resultJobId ? liveJobs.find((j) => j.id === resultJobId) : undefined
  if (resultJob && resultJob.id !== resultJobId) setResultJob(null)
  // Render-phase sync of the completed result (avoids setState-in-effect).
  if (liveResult?.state === "COMPLETED" && resultJob?.id !== liveResult.id) {
    setResultJob(liveResult)
    setViewerMode("result")
  }

  const reportedRun = useRef<string | null>(null)
  useEffect(() => {
    if (!liveResult || liveResult.state !== "COMPLETED" || liveResult.kind !== "run") return
    if (reportedRun.current === liveResult.id) return
    reportedRun.current = liveResult.id
    void fetchJobReport(liveResult.id)
      .then((report) => {
        setParams((prev) => {
          const next = applyReportToParams(prev, report as { userPrompt?: unknown; prompt?: unknown; targets?: unknown })
          if (prev.targets.some((t) => t.source === "manual")) {
            return { ...next, targets: prev.targets, parsedPrompt: prev.parsedPrompt }
          }
          return next
        })
      })
      .catch(() => {})
  }, [liveResult])

  const openConfig = () => {
    window.dispatchEvent(new CustomEvent("videoclean:open-config"))
  }

  return (
    <div className="grid h-full min-h-0 grid-cols-[260px_minmax(0,1fr)_340px] grid-rows-[minmax(0,1fr)_auto] gap-0">
      <aside className="row-span-2 min-h-0 min-w-0 border-r border-border/80 bg-card/40">
        <Library
          selectedId={sourceId}
          onSelect={setSource}
          beforeSourceChange={allowSourceChange}
          onClearSelection={() => {
            if (!allowSourceChange()) return
            setSource(null)
          }}
          activeJobs={activeJobs}
          onSelectJob={(job) => void selectJob(job)}
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
              src={videoUrl(source)}
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
          detect={detect}
          inpaint={inpaint}
          resultJob={resultJob}
          masks={masks}
          failedError={failedError}
          onClearFailed={() => setFailedError("")}
          onOpenConfig={openConfig}
          onOpenResult={() => setViewerMode("result")}
          onResultJobChange={setResultJob}
          phrase={interpret}
        />
      </aside>
    </div>
  )
}
