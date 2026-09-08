import { useEffect, useMemo, useState } from "react"
import { EditorViewer, type EditorMode } from "@widgets/editor-viewer"
import { Library } from "@widgets/library"
import {
  DEFAULT_PARAMS,
  StageRail,
  enabledTargets,
  toRunParams,
  type EditorParams,
} from "@widgets/stage-rail"
import { Timeline } from "@widgets/timeline"
import { useAnnotate } from "@features/annotate"
import { useDetectRun } from "@features/detect-run"
import { useInpaintRun } from "@features/inpaint-run"
import { useInterpret, type InterpretTarget } from "@features/interpret"
import { getJob, type Job } from "@/entities/job"
import { previewArtifactUrl } from "@/entities/preview"
import type { TargetKind, TargetRow } from "@/entities/targets"
import type { Source } from "@/entities/source"
import { formatTimecode } from "@/shared/lib/format"
import { Timecode } from "@/shared/ui/timecode"
import { ToggleGroup, ToggleGroupItem } from "@/shared/ui/toggle-group"

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

export function WorkspacePage() {
  const [source, setSource] = useState<Source | null>(null)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [viewerMode, setViewerMode] = useState<EditorMode>("annotate")
  const [params, setParams] = useState<EditorParams>(DEFAULT_PARAMS)
  const [maskOpacity, setMaskOpacity] = useState(0.6)
  const [runAllBusy, setRunAllBusy] = useState(false)
  const [runAllError, setRunAllError] = useState("")
  const [resultJob, setResultJob] = useState<Job | null>(null)

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

  const [prevSourceId, setPrevSourceId] = useState<string | null>(sourceId)
  if (prevSourceId !== sourceId) {
    setPrevSourceId(sourceId)
    setCurrentFrame(0)
    setViewerMode("annotate")
    setParams(DEFAULT_PARAMS)
    setMaskOpacity(0.6)
    setResultJob(null)
    setRunAllError("")
  }

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
    if (prompt && masks.length === 0) {
      await inpaint.run(
        { mode: "prompt", prompt, targets: enabledTargets(params) },
        toRunParams(params),
      )
      return
    }
    setRunAllBusy(true)
    try {
      const interp = await interpret.run(prompt)
      const targets = interp ? toTargetRows(interp.targets) : enabledTargets(params)
      if (targets.length === 0) {
        setRunAllError(
          interp
            ? "Таргеты не найдены — уточните промпт"
            : interpret.error || "Не удалось интерпретировать промпт",
        )
        return
      }
      await inpaint.run(
        { mode: "prompt", prompt: interp?.prompt || params.prompt, targets },
        toRunParams(params),
      )
    } finally {
      setRunAllBusy(false)
    }
  }

  const openConfig = () => {
    window.dispatchEvent(new CustomEvent("videoclean:open-config"))
  }

  return (
    <div className="grid h-full grid-cols-[280px_minmax(0,1fr)_360px] grid-rows-[minmax(0,1fr)_auto] gap-3 p-3">
      <div className="row-span-2 min-h-0 min-w-0">
        <Library selectedId={sourceId} onSelect={setSource} refreshKey={0} />
      </div>

      {source && probe ? (
        <>
          <div className="col-start-2 row-start-1 flex min-h-0 min-w-0 flex-col gap-2 overflow-hidden">
            <div className="flex items-center gap-3">
              <Timecode>{formatTimecode(currentFrame, probe.fps)}</Timecode>
              <span className="truncate text-sm text-muted-foreground">{source.name}</span>
              <span className="ml-auto shrink-0 text-xs text-muted-foreground">
                Кадр {currentFrame}
              </span>
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
                  <ToggleGroupItem key={m} value={m}>
                    {MODE_LABEL[m]}
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            </div>
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
              resultJobId={resultJobId}
            />
          </div>
          <div className="col-start-2 row-start-2 min-h-0 min-w-0">
            <Timeline
              src={source.video_url}
              fps={probe.fps}
              frameCount={probe.frame_count}
              currentFrame={currentFrame}
              onFrameChange={setCurrentFrame}
              annotatedFrames={annotatedFrames}
              maskedFrames={maskedFrames}
            />
          </div>
        </>
      ) : (
        <div className="col-start-2 row-start-1 flex min-h-0 items-center justify-center rounded-md border border-dashed bg-muted/40 p-8 text-center">
          <p className="text-sm text-muted-foreground">
            Выберите видео слева или загрузите новое
          </p>
        </div>
      )}

      <div className="col-start-3 row-span-2 min-h-0 min-w-0">
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
        />
      </div>
    </div>
  )
}
