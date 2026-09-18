import { useCallback, useEffect, useRef, useState } from "react"
import { timeToFrameIndex } from "@/entities/frame"
import { videoUrl, type Source } from "@/entities/source"
import type { Tool } from "@/entities/annotation"
import type { useAnnotate } from "@/features/annotate"
import { Brush, Eraser, Minus, Plus, Trash, Undo2 } from "lucide-react"
import { Button } from "@/shared/ui/button"
import { Slider } from "@/shared/ui/slider"
import { ToggleGroup, ToggleGroupItem } from "@/shared/ui/toggle-group"
import type { DetectBox } from "@features/detect-run"
import { AnnotationLayer } from "./annotation-layer"
import { Compare } from "./compare"
import { DetectBoxes, type OverlayTrack } from "./detect-boxes"
import { MaskOverlay } from "./mask-overlay"

export type EditorMode = "annotate" | "detect" | "result"

const SCRUB_PX_PER_FRAME = 4
const ZOOM_MIN = 1
const ZOOM_MAX = 8
const ZOOM_STEP = 1.25

function clampZoom(value: number): number {
  return Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, value))
}

type Props = {
  source: Source
  currentFrame: number
  onFrameChange: (frame: number) => void
  annotate: ReturnType<typeof useAnnotate>
  mode: EditorMode
  detectMaskUrl: string | null
  maskOpacity: number
  onMaskOpacityChange?: (v: number) => void
  detectBoxes: (number[] | null)[]
  detectTracks?: OverlayTrack[]
  selectedTrackId?: number | null
  onSelectTrack?: (id: number) => void
  onResizeTrack?: (id: number, box: DetectBox) => void
  resultJobId: string | null
}

export function EditorViewer({
  source,
  currentFrame,
  onFrameChange,
  annotate,
  mode,
  detectMaskUrl,
  maskOpacity,
  onMaskOpacityChange,
  detectBoxes,
  detectTracks,
  selectedTrackId = null,
  onSelectTrack,
  onResizeTrack,
  resultJobId,
}: Props) {
  const { fps, frame_count: frameCount, width, height } = source.probe
  const videoRef = useRef<HTMLVideoElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const playingRef = useRef(false)
  const scrubRef = useRef<{ x: number; frame: number } | null>(null)
  const panRef = useRef<{ x: number; y: number; sl: number; st: number } | null>(null)
  const { setActiveFrame, setTool, setSize, undo, clearFrame, isDirty } = annotate
  const [zoom, setZoom] = useState(1)

  const [prevSourceId, setPrevSourceId] = useState(source.id)
  if (prevSourceId !== source.id) {
    setPrevSourceId(source.id)
    setZoom(1)
  }

  const clamp = useCallback(
    (frame: number) => Math.min(Math.max(0, frame), Math.max(0, frameCount - 1)),
    [frameCount],
  )

  useEffect(() => {
    setActiveFrame(currentFrame)
  }, [currentFrame, setActiveFrame])

  useEffect(() => {
    const v = videoRef.current
    if (!v) return
    // Timeline / arrows own the clock while paused; play drives frames via timeupdate.
    if (!v.paused && playingRef.current) return
    if (!v.paused) v.pause()
    const target = currentFrame / Math.max(fps, 0.001)
    if (Math.abs(v.currentTime - target) > 0.5 / Math.max(fps, 0.001)) {
      try {
        v.currentTime = target
      } catch {
        // ignore seek before metadata
      }
    }
  }, [currentFrame, fps])

  const togglePlay = useCallback(() => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) void v.play().catch(() => {})
    else v.pause()
  }, [])

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault()
      const step = e.shiftKey ? 10 : 1
      onFrameChange(clamp(currentFrame + (e.key === "ArrowRight" ? step : -step)))
    } else if (e.code === "Space") {
      e.preventDefault()
      togglePlay()
    } else if (e.key === "+" || e.key === "=") {
      e.preventDefault()
      setZoom((z) => clampZoom(z * ZOOM_STEP))
    } else if (e.key === "-" || e.key === "_") {
      e.preventDefault()
      setZoom((z) => clampZoom(z / ZOOM_STEP))
    } else if (e.key === "0") {
      e.preventDefault()
      setZoom(1)
    }
  }

  const zoomAt = (next: number, clientX: number, clientY: number) => {
    const stage = stageRef.current
    if (!stage) {
      setZoom(next)
      return
    }
    const prev = zoom
    const rect = stage.getBoundingClientRect()
    const cx = clientX - rect.left + stage.scrollLeft
    const cy = clientY - rect.top + stage.scrollTop
    setZoom(next)
    requestAnimationFrame(() => {
      const el = stageRef.current
      if (!el || prev <= 0) return
      const ratio = next / prev
      el.scrollLeft = cx * ratio - (clientX - rect.left)
      el.scrollTop = cy * ratio - (clientY - rect.top)
    })
  }

  const scrubbing = mode !== "annotate" && zoom <= 1
  const panning = zoom > 1

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-2">
      {mode === "result" ? (
        resultJobId ? (
          <Compare
            inputUrl={videoUrl(source)}
            jobId={resultJobId}
            width={width}
            height={height}
            fps={fps}
            frameCount={frameCount}
            currentFrame={currentFrame}
            onFrameChange={onFrameChange}
          />
        ) : (
          <div className="flex min-h-48 flex-1 items-center justify-center rounded-lg border border-dashed border-border/80 text-sm text-muted-foreground">
            Результат ещё не готов
          </div>
        )
      ) : (
        <div className="relative min-h-0 flex-1">
        <div
          ref={stageRef}
          tabIndex={0}
          className="h-full min-h-0 overflow-auto rounded-lg border border-border/80 bg-black outline-none [container-type:size] focus-visible:ring-2 focus-visible:ring-ring/50"
          onKeyDown={onKeyDown}
          onWheel={(e) => {
            if (!e.ctrlKey && !e.metaKey) return
            e.preventDefault()
            const factor = e.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP
            zoomAt(clampZoom(zoom * factor), e.clientX, e.clientY)
          }}
          onPointerDown={(e) => {
            if (e.button === 1 || (panning && e.button === 0)) {
              const stage = stageRef.current
              if (!stage) return
              panRef.current = { x: e.clientX, y: e.clientY, sl: stage.scrollLeft, st: stage.scrollTop }
              e.currentTarget.setPointerCapture(e.pointerId)
              return
            }
            if (!scrubbing || e.button !== 0) return
            scrubRef.current = { x: e.clientX, frame: currentFrame }
            e.currentTarget.setPointerCapture(e.pointerId)
          }}
          onPointerMove={(e) => {
            const pan = panRef.current
            if (pan) {
              const stage = stageRef.current
              if (!stage) return
              stage.scrollLeft = pan.sl - (e.clientX - pan.x)
              stage.scrollTop = pan.st - (e.clientY - pan.y)
              return
            }
            const s = scrubRef.current
            if (!scrubbing || !s) return
            const next = clamp(s.frame + Math.round((e.clientX - s.x) / SCRUB_PX_PER_FRAME))
            if (next !== currentFrame) onFrameChange(next)
          }}
          onPointerUp={() => {
            scrubRef.current = null
            panRef.current = null
          }}
        >
          <div
            className="relative mx-auto"
            style={{
              aspectRatio: `${width} / ${height}`,
              width: `calc(min(100cqw, 100cqh * ${width} / ${height}) * ${zoom})`,
              height: `calc(min(100cqh, 100cqw * ${height} / ${width}) * ${zoom})`,
            }}
          >
            <video
              ref={videoRef}
              src={videoUrl(source)}
              preload="auto"
              playsInline
              className="absolute inset-0 h-full w-full object-contain"
              onLoadedMetadata={() => {
                const v = videoRef.current
                if (!v || playingRef.current) return
                const target = currentFrame / Math.max(fps, 0.001)
                if (Math.abs(v.currentTime - target) > 0.5 / Math.max(fps, 0.001)) {
                  v.currentTime = target
                }
              }}
              onPlay={() => {
                playingRef.current = true
              }}
              onPause={() => {
                playingRef.current = false
              }}
              onTimeUpdate={() => {
                const v = videoRef.current
                if (v && playingRef.current) {
                  onFrameChange(timeToFrameIndex(v.currentTime, fps, Math.max(0, frameCount - 1)))
                }
              }}
            />
            {mode === "annotate" && (
              <AnnotationLayer width={width} height={height} frame={currentFrame} annotate={annotate} />
            )}
            {mode === "detect" && <MaskOverlay url={detectMaskUrl} opacity={maskOpacity} />}
            {mode === "detect" &&
              (detectTracks && detectTracks.length > 0 ? (
                <DetectBoxes
                  tracks={detectTracks}
                  videoWidth={width}
                  videoHeight={height}
                  selectedId={selectedTrackId}
                  onSelect={(id) => onSelectTrack?.(id)}
                  onResize={onResizeTrack}
                />
              ) : (
                detectBoxes.map((box, i) =>
                  box && box.length >= 4 && width > 0 && height > 0 ? (
                    <div
                      key={i}
                      className="pointer-events-none absolute border border-mask"
                      style={{
                        left: `${(box[0] / width) * 100}%`,
                        top: `${(box[1] / height) * 100}%`,
                        width: `${((box[2] - box[0]) / width) * 100}%`,
                        height: `${((box[3] - box[1]) / height) * 100}%`,
                      }}
                    />
                  ) : null,
                )
              ))}
          </div>
        </div>
        <div
          className="absolute right-2 bottom-2 z-20 flex items-center gap-0.5 rounded-md bg-background/85 p-0.5"
          onPointerDown={(e) => e.stopPropagation()}
        >
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label="Уменьшить"
            disabled={zoom <= ZOOM_MIN}
            onClick={() => setZoom((z) => clampZoom(z / ZOOM_STEP))}
          >
            <Minus className="size-3.5" />
          </Button>
          <button
            type="button"
            className="min-w-10 px-1 text-center text-[11px] tabular-nums text-muted-foreground"
            onClick={() => setZoom(1)}
          >
            {Math.round(zoom * 100)}%
          </button>
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label="Увеличить"
            disabled={zoom >= ZOOM_MAX}
            onClick={() => setZoom((z) => clampZoom(z * ZOOM_STEP))}
          >
            <Plus className="size-3.5" />
          </Button>
        </div>
        </div>
      )}

      {mode === "annotate" && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border/60 bg-card/60 px-2.5 py-2 text-xs">
          <ToggleGroup
            variant="outline"
            size="sm"
            value={[annotate.tool]}
            onValueChange={(v) => {
              const next = v.at(-1)
              if (next) setTool(next as Tool)
            }}
          >
            <ToggleGroupItem value="brush" aria-label="Кисть">
              <Brush />
            </ToggleGroupItem>
            <ToggleGroupItem value="eraser" aria-label="Ластик">
              <Eraser />
            </ToggleGroupItem>
          </ToggleGroup>
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground">Размер</span>
            <Slider
              className="w-28"
              min={4}
              max={120}
              value={annotate.size}
              onValueChange={(v) => {
                if (typeof v === "number") setSize(v)
              }}
            />
            <span className="w-8 tabular-nums text-muted-foreground">{annotate.size}</span>
          </div>
          <Button variant="outline" size="icon-sm" aria-label="Отменить" onClick={undo}>
            <Undo2 />
          </Button>
          <Button variant="outline" size="icon-sm" aria-label="Очистить кадр" onClick={clearFrame}>
            <Trash />
          </Button>
          <span className="ml-auto text-muted-foreground">
            {isDirty ? "Сохранение…" : "Маска сохранена"}
          </span>
        </div>
      )}

      {mode === "detect" && (
        <div className="flex flex-wrap items-center gap-2 rounded-lg border border-border/60 bg-card/60 px-2.5 py-2 text-xs">
          {onMaskOpacityChange && (
            <>
              <span className="text-muted-foreground">Прозрачность маски</span>
              <Slider
                className="w-28"
                min={0}
                max={1}
                step={0.05}
                value={maskOpacity}
                onValueChange={(v) => {
                  if (typeof v === "number") onMaskOpacityChange(v)
                }}
              />
            </>
          )}
          <span className="ml-auto text-muted-foreground">Кадр {currentFrame}</span>
        </div>
      )}
    </div>
  )
}
