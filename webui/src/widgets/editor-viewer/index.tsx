import { useCallback, useEffect, useRef } from "react"
import { timeToFrameIndex } from "@/entities/frame"
import type { Source } from "@/entities/source"
import type { Tool } from "@/entities/annotation"
import type { useAnnotate } from "@/features/annotate"
import { Brush, Eraser, Trash, Undo2 } from "lucide-react"
import { Button } from "@/shared/ui/button"
import { Slider } from "@/shared/ui/slider"
import { ToggleGroup, ToggleGroupItem } from "@/shared/ui/toggle-group"
import { AnnotationLayer } from "./annotation-layer"
import { Compare } from "./compare"
import { MaskOverlay } from "./mask-overlay"

export type EditorMode = "annotate" | "detect" | "result"

const SCRUB_PX_PER_FRAME = 4

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
  resultJobId,
}: Props) {
  const { fps, frame_count: frameCount, width, height } = source.probe
  const videoRef = useRef<HTMLVideoElement>(null)
  const playingRef = useRef(false)
  const scrubRef = useRef<{ x: number; frame: number } | null>(null)
  const { setActiveFrame, setTool, setSize, undo, clearFrame, isDirty } = annotate

  const clamp = useCallback(
    (frame: number) => Math.min(Math.max(0, frame), Math.max(0, frameCount - 1)),
    [frameCount],
  )

  useEffect(() => {
    setActiveFrame(currentFrame)
  }, [currentFrame, setActiveFrame])

  useEffect(() => {
    const v = videoRef.current
    if (!v || playingRef.current) return
    const target = currentFrame / fps
    if (Math.abs(v.currentTime - target) > 0.4 / fps) v.currentTime = target
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
    }
  }

  const scrubbing = mode !== "annotate"

  return (
    <div className="flex min-w-0 flex-1 flex-col gap-2">
      {mode === "result" ? (
        resultJobId ? (
          <Compare inputUrl={source.video_url} jobId={resultJobId} width={width} height={height} />
        ) : (
          <div className="flex h-64 items-center justify-center rounded-md border text-sm text-muted-foreground">
            Результат недоступен
          </div>
        )
      ) : (
        <div
          tabIndex={0}
          className="relative shrink-0 overflow-hidden rounded-md border bg-black outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
          style={{ aspectRatio: `${width} / ${height}` }}
          onKeyDown={onKeyDown}
          onPointerDown={(e) => {
            if (!scrubbing || e.button !== 0) return
            scrubRef.current = { x: e.clientX, frame: currentFrame }
            e.currentTarget.setPointerCapture(e.pointerId)
          }}
          onPointerMove={(e) => {
            const s = scrubRef.current
            if (!scrubbing || !s) return
            const next = clamp(s.frame + Math.round((e.clientX - s.x) / SCRUB_PX_PER_FRAME))
            if (next !== currentFrame) onFrameChange(next)
          }}
          onPointerUp={() => {
            scrubRef.current = null
          }}
        >
          <video
            ref={videoRef}
            src={source.video_url}
            preload="auto"
            playsInline
            className="absolute inset-0 h-full w-full"
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
                >
                  <span className="absolute -top-4 left-0 bg-background/80 px-1 text-[10px] text-mask">
                    {i}
                  </span>
                </div>
              ) : null,
            )}
        </div>
      )}

      {mode === "annotate" && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
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
          <Slider
            className="w-32"
            min={4}
            max={120}
            value={annotate.size}
            onValueChange={(v) => {
              if (typeof v === "number") setSize(v)
            }}
          />
          <span className="text-muted-foreground">{annotate.size}px</span>
          <Button variant="outline" size="icon-sm" aria-label="Отменить" onClick={undo}>
            <Undo2 />
          </Button>
          <Button variant="outline" size="icon-sm" aria-label="Очистить кадр" onClick={clearFrame}>
            <Trash />
          </Button>
          <span className="ml-auto text-muted-foreground">
            Кадр {currentFrame} · {isDirty ? "сохранение…" : "маска сохранена"}
          </span>
        </div>
      )}

      {mode === "detect" && (
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {onMaskOpacityChange && (
            <>
              <span className="text-muted-foreground">Маска</span>
              <Slider
                className="w-32"
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
