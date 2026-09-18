import { useEffect, useMemo, useRef, useState } from "react"
import { cn } from "@/shared/lib/utils"
import { Button } from "@/shared/ui/button"
import { grabThumbs } from "./thumbs"

const THUMB_COUNT = 48

type Props = {
  src: string
  fps: number
  frameCount: number
  currentFrame: number
  onFrameChange: (frame: number) => void
  annotatedFrames: number[]
  maskedFrames: number[]
  /** Video-frame indices of manual box keys for the selected track. */
  keyframeFrames?: number[]
  onClearKey?: (frame: number) => void
  trackLabel?: string | null
}

function frameFromClientX(
  clientX: number,
  el: HTMLElement,
  frameCount: number,
): number {
  const rect = el.getBoundingClientRect()
  if (rect.width <= 0 || frameCount <= 1) return 0
  const t = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width))
  return Math.round(t * (frameCount - 1))
}

export function Timeline({
  src,
  fps,
  frameCount,
  currentFrame,
  onFrameChange,
  annotatedFrames,
  maskedFrames,
  keyframeFrames = [],
  onClearKey,
  trackLabel,
}: Props) {
  const [thumbs, setThumbs] = useState<Record<number, string>>({})
  const [media, setMedia] = useState({ src, fps, frameCount })
  const stripRef = useRef<HTMLDivElement>(null)
  if (media.src !== src || media.fps !== fps || media.frameCount !== frameCount) {
    setMedia({ src, fps, frameCount })
    setThumbs({})
  }

  useEffect(() => {
    return grabThumbs(src, fps, frameCount, THUMB_COUNT, (frame, dataUrl) => {
      setThumbs((prev) => ({ ...prev, [frame]: dataUrl }))
    })
  }, [src, fps, frameCount])

  const annotated = useMemo(() => new Set(annotatedFrames), [annotatedFrames])
  const masked = useMemo(() => new Set(maskedFrames), [maskedFrames])
  const keySet = useMemo(() => new Set(keyframeFrames), [keyframeFrames])
  const indexes = Object.keys(thumbs).map(Number).sort((a, b) => a - b)
  const atKey = keySet.has(currentFrame)
  const playheadPct =
    frameCount > 1 ? `${(currentFrame / (frameCount - 1)) * 100}%` : "0%"

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2 px-0.5">
        <p className="text-[11px] font-medium text-muted-foreground">Таймлайн</p>
        <div className="flex flex-wrap items-center justify-end gap-3 text-[10px] text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <span className="size-1.5 rounded-full bg-primary" /> разметка
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="size-1.5 rounded-full bg-destructive" /> маска
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="size-1.5 rotate-45 bg-amber-400" /> ключ рамки
          </span>
        </div>
      </div>

      {(keyframeFrames.length > 0 || trackLabel) && (
        <div className="flex items-center gap-2 px-0.5">
          <p className="min-w-0 flex-1 truncate text-[11px] text-muted-foreground">
            {trackLabel ? `Ключи: ${trackLabel}` : "Ключи выбранного трека"}
            {keyframeFrames.length > 0 ? ` · ${keyframeFrames.length}` : ""}
          </p>
          {atKey && onClearKey && (
            <Button size="xs" variant="ghost" onClick={() => onClearKey(currentFrame)}>
              Сбросить ключ
            </Button>
          )}
        </div>
      )}

      <div
        ref={stripRef}
        role="slider"
        aria-label="Ключи рамки по кадрам"
        aria-valuemin={0}
        aria-valuemax={Math.max(0, frameCount - 1)}
        aria-valuenow={currentFrame}
        tabIndex={0}
        className="relative h-5 cursor-pointer rounded-md border border-border/70 bg-muted/40"
        onClick={(e) => {
          const el = stripRef.current
          if (!el || frameCount <= 0) return
          onFrameChange(frameFromClientX(e.clientX, el, frameCount))
        }}
        onKeyDown={(e) => {
          if (e.key === "ArrowLeft") onFrameChange(Math.max(0, currentFrame - 1))
          if (e.key === "ArrowRight") onFrameChange(Math.min(frameCount - 1, currentFrame + 1))
        }}
      >
        {keyframeFrames.map((f) => {
          const left = frameCount > 1 ? `${(f / (frameCount - 1)) * 100}%` : "0%"
          return (
            <button
              key={f}
              type="button"
              title={`Ключ · кадр ${f}`}
              className={cn(
                "absolute top-1/2 z-10 size-2.5 -translate-x-1/2 -translate-y-1/2 rotate-45 border border-amber-600 bg-amber-400",
                f === currentFrame && "ring-2 ring-amber-300",
              )}
              style={{ left }}
              onClick={(e) => {
                e.stopPropagation()
                onFrameChange(f)
              }}
            />
          )
        })}
        <span
          className="pointer-events-none absolute inset-y-0 z-20 w-0.5 -translate-x-1/2 bg-primary"
          style={{ left: playheadPct }}
        />
      </div>

      <div className="flex gap-1 overflow-x-auto rounded-lg border border-border/70 bg-background/40 p-1.5">
        {indexes.map((idx) => (
          <button
            key={idx}
            type="button"
            className={cn(
              "relative h-14 w-24 shrink-0 overflow-hidden rounded-md border transition-shadow",
              currentFrame === idx
                ? "border-primary ring-2 ring-primary/40"
                : "border-transparent hover:border-border",
            )}
            onClick={() => onFrameChange(idx)}
          >
            <img src={thumbs[idx]} alt={`кадр ${idx}`} className="h-full w-full object-cover" />
            <span className="absolute right-0.5 bottom-0.5 rounded bg-background/85 px-1 font-mono text-[9px] tabular-nums">
              {idx}
            </span>
            {(annotated.has(idx) || masked.has(idx) || keySet.has(idx)) && (
              <span className="absolute bottom-1 left-1/2 flex -translate-x-1/2 gap-0.5">
                {annotated.has(idx) && <span className="size-1.5 rounded-full bg-primary" />}
                {masked.has(idx) && <span className="size-1.5 rounded-full bg-destructive" />}
                {keySet.has(idx) && <span className="size-1.5 rotate-45 bg-amber-400" />}
              </span>
            )}
          </button>
        ))}
        {indexes.length === 0 && (
          <div className="flex h-14 items-center px-3 text-xs text-muted-foreground">Строю миниатюры…</div>
        )}
      </div>
    </div>
  )
}
