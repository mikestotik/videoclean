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
  const width = Math.max(rect.width, el.scrollWidth)
  if (width <= 0 || frameCount <= 1) return 0
  const x = clientX - rect.left + el.scrollLeft
  const t = Math.min(1, Math.max(0, x / width))
  return Math.round(t * (frameCount - 1))
}

function nearestIndex(indexes: number[], frame: number): number | null {
  if (indexes.length === 0) return null
  let best = indexes[0]!
  let bestDist = Math.abs(best - frame)
  for (const i of indexes) {
    const d = Math.abs(i - frame)
    if (d < bestDist) {
      best = i
      bestDist = d
    }
  }
  return best
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
  const thumbsRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef(false)
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
  const focusThumb = nearestIndex(indexes, currentFrame)
  const atKey = keySet.has(currentFrame)
  const playheadPct =
    frameCount > 1 ? `${(currentFrame / (frameCount - 1)) * 100}%` : "0%"

  // Keep filmstrip scrolled to the playhead / nearest thumb.
  useEffect(() => {
    const root = thumbsRef.current
    if (!root || focusThumb == null) return
    const el = root.querySelector<HTMLElement>(`[data-frame="${focusThumb}"]`)
    if (!el) return
    el.scrollIntoView({
      inline: "center",
      block: "nearest",
      behavior: draggingRef.current ? "auto" : "smooth",
    })
  }, [focusThumb, currentFrame])

  const scrubFromEvent = (clientX: number, el: HTMLElement | null) => {
    if (!el || frameCount <= 0) return
    onFrameChange(frameFromClientX(clientX, el, frameCount))
  }

  const startScrub = (
    e: React.PointerEvent<HTMLElement>,
    el: HTMLElement | null,
  ) => {
    if (e.button !== 0 || !el) return
    draggingRef.current = true
    el.setPointerCapture(e.pointerId)
    scrubFromEvent(e.clientX, el)
  }

  const moveScrub = (e: React.PointerEvent<HTMLElement>, el: HTMLElement | null) => {
    if (!draggingRef.current || !el) return
    scrubFromEvent(e.clientX, el)
  }

  const endScrub = (e?: React.PointerEvent<HTMLElement>, el?: HTMLElement | null) => {
    draggingRef.current = false
    if (e && el?.hasPointerCapture?.(e.pointerId)) {
      el.releasePointerCapture(e.pointerId)
    }
  }

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between gap-2 px-0.5">
        <p className="text-[11px] font-medium text-muted-foreground">Кадры</p>
        <div className="flex flex-wrap items-center justify-end gap-3 text-[10px] text-muted-foreground">
          {annotatedFrames.length > 0 && (
            <span className="inline-flex items-center gap-1">
              <span className="size-1.5 rounded-full bg-primary" /> кисть
            </span>
          )}
          {maskedFrames.length > 0 && (
            <span className="inline-flex items-center gap-1">
              <span className="size-1.5 rounded-full bg-destructive" /> найденная маска
            </span>
          )}
          {keyframeFrames.length > 0 && (
            <span className="inline-flex items-center gap-1">
              <span className="size-1.5 rotate-45 bg-amber-400" /> правка рамки
            </span>
          )}
        </div>
      </div>

      {(keyframeFrames.length > 0 || trackLabel) && (
        <div className="flex items-center gap-2 px-0.5">
          <p className="min-w-0 flex-1 truncate text-[11px] text-muted-foreground">
            {trackLabel ? `Правки рамки: ${trackLabel}` : "Правки выбранной рамки"}
            {keyframeFrames.length > 0 ? ` · ${keyframeFrames.length}` : ""}
          </p>
          {atKey && onClearKey && (
            <Button size="xs" variant="ghost" onClick={() => onClearKey(currentFrame)}>
              Сбросить правку
            </Button>
          )}
        </div>
      )}

      <div
        ref={stripRef}
        role="slider"
        aria-label="Точная перемотка по кадрам"
        aria-valuemin={frameCount > 0 ? 1 : 0}
        aria-valuemax={Math.max(0, frameCount)}
        aria-valuenow={frameCount > 0 ? currentFrame + 1 : 0}
        tabIndex={0}
        className="relative h-5 cursor-ew-resize touch-none rounded-md border border-border/70 bg-muted/40 select-none"
        onPointerDown={(e) => startScrub(e, stripRef.current)}
        onPointerMove={(e) => moveScrub(e, stripRef.current)}
        onPointerUp={(e) => endScrub(e, stripRef.current)}
        onPointerCancel={(e) => endScrub(e, stripRef.current)}
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
              title={`Правка рамки · кадр ${f + 1}`}
              className={cn(
                "absolute top-1/2 z-10 size-2.5 -translate-x-1/2 -translate-y-1/2 rotate-45 border border-amber-600 bg-amber-400",
                f === currentFrame && "ring-2 ring-amber-300",
              )}
              style={{ left }}
              onPointerDown={(e) => {
                e.stopPropagation()
              }}
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

      <div
        ref={thumbsRef}
        className="relative flex gap-1 overflow-x-auto rounded-lg border border-border/70 bg-background/40 p-1.5"
      >
        {indexes.map((idx) => (
          <button
            key={idx}
            type="button"
            data-frame={idx}
            className={cn(
              "relative h-14 w-24 shrink-0 overflow-hidden rounded-md border transition-shadow",
              focusThumb === idx
                ? "border-primary ring-2 ring-primary/40"
                : "border-transparent hover:border-border",
            )}
            title={`Обзорный кадр ${idx + 1}. Точная перемотка — полосой выше.`}
            onClick={() => onFrameChange(idx)}
          >
            <img src={thumbs[idx]} alt={`кадр ${idx + 1}`} className="pointer-events-none h-full w-full object-cover" />
            <span className="pointer-events-none absolute right-0.5 bottom-0.5 rounded bg-background/85 px-1 font-mono text-[9px] tabular-nums">
              {idx + 1}
            </span>
            {(annotated.has(idx) || masked.has(idx) || keySet.has(idx)) && (
              <span className="pointer-events-none absolute bottom-1 left-1/2 flex -translate-x-1/2 gap-0.5">
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
