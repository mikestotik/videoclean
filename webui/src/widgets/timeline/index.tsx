import { useEffect, useMemo, useState } from "react"
import { cn } from "@/shared/lib/utils"
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
}

export function Timeline({ src, fps, frameCount, currentFrame, onFrameChange, annotatedFrames, maskedFrames }: Props) {
  const [thumbs, setThumbs] = useState<Record<number, string>>({})
  const [media, setMedia] = useState({ src, fps, frameCount })
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
  const indexes = Object.keys(thumbs).map(Number).sort((a, b) => a - b)

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between px-0.5">
        <p className="text-[11px] font-medium text-muted-foreground">Таймлайн</p>
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <span className="size-1.5 rounded-full bg-primary" /> разметка
          </span>
          <span className="inline-flex items-center gap-1">
            <span className="size-1.5 rounded-full bg-destructive" /> маска
          </span>
        </div>
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
            {(annotated.has(idx) || masked.has(idx)) && (
              <span className="absolute bottom-1 left-1/2 flex -translate-x-1/2 gap-0.5">
                {annotated.has(idx) && <span className="size-1.5 rounded-full bg-primary" />}
                {masked.has(idx) && <span className="size-1.5 rounded-full bg-destructive" />}
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
