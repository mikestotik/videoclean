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
    <div className="flex overflow-x-auto rounded-md border p-1">
      {indexes.map((idx) => (
        <button
          key={idx}
          className={cn(
            "relative h-16 w-28 shrink-0 overflow-hidden rounded-md border-2 border-transparent",
            currentFrame === idx && "ring-2 ring-primary",
          )}
          onClick={() => onFrameChange(idx)}
        >
          <img src={thumbs[idx]} alt={`frame ${idx}`} className="h-full w-full object-cover" />
          <span className="absolute right-0 bottom-0 bg-background/80 px-0.5 text-[9px]">{idx}</span>
          {(annotated.has(idx) || masked.has(idx)) && (
            <span className="absolute bottom-0.5 left-1/2 flex -translate-x-1/2 gap-0.5">
              {annotated.has(idx) && <span className="size-1.5 rounded-full bg-primary" />}
              {masked.has(idx) && <span className="size-1.5 rounded-full bg-destructive" />}
            </span>
          )}
        </button>
      ))}
      {indexes.length === 0 && (
        <div className="flex h-16 items-center px-3 text-xs text-muted-foreground">Миниатюры…</div>
      )}
    </div>
  )
}
