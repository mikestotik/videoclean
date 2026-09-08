import { useEffect, useState } from "react"
import { frameTimes, timeToFrameIndex } from "@/entities/frame"
import { cn } from "@/shared/lib/utils"

type Props = {
  src: string
  fps: number
  frameCount: number
  thumbnails: number
  selected: Set<number>
  current: number | null
  onToggle: (idx: number, additive: boolean, range: boolean) => void
  onCurrent: (idx: number) => void
}

export function Filmstrip({ src, fps, frameCount, thumbnails, selected, current, onToggle, onCurrent }: Props) {
  const [thumbs, setThumbs] = useState<Record<number, string>>({})

  useEffect(() => {
    if (fps <= 0 || frameCount <= 0) return
    const v = document.createElement("video")
    v.src = src
    v.muted = true
    v.preload = "auto"
    let cancelled = false

    const grabAll = () => {
      const duration =
        Number.isFinite(v.duration) && v.duration > 0 ? v.duration : frameCount / Math.max(fps, 0.001)
      const times = frameTimes(duration, thumbnails)
      const seen = new Set<number>()
      const targets: number[] = []
      for (const t of times) {
        const idx = timeToFrameIndex(t, fps, frameCount - 1)
        if (!seen.has(idx)) {
          seen.add(idx)
          targets.push(t)
        }
      }
      let i = 0
      const grab = () => {
        if (cancelled || i >= targets.length) return
        const target = targets[i]
        const frameIdx = timeToFrameIndex(target, fps, frameCount - 1)
        const onSeeked = () => {
          v.removeEventListener("seeked", onSeeked)
          const canvas = document.createElement("canvas")
          canvas.width = 160
          canvas.height = 90
          const ctx = canvas.getContext("2d")
          if (ctx) {
            ctx.drawImage(v, 0, 0, canvas.width, canvas.height)
            const data = canvas.toDataURL("image/jpeg", 0.6)
            setThumbs((prev) => ({ ...prev, [frameIdx]: data }))
          }
          i += 1
          setTimeout(grab, 0)
        }
        v.addEventListener("seeked", onSeeked)
        v.currentTime = target
      }
      grab()
    }

    v.addEventListener("loadedmetadata", grabAll)
    return () => {
      cancelled = true
      v.pause()
    }
  }, [src, fps, frameCount, thumbnails])

  const indexes = Object.keys(thumbs).map(Number).sort((a, b) => a - b)

  return (
    <div className="flex gap-1 overflow-x-auto rounded-md border p-2">
      {indexes.map((idx) => (
        <button
          key={idx}
          className={cn(
            "relative h-14 w-24 shrink-0 overflow-hidden rounded-md border-2",
            selected.has(idx) ? "border-primary" : "border-transparent",
            current === idx && "ring-2 ring-ring",
          )}
          onClick={(e) => {
            onToggle(idx, e.ctrlKey || e.metaKey, e.shiftKey)
            onCurrent(idx)
          }}
        >
          <img src={thumbs[idx]} alt={`frame ${idx}`} className="h-full w-full object-cover" />
          <span className="absolute right-0 bottom-0 bg-background/80 px-0.5 text-[9px]">{idx}</span>
        </button>
      ))}
    </div>
  )
}
