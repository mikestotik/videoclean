import { frameTimes, timeToFrameIndex } from "@/entities/frame"

export function grabThumbs(
  src: string,
  fps: number,
  frameCount: number,
  count: number,
  onThumb: (frame: number, dataUrl: string) => void,
): () => void {
  if (fps <= 0 || frameCount <= 0) return () => {}
  const v = document.createElement("video")
  v.src = src
  v.muted = true
  v.preload = "auto"
  let cancelled = false

  const grabAll = () => {
    if (cancelled) return
    const duration =
      Number.isFinite(v.duration) && v.duration > 0 ? v.duration : frameCount / Math.max(fps, 0.001)
    const times = frameTimes(duration, count)
    const seen = new Set<number>()
    const targets: { time: number; frame: number }[] = []
    for (const t of times) {
      const idx = timeToFrameIndex(t, fps, frameCount - 1)
      if (!seen.has(idx)) {
        seen.add(idx)
        targets.push({ time: t, frame: idx })
      }
    }
    let i = 0
    const grab = () => {
      if (cancelled || i >= targets.length) return
      const { time, frame } = targets[i]
      const onSeeked = () => {
        v.removeEventListener("seeked", onSeeked)
        if (cancelled) return
        const canvas = document.createElement("canvas")
        canvas.width = 160
        canvas.height = 90
        const ctx = canvas.getContext("2d")
        if (ctx) {
          ctx.drawImage(v, 0, 0, canvas.width, canvas.height)
          onThumb(frame, canvas.toDataURL("image/jpeg", 0.6))
        }
        i += 1
        setTimeout(grab, 0)
      }
      v.addEventListener("seeked", onSeeked)
      v.currentTime = time
    }
    grab()
  }

  v.addEventListener("loadedmetadata", grabAll)
  return () => {
    cancelled = true
    v.pause()
    v.removeEventListener("loadedmetadata", grabAll)
  }
}
