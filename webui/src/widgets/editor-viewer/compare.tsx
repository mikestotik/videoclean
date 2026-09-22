import { useEffect, useRef, useState } from "react"
import { Pause, Play } from "lucide-react"
import { timeToFrameIndex } from "@/entities/frame"
import { apiUrl } from "@/shared/api/client"
import { findCachedJob, useEventsOptional } from "@/shared/events"
import { Button } from "@/shared/ui/button"

type Props = {
  inputUrl: string
  jobId: string
  width: number
  height: number
  fps: number
  frameCount: number
  currentFrame: number
  onFrameChange: (frame: number) => void
}

export function Compare({
  inputUrl,
  jobId,
  width,
  height,
  fps,
  frameCount,
  currentFrame,
  onFrameChange,
}: Props) {
  const events = useEventsOptional()
  const live = events?.jobs.find((j) => j.id === jobId) ?? findCachedJob(jobId)
  const liveOutputUrl = live?.output_url ? apiUrl(live.output_url) : null
  const [loaded, setLoaded] = useState<{ jobId: string; url: string } | null>(null)
  const [pos, setPos] = useState(50)
  const [playing, setPlaying] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLVideoElement>(null)
  const outputRef = useRef<HTMLVideoElement>(null)
  const draggingRef = useRef(false)
  const playingRef = useRef(false)
  if (liveOutputUrl && (loaded?.jobId !== jobId || loaded.url !== liveOutputUrl)) {
    setLoaded({ jobId, url: liveOutputUrl })
  }
  const outputUrl = loaded?.jobId === jobId && loaded.url ? loaded.url : null

  useEffect(() => {
    const input = inputRef.current
    const output = outputRef.current
    if (!input || !output) return
    const syncTime = () => {
      if (Math.abs(output.currentTime - input.currentTime) > 0.05) {
        output.currentTime = input.currentTime
      }
    }
    const onPlay = () => {
      playingRef.current = true
      setPlaying(true)
      void output.play().catch(() => {})
    }
    const onPause = () => {
      playingRef.current = false
      setPlaying(false)
      output.pause()
      syncTime()
    }
    const onTimeUpdate = () => {
      syncTime()
      if (!playingRef.current) return
      onFrameChange(timeToFrameIndex(input.currentTime, fps, Math.max(0, frameCount - 1)))
    }
    input.addEventListener("play", onPlay)
    input.addEventListener("pause", onPause)
    input.addEventListener("seeked", syncTime)
    input.addEventListener("timeupdate", onTimeUpdate)
    return () => {
      input.removeEventListener("play", onPlay)
      input.removeEventListener("pause", onPause)
      input.removeEventListener("seeked", syncTime)
      input.removeEventListener("timeupdate", onTimeUpdate)
    }
  }, [outputUrl, fps, frameCount, onFrameChange])

  // Timeline / frame state → both preview videos
  useEffect(() => {
    const input = inputRef.current
    const output = outputRef.current
    if (!input) return
    if (playingRef.current && !input.paused) return
    if (!input.paused) input.pause()
    const target = currentFrame / Math.max(fps, 0.001)
    if (Math.abs(input.currentTime - target) > 0.5 / Math.max(fps, 0.001)) {
      try {
        input.currentTime = target
      } catch {
        // ignore seek before metadata
      }
    }
    if (output && Math.abs(output.currentTime - target) > 0.5 / Math.max(fps, 0.001)) {
      try {
        output.currentTime = target
      } catch {
        // ignore
      }
    }
  }, [currentFrame, fps])

  const updatePos = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return
    setPos(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)))
  }

  const togglePlay = () => {
    const input = inputRef.current
    if (!input) return
    if (input.paused) void input.play().catch(() => {})
    else input.pause()
  }

  if (!outputUrl) {
    return (
      <div className="flex h-64 items-center justify-center rounded-md border text-sm text-muted-foreground">
        Результат ещё не готов
      </div>
    )
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden rounded-md border bg-black [container-type:size]">
        <div
          ref={containerRef}
          className="relative select-none overflow-hidden"
          style={{
            aspectRatio: `${width} / ${height}`,
            width: `min(100cqw, calc(100cqh * ${width} / ${height}))`,
            height: `min(100cqh, calc(100cqw * ${height} / ${width}))`,
          }}
          onPointerDown={(e) => {
            if (e.button !== 0) return
            draggingRef.current = true
            e.currentTarget.setPointerCapture(e.pointerId)
            updatePos(e.clientX)
          }}
          onPointerMove={(e) => {
            if (draggingRef.current) updatePos(e.clientX)
          }}
          onPointerUp={() => {
            draggingRef.current = false
          }}
          onPointerCancel={() => {
            draggingRef.current = false
          }}
        >
          <video
            ref={inputRef}
            src={inputUrl}
            playsInline
            preload="auto"
            className="pointer-events-none absolute inset-0 h-full w-full object-contain"
          />
          <video
            ref={outputRef}
            src={outputUrl}
            playsInline
            muted
            preload="auto"
            className="pointer-events-none absolute inset-0 h-full w-full object-contain"
            style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
          />
          <div
            className="pointer-events-none absolute inset-y-0 z-10 w-0.5 bg-background/70"
            style={{ left: `${pos}%` }}
          >
            <div className="absolute top-1/2 left-1/2 size-8 -translate-x-1/2 -translate-y-1/2 rounded-full bg-background/80" />
          </div>
          <span className="pointer-events-none absolute top-2 left-2 z-10 rounded bg-background/80 px-1.5 text-xs">
            После
          </span>
          <span className="pointer-events-none absolute top-2 right-2 z-10 rounded bg-background/80 px-1.5 text-xs">
            До
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2 rounded-lg border border-border/60 bg-card/60 px-2.5 py-2">
        <Button size="sm" variant="outline" onClick={togglePlay} aria-label={playing ? "Пауза" : "Пуск"}>
          {playing ? <Pause className="size-3.5" /> : <Play className="size-3.5" />}
          {playing ? "Пауза" : "Пуск"}
        </Button>
      </div>
    </div>
  )
}
