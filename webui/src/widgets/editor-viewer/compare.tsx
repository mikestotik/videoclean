import { useEffect, useRef, useState } from "react"
import { getJob } from "@/entities/job"

type Props = {
  inputUrl: string
  jobId: string
  width: number
  height: number
}

export function Compare({ inputUrl, jobId, width, height }: Props) {
  const [loaded, setLoaded] = useState<{ jobId: string; url: string } | null>(null)
  const [pos, setPos] = useState(50)
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLVideoElement>(null)
  const outputRef = useRef<HTMLVideoElement>(null)
  const draggingRef = useRef(false)
  const outputUrl = loaded?.jobId === jobId && loaded.url ? loaded.url : null

  useEffect(() => {
    let alive = true
    getJob(jobId)
      .then((j) => {
        if (alive) setLoaded({ jobId, url: j.output_url ?? "" })
      })
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [jobId])

  useEffect(() => {
    const input = inputRef.current
    const output = outputRef.current
    if (!input || !output) return
    const syncTime = () => {
      if (Math.abs(output.currentTime - input.currentTime) > 0.05) {
        output.currentTime = input.currentTime
      }
    }
    const onPlay = () => void output.play().catch(() => {})
    const onPause = () => {
      output.pause()
      syncTime()
    }
    input.addEventListener("play", onPlay)
    input.addEventListener("pause", onPause)
    input.addEventListener("seeked", syncTime)
    input.addEventListener("timeupdate", syncTime)
    return () => {
      input.removeEventListener("play", onPlay)
      input.removeEventListener("pause", onPause)
      input.removeEventListener("seeked", syncTime)
      input.removeEventListener("timeupdate", syncTime)
    }
  }, [outputUrl])

  const updatePos = (clientX: number) => {
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect || rect.width === 0) return
    setPos(Math.min(100, Math.max(0, ((clientX - rect.left) / rect.width) * 100)))
  }

  if (!outputUrl) {
    return (
      <div className="flex h-64 items-center justify-center rounded-md border text-sm text-muted-foreground">
        Результат ещё не готов
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      className="relative shrink-0 select-none overflow-hidden rounded-md border bg-black"
      style={{ aspectRatio: `${width} / ${height}` }}
    >
      <video ref={inputRef} src={inputUrl} controls playsInline className="absolute inset-0 h-full w-full" />
      <video
        ref={outputRef}
        src={outputUrl}
        playsInline
        muted
        className="pointer-events-none absolute inset-0 h-full w-full"
        style={{ clipPath: `inset(0 ${100 - pos}% 0 0)` }}
      />
      <div className="absolute inset-y-0 z-10 w-0.5 bg-background/70" style={{ left: `${pos}%` }}>
        <div
          role="separator"
          aria-label="Сравнение до и после"
          className="absolute top-1/2 left-1/2 size-8 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize touch-none rounded-full bg-background/80"
          onPointerDown={(e) => {
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
        />
      </div>
      <span className="pointer-events-none absolute top-2 left-2 z-10 rounded bg-background/80 px-1.5 text-xs">
        После
      </span>
      <span className="pointer-events-none absolute top-2 right-2 z-10 rounded bg-background/80 px-1.5 text-xs">
        До
      </span>
    </div>
  )
}
