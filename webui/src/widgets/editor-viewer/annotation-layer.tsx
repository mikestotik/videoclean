import { useCallback, useEffect, useMemo, useRef } from "react"
import type { useAnnotate } from "@/features/annotate"

type Props = {
  width: number
  height: number
  frame: number
  annotate: ReturnType<typeof useAnnotate>
}

export function AnnotationLayer({ width, height, frame, annotate }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rafRef = useRef(0)
  const drawingRef = useRef(false)
  const strokes = useMemo(() => annotate.strokesByFrame[frame] ?? [], [annotate.strokesByFrame, frame])

  const draw = useCallback(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext("2d")
    if (!canvas || !ctx) return
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.lineCap = "round"
    ctx.lineJoin = "round"
    for (const stroke of strokes) {
      if (stroke.points.length === 0) continue
      ctx.globalCompositeOperation = stroke.tool === "eraser" ? "destination-out" : "source-over"
      ctx.strokeStyle = "#ffffff"
      ctx.fillStyle = "#ffffff"
      ctx.lineWidth = stroke.size
      if (stroke.points.length === 1) {
        const [x, y] = stroke.points[0]
        ctx.beginPath()
        ctx.arc(x, y, stroke.size / 2, 0, Math.PI * 2)
        ctx.fill()
        continue
      }
      ctx.beginPath()
      ctx.moveTo(stroke.points[0][0], stroke.points[0][1])
      for (const [x, y] of stroke.points.slice(1)) ctx.lineTo(x, y)
      ctx.stroke()
    }
    ctx.globalCompositeOperation = "source-over"
  }, [strokes])

  useEffect(() => {
    cancelAnimationFrame(rafRef.current)
    rafRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(rafRef.current)
  }, [draw])

  const toVideo = (e: React.PointerEvent<HTMLCanvasElement>): [number, number] => {
    const rect = e.currentTarget.getBoundingClientRect()
    return [
      ((e.clientX - rect.left) / rect.width) * width,
      ((e.clientY - rect.top) / rect.height) * height,
    ]
  }

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return
    e.currentTarget.setPointerCapture(e.pointerId)
    drawingRef.current = true
    annotate.beginStroke(toVideo(e))
  }

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (drawingRef.current) annotate.extendStroke(toVideo(e))
  }

  const onPointerUp = () => {
    if (!drawingRef.current) return
    drawingRef.current = false
    annotate.endStroke()
  }

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute inset-0 h-full w-full cursor-crosshair touch-none"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    />
  )
}
