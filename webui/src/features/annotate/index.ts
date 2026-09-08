import { useCallback, useEffect, useRef, useState } from "react"
import {
  canvasToPngBlob,
  deleteMask,
  fetchAnnotations,
  putMask,
  renderStrokesToCanvas,
} from "@/entities/annotation"
import type { Source } from "@/entities/source"
import type { Stroke, Tool } from "@/entities/annotation"

const AUTOSAVE_DEBOUNCE_MS = 600

export type VideoSize = { width: number; height: number }

export function useAnnotate(source: Source | null, videoSize: VideoSize | null) {
  const [strokesByFrame, setStrokesByFrame] = useState<Record<number, Stroke[]>>({})
  const [activeFrame, setActiveFrame] = useState(0)
  const [tool, setTool] = useState<Tool>("brush")
  const [size, setSize] = useState(24)
  const [pendingFrame, setPendingFrame] = useState<number | null>(null)
  const strokesRef = useRef<Record<number, Stroke[]>>({})
  const drawingRef = useRef(false)

  const [prevSourceId, setPrevSourceId] = useState(source?.id)
  if (prevSourceId !== source?.id) {
    setPrevSourceId(source?.id)
    setStrokesByFrame({})
    setPendingFrame(null)
  }

  useEffect(() => {
    strokesRef.current = strokesByFrame
  })

  useEffect(() => {
    if (!source) return
    let alive = true
    fetchAnnotations(source.id)
      .then(({ frames }) => {
        if (!alive) return
        const restored: Record<number, Stroke[]> = {}
        for (const f of frames) {
          if (f.strokes.length > 0) restored[f.frame] = f.strokes
        }
        if (Object.keys(restored).length > 0) {
          strokesRef.current = restored
          setStrokesByFrame(restored)
        }
      })
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [source])

  const saveFrameMask = useCallback(
    async (frame: number, strokes: Stroke[]) => {
      if (!source || !videoSize || videoSize.width <= 0 || videoSize.height <= 0) return
      if (strokes.length === 0) {
        await deleteMask(source.id, frame).catch(() => {})
        return
      }
      const canvas = renderStrokesToCanvas(strokes, videoSize.width, videoSize.height)
      const png = await canvasToPngBlob(canvas)
      await putMask(source.id, frame, png, strokes)
    },
    [source, videoSize],
  )

  useEffect(() => {
    if (pendingFrame === null) return
    const frame = pendingFrame
    const timer = setTimeout(() => {
      setPendingFrame(null)
      saveFrameMask(frame, strokesRef.current[frame] ?? []).catch(() => {})
    }, AUTOSAVE_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [pendingFrame, saveFrameMask])

  const setStrokes = useCallback((frame: number, updater: (prev: Stroke[]) => Stroke[]) => {
    const next = updater(strokesRef.current[frame] ?? [])
    strokesRef.current = { ...strokesRef.current, [frame]: next }
    setStrokesByFrame(strokesRef.current)
    setPendingFrame(frame)
  }, [])

  const beginStroke = useCallback(
    (pt: [number, number]) => {
      drawingRef.current = true
      setStrokes(activeFrame, (prev) => [...prev, { tool, size, points: [pt] }])
    },
    [activeFrame, setStrokes, size, tool],
  )

  const extendStroke = useCallback(
    (pt: [number, number]) => {
      if (!drawingRef.current) return
      setStrokes(activeFrame, (prev) => {
        if (prev.length === 0) return prev
        const last = prev[prev.length - 1]
        return [...prev.slice(0, -1), { ...last, points: [...last.points, pt] }]
      })
    },
    [activeFrame, setStrokes],
  )

  const endStroke = useCallback(() => {
    drawingRef.current = false
  }, [])

  const undo = useCallback(() => {
    setStrokes(activeFrame, (prev) => prev.slice(0, -1))
  }, [activeFrame, setStrokes])

  const clearFrame = useCallback(() => {
    setStrokes(activeFrame, () => [])
  }, [activeFrame, setStrokes])

  return {
    strokesByFrame,
    activeFrame,
    setActiveFrame,
    tool,
    setTool,
    size,
    setSize,
    beginStroke,
    extendStroke,
    endStroke,
    undo,
    clearFrame,
    isDirty: pendingFrame !== null,
  }
}
