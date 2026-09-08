export type Tool = "brush" | "eraser"

export type Stroke = {
  tool: Tool
  size: number
  points: [number, number][]
}

export type AnnotatedFrame = {
  frame: number
  url: string
  strokes: Stroke[]
  updatedAt: string
}

export function parseStrokes(value: unknown): Stroke[] {
  if (!Array.isArray(value)) return []
  const strokes: Stroke[] = []
  for (const item of value) {
    const r = (item ?? {}) as Record<string, unknown>
    const tool = r.tool === "eraser" ? "eraser" : "brush"
    const size = Number(r.size ?? 0)
    const pointsRaw = Array.isArray(r.points) ? r.points : []
    const points: [number, number][] = []
    for (const p of pointsRaw) {
      if (Array.isArray(p) && p.length >= 2) points.push([Number(p[0]), Number(p[1])])
    }
    if (points.length > 0 && size > 0) strokes.push({ tool, size, points })
  }
  return strokes
}
