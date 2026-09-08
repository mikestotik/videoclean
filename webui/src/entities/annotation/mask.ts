import type { Stroke } from "./types"

export function renderStrokesToCanvas(strokes: Stroke[], w: number, h: number): HTMLCanvasElement {
  const canvas = document.createElement("canvas")
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext("2d")
  if (!ctx) return canvas
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
  return canvas
}

export function canvasToPngBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob)
      else reject(new Error("canvas toBlob failed"))
    }, "image/png")
  })
}

export function imageToCanvas(url: string): Promise<HTMLCanvasElement> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement("canvas")
      canvas.width = img.naturalWidth
      canvas.height = img.naturalHeight
      const ctx = canvas.getContext("2d")
      if (!ctx) {
        reject(new Error("canvas 2d context unavailable"))
        return
      }
      ctx.drawImage(img, 0, 0)
      resolve(canvas)
    }
    img.onerror = () => reject(new Error(`failed to load image ${url}`))
    img.src = url
  })
}
