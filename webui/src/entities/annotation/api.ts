import { api } from "@/shared/api/client"
import { parseStrokes, type AnnotatedFrame, type Stroke } from "./types"

export const fetchAnnotations = (sourceId: string) =>
  api<{ frames: AnnotatedFrame[] }>(`/api/sources/${sourceId}/annotations`).then((r) => ({
    frames: (r.frames ?? []).map((f) => ({ ...f, strokes: parseStrokes(f.strokes) })),
  }))

export const putMask = (sourceId: string, frame: number, png: Blob, strokes: Stroke[]) => {
  const form = new FormData()
  form.append("mask", png, `${String(frame).padStart(6, "0")}.png`)
  form.append("strokes", JSON.stringify(strokes))
  return api<{ ok: boolean }>(`/api/sources/${sourceId}/masks/${frame}`, {
    method: "PUT",
    body: form,
  })
}

export const deleteMask = (sourceId: string, frame: number) =>
  api<{ ok: boolean }>(`/api/sources/${sourceId}/masks/${frame}`, { method: "DELETE" })

export const maskUrl = (sourceId: string, frame: number) =>
  `/api/sources/${sourceId}/masks/${frame}`
