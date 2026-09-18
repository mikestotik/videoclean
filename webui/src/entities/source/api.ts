import { api, apiUrl } from "@/shared/api/client"
import type { Source } from "./types"

export const listSources = () => api<Source[]>("/api/sources")

export const getSource = (sourceId: string) => api<Source>(`/api/sources/${sourceId}`)

export const uploadSource = (video: File) => {
  const form = new FormData()
  form.append("video", video)
  return api<Source>("/api/sources", { method: "POST", body: form })
}

export const deleteSource = (sourceId: string) =>
  api<{ ok: boolean; id: string }>(`/api/sources/${sourceId}`, { method: "DELETE" })

export type CropSourceParams = {
  start_s?: number | null
  end_s?: number | null
  left?: number
  right?: number
  top?: number
  bottom?: number
  name?: string
}

export const cropSource = (sourceId: string, params: CropSourceParams) =>
  api<Source>(`/api/sources/${sourceId}/crop`, {
    method: "POST",
    body: JSON.stringify(params),
  })

export const videoUrl = (source: Source) => apiUrl(source.video_url)

export const frameUrl = (source: Source, frame: number) =>
  apiUrl(`/api/sources/${source.id}/frames/${frame}`)
