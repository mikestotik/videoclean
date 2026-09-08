import { api } from "@/shared/api/client"
import type { Source } from "./types"

export const listSources = () => api<Source[]>("/api/sources")

export const uploadSource = (video: File) => {
  const form = new FormData()
  form.append("video", video)
  return api<Source>("/api/sources", { method: "POST", body: form })
}

export const deleteSource = (sourceId: string) =>
  api<{ ok: boolean; id: string }>(`/api/sources/${sourceId}`, { method: "DELETE" })

export const videoUrl = (source: Source) => source.video_url

export const frameUrl = (source: Source, frame: number) =>
  `/api/sources/${source.id}/frames/${frame}`
