import type { Job } from "@/entities/job"
import { api } from "@/shared/api/client"
import { parsePreviewManifest } from "./types"

export const fetchPreviewManifest = (jobId: string) =>
  api<unknown>(`/api/jobs/${jobId}/preview/preview.json`).then(parsePreviewManifest)

export const previewArtifactUrl = (jobId: string, name: string) =>
  `/api/jobs/${jobId}/preview/${encodeURIComponent(name)}`

export type PreviewFromJobBody = {
  job_id: string
  prompt?: string
  mode?: "parse" | "detect"
  indices?: number[]
  targets?: { kind: string; query: string; where?: string | null }[]
  mask_dilate_px?: string
}

export const runPreviewFromJob = (body: PreviewFromJobBody) =>
  api<Job>("/api/preview/from-job", { method: "POST", body: JSON.stringify(body) })
