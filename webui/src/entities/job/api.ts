import { api } from "@/shared/api/client"
import type { Job, MediaProbe } from "./types"

export const listJobs = () => api<{ jobs: Job[] }>("/api/poll").then((r) => r.jobs)
export const getJob = (id: string) => api<Job>(`/api/jobs/${id}`)
export const cancelJob = (id: string) => api<Job>(`/api/jobs/${id}/cancel`, { method: "POST" })
export const retryJob = (id: string) => api<Job>(`/api/jobs/${id}/retry`, { method: "POST" })
export const deleteJob = (id: string) => api<{ ok: boolean }>(`/api/jobs/${id}`, { method: "DELETE" })
export const probeJob = (id: string) => api<MediaProbe>(`/api/jobs/${id}/probe`)
export const fetchJobReport = (id: string) => api<unknown>(`/api/jobs/${id}/report`)

export const saveJobTracks = (id: string, tracks: unknown[]) =>
  api<{ ok: boolean; id: string; tracks: number }>(`/api/jobs/${id}/report/tracks`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tracks }),
  })

export type SubmitJobFields = {
  kind: "run" | "preview" | "prompt"
  source_id?: string
  prompt?: string
  targets?: unknown[]
  tracks?: unknown[]
  masks?: number[]
  mode?: "parse" | "detect"
  start?: number
  count?: number
  stride?: number
  indices?: number[]
  all?: boolean
  params?: Record<string, string | number | boolean>
  video?: File
}

export function submitJob(fields: SubmitJobFields): Promise<Job> {
  const form = new FormData()
  if (fields.video) form.append("video", fields.video)
  form.append("kind", fields.kind)
  if (fields.source_id) form.append("source_id", fields.source_id)
  if (fields.prompt) form.append("prompt", fields.prompt)
  if (fields.targets) form.append("targets", JSON.stringify(fields.targets))
  if (fields.tracks) form.append("tracks", JSON.stringify(fields.tracks))
  if (fields.masks && fields.masks.length > 0) form.append("masks", fields.masks.join(","))
  if (fields.mode) form.append("mode", fields.mode)
  if (fields.start !== undefined) form.append("start", String(fields.start))
  if (fields.count !== undefined) form.append("count", String(fields.count))
  if (fields.stride !== undefined) form.append("stride", String(fields.stride))
  if (fields.indices) form.append("indices", fields.indices.join(","))
  if (fields.all) form.append("all", "1")
  for (const [k, v] of Object.entries(fields.params ?? {})) {
    if (v === "" || v === undefined || v === null) continue
    form.append(k, String(v))
  }
  return api<Job>("/api/jobs", { method: "POST", body: form })
}

/** Download a completed job artifact via fetch (keeps same-origin cookies/auth). */
export async function downloadJobOutput(url: string, filename: string): Promise<void> {
  const res = await fetch(url)
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`
    try {
      const data = await res.json()
      if (data?.detail) message = String(data.detail)
    } catch {
      // keep default
    }
    throw new ApiError(res.status, message)
  }
  const blob = await res.blob()
  const objectUrl = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = objectUrl
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(objectUrl)
}

export function outputDownloadName(fmt: string, jobId: string): string {
  const key = (fmt || "mp4").toLowerCase()
  if (key === "hls-fmp4" || key === "hls-ts" || key === "dash" || key === "default") {
    const stem = key === "default" ? "cleaned" : key
    return `${stem}-${jobId.slice(0, 8)}.zip`
  }
  return `cleaned-${jobId.slice(0, 8)}.${key}`
}

export async function pollJobToCompletion(
  id: string,
  onProgress?: (j: Job) => void,
  intervalMs = 1000,
  maxSeconds = 1800,
): Promise<Job> {
  const deadline = Date.now() + maxSeconds * 1000
  for (;;) {
    let job: Job
    try {
      job = await getJob(id)
    } catch (e) {
      if (Date.now() >= deadline) throw e
      await sleep(intervalMs)
      continue
    }
    onProgress?.(job)
    if (job.state === "COMPLETED") return job
    if (job.state === "FAILED" || job.state === "CANCELLED") {
      throw new Error(job.error || `job ${id} ${job.state.toLowerCase()}`)
    }
    if (Date.now() >= deadline) throw new Error(`job ${id} poll timed out`)
    await sleep(intervalMs)
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}
