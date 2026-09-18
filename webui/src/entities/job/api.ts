import { api, apiUrl, ApiError } from "@/shared/api/client"
import { findCachedJob, subscribeJobs } from "@/shared/events/jobBus"
import type { Job, MediaProbe } from "./types"

export const listJobs = () => api<Job[]>("/api/jobs")
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
export type PackageJobFields = {
  formats: string[]
  webm_crf?: number
  segment_seconds?: number
  overwrite?: boolean
}

/** Queue on-demand packaging from a completed cleanup job's mezzanine. */
export function packageJob(jobId: string, fields: PackageJobFields): Promise<Job> {
  const form = new FormData()
  form.append("formats", fields.formats.join(","))
  if (fields.webm_crf !== undefined) form.append("webm_crf", String(fields.webm_crf))
  if (fields.segment_seconds !== undefined) {
    form.append("segment_seconds", String(fields.segment_seconds))
  }
  if (fields.overwrite !== undefined) form.append("overwrite", fields.overwrite ? "1" : "0")
  return api<Job>(`/api/jobs/${jobId}/package`, { method: "POST", body: form })
}

export async function downloadJobOutput(url: string, filename: string): Promise<void> {
  const res = await fetch(apiUrl(url))
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

/**
 * Wait until a job reaches a terminal state via SSE job-list pushes (jobBus).
 * No HTTP polling. Pass `seed` from submitJob to avoid a cold start.
 */
export async function waitJobToCompletion(
  id: string,
  onProgress?: (j: Job) => void,
  opts?: { maxSeconds?: number; seed?: Job },
): Promise<Job> {
  const maxSeconds = opts?.maxSeconds ?? 1800
  const deadlineMs = Math.max(0, maxSeconds) * 1000

  const settle = (job: Job): "done" | "fail" | "run" => {
    onProgress?.(job)
    if (job.state === "COMPLETED") return "done"
    if (job.state === "FAILED" || job.state === "CANCELLED") return "fail"
    return "run"
  }

  const seed = opts?.seed?.id === id ? opts.seed : undefined
  const initial = seed ?? findCachedJob(id)
  if (initial) {
    const s = settle(initial)
    if (s === "done") return initial
    if (s === "fail") throw new Error(initial.error || `job ${id} ${initial.state.toLowerCase()}`)
  }

  return new Promise<Job>((resolve, reject) => {
    let timeoutId: ReturnType<typeof setTimeout> | undefined
    let unsub = () => {}
    let settled = false

    const cleanup = () => {
      unsub()
      if (timeoutId !== undefined) clearTimeout(timeoutId)
    }
    const finishOk = (job: Job) => {
      if (settled) return
      settled = true
      cleanup()
      resolve(job)
    }
    const finishErr = (err: unknown) => {
      if (settled) return
      settled = true
      cleanup()
      reject(err instanceof Error ? err : new Error(String(err)))
    }

    const consider = (job: Job | undefined) => {
      if (!job) return
      const s = settle(job)
      if (s === "done") finishOk(job)
      else if (s === "fail") finishErr(new Error(job.error || `job ${id} ${job.state.toLowerCase()}`))
    }

    unsub = subscribeJobs((jobs) => {
      consider(jobs.find((j) => j.id === id) ?? findCachedJob(id))
    })

    timeoutId = setTimeout(() => {
      const cached = findCachedJob(id)
      if (cached) {
        const s = settle(cached)
        if (s === "done") {
          finishOk(cached)
          return
        }
        if (s === "fail") {
          finishErr(new Error(cached.error || `job ${id} ${cached.state.toLowerCase()}`))
          return
        }
      }
      finishErr(new Error(`job ${id} wait timed out (no SSE terminal state)`))
    }, deadlineMs)

    consider(findCachedJob(id) ?? seed)
  })
}
