import { api } from "@/shared/api/client"
import type { Job, MediaProbe } from "./types"

export const listJobs = () => api<{ jobs: Job[] }>("/api/poll").then((r) => r.jobs)
export const getJob = (id: string) => api<Job>(`/api/jobs/${id}`)
export const cancelJob = (id: string) => api<Job>(`/api/jobs/${id}/cancel`, { method: "POST" })
export const retryJob = (id: string) => api<Job>(`/api/jobs/${id}/retry`, { method: "POST" })
export const deleteJob = (id: string) => api<{ ok: boolean }>(`/api/jobs/${id}`, { method: "DELETE" })
export const probeJob = (id: string) => api<MediaProbe>(`/api/jobs/${id}/probe`)

export function submitRun(video: File, prompt: string, fields: Record<string, string>): Promise<Job> {
  const form = new FormData()
  form.append("video", video)
  form.append("prompt", prompt)
  for (const [k, v] of Object.entries(fields)) if (v) form.append(k, v)
  return api<Job>("/api/jobs", { method: "POST", body: form })
}
