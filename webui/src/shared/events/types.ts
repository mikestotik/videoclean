import type { Job } from "@/entities/job"
import type { Source } from "@/entities/source"

export type PollSnapshot = {
  jobs: Job[]
  sources?: Source[]
  downloads?: unknown[]
  models: Record<string, unknown[]>
  doctor: Record<string, string>
  ollama: { ok: boolean; base_url: string; models: string[] }
  providers?: unknown[]
  options?: {
    families?: Record<string, unknown[]>
    providers?: unknown[]
    llm_model_options?: {
      id: string
      title: string
      model: string
      provider_id: string
      ready: boolean
      base_url?: string
    }[]
  }
  device?: string
}

export type EventsStatus = "connecting" | "live" | "reconnecting" | "idle"
