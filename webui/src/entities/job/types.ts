export type JobState = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED"

export type JobKind = "run" | "preview" | "prompt" | "package"

export type Job = {
  id: string
  state: JobState
  prompt: string
  created_at: string
  updated_at: string
  error: string
  stage: string
  fraction: number
  detail: string
  eta: string
  kind: JobKind
  source_id: string | null
  source_name: string | null
  has_output: boolean
  has_input: boolean
  /** True when a completed cleanup job still has a mezzanine for on-demand convert. */
  can_package?: boolean
  parent_job_id?: string | null
  output_url: string | null
  /** fmt → download URL (mp4/… or zip for HLS/DASH packages). */
  outputs?: Record<string, string>
  input_url: string | null
  status_url: string
}

export type JobReport = {
  kind: JobKind
  prompt: string
  targets: { kind: string; query: string; where: string | null; motion?: string }[]
  tracks?: unknown[]
  [key: string]: unknown
}

export type MediaProbe = {
  fps: number
  duration_s: number
  width: number
  height: number
  frame_count: number
}
