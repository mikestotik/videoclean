export type JobState = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED"

export type JobKind = "run" | "preview" | "prompt"

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
  output_url: string | null
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
