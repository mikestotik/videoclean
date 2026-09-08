export type JobState = "QUEUED" | "RUNNING" | "COMPLETED" | "FAILED" | "CANCELLED"

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
  kind: "run" | "preview"
  has_output: boolean
  has_input: boolean
  output_url: string | null
  input_url: string | null
  status_url: string
}

export type MediaProbe = {
  fps: number
  duration_s: number
  width: number
  height: number
  frame_count: number
}
