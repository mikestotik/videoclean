import { useCallback, useRef, useState } from "react"
import { pollJobToCompletion, submitJob } from "@/entities/job"
import type { Source } from "@/entities/source"

export type InpaintMode = "tracks" | "masks" | "prompt"

export type InpaintPayload = {
  mode: InpaintMode
  prompt?: string
  targets?: unknown[]
  tracks?: unknown[]
  masks?: number[]
}

export type InpaintProgress = { fraction: number; detail: string; eta: string }

export function useInpaintRun(source: Source | null) {
  const [running, setRunning] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<InpaintProgress>({ fraction: 0, detail: "", eta: "" })
  const [error, setError] = useState("")
  const runIdRef = useRef(0)

  const run = useCallback(
    async (payload: InpaintPayload, params: Record<string, string | number | boolean> = {}) => {
      if (!source || running) return null
      const runId = ++runIdRef.current
      setRunning(true)
      setError("")
      setProgress({ fraction: 0, detail: "", eta: "" })
      try {
        const job = await submitJob({
          kind: "run",
          source_id: source.id,
          prompt: payload.prompt,
          targets: payload.mode === "tracks" ? undefined : payload.targets,
          tracks: payload.mode === "tracks" ? payload.tracks : undefined,
          masks: payload.masks,
          params,
        })
        if (runId === runIdRef.current) setJobId(job.id)
        const done = await pollJobToCompletion(job.id, (j) => {
          if (runId === runIdRef.current) {
            setProgress({ fraction: j.fraction, detail: j.detail, eta: j.eta })
          }
        })
        if (runId !== runIdRef.current) return null
        return done.id
      } catch (e) {
        if (runId === runIdRef.current) setError(e instanceof Error ? e.message : String(e))
        return null
      } finally {
        if (runId === runIdRef.current) {
          setRunning(false)
          setJobId(null)
        }
      }
    },
    [running, source],
  )

  return { run, running, jobId, progress, error }
}
