import { useCallback, useRef, useState } from "react"
import { waitJobToCompletion, submitJob } from "@/entities/job"
import type { Source } from "@/entities/source"

export type InpaintMode = "tracks" | "masks" | "prompt"

export type InpaintPayload = {
  mode: InpaintMode
  prompt?: string
  targets?: unknown[]
  tracks?: unknown[]
  masks?: number[]
}

export type InpaintProgress = {
  fraction: number
  detail: string
  eta: string
  stage: string
  stageTitle: string
}

export function useInpaintRun(source: Source | null) {
  const [running, setRunning] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [lastJobId, setLastJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<InpaintProgress>({
    fraction: 0,
    detail: "",
    eta: "",
    stage: "",
    stageTitle: "",
  })
  const [error, setError] = useState("")
  const runIdRef = useRef(0)

  const [prevSourceId, setPrevSourceId] = useState(source?.id)
  if (prevSourceId !== source?.id) {
    setPrevSourceId(source?.id)
    setJobId(null)
    setLastJobId(null)
    setError("")
  }

  const run = useCallback(
    async (payload: InpaintPayload, params: Record<string, string | number | boolean> = {}) => {
      if (!source || running) return null
      const runId = ++runIdRef.current
      setRunning(true)
      setError("")
      setProgress({ fraction: 0, detail: "", eta: "", stage: "", stageTitle: "" })
      try {
        const job = await submitJob({
          kind: "run",
          source_id: source.id,
          prompt: payload.prompt,
          // Entry C: masks + targets together; tracks stay exclusive.
          targets: payload.mode === "tracks" ? undefined : payload.targets,
          tracks: payload.mode === "tracks" ? payload.tracks : undefined,
          masks: payload.mode === "tracks" ? undefined : payload.masks,
          params,
        })
        if (runId === runIdRef.current) setJobId(job.id)
        const done = await waitJobToCompletion(
          job.id,
          (j) => {
            if (runId === runIdRef.current) {
              setProgress({
                fraction: j.fraction,
                detail: j.detail,
                eta: j.eta,
                stage: j.stage,
                stageTitle: j.stageTitle ?? "",
              })
            }
          },
          { seed: job },
        )
        if (runId !== runIdRef.current) return null
        setLastJobId(done.id)
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

  const loadFromJob = useCallback((jobId: string) => {
    setLastJobId(jobId)
    setError("")
    setProgress({ fraction: 1, detail: "", eta: "", stage: "", stageTitle: "" })
  }, [])

  return { run, running, jobId, lastJobId, progress, error, loadFromJob }
}
