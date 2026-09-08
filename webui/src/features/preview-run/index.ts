import { useCallback, useState } from "react"
import { getJob, type Job } from "@/entities/job"
import { fetchPreviewManifest, runPreviewFromJob } from "@/entities/preview"
import type { PreviewManifest } from "@/entities/preview"

export type PreviewTarget = { kind: string; query: string; where?: string | null }

export function framesToIndices(sel: Set<number>): number[] {
  return [...sel].sort((a, b) => a - b)
}

export function usePreviewRun(job: Job | null, selectedFrames: Set<number>, targets: PreviewTarget[]) {
  const [jobId, setJobId] = useState<string | null>(null)
  const [manifest, setManifest] = useState<PreviewManifest | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState("")

  const run = useCallback(
    async (mode: "parse" | "detect", prompt: string) => {
      if (!job || running) return
      const indices = framesToIndices(selectedFrames)
      if (indices.length === 0) {
        setError("Выберите кадры на филмстрипе")
        return
      }
      if (mode === "parse" && !prompt.trim()) {
        setError("Введите промпт для разбора")
        return
      }
      if (mode === "detect" && targets.length === 0) {
        setError("Добавьте таргеты для режима detect")
        return
      }
      setRunning(true)
      setError("")
      setManifest(null)
      try {
        const res = await runPreviewFromJob({
          job_id: job.id,
          prompt: prompt.trim(),
          mode,
          indices,
          targets: mode === "detect" ? targets : undefined,
        })
        setJobId(res.id)
        for (let i = 0; i < 600; i++) {
          await new Promise((r) => setTimeout(r, 1000))
          const status = await getJob(res.id)
          if (status.state === "COMPLETED") {
            setManifest(await fetchPreviewManifest(res.id))
            break
          }
          if (status.state === "FAILED" || status.state === "CANCELLED") {
            setError(status.error || status.state)
            break
          }
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e))
      } finally {
        setRunning(false)
      }
    },
    [job, running, selectedFrames, targets],
  )

  return { run, jobId, manifest, running, error }
}
