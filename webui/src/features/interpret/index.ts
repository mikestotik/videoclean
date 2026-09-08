import { useCallback, useRef, useState } from "react"
import { fetchJobReport, pollJobToCompletion, submitJob } from "@/entities/job"
import type { Source } from "@/entities/source"

export type InterpretTarget = { kind: string; query: string; where: string | null; motion?: string }

export type InterpretResult = { prompt: string; targets: InterpretTarget[] }

type ReportBody = { prompt?: unknown; targets?: unknown }

function parseReportTargets(value: unknown): InterpretTarget[] {
  if (!Array.isArray(value)) return []
  const targets: InterpretTarget[] = []
  for (const item of value) {
    const r = (item ?? {}) as Record<string, unknown>
    const query = String(r.query ?? "").trim()
    if (!query) continue
    targets.push({
      kind: String(r.kind ?? "object"),
      query,
      where: r.where ? String(r.where) : null,
      motion: r.motion === undefined ? undefined : String(r.motion),
    })
  }
  return targets
}

export function useInterpret(source: Source | null) {
  const [running, setRunning] = useState(false)
  const [error, setError] = useState("")
  const [result, setResult] = useState<InterpretResult | null>(null)
  const runIdRef = useRef(0)

  const run = useCallback(
    async (prompt: string) => {
      if (!source || running) return
      const runId = ++runIdRef.current
      setRunning(true)
      setError("")
      setResult(null)
      try {
        const job = await submitJob({ kind: "prompt", source_id: source.id, prompt })
        await pollJobToCompletion(job.id, undefined, 1000, 600)
        if (runId !== runIdRef.current) return
        const report = (await fetchJobReport(job.id)) as ReportBody
        if (runId !== runIdRef.current) return
        setResult({
          prompt: String(report.prompt ?? prompt),
          targets: parseReportTargets(report.targets),
        })
      } catch (e) {
        if (runId === runIdRef.current) setError(e instanceof Error ? e.message : String(e))
      } finally {
        if (runId === runIdRef.current) setRunning(false)
      }
    },
    [running, source],
  )

  return { run, running, error, result }
}
