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
  const [lastJobId, setLastJobId] = useState<string | null>(null)
  const runIdRef = useRef(0)

  const [prevSourceId, setPrevSourceId] = useState(source?.id)
  if (prevSourceId !== source?.id) {
    setPrevSourceId(source?.id)
    setResult(null)
    setLastJobId(null)
    setError("")
  }

  const run = useCallback(
    async (prompt: string, llmModel = ""): Promise<InterpretResult | null> => {
      if (!source || running) return null
      const runId = ++runIdRef.current
      setRunning(true)
      setError("")
      setResult(null)
      try {
        const job = await submitJob({
          kind: "prompt",
          source_id: source.id,
          prompt,
          params: { llm_model: llmModel },
        })
        await pollJobToCompletion(job.id, undefined, 1000, 600)
        if (runId !== runIdRef.current) return null
        const report = (await fetchJobReport(job.id)) as ReportBody
        if (runId !== runIdRef.current) return null
        const parsed: InterpretResult = {
          prompt: String(report.prompt ?? prompt),
          targets: parseReportTargets(report.targets),
        }
        setResult(parsed)
        setLastJobId(job.id)
        return parsed
      } catch (e) {
        if (runId === runIdRef.current) setError(e instanceof Error ? e.message : String(e))
        return null
      } finally {
        if (runId === runIdRef.current) setRunning(false)
      }
    },
    [running, source],
  )

  const loadFromJob = useCallback(async (jobId: string): Promise<InterpretResult | null> => {
    setError("")
    try {
      const report = (await fetchJobReport(jobId)) as ReportBody
      const parsed: InterpretResult = {
        prompt: String(report.prompt ?? ""),
        targets: parseReportTargets(report.targets),
      }
      setResult(parsed)
      setLastJobId(jobId)
      return parsed
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      return null
    }
  }, [])

  return { run, running, error, result, lastJobId, loadFromJob }
}
