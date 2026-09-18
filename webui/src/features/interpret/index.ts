import { useCallback, useRef, useState } from "react"
import { fetchJobReport, waitJobToCompletion, submitJob } from "@/entities/job"
import type { Source } from "@/entities/source"

export type InterpretTarget = { kind: string; query: string; where: string | null; motion?: string }

export type InterpretResult = {
  prompt: string
  targets: InterpretTarget[]
  parseMode?: string
  defaulted?: boolean
  visionFrameIndices?: number[]
  framesUsed?: number[]
}

type ReportBody = {
  prompt?: unknown
  targets?: unknown
  parseMode?: unknown
  defaulted?: unknown
  visionFrameIndices?: unknown
  framesUsed?: unknown
}

function parseFrameIndices(value: unknown): number[] | undefined {
  if (!Array.isArray(value)) return undefined
  const out = value.map((x) => Number(x)).filter((n) => Number.isFinite(n))
  return out.length ? out : undefined
}

function parseInterpretMeta(report: ReportBody): Pick<InterpretResult, "parseMode" | "defaulted" | "visionFrameIndices" | "framesUsed"> {
  return {
    parseMode: report.parseMode === undefined ? undefined : String(report.parseMode),
    defaulted: typeof report.defaulted === "boolean" ? report.defaulted : undefined,
    visionFrameIndices: parseFrameIndices(report.visionFrameIndices),
    framesUsed: parseFrameIndices(report.framesUsed),
  }
}

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
        await waitJobToCompletion(job.id, undefined, { maxSeconds: 600, seed: job })
        if (runId !== runIdRef.current) return null
        const report = (await fetchJobReport(job.id)) as ReportBody
        if (runId !== runIdRef.current) return null
        const parsed: InterpretResult = {
          prompt: String(report.prompt ?? prompt),
          targets: parseReportTargets(report.targets),
          ...parseInterpretMeta(report),
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
        ...parseInterpretMeta(report),
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
