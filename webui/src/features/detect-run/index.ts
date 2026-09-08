import { useCallback, useRef, useState } from "react"
import { fetchJobReport, pollJobToCompletion, submitJob } from "@/entities/job"
import { fetchPreviewManifest, type PreviewManifest } from "@/entities/preview"
import type { Source } from "@/entities/source"

export type DetectTarget = { kind: string; query: string; where?: string | null }

export type DetectTrack = {
  id: number
  label: string
  motion?: string
  boxes: [number, number, number, number][]
}

type ReportBody = { tracks?: unknown }

function parseTracks(value: unknown): DetectTrack[] {
  if (!Array.isArray(value)) return []
  const tracks: DetectTrack[] = []
  for (const item of value) {
    const r = (item ?? {}) as Record<string, unknown>
    const boxesRaw = Array.isArray(r.boxes) ? r.boxes : []
    const boxes: [number, number, number, number][] = []
    for (const b of boxesRaw) {
      if (Array.isArray(b) && b.length >= 4) {
        boxes.push([Number(b[0]), Number(b[1]), Number(b[2]), Number(b[3])])
      }
    }
    tracks.push({
      id: Number(r.id ?? tracks.length),
      label: String(r.label ?? ""),
      motion: r.motion === undefined ? undefined : String(r.motion),
      boxes,
    })
  }
  return tracks
}

export type DetectRunPayload = {
  mode: "parse" | "detect"
  prompt?: string
  targets?: DetectTarget[]
  stride?: number
  all?: boolean
}

export function useDetectRun(source: Source | null) {
  const [running, setRunning] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ fraction: number; detail: string; eta: string }>({
    fraction: 0,
    detail: "",
    eta: "",
  })
  const [error, setError] = useState("")
  const [manifest, setManifest] = useState<PreviewManifest | null>(null)
  const [tracks, setTracks] = useState<DetectTrack[]>([])
  const runIdRef = useRef(0)

  const run = useCallback(
    async (payload: DetectRunPayload) => {
      if (!source || running) return
      const runId = ++runIdRef.current
      setRunning(true)
      setError("")
      setManifest(null)
      setTracks([])
      setProgress({ fraction: 0, detail: "", eta: "" })
      try {
        const job = await submitJob({
          kind: "preview",
          source_id: source.id,
          mode: payload.mode,
          prompt: payload.prompt,
          targets: payload.targets,
          stride: payload.all ? undefined : payload.stride,
          all: payload.all,
        })
        if (runId === runIdRef.current) setJobId(job.id)
        const done = await pollJobToCompletion(job.id, (j) => {
          if (runId === runIdRef.current) {
            setProgress({ fraction: j.fraction, detail: j.detail, eta: j.eta })
          }
        })
        if (runId !== runIdRef.current) return
        const [m, report] = await Promise.all([
          fetchPreviewManifest(done.id),
          fetchJobReport(done.id).catch(() => null),
        ])
        if (runId !== runIdRef.current) return
        setManifest(m)
        setTracks(parseTracks((report as ReportBody | null)?.tracks))
      } catch (e) {
        if (runId === runIdRef.current) setError(e instanceof Error ? e.message : String(e))
      } finally {
        if (runId === runIdRef.current) {
          setRunning(false)
          setJobId(null)
        }
      }
    },
    [running, source],
  )

  return { run, running, jobId, progress, error, manifest, tracks }
}
