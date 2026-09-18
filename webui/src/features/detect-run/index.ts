import { useCallback, useRef, useState } from "react"
import { fetchJobReport, pollJobToCompletion, submitJob } from "@/entities/job"
import { fetchPreviewManifest, type PreviewManifest } from "@/entities/preview"
import type { Source } from "@/entities/source"

export type DetectTarget = { kind: string; query: string; where?: string | null }

export type DetectBox = [number, number, number, number]

export type DetectTrack = {
  id: number
  label: string
  motion?: string
  boxes: (DetectBox | null)[]
}

type ReportBody = { tracks?: unknown }

export function parseTracks(value: unknown): DetectTrack[] {
  if (!Array.isArray(value)) return []
  const tracks: DetectTrack[] = []
  for (const item of value) {
    const r = (item ?? {}) as Record<string, unknown>
    const boxesRaw = Array.isArray(r.boxes) ? r.boxes : []
    const boxes: (DetectBox | null)[] = []
    for (const b of boxesRaw) {
      if (Array.isArray(b) && b.length >= 4) {
        boxes.push([Number(b[0]), Number(b[1]), Number(b[2]), Number(b[3])])
      } else {
        boxes.push(null)
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

export function tracksAreFullLength(tracks: DetectTrack[], frameCount: number): boolean {
  if (frameCount <= 0 || tracks.length === 0) return false
  return tracks.every(
    (t) => t.boxes.length === frameCount && t.boxes.some((b) => b != null),
  )
}

export type DetectRunPayload = {
  mode: "parse" | "detect"
  prompt?: string
  targets?: DetectTarget[]
  stride?: number
  all?: boolean
  params?: Record<string, string | number | boolean>
}

export function useDetectRun(source: Source | null) {
  const [running, setRunning] = useState(false)
  const [jobId, setJobId] = useState<string | null>(null)
  const [lastJobId, setLastJobId] = useState<string | null>(null)
  const [progress, setProgress] = useState<{ fraction: number; detail: string; eta: string }>({
    fraction: 0,
    detail: "",
    eta: "",
  })
  const [error, setError] = useState("")
  const [manifest, setManifest] = useState<PreviewManifest | null>(null)
  const [tracks, setTracks] = useState<DetectTrack[]>([])
  const [excludedIds, setExcludedIds] = useState<number[]>([])
  const [selectedTrackId, setSelectedTrackId] = useState<number | null>(null)
  const runIdRef = useRef(0)

  const [prevSourceId, setPrevSourceId] = useState(source?.id)
  if (prevSourceId !== source?.id) {
    setPrevSourceId(source?.id)
    setJobId(null)
    setLastJobId(null)
    setManifest(null)
    setTracks([])
    setExcludedIds([])
    setSelectedTrackId(null)
    setError("")
  }

  const run = useCallback(
    async (payload: DetectRunPayload): Promise<DetectTrack[] | null> => {
      if (!source || running) return null
      const runId = ++runIdRef.current
      setRunning(true)
      setError("")
      setManifest(null)
      setTracks([])
      setExcludedIds([])
      setSelectedTrackId(null)
      setProgress({ fraction: 0, detail: "", eta: "" })
      try {
        const stride = payload.all ? undefined : payload.stride
        const count =
          stride && source.probe.frame_count > 0
            ? Math.ceil(source.probe.frame_count / stride)
            : undefined
        const job = await submitJob({
          kind: "preview",
          source_id: source.id,
          mode: payload.mode,
          prompt: payload.prompt,
          targets: payload.targets,
          stride,
          count,
          all: payload.all,
          params: payload.params,
        })
        if (runId === runIdRef.current) setJobId(job.id)
        const done = await pollJobToCompletion(job.id, (j) => {
          if (runId === runIdRef.current) {
            setProgress({ fraction: j.fraction, detail: j.detail, eta: j.eta })
          }
        })
        if (runId !== runIdRef.current) return null
        const [m, report] = await Promise.all([
          fetchPreviewManifest(done.id),
          fetchJobReport(done.id).catch(() => null),
        ])
        if (runId !== runIdRef.current) return null
        const parsed = parseTracks((report as ReportBody | null)?.tracks)
        setManifest(m)
        setTracks(parsed)
        setExcludedIds([])
        setSelectedTrackId(parsed[0]?.id ?? null)
        setLastJobId(done.id)
        return parsed
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

  const loadFromJob = useCallback(async (jobId: string) => {
    setError("")
    try {
      const [m, report] = await Promise.all([
        fetchPreviewManifest(jobId),
        fetchJobReport(jobId).catch(() => null),
      ])
      const parsed = parseTracks((report as ReportBody | null)?.tracks)
      setManifest(m)
      setTracks(parsed)
      setExcludedIds([])
      setSelectedTrackId(parsed[0]?.id ?? null)
      setLastJobId(jobId)
      setProgress({ fraction: 1, detail: "", eta: "" })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  const toggleTrack = useCallback((id: number) => {
    setExcludedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
  }, [])

  const patchBox = useCallback((id: number, frame: number, box: DetectBox) => {
    setTracks((prev) =>
      prev.map((t) => {
        if (t.id !== id || frame < 0 || frame >= t.boxes.length) return t
        const boxes = t.boxes.slice()
        boxes[frame] = box
        return { ...t, boxes }
      }),
    )
  }, [])

  const enabledTracks = tracks.filter((t) => !excludedIds.includes(t.id))

  return {
    run,
    running,
    jobId,
    lastJobId,
    progress,
    error,
    manifest,
    tracks,
    enabledTracks,
    excludedIds,
    selectedTrackId,
    setSelectedTrackId,
    toggleTrack,
    patchBox,
    loadFromJob,
  }
}
