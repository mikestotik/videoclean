import { useCallback, useRef, useState } from "react"
import { fetchJobReport, waitJobToCompletion, saveJobTracks, submitJob } from "@/entities/job"
import { fetchPreviewManifest, type PreviewManifest } from "@/entities/preview"
import type { Source } from "@/entities/source"

export type DetectTarget = { kind: string; query: string; where?: string | null }

export type DetectBox = [number, number, number, number]

export type DetectTrack = {
  id: number
  label: string
  motion?: string
  boxes: (DetectBox | null)[]
  /** Manual (or restored) anchor frames into `boxes` indices. */
  keyframes: number[]
}

export type BoxEditMode = "frame" | "hold"

type ReportBody = { tracks?: unknown }

function sortedUnique(nums: number[]): number[] {
  return [...new Set(nums.filter((n) => Number.isFinite(n)))].sort((a, b) => a - b)
}

function parseKeyframes(raw: unknown, boxCount: number): number[] {
  if (!Array.isArray(raw)) return []
  return sortedUnique(
    raw.map((v) => Number(v)).filter((n) => Number.isInteger(n) && n >= 0 && n < boxCount),
  )
}

/** Hold `box` from `frame` until the next keyframe (exclusive), or end. */
export function applyBoxEdit(
  track: DetectTrack,
  frame: number,
  box: DetectBox,
  mode: BoxEditMode,
): DetectTrack {
  if (frame < 0 || frame >= track.boxes.length) return track
  const boxes = track.boxes.slice()
  const keyframes = new Set(track.keyframes)
  keyframes.add(frame)
  boxes[frame] = box
  if (mode === "hold") {
    const nextKey = [...keyframes].filter((k) => k > frame).sort((a, b) => a - b)[0]
    const end = nextKey ?? boxes.length
    for (let i = frame + 1; i < end; i++) boxes[i] = box
  }
  return { ...track, boxes, keyframes: sortedUnique([...keyframes]) }
}

/** Drop a key at `frame` and re-hold from the previous key until the next. */
export function clearTrackKey(track: DetectTrack, frame: number): DetectTrack {
  if (!track.keyframes.includes(frame)) return track
  const keyframes = track.keyframes.filter((k) => k !== frame)
  const prev = [...keyframes].filter((k) => k < frame).sort((a, b) => a - b).at(-1)
  if (prev === undefined) return { ...track, keyframes }
  const prevBox = track.boxes[prev]
  if (!prevBox) return { ...track, keyframes }
  const nextKey = keyframes.find((k) => k > frame)
  const end = nextKey ?? track.boxes.length
  const boxes = track.boxes.slice()
  for (let i = prev + 1; i < end; i++) boxes[i] = prevBox
  return { ...track, boxes, keyframes }
}

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
      keyframes: parseKeyframes(r.keyframes, boxes.length),
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

export function tracksToJson(tracks: DetectTrack[]) {
  return tracks.map((t) => ({
    id: t.id,
    label: t.label,
    motion: t.motion,
    boxes: t.boxes.map((b) => (b ? ([b[0], b[1], b[2], b[3]] as DetectBox) : null)),
    keyframes: t.keyframes.slice(),
  }))
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
  const [boxEditMode, setBoxEditMode] = useState<BoxEditMode>("hold")
  const [tracksDirty, setTracksDirty] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState("")
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
    setBoxEditMode("hold")
    setTracksDirty(false)
    setSaving(false)
    setSaveError("")
    setError("")
  }

  const run = useCallback(
    async (payload: DetectRunPayload): Promise<DetectTrack[] | null> => {
      if (!source || running) return null
      const runId = ++runIdRef.current
      setRunning(true)
      setError("")
      setSaveError("")
      setManifest(null)
      setTracks([])
      setExcludedIds([])
      setSelectedTrackId(null)
      setTracksDirty(false)
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
        const done = await waitJobToCompletion(
          job.id,
          (j) => {
            if (runId === runIdRef.current) {
              setProgress({ fraction: j.fraction, detail: j.detail, eta: j.eta })
            }
          },
          { seed: job },
        )
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
    setSaveError("")
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
      setTracksDirty(false)
      setProgress({ fraction: 1, detail: "", eta: "" })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [])

  const toggleTrack = useCallback((id: number) => {
    setExcludedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]))
  }, [])

  const patchBox = useCallback(
    (id: number, frame: number, box: DetectBox) => {
      setTracks((prev) =>
        prev.map((t) => (t.id === id ? applyBoxEdit(t, frame, box, boxEditMode) : t)),
      )
      setTracksDirty(true)
      setSaveError("")
    },
    [boxEditMode],
  )

  const clearKey = useCallback((id: number, frame: number) => {
    setTracks((prev) => prev.map((t) => (t.id === id ? clearTrackKey(t, frame) : t)))
    setTracksDirty(true)
    setSaveError("")
  }, [])

  const saveTracks = useCallback(async () => {
    if (!lastJobId || tracks.length === 0 || saving) return false
    setSaving(true)
    setSaveError("")
    try {
      await saveJobTracks(lastJobId, tracksToJson(tracks))
      setTracksDirty(false)
      return true
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : String(e))
      return false
    } finally {
      setSaving(false)
    }
  }, [lastJobId, tracks, saving])

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
    clearKey,
    boxEditMode,
    setBoxEditMode,
    tracksDirty,
    saving,
    saveError,
    saveTracks,
    loadFromJob,
  }
}
