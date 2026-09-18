export type PreviewFrame = {
  index: number
  maskCoverage: number
  boxes: (number[] | null)[]
  artifacts: { raw: string; boxes: string; mask: string }
}

export type PreviewManifest = {
  state: string
  mode?: string
  framesRequested: number[]
  frames: PreviewFrame[]
  meanMaskCoverage?: number
  targets?: { kind: string; query: string }[]
  parseMode?: string
  selectRelaxed?: boolean
  note?: string
  error?: string
}

export function parsePreviewManifest(data: unknown): PreviewManifest {
  const raw = (data ?? {}) as Record<string, unknown>
  const framesRaw = Array.isArray(raw.frames) ? raw.frames : []
  const frames: PreviewFrame[] = framesRaw.map((f) => {
    const fr = (f ?? {}) as Record<string, unknown>
    const index = Number(fr.index ?? 0)
    return {
      index,
      maskCoverage: Number(fr.maskCoverage ?? 0),
      boxes: Array.isArray(fr.boxes) ? (fr.boxes as (number[] | null)[]) : [],
      artifacts: {
        raw: `${String(index).padStart(6, "0")}_raw.jpg`,
        boxes: `${String(index).padStart(6, "0")}_boxes.jpg`,
        mask: `${String(index).padStart(6, "0")}_mask.jpg`,
      },
    }
  })
  return {
    state: String(raw.state ?? ""),
    mode: raw.mode === undefined ? undefined : String(raw.mode),
    framesRequested: Array.isArray(raw.framesRequested) ? (raw.framesRequested as number[]) : [],
    frames,
    meanMaskCoverage: raw.meanMaskCoverage === undefined ? undefined : Number(raw.meanMaskCoverage),
    targets: Array.isArray(raw.targets) ? (raw.targets as { kind: string; query: string }[]) : [],
    parseMode: raw.parseMode === undefined ? undefined : String(raw.parseMode),
    selectRelaxed: typeof raw.selectRelaxed === "boolean" ? raw.selectRelaxed : undefined,
    note: raw.note === undefined ? undefined : String(raw.note),
    error: raw.error === undefined ? undefined : String(raw.error),
  }
}
