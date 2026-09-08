export function frameTimes(durationS: number, count: number): number[] {
  const n = Math.max(1, count)
  const d = Math.max(0, durationS)
  if (n === 1) return [0]
  const step = d / (n - 1)
  return Array.from({ length: n }, (_, i) => Math.min(d, i * step))
}

export function timeToFrameIndex(t: number, fps: number, maxFrame: number): number {
  const idx = Math.round(t * Math.max(fps, 0.001))
  return Math.min(Math.max(0, maxFrame), Math.max(0, idx))
}

export function toggleSelect(sel: Set<number>, idx: number): Set<number> {
  const next = new Set(sel)
  if (next.has(idx)) next.delete(idx)
  else next.add(idx)
  return next
}

export function rangeSelect(sel: Set<number>, idx: number): Set<number> {
  const next = new Set(sel)
  const anchor = [...sel].sort((a, b) => a - b).at(-1) ?? idx
  const [from, to] = anchor <= idx ? [anchor, idx] : [idx, anchor]
  for (let i = from; i <= to; i++) next.add(i)
  return next
}
