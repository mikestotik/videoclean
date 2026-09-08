export function formatTimecode(frame: number, fps: number): string {
  const fpsRounded = Math.round(fps)
  const ff = ((frame % fpsRounded) + fpsRounded) % fpsRounded
  const totalSeconds = Math.floor(frame / fpsRounded)
  const ss = totalSeconds % 60
  const mm = Math.floor(totalSeconds / 60) % 60
  const hh = Math.floor(totalSeconds / 3600)
  const p = (n: number) => String(n).padStart(2, "0")
  return `${p(hh)}:${p(mm)}:${p(ss)}:${p(ff)}`
}

export function formatEta(seconds: number): string {
  if (seconds <= 0) return ""
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  if (m === 0) return `${s}с`
  return `${m}м ${s}с`
}

export function formatBytes(n: number): string {
  if (n <= 0) return "0 Б"
  const units = ["Б", "КБ", "МБ", "ГБ", "ТБ"]
  const i = Math.min(Math.floor(Math.log(n) / Math.log(1024)), units.length - 1)
  const value = n / 1024 ** i
  const rounded = i === 0 ? Math.round(value) : Math.round(value * 10) / 10
  return `${rounded} ${units[i]}`
}
