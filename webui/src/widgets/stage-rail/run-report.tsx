import { useEffect, useState } from "react"
import { fetchJobReport } from "@/entities/job"

const TIMING_ROWS: [string, string][] = [
  ["load", "Загрузка"],
  ["decode", "Кадры"],
  ["parse", "Разбор фразы"],
  ["detect", "Поиск"],
  ["track", "Трекинг"],
  ["segment", "Сегментация"],
  ["inpaint", "Заливка"],
  ["verify", "Проверка"],
  ["encode", "Сборка"],
  ["package", "Форматы"],
]

function secondsLabel(value: number): string {
  if (value < 10) return `${value.toFixed(1)} с`
  if (value < 60) return `${Math.round(value)} с`
  const minutes = Math.floor(value / 60)
  const seconds = Math.round(value % 60)
  return `${minutes} мин ${seconds.toString().padStart(2, "0")} с`
}

export function RunReport({ jobId }: { jobId: string }) {
  const [report, setReport] = useState<Record<string, unknown> | null>(null)
  const [failed, setFailed] = useState(false)

  useEffect(() => {
    let alive = true
    fetchJobReport(jobId)
      .then((body) => {
        if (alive) setReport((body ?? null) as Record<string, unknown> | null)
      })
      .catch(() => {
        if (alive) setFailed(true)
      })
    return () => {
      alive = false
    }
  }, [jobId])

  if (failed) return <p className="text-xs text-destructive">Отчёт не прочитался.</p>
  if (!report) return <p className="text-xs text-muted-foreground">Читаю отчёт…</p>

  const timings = (report.timings ?? {}) as Record<string, unknown>
  const rows = TIMING_ROWS.map(([key, label]) => {
    const raw = timings[key]
    return { key, label, seconds: typeof raw === "number" && Number.isFinite(raw) ? raw : null }
  })
  const total = rows.reduce((sum, row) => sum + (row.seconds ?? 0), 0)
  const frames = typeof report.frames === "number" ? report.frames : null
  const coverage = typeof report.meanMaskCoverage === "number" ? report.meanMaskCoverage : null
  const budget = (report.budget ?? {}) as Record<string, unknown>
  const device = typeof budget.device === "string" ? budget.device : ""
  const peak = typeof budget.vramPeakInpaintMb === "number" ? budget.vramPeakInpaintMb : null

  return (
    <div className="flex flex-col gap-1.5 text-xs">
      <p className="font-medium">Отчёт прогона</p>
      <div className="grid grid-cols-[1fr_auto] gap-x-3 gap-y-0.5">
        {rows.map((row) => (
          <div key={row.key} className="contents">
            <span className="text-muted-foreground">{row.label}</span>
            <span className="text-right tabular-nums">{row.seconds == null ? "—" : secondsLabel(row.seconds)}</span>
          </div>
        ))}
        <span>Всего по шагам</span>
        <span className="text-right tabular-nums">{secondsLabel(total)}</span>
      </div>
      <p className="text-muted-foreground">
        {[
          frames != null ? `${frames} кадров` : "",
          coverage != null ? `маска ${ (coverage * 100).toFixed(2) }%` : "",
          device ? device : "",
          peak != null && peak > 0 ? `пик заливки ${peak} МБ` : "",
        ]
          .filter(Boolean)
          .join(" · ")}
      </p>
    </div>
  )
}
