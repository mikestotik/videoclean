import { useCallback, useEffect, useState } from "react"
import { cancelJob, deleteJob, listJobs, retryJob, type Job, type JobKind, type JobState } from "@/entities/job"
import { listSources, type Source } from "@/entities/source"
import { usePoll } from "@/shared/hooks/usePoll"
import { formatTimecode } from "@/shared/lib/format"
import { Button } from "@/shared/ui/button"
import { ScrollArea } from "@/shared/ui/scroll-area"
import { UploadButton } from "./upload"

const KIND_LABEL: Record<JobKind, string> = { prompt: "Промпт", preview: "Маски", run: "Inpaint" }

function stateClass(state: JobState): string {
  if (state === "COMPLETED") return "text-ok"
  if (state === "FAILED" || state === "CANCELLED") return "text-destructive"
  return state === "RUNNING" ? "text-primary animate-pulse" : "text-primary"
}

type ActFn = (fn: (id: string) => Promise<unknown>, id: string) => void

function JobRow({ job, onAct }: { job: Job; onAct: ActFn }) {
  return (
    <div className="flex flex-col gap-1 rounded-md border p-2 text-xs">
      <div className="flex items-center justify-between gap-2">
        <span className={stateClass(job.state)}>
          {KIND_LABEL[job.kind]} · {job.state}
        </span>
        <span className="shrink-0 text-muted-foreground">
          {new Date(job.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>
      {job.state === "RUNNING" && job.stage && (
        <span className="text-muted-foreground">
          {job.stage} {Math.round(job.fraction * 100)}%
        </span>
      )}
      <div className="flex flex-wrap gap-1">
        {(job.state === "QUEUED" || job.state === "RUNNING") && (
          <Button size="xs" variant="outline" onClick={() => onAct(cancelJob, job.id)}>
            Отмена
          </Button>
        )}
        {(job.state === "FAILED" || job.state === "CANCELLED") && (
          <Button size="xs" variant="outline" onClick={() => onAct(retryJob, job.id)}>
            Повторить
          </Button>
        )}
        {job.has_output && (
          <a href={job.output_url ?? "#"} download>
            <Button size="xs" variant="outline">Скачать</Button>
          </a>
        )}
        <Button size="xs" variant="destructive" onClick={() => onAct(deleteJob, job.id)}>
          Удалить
        </Button>
      </div>
      {job.error && <p className="text-destructive">{job.error.slice(0, 120)}</p>}
    </div>
  )
}

type Props = {
  selectedId: string | null
  onSelect: (s: Source) => void
  refreshKey: number
}

export function Library({ selectedId, onSelect, refreshKey }: Props) {
  const [sources, setSources] = useState<Source[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [tick, setTick] = useState(0)
  const [error, setError] = useState("")

  const refresh = useCallback(() => setTick((t) => t + 1), [])

  const loadSources = useCallback(() => {
    listSources()
      .then(setSources)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])
  const loadJobs = useCallback(() => {
    listJobs()
      .then(setJobs)
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  usePoll(loadSources, 5000)
  usePoll(loadJobs, 3000)

  useEffect(() => {
    loadSources()
    loadJobs()
  }, [loadSources, loadJobs, refreshKey, tick])

  const act: ActFn = async (fn, id) => {
    try {
      await fn(id)
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const freeJobs = jobs.filter((j) => j.source_id === null)
  const sourceJobs = selectedId ? jobs.filter((j) => j.source_id === selectedId) : []

  return (
    <div className="flex h-full w-[280px] shrink-0 flex-col gap-2">
      <UploadButton onUploaded={refresh} />
      {error && <p className="text-xs text-destructive">{error}</p>}
      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-1 pr-2">
          <p className="text-xs font-medium text-muted-foreground">Источники</p>
          {sources.map((s) => (
            <div key={s.id} className="flex flex-col gap-1">
              <button
                className={`rounded-md border p-2 text-left text-xs ${
                  s.id === selectedId ? "border-primary" : ""
                }`}
                onClick={() => onSelect(s)}
              >
                <span className="block truncate font-medium">{s.name}</span>
                <span className="text-muted-foreground">
                  {formatTimecode(Math.round(s.probe.duration_s * s.probe.fps), s.probe.fps)}
                </span>
              </button>
              {s.id === selectedId && (
                <div className="flex flex-col gap-1 pb-1 pl-2">
                  {sourceJobs.map((j) => (
                    <JobRow key={j.id} job={j} onAct={act} />
                  ))}
                  {sourceJobs.length === 0 && (
                    <p className="text-xs text-muted-foreground">Задач нет</p>
                  )}
                </div>
              )}
            </div>
          ))}
          {sources.length === 0 && <p className="text-xs text-muted-foreground">Пусто</p>}
          <p className="mt-2 text-xs font-medium text-muted-foreground">Без источника</p>
          {freeJobs.map((j) => (
            <div key={j.id} className="flex items-center justify-between gap-2 rounded-md border p-2 text-xs">
              <span className={stateClass(j.state)}>{j.state}</span>
              {j.has_output && (
                <a href={j.output_url ?? "#"} download>
                  <Button size="xs" variant="outline">Скачать</Button>
                </a>
              )}
            </div>
          ))}
          {freeJobs.length === 0 && <p className="text-xs text-muted-foreground">Пусто</p>}
        </div>
      </ScrollArea>
    </div>
  )
}
