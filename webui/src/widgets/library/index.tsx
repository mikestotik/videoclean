import { useCallback, useEffect, useState } from "react"
import { Download, Film, Loader2, RotateCcw, Trash2, X } from "lucide-react"
import { cancelJob, deleteJob, listJobs, retryJob, type Job, type JobKind, type JobState } from "@/entities/job"
import { deleteSource, listSources, type Source } from "@/entities/source"
import { usePoll } from "@/shared/hooks/usePoll"
import { formatTimecode } from "@/shared/lib/format"
import { Button } from "@/shared/ui/button"
import { ScrollArea } from "@/shared/ui/scroll-area"
import { cn } from "@/shared/lib/utils"
import { UploadButton } from "./upload"

const KIND_LABEL: Record<JobKind, string> = { prompt: "Промпт", preview: "Маски", run: "Inpaint" }

function stateClass(state: JobState): string {
  if (state === "COMPLETED") return "text-ok"
  if (state === "FAILED" || state === "CANCELLED") return "text-destructive"
  return state === "RUNNING" ? "text-primary" : "text-primary"
}

type ActFn = (fn: (id: string) => Promise<unknown>, id: string) => void

function JobRow({ job, onAct }: { job: Job; onAct: ActFn }) {
  return (
    <div className="rounded-md border border-border/70 bg-background/50 p-2.5 text-xs">
      <div className="flex items-center justify-between gap-2">
        <span className={cn("font-medium", stateClass(job.state))}>
          {KIND_LABEL[job.kind]} · {job.state}
        </span>
        <span className="shrink-0 tabular-nums text-muted-foreground">
          {new Date(job.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
        </span>
      </div>
      {job.state === "RUNNING" && job.stage && (
        <div className="mt-1.5">
          <div className="mb-1 flex justify-between text-[11px] text-muted-foreground">
            <span className="truncate">{job.stage}</span>
            <span>{Math.round(job.fraction * 100)}%</span>
          </div>
          <div className="h-1 overflow-hidden rounded-full bg-muted">
            <div
              className="h-full rounded-full bg-primary transition-[width]"
              style={{ width: `${Math.round(job.fraction * 100)}%` }}
            />
          </div>
        </div>
      )}
      <div className="mt-2 flex flex-wrap gap-1">
        {(job.state === "QUEUED" || job.state === "RUNNING") && (
          <Button size="xs" variant="outline" onClick={() => onAct(cancelJob, job.id)}>
            <X className="size-3" />
            Отмена
          </Button>
        )}
        {(job.state === "FAILED" || job.state === "CANCELLED") && (
          <Button size="xs" variant="outline" onClick={() => onAct(retryJob, job.id)}>
            <RotateCcw className="size-3" />
            Повторить
          </Button>
        )}
        {job.has_output && (
          <a href={job.output_url ?? "#"} download>
            <Button size="xs" variant="outline">
              <Download className="size-3" />
              Скачать
            </Button>
          </a>
        )}
        <Button size="xs" variant="ghost" onClick={() => onAct(deleteJob, job.id)}>
          <Trash2 className="size-3" />
        </Button>
      </div>
      {job.error && <p className="mt-1.5 text-destructive">{job.error.slice(0, 120)}</p>}
    </div>
  )
}

type Props = {
  selectedId: string | null
  onSelect: (s: Source) => void
  onClearSelection?: () => void
  refreshKey: number
  onUploaded?: () => void
}

export function Library({ selectedId, onSelect, onClearSelection, refreshKey, onUploaded }: Props) {
  const [sources, setSources] = useState<Source[]>([])
  const [jobs, setJobs] = useState<Job[]>([])
  const [tick, setTick] = useState(0)
  const [error, setError] = useState("")
  const [loading, setLoading] = useState(true)
  const [deletingId, setDeletingId] = useState("")

  const refresh = useCallback(() => setTick((t) => t + 1), [])

  const loadSources = useCallback(() => {
    listSources()
      .then((s) => {
        setSources(s)
        setLoading(false)
      })
      .catch((e) => {
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      })
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

  const removeSource = async (source: Source) => {
    if (deletingId) return
    const ok = window.confirm(`Удалить «${source.name}» и связанные данные?`)
    if (!ok) return
    setDeletingId(source.id)
    setError("")
    try {
      await deleteSource(source.id)
      if (selectedId === source.id) onClearSelection?.()
      refresh()
    } catch (e) {
      const raw = e instanceof Error ? e.message : String(e)
      setError(
        /active jobs|cancel them first/i.test(raw)
          ? "Сначала отмените активные задачи по этому видео"
          : raw,
      )
    } finally {
      setDeletingId("")
    }
  }

  const freeJobs = jobs.filter((j) => j.source_id === null)
  const sourceJobs = selectedId ? jobs.filter((j) => j.source_id === selectedId) : []

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 space-y-3 border-b border-border/70 p-3">
        <div>
          <h2 className="text-sm font-semibold">Библиотека</h2>
          <p className="text-[11px] text-muted-foreground">Источники и задачи</p>
        </div>
        <UploadButton
          onUploaded={() => {
            refresh()
            onUploaded?.()
          }}
        />
        {error && <p className="text-xs text-destructive">{error}</p>}
      </div>

      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-4 p-3">
          <section className="flex flex-col gap-1.5">
            <p className="px-0.5 text-[11px] font-medium tracking-wide text-muted-foreground">
              Источники
            </p>
            {loading && (
              <div className="flex items-center gap-2 px-1 py-3 text-xs text-muted-foreground">
                <Loader2 className="size-3.5 animate-spin" />
                Загрузка…
              </div>
            )}
            {!loading && sources.length === 0 && (
              <div className="rounded-lg border border-dashed border-border/80 px-3 py-6 text-center">
                <Film className="mx-auto mb-2 size-5 text-muted-foreground/70" />
                <p className="text-xs text-muted-foreground">Пока пусто. Загрузите первое видео.</p>
              </div>
            )}
            {sources.map((s) => {
              const selected = s.id === selectedId
              return (
                <div key={s.id} className="flex flex-col gap-1.5">
                  <div
                    className={cn(
                      "group flex items-stretch gap-0.5 rounded-lg border transition-colors",
                      selected
                        ? "border-primary/50 bg-primary/10"
                        : "border-border/70 bg-background/40 hover:border-border hover:bg-muted/40",
                    )}
                  >
                    <button
                      type="button"
                      className="min-w-0 flex-1 px-3 py-2.5 text-left"
                      onClick={() => onSelect(s)}
                    >
                      <span className="block truncate text-sm font-medium">{s.name}</span>
                      <span className="mt-0.5 block text-[11px] tabular-nums text-muted-foreground">
                        {formatTimecode(Math.round(s.probe.duration_s * s.probe.fps), s.probe.fps)}
                        {" · "}
                        {s.probe.width}×{s.probe.height}
                      </span>
                    </button>
                    <Button
                      size="icon-xs"
                      variant="ghost"
                      className="m-1.5 shrink-0 self-start text-muted-foreground opacity-70 hover:text-destructive group-hover:opacity-100"
                      aria-label={`Удалить ${s.name}`}
                      disabled={deletingId === s.id}
                      onClick={(e) => {
                        e.stopPropagation()
                        void removeSource(s)
                      }}
                    >
                      {deletingId === s.id ? (
                        <Loader2 className="size-3.5 animate-spin" />
                      ) : (
                        <Trash2 className="size-3.5" />
                      )}
                    </Button>
                  </div>
                  {selected && (
                    <div className="flex flex-col gap-1.5 pl-1">
                      {sourceJobs.length === 0 ? (
                        <p className="px-1 py-1 text-[11px] text-muted-foreground">Задач по этому ролику нет</p>
                      ) : (
                        sourceJobs.map((j) => <JobRow key={j.id} job={j} onAct={act} />)
                      )}
                    </div>
                  )}
                </div>
              )
            })}
          </section>

          {freeJobs.length > 0 && (
            <section className="flex flex-col gap-1.5">
              <p className="px-0.5 text-[11px] font-medium tracking-wide text-muted-foreground">
                Прочие задачи
              </p>
              {freeJobs.map((j) => (
                <JobRow key={j.id} job={j} onAct={act} />
              ))}
            </section>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
