import { useCallback, useEffect, useMemo, useState } from "react"
import {
  Check,
  Download,
  Eraser,
  Film,
  Loader2,
  RotateCcw,
  ScanSearch,
  Sparkles,
  Trash2,
  X,
} from "lucide-react"
import { cancelJob, deleteJob, listJobs, retryJob, type Job, type JobKind, type JobState } from "@/entities/job"
import { deleteSource, listSources, type Source } from "@/entities/source"
import { usePoll } from "@/shared/hooks/usePoll"
import { formatTimecode } from "@/shared/lib/format"
import { Button } from "@/shared/ui/button"
import { ScrollArea } from "@/shared/ui/scroll-area"
import { cn } from "@/shared/lib/utils"
import { UploadButton } from "./upload"

const KIND_META: Record<JobKind, { label: string; Icon: typeof Sparkles }> = {
  prompt: { label: "Промпт", Icon: Sparkles },
  preview: { label: "Маски", Icon: ScanSearch },
  run: { label: "Результат", Icon: Eraser },
}

const KIND_ORDER: JobKind[] = ["prompt", "preview", "run"]

const STATE_LABEL: Record<JobState, string> = {
  QUEUED: "В очереди",
  RUNNING: "Идёт",
  COMPLETED: "Готово",
  FAILED: "Ошибка",
  CANCELLED: "Отменено",
}

function stateTone(state: JobState): string {
  if (state === "COMPLETED") return "text-ok"
  if (state === "FAILED" || state === "CANCELLED") return "text-destructive"
  return "text-primary"
}

export type ActivePipelineJobs = {
  prompt: string | null
  preview: string | null
  run: string | null
}

type ActFn = (fn: (id: string) => Promise<unknown>, id: string) => void

function JobRow({
  job,
  active,
  onAct,
  onSelect,
}: {
  job: Job
  active: boolean
  onAct: ActFn
  onSelect?: (job: Job) => void
}) {
  const selectable = job.state === "COMPLETED" && Boolean(onSelect)
  const time = new Date(job.created_at).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  })
  const summary = job.prompt.trim() || KIND_META[job.kind].label

  return (
    <div
      role={selectable ? "button" : undefined}
      tabIndex={selectable ? 0 : undefined}
      aria-pressed={selectable ? active : undefined}
      className={cn(
        "grid grid-cols-[14px_minmax(0,1fr)_auto] items-start gap-x-2 rounded-md px-2 py-1.5 text-xs",
        active && "bg-primary/10",
        selectable && "cursor-pointer hover:bg-muted/50",
      )}
      onClick={selectable ? () => onSelect?.(job) : undefined}
      onKeyDown={
        selectable
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault()
                onSelect?.(job)
              }
            }
          : undefined
      }
    >
      <span className="mt-0.5 flex size-3.5 items-center justify-center" aria-hidden>
        {selectable ? (
          <span
            className={cn(
              "flex size-3.5 items-center justify-center rounded-full border",
              active
                ? "border-primary bg-primary text-primary-foreground"
                : "border-muted-foreground/40 bg-transparent",
            )}
          >
            {active ? <Check className="size-2.5" /> : null}
          </span>
        ) : (
          <span className="size-3.5" />
        )}
      </span>

      <div className="min-w-0">
        <div className="flex h-4 items-center gap-2">
          <span className={cn("shrink-0 font-medium", stateTone(job.state))}>
            {STATE_LABEL[job.state]}
          </span>
          <span className="shrink-0 tabular-nums text-muted-foreground">{time}</span>
        </div>
        <p className="mt-0.5 h-4 truncate text-[11px] leading-4 text-muted-foreground">{summary}</p>
        {job.state === "RUNNING" && job.stage ? (
          <div className="mt-1 flex h-3 items-center gap-2 text-[10px] text-muted-foreground">
            <span className="min-w-0 truncate">{job.stage}</span>
            <span className="shrink-0 tabular-nums">{Math.round(job.fraction * 100)}%</span>
          </div>
        ) : null}
      </div>

      <div
        className="flex h-7 items-center gap-0.5"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => e.stopPropagation()}
      >
        {(job.state === "QUEUED" || job.state === "RUNNING") && (
          <Button size="icon-xs" variant="ghost" aria-label="Отмена" onClick={() => onAct(cancelJob, job.id)}>
            <X className="size-3" />
          </Button>
        )}
        {(job.state === "FAILED" || job.state === "CANCELLED") && (
          <Button size="icon-xs" variant="ghost" aria-label="Повторить" onClick={() => onAct(retryJob, job.id)}>
            <RotateCcw className="size-3" />
          </Button>
        )}
        {job.has_output && (
          <a href={job.output_url ?? "#"} download aria-label="Скачать">
            <Button size="icon-xs" variant="ghost">
              <Download className="size-3" />
            </Button>
          </a>
        )}
        <Button
          size="icon-xs"
          variant="ghost"
          aria-label="Удалить"
          className="text-muted-foreground hover:text-destructive"
          onClick={() => onAct(deleteJob, job.id)}
        >
          <Trash2 className="size-3" />
        </Button>
      </div>
    </div>
  )
}

function StageGroup({
  kind,
  jobs,
  activeId,
  onAct,
  onSelectJob,
}: {
  kind: JobKind
  jobs: Job[]
  activeId: string | null
  onAct: ActFn
  onSelectJob?: (job: Job) => void
}) {
  const meta = KIND_META[kind]
  const Icon = meta.Icon

  return (
    <div className="space-y-0.5">
      <div className="flex h-5 items-center gap-1.5 px-2 text-[11px] font-medium text-muted-foreground">
        <Icon className="size-3.5 shrink-0" />
        <span>{meta.label}</span>
      </div>
      {jobs.map((job) => (
        <JobRow
          key={job.id}
          job={job}
          active={job.id === activeId}
          onAct={onAct}
          onSelect={onSelectJob}
        />
      ))}
    </div>
  )
}

type Props = {
  selectedId: string | null
  onSelect: (s: Source) => void
  onClearSelection?: () => void
  activeJobs?: ActivePipelineJobs
  onSelectJob?: (job: Job) => void
  refreshKey: number
  onUploaded?: () => void
}

const EMPTY_ACTIVE: ActivePipelineJobs = { prompt: null, preview: null, run: null }

export function Library({
  selectedId,
  onSelect,
  onClearSelection,
  activeJobs = EMPTY_ACTIVE,
  onSelectJob,
  refreshKey,
  onUploaded,
}: Props) {
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
  const sourceJobs = useMemo(
    () => (selectedId ? jobs.filter((j) => j.source_id === selectedId) : []),
    [jobs, selectedId],
  )
  const jobsByKind = useMemo(() => {
    const map: Record<JobKind, Job[]> = { prompt: [], preview: [], run: [] }
    for (const job of sourceJobs) map[job.kind].push(job)
    for (const kind of KIND_ORDER) {
      map[kind].sort((a, b) => b.created_at.localeCompare(a.created_at))
    }
    return map
  }, [sourceJobs])

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="shrink-0 space-y-3 border-b border-border/70 p-3">
        <div>
          <h2 className="text-sm font-semibold">Библиотека</h2>
          <p className="text-[11px] text-muted-foreground">Видео и прогоны</p>
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
        <div className="flex flex-col gap-1 p-2">
          {loading && (
            <div className="flex items-center gap-2 px-2 py-3 text-xs text-muted-foreground">
              <Loader2 className="size-3.5 animate-spin" />
              Загрузка…
            </div>
          )}
          {!loading && sources.length === 0 && (
            <div className="px-2 py-6 text-center">
              <Film className="mx-auto mb-2 size-5 text-muted-foreground/70" />
              <p className="text-xs text-muted-foreground">Пока пусто. Загрузите первое видео.</p>
            </div>
          )}

          {sources.map((s) => {
            const selected = s.id === selectedId
            return (
              <div key={s.id}>
                <div
                  className={cn(
                    "group flex items-center gap-1 rounded-md",
                    selected ? "bg-primary/10" : "hover:bg-muted/40",
                  )}
                >
                  <button
                    type="button"
                    className="flex min-w-0 flex-1 items-center gap-2.5 px-2 py-2 text-left"
                    onClick={() => onSelect(s)}
                  >
                    <Film
                      className={cn(
                        "size-4 shrink-0",
                        selected ? "text-primary" : "text-muted-foreground",
                      )}
                    />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium leading-5">{s.name}</span>
                      <span className="block h-4 truncate text-[11px] leading-4 tabular-nums text-muted-foreground">
                        {formatTimecode(Math.round(s.probe.duration_s * s.probe.fps), s.probe.fps)}
                        {" · "}
                        {s.probe.width}×{s.probe.height}
                      </span>
                    </span>
                  </button>
                  <Button
                    size="icon-xs"
                    variant="ghost"
                    className="mr-1 shrink-0 text-muted-foreground opacity-0 hover:text-destructive group-hover:opacity-100 focus-visible:opacity-100"
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

                {selected && sourceJobs.length > 0 && (
                  <div className="ml-6 mt-0.5 space-y-2">
                    {KIND_ORDER.map((kind) =>
                      jobsByKind[kind].length > 0 ? (
                        <StageGroup
                          key={kind}
                          kind={kind}
                          jobs={jobsByKind[kind]}
                          activeId={activeJobs[kind]}
                          onAct={act}
                          onSelectJob={onSelectJob}
                        />
                      ) : null,
                    )}
                  </div>
                )}
              </div>
            )
          })}

          {freeJobs.length > 0 && (
            <div className="mt-3 space-y-0.5">
              <p className="px-2 text-[11px] font-medium text-muted-foreground">Без видео</p>
              {freeJobs.map((j) => (
                <JobRow
                  key={j.id}
                  job={j}
                  active={j.id === activeJobs[j.kind]}
                  onAct={act}
                  onSelect={onSelectJob}
                />
              ))}
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  )
}
