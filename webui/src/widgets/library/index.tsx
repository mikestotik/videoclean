import { useMemo, useState } from "react"
import {
  ChevronDown,
  Download,
  Eraser,
  Film,
  Loader2,
  Package,
  RotateCcw,
  ScanSearch,
  Sparkles,
  Trash2,
  X,
} from "lucide-react"
import {
  cancelJob,
  deleteJob,
  downloadJobOutput,
  outputDownloadName,
  retryJob,
  type Job,
  type JobKind,
  type JobState,
} from "@/entities/job"
import { deleteSource, videoUrl, type Source } from "@/entities/source"
import { useEvents } from "@/shared/events"
import { formatTimecode } from "@/shared/lib/format"
import { Button } from "@/shared/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/dialog"
import { Input } from "@/shared/ui/input"
import { ScrollArea } from "@/shared/ui/scroll-area"
import { cn } from "@/shared/lib/utils"
import { CropDialog } from "./crop-dialog"
import { UploadButton } from "./upload"
import type { DetectTrack } from "@features/detect-run"

const KIND_META: Record<JobKind, { label: string; Icon: typeof Sparkles }> = {
  prompt: { label: "Фраза", Icon: Sparkles },
  preview: { label: "Рамки", Icon: ScanSearch },
  run: { label: "Результат", Icon: Eraser },
  package: { label: "Пакет", Icon: Package },
}

const KIND_ORDER: Array<"prompt" | "preview" | "run"> = ["prompt", "preview", "run"]

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

function jobWhen(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ""
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
  if (d.toDateString() === new Date().toDateString()) return time
  return `${d.toLocaleDateString([], { day: "numeric", month: "short" })} ${time}`
}

function stageLine(job: Job): string {
  const title = (job.stageTitle ?? "").trim()
  if (title) return title
  const raw = job.stage.trim()
  if (!raw) return "Обработка"
  return raw
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
  onDownload,
  onAskDelete,
}: {
  job: Job
  active: boolean
  onAct: ActFn
  onSelect?: (job: Job) => void
  onDownload?: (job: Job) => void
  onAskDelete: (job: Job) => void
}) {
  const selectable = (job.state === "COMPLETED" || job.state === "FAILED") && Boolean(onSelect)
  const time = jobWhen(job.created_at)
  const summary =
    job.state === "FAILED" && job.error.trim()
      ? job.error.trim()
      : job.prompt.trim() || KIND_META[job.kind].label

  return (
    <div
      role={selectable ? "button" : undefined}
      tabIndex={selectable ? 0 : undefined}
      aria-pressed={selectable ? active : undefined}
      className={cn(
        "grid grid-cols-[minmax(0,1fr)_auto] items-start gap-x-2 rounded-md px-2 py-1.5 text-xs",
        active && "border-l-2 border-primary bg-primary/10",
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
      <div className="min-w-0">
        <div className="flex h-4 items-center gap-2">
          <span className={cn("shrink-0 font-medium", stateTone(job.state))}>
            {STATE_LABEL[job.state]}
          </span>
          <span className="shrink-0 tabular-nums text-muted-foreground">{time}</span>
          {active && <span className="truncate text-[10px] text-primary">на экране</span>}
        </div>
        <p className="mt-0.5 h-4 truncate text-[11px] leading-4 text-muted-foreground" title={summary}>
          {summary}
        </p>
        {job.state === "RUNNING" ? (
          <div className="mt-1 flex h-3 items-center gap-2 text-[10px] text-muted-foreground">
            <span className="min-w-0 truncate">{stageLine(job)}</span>
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
        {job.has_output && (job.outputs?.mp4 || job.output_url) && (
          <Button size="icon-xs" variant="ghost" aria-label="Скачать" onClick={() => onDownload?.(job)}>
            <Download className="size-3" />
          </Button>
        )}
        <Button
          size="icon-xs"
          variant="ghost"
          aria-label="Удалить"
          className="text-muted-foreground hover:text-destructive"
          onClick={() => onAskDelete(job)}
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
  onDownload,
  maskTracks,
  excludedIds,
  selectedTrackId,
  onToggleTrack,
  onSelectTrack,
  onAskDelete,
}: {
  kind: JobKind
  jobs: Job[]
  activeId: string | null
  onAct: ActFn
  onSelectJob?: (job: Job) => void
  onDownload?: (job: Job) => void
  maskTracks?: DetectTrack[]
  excludedIds?: number[]
  selectedTrackId?: number | null
  onToggleTrack?: (id: number) => void
  onSelectTrack?: (id: number) => void
  onAskDelete: (job: Job) => void
}) {
  const meta = KIND_META[kind]
  const Icon = meta.Icon

  return (
    <div className="space-y-0.5">
      <div className="flex h-5 items-center gap-1.5 text-[11px] font-medium text-muted-foreground">
        <Icon className="size-3.5 shrink-0" />
        <span>{meta.label}</span>
      </div>
      {jobs.map((job) => (
        <div key={job.id}>
          <JobRow
            job={job}
            active={job.id === activeId}
            onAct={onAct}
            onSelect={onSelectJob}
            onDownload={onDownload}
            onAskDelete={onAskDelete}
          />
          {kind === "preview" && job.id === activeId && (maskTracks?.length ?? 0) > 0 && (
            <>
            <p className="ml-3.5 mt-1 text-[10px] text-muted-foreground">
              Галочка — удалять объект. Число — на скольких кадрах он найден.
            </p>
            <ul className="ml-3.5 mt-0.5 space-y-0.5 border-l border-border/60 pl-2">
              {maskTracks!.map((t) => {
                const on = !excludedIds?.includes(t.id)
                const hits = t.boxes.filter((b) => b != null).length
                const selected = t.id === selectedTrackId
                return (
                  <li key={t.id}>
                    <button
                      type="button"
                      className={cn(
                        "flex w-full items-center gap-1.5 rounded-md px-1.5 py-1 text-left text-[11px]",
                        selected ? "bg-muted" : "hover:bg-muted/50",
                        !on && "opacity-50",
                      )}
                      onClick={() => onSelectTrack?.(t.id)}
                    >
                      <input
                        type="checkbox"
                        className="size-3 shrink-0 accent-primary"
                        checked={on}
                        onChange={() => onToggleTrack?.(t.id)}
                        onClick={(e) => e.stopPropagation()}
                        aria-label={on ? "Не удалять этот объект" : "Удалять этот объект"}
                      />
                      <span className="min-w-0 flex-1 truncate">
                        {t.label || `объект ${t.id}`}
                      </span>
                      <span
                        className="shrink-0 tabular-nums text-muted-foreground"
                        title="Кадров, где объект найден"
                      >
                        {hits}
                      </span>
                    </button>
                  </li>
                )
              })}
            </ul>
            </>
          )}
        </div>
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
  onUploaded?: (source: Source) => void
  beforeSourceChange?: () => boolean | Promise<boolean>
  maskTracks?: DetectTrack[]
  excludedIds?: number[]
  selectedTrackId?: number | null
  onToggleTrack?: (id: number) => void
  onSelectTrack?: (id: number) => void
}

const EMPTY_ACTIVE: ActivePipelineJobs = { prompt: null, preview: null, run: null }

export { CropDialog }

export function Library({
  selectedId,
  onSelect,
  onClearSelection,
  activeJobs = EMPTY_ACTIVE,
  onSelectJob,
  onUploaded,
  beforeSourceChange,
  maskTracks,
  excludedIds,
  selectedTrackId,
  onToggleTrack,
  onSelectTrack,
}: Props) {
  const { jobs, sources, status } = useEvents()
  const [error, setError] = useState("")
  const [deletingId, setDeletingId] = useState("")
  const [query, setQuery] = useState("")
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})
  const [confirm, setConfirm] = useState<
    { title: string; body: string; confirm: string; run: () => void } | null
  >(null)
  const loading = status === "connecting" && sources.length === 0

  const gate = async () => {
    if (!beforeSourceChange) return true
    return await beforeSourceChange()
  }

  const act: ActFn = async (fn, id) => {
    try {
      await fn(id)
      // Job/source lists update via SSE — no client refresh.
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  const downloadOutput = (job: Job) => {
    const url = job.outputs?.mp4 || job.output_url
    if (!url) return
    const fmt = job.outputs?.mp4 ? "mp4" : "default"
    void downloadJobOutput(url, outputDownloadName(fmt, job.id)).catch((e: unknown) => {
      setError(e instanceof Error ? e.message : String(e))
    })
  }

  const removeSource = async (source: Source) => {
    if (deletingId) return
    setDeletingId(source.id)
    setError("")
    try {
      await deleteSource(source.id)
      if (selectedId === source.id) onClearSelection?.()
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

  const askDeleteJob = (job: Job) => {
    const label = job.prompt.trim() || KIND_META[job.kind].label
    setConfirm({
      title: "Удалить задачу",
      body: `«${label}» и связанные данные пропадут из библиотеки.`,
      confirm: "Удалить",
      run: () => act(deleteJob, job.id),
    })
  }

  const askDeleteSource = (source: Source) => {
    setConfirm({
      title: "Удалить ролик",
      body: `«${source.name}» и связанные задачи пропадут из библиотеки.`,
      confirm: "Удалить",
      run: () => void removeSource(source),
    })
  }

  const freeJobs = jobs.filter((j) => j.source_id === null)
  const visibleSources = useMemo(() => {
    const q = query.trim().toLowerCase()
    const rows = [...sources].sort((a, b) => b.createdAt.localeCompare(a.createdAt))
    if (!q) return rows
    return rows.filter((s) => s.name.toLowerCase().includes(q))
  }, [query, sources])
  const sourceJobs = useMemo(
    () => (selectedId ? jobs.filter((j) => j.source_id === selectedId) : []),
    [jobs, selectedId],
  )
  const jobsByKind = useMemo(() => {
    const map: Record<JobKind, Job[]> = {
      prompt: [],
      preview: [],
      run: [],
      package: [],
    }
    for (const job of sourceJobs) {
      if (job.kind === "package") continue
      map[job.kind].push(job)
    }
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
          <p className="text-[11px] text-muted-foreground">Ролики и прошлые запуски</p>
        </div>
        <UploadButton
          onUploaded={(created) => {
            void (async () => {
              if (!(await gate())) return
              onSelect(created)
              onUploaded?.(created)
            })()
          }}
        />
        {sources.length > 3 && (
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Найти ролик"
            aria-label="Найти ролик"
            className="h-8"
          />
        )}
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

          {visibleSources.map((s) => {
            const selected = s.id === selectedId
            const folded = Boolean(collapsed[s.id])
            return (
              <div key={s.id}>
                <div
                  className={cn(
                    "flex items-center gap-1 rounded-md",
                    selected ? "bg-primary/10" : "hover:bg-muted/40",
                  )}
                >
                  <button
                    type="button"
                    className="flex min-w-0 flex-1 items-center gap-2.5 px-2 py-2 text-left"
                    onClick={() => {
                      void (async () => {
                        if (s.id === selectedId) {
                          setCollapsed((prev) => ({ ...prev, [s.id]: !prev[s.id] }))
                          return
                        }
                        if (!(await gate())) return
                        onSelect(s)
                      })()
                    }}
                  >
                    <video
                      src={`${videoUrl(s)}#t=0.1`}
                      muted
                      playsInline
                      preload="metadata"
                      aria-hidden
                      className="size-10 shrink-0 rounded bg-black object-cover"
                    />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium leading-5" title={s.name}>
                        {s.name}
                      </span>
                      <span className="block h-4 truncate text-[11px] leading-4 tabular-nums text-muted-foreground">
                        {formatTimecode(Math.round(s.probe.duration_s * s.probe.fps), s.probe.fps)}
                        {" · "}
                        {s.probe.width}×{s.probe.height}
                      </span>
                    </span>
                  </button>
                  {selected && (
                    <Button
                      size="icon-xs"
                      variant="ghost"
                      className="shrink-0 text-muted-foreground"
                      aria-label={folded ? "Показать запуски" : "Скрыть запуски"}
                      onClick={() => setCollapsed((prev) => ({ ...prev, [s.id]: !prev[s.id] }))}
                    >
                      <ChevronDown className={cn("size-3.5 transition-transform", folded && "-rotate-90")} />
                    </Button>
                  )}
                  <Button
                    size="icon-xs"
                    variant="ghost"
                    className="mr-1 shrink-0 text-muted-foreground hover:text-destructive"
                    aria-label={`Удалить ${s.name}`}
                    disabled={deletingId === s.id}
                    onClick={(e) => {
                      e.stopPropagation()
                      askDeleteSource(s)
                    }}
                  >
                    {deletingId === s.id ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      <Trash2 className="size-3.5" />
                    )}
                  </Button>
                </div>

                {selected && !folded && sourceJobs.length > 0 && (
                  <div className="ml-3 mt-1 space-y-2">
                    {KIND_ORDER.map((kind) =>
                        jobsByKind[kind].length > 0 ? (
                          <StageGroup
                            key={kind}
                            kind={kind}
                            jobs={jobsByKind[kind]}
                            activeId={activeJobs[kind]}
                            onAct={act}
                            onSelectJob={onSelectJob}
                            onDownload={downloadOutput}
                            maskTracks={maskTracks}
                            excludedIds={excludedIds}
                            selectedTrackId={selectedTrackId}
                            onToggleTrack={onToggleTrack}
                            onSelectTrack={onSelectTrack}
                            onAskDelete={askDeleteJob}
                          />
                        ) : null
                    )}
                  </div>
                )}
              </div>
            )
          })}

          {query.trim() && visibleSources.length === 0 && sources.length > 0 && (
            <p className="px-2 py-4 text-center text-xs text-muted-foreground">Ничего не найдено</p>
          )}

          {freeJobs.length > 0 && (
            <div className="mt-3 space-y-0.5">
              <p className="px-2 text-[11px] font-medium text-muted-foreground">Задачи без ролика</p>
              <p className="px-2 pb-1 text-[10px] text-muted-foreground">Их не к чему открыть в плеере.</p>
              {freeJobs.map((j) => (
                <JobRow
                  key={j.id}
                  job={j}
                  active={j.kind !== "package" && j.id === activeJobs[j.kind]}
                  onAct={act}
                  onSelect={onSelectJob}
                  onDownload={downloadOutput}
                  onAskDelete={askDeleteJob}
                />
              ))}
            </div>
          )}
        </div>
      </ScrollArea>

      <Dialog
        open={confirm !== null}
        onOpenChange={(open) => {
          if (!open) setConfirm(null)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{confirm?.title}</DialogTitle>
            <DialogDescription>{confirm?.body}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setConfirm(null)}>
              Отмена
            </Button>
            <Button
              variant="destructive"
              onClick={() => {
                const run = confirm?.run
                setConfirm(null)
                run?.()
              }}
            >
              {confirm?.confirm ?? "Удалить"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
