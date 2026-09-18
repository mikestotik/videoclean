import { useCallback, useEffect, useMemo, useState } from "react"
import {
  AlertCircle,
  CheckCircle2,
  Download,
  HardDrive,
  Loader2,
  RefreshCw,
  Server,
  XCircle,
} from "lucide-react"
import { api } from "@/shared/api/client"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import { ScrollArea } from "@/shared/ui/scroll-area"
import { Separator } from "@/shared/ui/separator"
import { cn } from "@/shared/lib/utils"

type ModelInfo = {
  id: string
  title: string
  kind: string
  state?: string
  progress?: number
  downloadable?: boolean
  size_hint?: string
  message?: string
  backend?: string
}

type PollData = {
  models: Record<string, ModelInfo[]>
  doctor: Record<string, string>
  ollama: { ok: boolean; base_url: string; models: string[] }
}

const KIND_META: Record<string, { title: string; hint: string }> = {
  detector: {
    title: "Детекторы",
    hint: "Находят объекты и текст на кадрах по запросу",
  },
  segmenter: {
    title: "Сегментеры",
    hint: "Строят точные маски вокруг найденных объектов",
  },
  inpainter: {
    title: "Инпейнтеры",
    hint: "Заполняют удалённые области соседними пикселями",
  },
  llm: {
    title: "Языковые модели",
    hint: "Разбирают текстовый промпт в список целей",
  },
}

const DOCTOR_LABELS: Record<string, string> = {
  python: "Python",
  ffmpeg: "FFmpeg",
  ffprobe: "ffprobe",
  opencv: "OpenCV",
  torch: "PyTorch",
  cuda: "CUDA",
  mps: "Apple MPS",
}

const STATE_LABEL: Record<string, string> = {
  ready: "Готова",
  installed: "Установлена",
  missing: "Не скачана",
  downloading: "Скачивается",
  error: "Ошибка",
  unavailable: "Недоступна",
}

function stateTone(state?: string): "ok" | "warn" | "bad" | "muted" {
  if (!state) return "muted"
  if (state === "ready" || state === "installed") return "ok"
  if (state === "downloading") return "warn"
  if (state === "error" || state === "missing" || state === "unavailable") return "bad"
  return "muted"
}

function DoctorRow({ label, value }: { label: string; value: string }) {
  const bad =
    value === "missing" ||
    value === "no" ||
    value === "not imported" ||
    value.toLowerCase().includes("not available")
  const ok = value === "yes" || (!bad && value.length > 0)

  return (
    <div className="flex items-start justify-between gap-4 rounded-md bg-muted/30 px-3 py-2.5">
      <div className="min-w-0">
        <p className="text-sm font-medium">{label}</p>
        <p className="mt-0.5 font-mono text-xs break-all text-muted-foreground">{value}</p>
      </div>
      {ok && !bad ? (
        <CheckCircle2 className="size-4 shrink-0 text-ok" aria-label="ок" />
      ) : bad ? (
        <XCircle className="size-4 shrink-0 text-destructive" aria-label="проблема" />
      ) : (
        <AlertCircle className="size-4 shrink-0 text-muted-foreground" aria-label="неизвестно" />
      )}
    </div>
  )
}

function ModelRow({
  model,
  busy,
  onDownload,
  onCancel,
}: {
  model: ModelInfo
  busy: string
  onDownload: (id: string) => void
  onCancel: () => void
}) {
  const tone = stateTone(model.state)
  const downloading = model.state === "downloading"
  const progress = Math.round((model.progress ?? 0) * 100)
  const canDownload = model.downloadable !== false && !downloading && busy !== model.id

  return (
    <div className="rounded-lg border border-border/70 bg-background/40 p-3">
      <div className="flex flex-wrap items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-medium">{model.title}</p>
            <Badge
              variant="secondary"
              className={cn(
                "text-[10px]",
                tone === "ok" && "bg-ok/15 text-ok",
                tone === "warn" && "bg-primary/15 text-primary",
                tone === "bad" && "bg-destructive/15 text-destructive",
              )}
            >
              {STATE_LABEL[model.state ?? ""] ?? model.state ?? "—"}
              {downloading ? ` ${progress}%` : ""}
            </Badge>
          </div>
          <p className="mt-1 font-mono text-[11px] text-muted-foreground">{model.id}</p>
          {(model.size_hint || model.backend) && (
            <p className="mt-1 text-xs text-muted-foreground">
              {[model.backend, model.size_hint].filter(Boolean).join(" · ")}
            </p>
          )}
          {model.message && (
            <p className="mt-1 text-xs text-muted-foreground">{model.message}</p>
          )}
        </div>
        <div className="flex shrink-0 items-center gap-1.5">
          {downloading && (
            <Button size="sm" variant="outline" disabled={busy === "cancel"} onClick={onCancel}>
              {busy === "cancel" ? <Loader2 className="size-3.5 animate-spin" /> : "Отменить"}
            </Button>
          )}
          {model.downloadable !== false && (
            <Button
              size="sm"
              variant={tone === "ok" ? "outline" : "default"}
              disabled={!canDownload}
              onClick={() => onDownload(model.id)}
            >
              {busy === model.id ? (
                <Loader2 className="size-3.5 animate-spin" />
              ) : (
                <>
                  <Download className="size-3.5" />
                  {tone === "ok" ? "Перекачать" : "Скачать"}
                </>
              )}
            </Button>
          )}
        </div>
      </div>
      {downloading && (
        <div className="mt-3 h-1.5 overflow-hidden rounded-full bg-muted">
          <div
            className="h-full rounded-full bg-primary transition-[width] duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
    </div>
  )
}

export function ConfigPage() {
  const [data, setData] = useState<PollData | null>(null)
  const [busy, setBusy] = useState("")
  const [error, setError] = useState("")

  const refresh = useCallback(() => {
    api<PollData>("/api/poll")
      .then((d) => {
        setData(d)
        setError("")
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [refresh])

  const download = async (id: string) => {
    setBusy(id)
    setError("")
    try {
      await api("/api/models/download", { method: "POST", body: JSON.stringify({ id }) })
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy("")
    }
  }

  const cancelDownloads = async () => {
    setBusy("cancel")
    setError("")
    try {
      await api("/api/models/cancel", { method: "POST" })
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy("")
    }
  }

  const doctorEntries = useMemo(() => {
    if (!data?.doctor) return []
    const preferred = ["python", "ffmpeg", "ffprobe", "opencv", "torch", "cuda", "mps"]
    const keys = [
      ...preferred.filter((k) => k in data.doctor),
      ...Object.keys(data.doctor).filter((k) => !preferred.includes(k)),
    ]
    return keys.map((k) => ({ key: k, label: DOCTOR_LABELS[k] ?? k, value: data.doctor[k] }))
  }, [data])

  const modelGroups = useMemo(() => {
    if (!data?.models) return []
    const order = ["detector", "segmenter", "inpainter", "llm"]
    const keys = [
      ...order.filter((k) => k in data.models),
      ...Object.keys(data.models).filter((k) => !order.includes(k)),
    ]
    return keys.map((kind) => ({
      kind,
      meta: KIND_META[kind] ?? { title: kind, hint: "" },
      models: data.models[kind] ?? [],
    }))
  }, [data])

  if (!data) {
    return (
      <div className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        {error || "Загрузка системы…"}
      </div>
    )
  }

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-6 py-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Система</h1>
            <p className="mt-1 max-w-xl text-sm text-muted-foreground">
              Модели, окружение и LLM. Здесь проверяют готовность машины перед прогоном.
            </p>
          </div>
          <Button size="sm" variant="outline" onClick={refresh}>
            <RefreshCw className="size-3.5" />
            Обновить
          </Button>
        </div>

        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            {error}
          </div>
        )}

        <section className="panel p-5">
          <div className="flex items-start gap-3">
            <span className="flex size-9 items-center justify-center rounded-md bg-primary/10 text-primary">
              <Server className="size-4" />
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-base font-semibold">Ollama</h2>
                <Badge
                  variant="secondary"
                  className={cn(
                    "text-[10px]",
                    data.ollama.ok ? "bg-ok/15 text-ok" : "bg-destructive/15 text-destructive",
                  )}
                >
                  {data.ollama.ok ? "Доступна" : "Недоступна"}
                </Badge>
              </div>
              <p className="mt-1 font-mono text-xs text-muted-foreground">{data.ollama.base_url}</p>
              {!data.ollama.ok ? (
                <p className="mt-3 text-sm text-muted-foreground">
                  Запустите Ollama локально, чтобы интерпретировать промпты. Без неё можно работать
                  с ручными таргетами и масками.
                </p>
              ) : data.ollama.models.length === 0 ? (
                <p className="mt-3 text-sm text-muted-foreground">
                  Сервер отвечает, но моделей нет — скачайте vision-модель в Ollama.
                </p>
              ) : (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {data.ollama.models.map((m) => (
                    <Badge key={m} variant="outline" className="font-mono text-[11px]">
                      {m}
                    </Badge>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>

        <section className="flex flex-col gap-4">
          <div className="flex items-center gap-3">
            <span className="flex size-9 items-center justify-center rounded-md bg-primary/10 text-primary">
              <HardDrive className="size-4" />
            </span>
            <div>
              <h2 className="text-base font-semibold">Модели пайплайна</h2>
              <p className="text-sm text-muted-foreground">
                Скачиваются один раз и используются локально
              </p>
            </div>
          </div>

          {modelGroups.map(({ kind, meta, models }) => (
            <div key={kind} className="panel overflow-hidden">
              <div className="border-b border-border/70 px-5 py-4">
                <h3 className="text-sm font-semibold">{meta.title}</h3>
                {meta.hint && <p className="mt-0.5 text-xs text-muted-foreground">{meta.hint}</p>}
              </div>
              <div className="flex flex-col gap-2 p-3">
                {models.length === 0 ? (
                  <p className="px-2 py-3 text-sm text-muted-foreground">Моделей в этой группе нет</p>
                ) : (
                  models.map((m) => (
                    <ModelRow
                      key={m.id}
                      model={m}
                      busy={busy}
                      onDownload={(id) => void download(id)}
                      onCancel={() => void cancelDownloads()}
                    />
                  ))
                )}
              </div>
            </div>
          ))}
        </section>

        <section className="panel p-5">
          <div className="mb-4">
            <h2 className="text-base font-semibold">Окружение</h2>
            <p className="mt-0.5 text-sm text-muted-foreground">
              Проверка зависимостей на этой машине
            </p>
          </div>
          <div className="grid gap-2 sm:grid-cols-2">
            {doctorEntries.map((row) => (
              <DoctorRow key={row.key} label={row.label} value={row.value} />
            ))}
          </div>
          <Separator className="my-4" />
          <p className="text-xs text-muted-foreground">
            Данные обновляются автоматически каждые 5 секунд.
          </p>
        </section>
      </div>
    </ScrollArea>
  )
}
