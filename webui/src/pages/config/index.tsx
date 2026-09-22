import { useMemo, useState } from "react"
import {
  AlertCircle,
  CheckCircle2,
  Download,
  HardDrive,
  Loader2,
  Plus,
  Server,
  Trash2,
  XCircle,
} from "lucide-react"
import { api } from "@/shared/api/client"
import { useEvents } from "@/shared/events"
import type { PollSnapshot } from "@/shared/events"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/dialog"
import { Input } from "@/shared/ui/input"
import { Label } from "@/shared/ui/label"
import { ScrollArea } from "@/shared/ui/scroll-area"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select"
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
  model_ref?: string
  source?: string
}

type FamilyBackend = {
  id: string
  label: string
  ref_kind: string
  example?: string
}

type ProviderInfo = {
  id: string
  title: string
  base_url: string
  has_api_key?: boolean
  models: string[]
}

type PollData = {
  models: Record<string, ModelInfo[]>
  doctor: Record<string, string>
  ollama: { ok: boolean; base_url: string; models: string[] }
  providers?: ProviderInfo[]
  options?: {
    families?: Record<string, FamilyBackend[]>
    providers?: ProviderInfo[]
  }
}

function asPollData(snap: PollSnapshot | null): PollData | null {
  if (!snap) return null
  return {
    models: snap.models as Record<string, ModelInfo[]>,
    doctor: snap.doctor,
    ollama: snap.ollama,
    providers: snap.providers as ProviderInfo[] | undefined,
    options: snap.options as PollData["options"],
  }
}

const KIND_META: Record<string, { title: string }> = {
  detector: { title: "Детекторы" },
  segmenter: { title: "Сегментеры" },
  inpainter: { title: "Инпейнтеры" },
  llm: { title: "Языковые модели" },
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
  onRemove,
}: {
  model: ModelInfo
  busy: string
  onDownload: (id: string) => void
  onCancel: () => void
  onRemove?: (id: string) => void
}) {
  const tone = stateTone(model.state)
  const downloading = model.state === "downloading"
  const progress = Math.round((model.progress ?? 0) * 100)
  const canDownload = model.downloadable !== false && !downloading && busy !== model.id
  const removable = model.source === "extra" && onRemove

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
          <p className="mt-1 font-mono text-[11px] text-muted-foreground">
            {model.model_ref || model.id}
          </p>
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
          {removable && (
            <Button
              size="sm"
              variant="ghost"
              disabled={busy === model.id}
              onClick={() => onRemove?.(model.id)}
              aria-label="Удалить"
            >
              <Trash2 className="size-3.5" />
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

function AddModelDialog({
  kind,
  open,
  onOpenChange,
  families,
  ollamaModels,
  busy,
  onSubmit,
}: {
  kind: string
  open: boolean
  onOpenChange: (v: boolean) => void
  families: FamilyBackend[]
  ollamaModels: string[]
  busy: boolean
  onSubmit: (payload: { backend: string; model_ref: string; title: string; download: boolean }) => Promise<void>
}) {
  const [backend, setBackend] = useState(families[0]?.id ?? "")
  const [modelRef, setModelRef] = useState("")
  const [title, setTitle] = useState("")
  const [error, setError] = useState("")

  const selected = families.find((f) => f.id === backend) ?? families[0]
  const isOllama = selected?.ref_kind === "ollama"
  const isProvider = selected?.ref_kind === "provider"

  const submit = async () => {
    setError("")
    if (isProvider) {
      setError("OpenAI-compatible подключается в блоке провайдеров ниже")
      return
    }
    if (!backend || !modelRef.trim()) {
      setError(isOllama ? "Укажите тег Ollama" : "Укажите семейство и модель")
      return
    }
    try {
      await onSubmit({
        backend,
        model_ref: modelRef.trim(),
        title: title.trim(),
        download: true,
      })
      onOpenChange(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Добавить · {KIND_META[kind]?.title ?? kind}</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-3 py-1">
          <div className="flex flex-col gap-1.5">
            <Label>Семейство</Label>
            <Select value={backend} onValueChange={(v) => { if (v) setBackend(v) }}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {families.map((f) => (
                  <SelectItem key={f.id} value={f.id}>
                    {f.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>{isOllama ? "Тег Ollama" : "Модель"}</Label>
            {isOllama && ollamaModels.length > 0 && (
              <Select
                value={ollamaModels.includes(modelRef) ? modelRef : undefined}
                onValueChange={(v) => { if (v) setModelRef(v) }}
              >
                <SelectTrigger>
                  <SelectValue placeholder="Из установленных" />
                </SelectTrigger>
                <SelectContent>
                  {ollamaModels.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            )}
            <Input
              value={modelRef}
              onChange={(e) => setModelRef(e.target.value)}
              placeholder={selected?.example || (isOllama ? "qwen2.5vl:3b" : "org/name")}
              className="font-mono text-xs"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>Название</Label>
            <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Необязательно" />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={busy}>
            Отмена
          </Button>
          <Button onClick={() => void submit()} disabled={busy || isProvider}>
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : "Добавить"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function AddProviderDialog({
  open,
  onOpenChange,
  busy,
  onSubmit,
}: {
  open: boolean
  onOpenChange: (v: boolean) => void
  busy: boolean
  onSubmit: (payload: {
    title: string
    base_url: string
    api_key: string
    models: string[]
  }) => Promise<void>
}) {
  const [title, setTitle] = useState("")
  const [baseUrl, setBaseUrl] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [models, setModels] = useState("")
  const [error, setError] = useState("")

  const submit = async () => {
    setError("")
    const list = models
      .split(/[\n,]+/)
      .map((s) => s.trim())
      .filter(Boolean)
    try {
      await onSubmit({
        title: title.trim(),
        base_url: baseUrl.trim(),
        api_key: apiKey.trim(),
        models: list,
      })
      onOpenChange(false)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>OpenAI-compatible провайдер</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-3 py-1">
          <div className="flex flex-col gap-1.5">
            <Label>Название</Label>
            <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="vLLM / OpenAI / …" />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>Base URL</Label>
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="http://127.0.0.1:8000/v1"
              className="font-mono text-xs"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>API key</Label>
            <Input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Необязательно для локальных"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label>Модели</Label>
            <Input
              value={models}
              onChange={(e) => setModels(e.target.value)}
              placeholder="gpt-4o-mini, my-model"
              className="font-mono text-xs"
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={busy}>
            Отмена
          </Button>
          <Button onClick={() => void submit()} disabled={busy}>
            {busy ? <Loader2 className="size-3.5 animate-spin" /> : "Подключить"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export function ConfigPage() {
  const { snapshot } = useEvents()
  const data = asPollData(snapshot)
  const [busy, setBusy] = useState("")
  const [error, setError] = useState("")
  const [addKind, setAddKind] = useState<string | null>(null)
  const [addProviderOpen, setAddProviderOpen] = useState(false)

  const download = async (id: string) => {
    setBusy(id)
    setError("")
    try {
      await api("/api/models/download", { method: "POST", body: JSON.stringify({ id }) })
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
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy("")
    }
  }

  const removeModel = async (id: string) => {
    setBusy(id)
    setError("")
    try {
      await api(`/api/models/${encodeURIComponent(id)}`, { method: "DELETE" })
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy("")
    }
  }

  const addModel = async (kind: string, payload: {
    backend: string
    model_ref: string
    title: string
    download: boolean
  }) => {
    setBusy(`add:${kind}`)
    setError("")
    try {
      await api("/api/models/custom", {
        method: "POST",
        body: JSON.stringify({ kind, ...payload }),
      })
    } finally {
      setBusy("")
    }
  }

  const addProvider = async (payload: {
    title: string
    base_url: string
    api_key: string
    models: string[]
  }) => {
    setBusy("add:provider")
    setError("")
    try {
      await api("/api/providers", { method: "POST", body: JSON.stringify(payload) })
    } finally {
      setBusy("")
    }
  }

  const removeProvider = async (id: string) => {
    setBusy(`provider:${id}`)
    setError("")
    try {
      await api(`/api/providers/${encodeURIComponent(id)}`, { method: "DELETE" })
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
      meta: KIND_META[kind] ?? { title: kind },
      models: data.models[kind] ?? [],
      families: data.options?.families?.[kind] ?? [],
    }))
  }, [data])

  const providers = data?.providers ?? data?.options?.providers ?? []
  const familiesForAdd = addKind ? (data?.options?.families?.[addKind] ?? []) : []

  if (!data) {
    return (
      <div className="flex h-full items-center justify-center gap-2 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        {error || "Загрузка…"}
      </div>
    )
  }

  return (
    <ScrollArea className="h-full">
      <div className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-6 py-8">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Настройки</h1>
            <p className="mt-1 max-w-xl text-sm text-muted-foreground">
              Подключённые модели и провайдеры. В пайплайне доступны только они.
            </p>
          </div>
        </div>

        {error && (
          <div className="flex items-start gap-2 rounded-lg border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            <AlertCircle className="mt-0.5 size-4 shrink-0" />
            {error}
          </div>
        )}

        <section className="flex flex-col gap-5">
          <div className="flex items-center gap-3">
            <span className="flex size-9 items-center justify-center rounded-md bg-primary/10 text-primary">
              <HardDrive className="size-4" />
            </span>
            <h2 className="text-base font-semibold">Модели</h2>
          </div>

          {modelGroups.map(({ kind, meta, models, families }) => (
            <div key={kind} className="flex flex-col gap-2">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-semibold">{meta.title}</h3>
                {families.length > 0 && (
                  <Button size="sm" variant="outline" onClick={() => setAddKind(kind)}>
                    <Plus className="size-3.5" />
                    Добавить
                  </Button>
                )}
              </div>
              <div className="flex flex-col gap-2">
                {models.length === 0 ? (
                  <p className="px-1 py-2 text-sm text-muted-foreground">Пусто</p>
                ) : (
                  models.map((m) => (
                    <ModelRow
                      key={m.id}
                      model={m}
                      busy={busy}
                      onDownload={(id) => void download(id)}
                      onCancel={() => void cancelDownloads()}
                      onRemove={(id) => void removeModel(id)}
                    />
                  ))
                )}
              </div>
            </div>
          ))}
        </section>

        <section className="flex flex-col gap-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <span className="flex size-9 items-center justify-center rounded-md bg-primary/10 text-primary">
                <Server className="size-4" />
              </span>
              <div>
                <h2 className="text-base font-semibold">LLM-провайдеры</h2>
                <p className="font-mono text-xs text-muted-foreground">
                  Ollama · {data.ollama.base_url}
                  {data.ollama.ok ? " · ok" : " · недоступна"}
                </p>
              </div>
            </div>
            <Button size="sm" variant="outline" onClick={() => setAddProviderOpen(true)}>
              <Plus className="size-3.5" />
              Провайдер
            </Button>
          </div>

          <div className="rounded-lg border border-border/70 bg-background/40 p-3">
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-sm font-medium">Ollama</p>
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
            {data.ollama.ok && data.ollama.models.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5">
                {data.ollama.models.map((m) => (
                  <Badge key={m} variant="outline" className="font-mono text-[11px]">
                    {m}
                  </Badge>
                ))}
              </div>
            )}
          </div>

          {providers.map((p) => (
            <div key={p.id} className="rounded-lg border border-border/70 bg-background/40 p-3">
              <div className="flex flex-wrap items-start gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="text-sm font-medium">{p.title}</p>
                    {p.has_api_key && (
                      <Badge variant="secondary" className="text-[10px]">
                        key
                      </Badge>
                    )}
                  </div>
                  <p className="mt-1 font-mono text-[11px] text-muted-foreground">{p.base_url}</p>
                  {p.models.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1.5">
                      {p.models.map((m) => (
                        <Badge key={m} variant="outline" className="font-mono text-[11px]">
                          {m}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={busy === `provider:${p.id}`}
                  onClick={() => void removeProvider(p.id)}
                  aria-label="Удалить провайдера"
                >
                  <Trash2 className="size-3.5" />
                </Button>
              </div>
            </div>
          ))}
        </section>

        <section className="flex flex-col gap-3">
          <h2 className="text-base font-semibold">Окружение</h2>
          <div className="grid gap-2 sm:grid-cols-2">
            {doctorEntries.map((row) => (
              <DoctorRow key={row.key} label={row.label} value={row.value} />
            ))}
          </div>
          <Separator className="my-1" />
          <p className="text-xs text-muted-foreground">Обновление каждые 5 с</p>
        </section>
      </div>

      {/* Remount on open resets the form; `families` identity changes on every
          SSE push, so it must not drive the reset. */}
      <AddModelDialog
        key={addKind ?? "closed"}
        kind={addKind ?? "detector"}
        open={Boolean(addKind)}
        onOpenChange={(v) => { if (!v) setAddKind(null) }}
        families={familiesForAdd}
        ollamaModels={data.ollama.models}
        busy={busy.startsWith("add:")}
        onSubmit={async (payload) => {
          if (!addKind) return
          await addModel(addKind, payload)
        }}
      />

      <AddProviderDialog
        key={addProviderOpen ? "open" : "closed"}
        open={addProviderOpen}
        onOpenChange={setAddProviderOpen}
        busy={busy === "add:provider"}
        onSubmit={addProvider}
      />
    </ScrollArea>
  )
}
