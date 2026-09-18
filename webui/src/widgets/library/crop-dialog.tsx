import { useEffect, useMemo, useState } from "react"
import { Loader2 } from "lucide-react"
import { cropSource, frameUrl, type Source } from "@/entities/source"
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

type Props = {
  source: Source | null
  open: boolean
  onOpenChange: (open: boolean) => void
  onDone: (source: Source) => void
}

function numOrEmpty(raw: string): number | null {
  const t = raw.trim()
  if (!t) return null
  const n = Number(t)
  return Number.isFinite(n) ? n : NaN
}

function intPx(raw: string): number {
  return Math.max(0, Math.floor(Number(raw) || 0))
}

export function CropDialog({ source, open, onOpenChange, onDone }: Props) {
  const [startS, setStartS] = useState("")
  const [endS, setEndS] = useState("")
  const [left, setLeft] = useState("0")
  const [right, setRight] = useState("0")
  const [top, setTop] = useState("0")
  const [bottom, setBottom] = useState("0")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    if (!open || !source) return
    setStartS("")
    setEndS("")
    setLeft("0")
    setRight("0")
    setTop("0")
    setBottom("0")
    setBusy(false)
    setError("")
  }, [open, source])

  const edges = useMemo(
    () => ({
      left: intPx(left),
      right: intPx(right),
      top: intPx(top),
      bottom: intPx(bottom),
    }),
    [left, right, top, bottom],
  )

  const preview = useMemo(() => {
    if (!source) return null
    const w = source.probe.width
    const h = source.probe.height
    const outW = Math.max(0, w - edges.left - edges.right)
    const outH = Math.max(0, h - edges.top - edges.bottom)
    const mid = Math.max(0, Math.floor((source.probe.frame_count || 1) / 2))
    return {
      width: outW,
      height: outH,
      duration: source.probe.duration_s,
      frame: mid,
      inset: {
        left: w > 0 ? (edges.left / w) * 100 : 0,
        right: w > 0 ? (edges.right / w) * 100 : 0,
        top: h > 0 ? (edges.top / h) * 100 : 0,
        bottom: h > 0 ? (edges.bottom / h) * 100 : 0,
      },
    }
  }, [source, edges])

  const submit = async () => {
    if (!source || busy) return
    const start = numOrEmpty(startS)
    const end = numOrEmpty(endS)
    if (start !== null && Number.isNaN(start)) {
      setError("Начало: укажите число (секунды)")
      return
    }
    if (end !== null && Number.isNaN(end)) {
      setError("Конец: укажите число (секунды)")
      return
    }
    setBusy(true)
    setError("")
    try {
      const created = await cropSource(source.id, {
        start_s: start,
        end_s: end,
        ...edges,
      })
      onOpenChange(false)
      onDone(created)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Обрезать · {source?.name ?? ""}</DialogTitle>
        </DialogHeader>
        <div className="grid gap-4">
          {source && preview && (
            <div
              className="relative mx-auto w-full overflow-hidden rounded-md border border-border/70 bg-black"
              style={{
                aspectRatio: `${source.probe.width} / ${source.probe.height}`,
                maxHeight: 160,
              }}
            >
              <img
                src={frameUrl(source, preview.frame)}
                alt=""
                className="absolute inset-0 h-full w-full object-contain"
              />
              <div
                className="pointer-events-none absolute border-2 border-primary shadow-[0_0_0_9999px_rgba(0,0,0,0.55)]"
                style={{
                  left: `${preview.inset.left}%`,
                  right: `${preview.inset.right}%`,
                  top: `${preview.inset.top}%`,
                  bottom: `${preview.inset.bottom}%`,
                }}
              />
            </div>
          )}

          <div className="space-y-2">
            <p className="text-[11px] font-medium text-muted-foreground">Время</p>
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <Label htmlFor="crop-start">С, сек</Label>
                <Input
                  id="crop-start"
                  inputMode="decimal"
                  placeholder="0"
                  value={startS}
                  onChange={(e) => setStartS(e.target.value)}
                  disabled={busy}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="crop-end">По, сек</Label>
                <Input
                  id="crop-end"
                  inputMode="decimal"
                  placeholder={
                    preview ? String(Math.round(preview.duration * 1000) / 1000) : ""
                  }
                  value={endS}
                  onChange={(e) => setEndS(e.target.value)}
                  disabled={busy}
                />
              </div>
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-[11px] font-medium text-muted-foreground">Края кадра, px</p>
            <div className="grid grid-cols-2 gap-2">
              <div className="space-y-1">
                <Label htmlFor="crop-left">Слева</Label>
                <Input
                  id="crop-left"
                  inputMode="numeric"
                  value={left}
                  onChange={(e) => setLeft(e.target.value)}
                  disabled={busy}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="crop-right">Справа</Label>
                <Input
                  id="crop-right"
                  inputMode="numeric"
                  value={right}
                  onChange={(e) => setRight(e.target.value)}
                  disabled={busy}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="crop-top">Сверху</Label>
                <Input
                  id="crop-top"
                  inputMode="numeric"
                  value={top}
                  onChange={(e) => setTop(e.target.value)}
                  disabled={busy}
                />
              </div>
              <div className="space-y-1">
                <Label htmlFor="crop-bottom">Снизу</Label>
                <Input
                  id="crop-bottom"
                  inputMode="numeric"
                  value={bottom}
                  onChange={(e) => setBottom(e.target.value)}
                  disabled={busy}
                />
              </div>
            </div>
          </div>

          {source && preview && (
            <p className="text-[11px] tabular-nums text-muted-foreground">
              Кадр {source.probe.width}×{source.probe.height} → {preview.width}×{preview.height}
              {" · "}
              {Math.round(source.probe.duration_s * 10) / 10}с
            </p>
          )}
          {error && <p className="text-xs text-destructive">{error}</p>}
        </div>
        <DialogFooter>
          <Button variant="outline" disabled={busy} onClick={() => onOpenChange(false)}>
            Отмена
          </Button>
          <Button disabled={busy || !source} onClick={() => void submit()}>
            {busy ? (
              <>
                <Loader2 className="size-3.5 animate-spin" />
                Обрезаю…
              </>
            ) : (
              "Создать клип"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
