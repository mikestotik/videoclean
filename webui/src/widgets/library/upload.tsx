import { useEffect, useRef, useState } from "react"
import { Loader2, Upload } from "lucide-react"
import { downscaleSource, uploadSource, type Source } from "@/entities/source"
import { Button } from "@/shared/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/shared/ui/dialog"

type Props = { onUploaded: (source: Source) => void }

export function UploadButton({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const [dragging, setDragging] = useState(false)
  const [pending, setPending] = useState<Source | null>(null)
  const [converting, setConverting] = useState<720 | 480 | null>(null)
  const kept = useRef(false)

  useEffect(() => {
    const pick = () => inputRef.current?.click()
    window.addEventListener("videoclean:pick-upload", pick)
    return () => window.removeEventListener("videoclean:pick-upload", pick)
  }, [])

  const onFile = async (file: File | undefined) => {
    if (!file) return
    setBusy(true)
    setError("")
    try {
      const created = await uploadSource(file)
      const short = Math.min(created.probe.width, created.probe.height)
      if (short > 480) {
        kept.current = false
        setPending(created)
      } else {
        onUploaded(created)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      if (inputRef.current) inputRef.current.value = ""
    }
  }

  return (
    <div className="flex flex-col gap-1.5">
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => void onFile(e.target.files?.[0])}
      />
      <button
        type="button"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
        onDragEnter={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault()
          setDragging(false)
          void onFile(e.dataTransfer.files?.[0])
        }}
        className={`flex w-full flex-col items-center gap-1.5 rounded-lg border border-dashed px-3 py-4 text-center transition-colors ${
          dragging
            ? "border-primary bg-primary/10 text-primary"
            : "border-border/80 bg-muted/20 text-muted-foreground hover:border-border hover:bg-muted/40 hover:text-foreground"
        } disabled:opacity-60`}
      >
        {busy ? <Loader2 className="size-4 animate-spin" /> : <Upload className="size-4" />}
        <span className="text-xs font-medium text-foreground">
          {busy ? "Загрузка…" : "Загрузить видео"}
        </span>
        <span className="text-[10px]">mp4, mov, webm, mkv — перетащите или нажмите</span>
      </button>
      {error && (
        <p className="text-xs text-destructive">
          {error}{" "}
          <Button size="xs" variant="link" className="h-auto p-0" onClick={() => inputRef.current?.click()}>
            Повторить
          </Button>
        </p>
      )}
      <Dialog
        open={pending !== null}
        onOpenChange={(open) => {
          if (open || !pending || converting) return
          if (kept.current) return
          kept.current = true
          const source = pending
          setPending(null)
          onUploaded(source)
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Уменьшить перед работой?</DialogTitle>
            <DialogDescription>
              {pending
                ? `Ролик ${pending.probe.width}×${pending.probe.height}. Меньшая копия быстрее ищет и удаляет объекты. Исходник останется в библиотеке.`
                : ""}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              disabled={converting !== null}
              onClick={() => {
                if (!pending || kept.current) return
                kept.current = true
                const source = pending
                setPending(null)
                onUploaded(source)
              }}
            >
              Оставить как есть
            </Button>
            {pending && Math.min(pending.probe.width, pending.probe.height) > 720 && (
              <Button
                variant="outline"
                disabled={converting !== null}
                onClick={() => void convert(720)}
              >
                {converting === 720 ? <Loader2 className="animate-spin" /> : null}
                720p
              </Button>
            )}
            <Button disabled={converting !== null} onClick={() => void convert(480)}>
              {converting === 480 ? <Loader2 className="animate-spin" /> : null}
              480p
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )

  async function convert(shortSide: 720 | 480) {
    if (!pending) return
    setConverting(shortSide)
    setError("")
    try {
      const created = await downscaleSource(pending.id, shortSide)
      kept.current = true
      setPending(null)
      onUploaded(created)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setConverting(null)
    }
  }
}
