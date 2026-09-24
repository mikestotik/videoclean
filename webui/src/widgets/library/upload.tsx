import { useRef, useState } from "react"
import { Loader2, Upload } from "lucide-react"
import { uploadSource, type Source } from "@/entities/source"
import { Button } from "@/shared/ui/button"

type Props = { onUploaded: (source: Source) => void }

export function UploadButton({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const [dragging, setDragging] = useState(false)

  const onFile = async (file: File | undefined) => {
    if (!file) return
    setBusy(true)
    setError("")
    try {
      const created = await uploadSource(file)
      onUploaded(created)
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
        <span className="text-[10px]">Перетащите файл или нажмите</span>
      </button>
      {error && (
        <p className="text-xs text-destructive">
          {error}{" "}
          <Button size="xs" variant="link" className="h-auto p-0" onClick={() => inputRef.current?.click()}>
            Повторить
          </Button>
        </p>
      )}
    </div>
  )
}
