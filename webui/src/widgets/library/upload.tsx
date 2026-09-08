import { useRef, useState } from "react"
import { uploadSource } from "@/entities/source"
import { Button } from "@/shared/ui/button"

type Props = { onUploaded: () => void }

export function UploadButton({ onUploaded }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")

  const onChange = async (file: File | undefined) => {
    if (!file) return
    setBusy(true)
    setError("")
    try {
      await uploadSource(file)
      onUploaded()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
      if (inputRef.current) inputRef.current.value = ""
    }
  }

  return (
    <div className="flex flex-col gap-1">
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => void onChange(e.target.files?.[0])}
      />
      <Button variant="outline" disabled={busy} onClick={() => inputRef.current?.click()}>
        {busy ? "Загрузка…" : "Загрузить видео"}
      </Button>
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  )
}
