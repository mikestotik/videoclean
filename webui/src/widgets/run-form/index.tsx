import { useRef, useState } from "react"
import { submitJob, type Job } from "@/entities/job"
import { Button } from "@/shared/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card"
import { Input } from "@/shared/ui/input"
import { Label } from "@/shared/ui/label"
import { Separator } from "@/shared/ui/separator"
import { Textarea } from "@/shared/ui/textarea"

const ADVANCED_FIELDS = [
  "detector_threshold", "mask_dilate_px", "telea_radius", "verify",
  "prompt_frame_stride", "prompt_frame_max", "parse_chunk_frames", "vision_batch",
  "detector_keyframes", "detector_nms_iou", "detector_max_box_area",
  "tracker_min_score", "tracker_max_template_area",
  "propainter_mask_dilation", "propainter_ref_stride",
  "propainter_neighbor_length", "propainter_subvideo_length", "propainter_raft_iter",
] as const

type Props = { onSubmitted: (job: Job) => void }

export function RunForm({ onSubmitted }: Props) {
  const [file, setFile] = useState<File | null>(null)
  const [prompt, setPrompt] = useState("")
  const [advanced, setAdvanced] = useState(false)
  const [values, setValues] = useState<Record<string, string>>({})
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState("")
  const fileRef = useRef<HTMLInputElement>(null)

  const submit = async () => {
    if (!file || !prompt.trim() || busy) return
    setBusy(true)
    setError("")
    try {
      const job = await submitJob({ kind: "run", video: file, prompt: prompt.trim(), params: values })
      onSubmitted(job)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Полный прогон</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <div className="flex gap-2">
          <Input
            ref={fileRef}
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          {file && (
            <Button
              variant="ghost"
              onClick={() => {
                setFile(null)
                if (fileRef.current) fileRef.current.value = ""
              }}
            >
              Сбросить
            </Button>
          )}
        </div>
        <Textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Что удалить: напр. «текст в правом нижнем углу и красный логотип»"
        />
        <Button variant="outline" size="sm" className="self-start" onClick={() => setAdvanced((v) => !v)}>
          {advanced ? "Скрыть параметры" : "Дополнительно"}
        </Button>
        {advanced && (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3">
            {ADVANCED_FIELDS.map((f) => (
              <div key={f} className="flex flex-col gap-1">
                <Label className="text-muted-foreground">{f}</Label>
                <Input
                  value={values[f] ?? ""}
                  onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
                />
              </div>
            ))}
          </div>
        )}
        {error && <p className="text-sm text-destructive">{error}</p>}
        <Separator />
        <Button disabled={!file || !prompt.trim() || busy} onClick={submit} className="self-start">
          {busy ? "Запуск…" : "Запустить"}
        </Button>
      </CardContent>
    </Card>
  )
}
