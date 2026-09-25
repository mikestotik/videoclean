import { useState } from "react"
import { cancelJob, downloadJobOutput, outputDownloadName, packageJob, waitJobToCompletion, type Job } from "@/entities/job"
import { findCachedJob } from "@/shared/events"
import { Button } from "@/shared/ui/button"
import { Switch } from "@/shared/ui/switch"
import { FORMAT_META } from "./param-meta"
import { OUTPUT_FORMATS, type EditorParams } from "./params"
import { FieldLabel, ParamHint, ParamSlider } from "./stage-parts"

const PACKAGE_FORMATS = new Set(["hls-fmp4", "hls-ts", "dash"])

export function OutputFormatBlock({
  params,
  onChange,
  disabled,
  resultJob,
  onResultJobChange,
  showWebm,
  showSegment,
}: {
  params: EditorParams
  onChange: (next: EditorParams) => void
  disabled?: boolean
  resultJob?: Job | null
  onResultJobChange?: (job: Job) => void
  showWebm: boolean
  showSegment: boolean
}) {
  const [packBusy, setPackBusy] = useState(false)
  const [packJobId, setPackJobId] = useState<string | null>(null)
  const [packProgress, setPackProgress] = useState({ fraction: 0, detail: "", eta: "" })
  const [packError, setPackError] = useState("")
  const [busyFmt, setBusyFmt] = useState<string | null>(null)
  const [dlError, setDlError] = useState("")

  const setRun = (patch: Partial<EditorParams["run"]>) =>
    onChange({ ...params, run: { ...params.run, ...patch } })

  const done = resultJob?.state === "COMPLETED" ? resultJob : null
  const built = new Set(Object.keys(done?.outputs ?? {}))
  const missing = done ? params.run.formats.filter((f) => f !== "mp4" && !built.has(f)) : []
  const canPackage = done?.can_package !== false
  const needsWebm = params.run.formats.includes("webm")
  const needsSegment = params.run.formats.some((f) => PACKAGE_FORMATS.has(f))

  const onDownload = async (fmt: string, url: string) => {
    if (!done) return
    setDlError("")
    setBusyFmt(fmt)
    try {
      await downloadJobOutput(url, outputDownloadName(fmt, done.id))
    } catch (e) {
      setDlError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusyFmt(null)
    }
  }

  const onConvert = async () => {
    if (!done || !canPackage || packBusy || missing.length === 0) return
    setPackError("")
    setPackBusy(true)
    setPackProgress({ fraction: 0, detail: "", eta: "" })
    try {
      const queued = await packageJob(done.id, {
        formats: missing,
        webm_crf: needsWebm ? params.run.webm_crf : undefined,
        segment_seconds: needsSegment ? params.run.segment_seconds : undefined,
        overwrite: true,
      })
      setPackJobId(queued.id)
      await waitJobToCompletion(
        queued.id,
        (j) => setPackProgress({ fraction: j.fraction, detail: j.detail, eta: j.eta }),
        { seed: queued },
      )
      onResultJobChange?.(findCachedJob(done.id) ?? done)
    } catch (e) {
      setPackError(e instanceof Error ? e.message : String(e))
    } finally {
      setPackBusy(false)
      setPackJobId(null)
    }
  }

  return (
    <div className="flex flex-col gap-2 text-xs">
      <div data-param="formats" className="flex flex-col gap-1.5">
        <FieldLabel hint="MP4 собирается всегда. Остальные можно включить в следующий прогон и добрать у уже готового ролика.">
          Что сохранить
        </FieldLabel>
        {OUTPUT_FORMATS.map((f) => {
          const url = done?.outputs?.[f] || (f === "mp4" ? done?.output_url || "" : "")
          return (
            <div key={f} className="flex items-center justify-between gap-2">
              <span className="flex min-w-0 items-center gap-1">
                <span>{FORMAT_META[f]?.label ?? f}</span>
                {FORMAT_META[f]?.hint ? <ParamHint text={FORMAT_META[f].hint} /> : null}
                {url ? <span className="text-ok">· готов</span> : null}
              </span>
              <span className="flex shrink-0 items-center gap-1.5">
                {url ? (
                  <Button
                    size="xs"
                    variant="outline"
                    disabled={busyFmt === f}
                    onClick={() => void onDownload(f, url)}
                  >
                    {busyFmt === f ? "…" : "Скачать"}
                  </Button>
                ) : null}
                <Switch
                  size="sm"
                  checked={f === "mp4" || params.run.formats.includes(f)}
                  disabled={disabled || f === "mp4" || packBusy}
                  onCheckedChange={(checked) => {
                    if (f === "mp4") return
                    const without = params.run.formats.filter((x) => x !== f && x !== "mp4")
                    setRun({ formats: checked ? ["mp4", ...without, f] : ["mp4", ...without] })
                  }}
                />
              </span>
            </div>
          )
        })}
      </div>
      {showWebm && (
        <ParamSlider
          param="webm_crf"
          label="Качество WebM"
          hint="Меньше — лучше качество и больше файл."
          value={params.run.webm_crf}
          min={18}
          max={45}
          step={1}
          disabled={disabled || packBusy || !params.run.formats.includes("webm")}
          onChange={(v) => setRun({ webm_crf: v })}
        />
      )}
      {showSegment && (
        <ParamSlider
          param="segment_seconds"
          label="Длина сегмента"
          hint="Длина куска HLS или DASH в секундах."
          value={params.run.segment_seconds}
          min={2}
          max={12}
          step={1}
          disabled={disabled || packBusy || !params.run.formats.some((f) => PACKAGE_FORMATS.has(f))}
          formatValue={(v) => `${v} с`}
          onChange={(v) => setRun({ segment_seconds: v })}
        />
      )}
      {done && missing.length > 0 && (
        <Button
          size="sm"
          variant={packBusy ? "outline" : "default"}
          disabled={packBusy ? !packJobId : !canPackage}
          onClick={() => {
            if (packBusy && packJobId) void cancelJob(packJobId)
            else void onConvert()
          }}
        >
          {packBusy ? "Отменить" : "Сконвертировать готовое"}
        </Button>
      )}
      {packBusy && (
        <p className="text-muted-foreground">
          {[`${Math.round(packProgress.fraction * 100)}%`, packProgress.detail, packProgress.eta ? `ETA ${packProgress.eta}` : ""]
            .filter(Boolean)
            .join(" · ")}
        </p>
      )}
      {packError && <p className="text-destructive">{packError}</p>}
      {dlError && <p className="text-destructive">{dlError}</p>}
      {done && done.can_package === false && missing.length > 0 && (
        <p className="text-muted-foreground">Этот результат уже нельзя перекодировать.</p>
      )}
    </div>
  )
}
