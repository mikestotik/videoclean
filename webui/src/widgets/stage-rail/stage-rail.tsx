import { useEffect, useRef, useState } from "react"
import { useInterpret } from "@features/interpret"
import { useDetectRun } from "@features/detect-run"
import { useInpaintRun } from "@features/inpaint-run"
import { cancelJob, type Job } from "@/entities/job"
import type { Source } from "@/entities/source"
import { Button } from "@/shared/ui/button"
import { Label } from "@/shared/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select"
import { Slider } from "@/shared/ui/slider"
import { Switch } from "@/shared/ui/switch"
import { Textarea } from "@/shared/ui/textarea"
import { ToggleGroup, ToggleGroupItem } from "@/shared/ui/toggle-group"
import { TargetsEditor } from "./targets-editor"
import { AdvancedFields, LlmChip, ParamSlider, PresetsPopover, StageSection } from "./stage-parts"
import {
  INPAINTER_OPTIONS,
  OUTPUT_FORMATS,
  autoStride,
  enabledTargets,
  resetParams,
  toRunParams,
  type EditorParams,
  type InpaintMode,
} from "./params"

type Props = {
  source: Source | null
  frameCount: number
  params: EditorParams
  onParamsChange: (p: EditorParams) => void
  interpret: ReturnType<typeof useInterpret>
  detect: ReturnType<typeof useDetectRun>
  inpaint: ReturnType<typeof useInpaintRun>
  resultJob: Job | null
  masks?: number[]
  onOpenConfig?: () => void
  onOpenResult?: () => void
}

export function StageRail({
  source,
  frameCount,
  params,
  onParamsChange,
  interpret,
  detect,
  inpaint,
  resultJob,
  masks,
  onOpenConfig,
  onOpenResult,
}: Props) {
  const [inpaintMode, setInpaintMode] = useState<InpaintMode>("tracks")
  const [advancedOpen, setAdvancedOpen] = useState(false)
  const noSource = !source
  const set = (patch: Partial<EditorParams>) => onParamsChange({ ...params, ...patch })

  const activeStage = resultJob?.state === "COMPLETED"
    ? 5
    : inpaint.running
      ? 4
      : detect.running
        ? 3
        : interpret.result || params.targets.length > 0
          ? 2
          : 1

  const appliedResultRef = useRef<unknown>(null)
  useEffect(() => {
    if (!interpret.result || appliedResultRef.current === interpret.result) return
    appliedResultRef.current = interpret.result
    onParamsChange({
      ...params,
      prompt: interpret.result.prompt || params.prompt,
      targets: interpret.result.targets.map((t) => ({
        kind: t.kind === "watermark" || t.kind === "text_overlay" ? t.kind : "object",
        query: t.query,
        where: t.where,
        enabled: true,
        source: "auto" as const,
      })),
    })
  }, [interpret.result, params, onParamsChange])

  const runDetect = () =>
    detect.run({
      mode: params.detect.mode === "targets" ? "detect" : "parse",
      prompt: params.prompt,
      targets: enabledTargets(params),
      all: params.detect.all,
      stride: params.detect.stride,
    })

  const runInpaint = () => {
    const payload =
      inpaintMode === "tracks"
        ? { mode: "tracks" as const, tracks: detect.tracks }
        : inpaintMode === "masks"
          ? { mode: "masks" as const, masks: masks ?? [] }
          : { mode: "prompt" as const, prompt: params.prompt, targets: enabledTargets(params) }
    void inpaint.run(payload, toRunParams(params))
  }

  return (
    <aside className="flex w-[360px] shrink-0 flex-col overflow-y-auto rounded-md border bg-card">
      <StageSection n={1} title="Вход" active={activeStage === 1}>
        <Textarea
          value={params.prompt}
          onChange={(e) => set({ prompt: e.target.value })}
          placeholder="Опишите, что удалить (любой язык)"
          className="min-h-20 text-xs"
          disabled={noSource}
        />
        <LlmChip onOpenConfig={onOpenConfig} />
        <Button
          size="sm"
          disabled={noSource || interpret.running || (!params.prompt.trim() && !(masks && masks.length > 0))}
          onClick={() => void interpret.run(params.prompt)}
        >
          Интерпретировать
        </Button>
        {interpret.running && <p className="text-xs text-muted-foreground">Разбираю кадры через LLM…</p>}
        {interpret.error && <p className="text-xs text-destructive">{interpret.error}</p>}
      </StageSection>

      <StageSection n={2} title="Таргеты" active={activeStage === 2}>
        <TargetsEditor targets={params.targets} onChange={(targets) => set({ targets })} disabled={noSource} />
      </StageSection>

      <StageSection n={3} title="Маски" active={activeStage === 3}>
        <ToggleGroup
          variant="outline"
          size="sm"
          value={[params.detect.all ? "all" : "stride"]}
          onValueChange={(v) => {
            const next = v.at(-1)
            if (next) set({ detect: { ...params.detect, all: next === "all" } })
          }}
        >
          <ToggleGroupItem value="all">Всё видео</ToggleGroupItem>
          <ToggleGroupItem value="stride">Каждый N-й кадр</ToggleGroupItem>
        </ToggleGroup>
        {!params.detect.all && (
          <div className="flex items-center gap-2 text-xs">
            <Slider
              className="w-32"
              min={1}
              max={30}
              value={params.detect.stride}
              onValueChange={(v) => {
                if (typeof v === "number") set({ detect: { ...params.detect, stride: v } })
              }}
            />
            <span className="text-muted-foreground">шаг {params.detect.stride}</span>
            <Button
              size="xs"
              variant="ghost"
              disabled={frameCount <= 0}
              onClick={() => set({ detect: { ...params.detect, stride: autoStride(frameCount) } })}
            >
              Авто: {autoStride(frameCount)}
            </Button>
          </div>
        )}
        <ToggleGroup
          variant="outline"
          size="sm"
          value={[params.detect.mode]}
          onValueChange={(v) => {
            const next = v.at(-1)
            if (next === "targets" || next === "prompt") set({ detect: { ...params.detect, mode: next } })
          }}
        >
          <ToggleGroupItem value="targets">По таргетам</ToggleGroupItem>
          <ToggleGroupItem value="prompt">По промпту</ToggleGroupItem>
        </ToggleGroup>
        <Button size="sm" disabled={noSource || detect.running} onClick={runDetect}>
          Найти маски
        </Button>
        {detect.running && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">
              {Math.round(detect.progress.fraction * 100)}% {detect.progress.eta && `· ETA ${detect.progress.eta}`}
            </span>
            <Button size="xs" variant="outline" disabled={!detect.jobId} onClick={() => detect.jobId && void cancelJob(detect.jobId)}>
              Стоп
            </Button>
          </div>
        )}
        {detect.error && <p className="text-xs text-destructive">{detect.error}</p>}
        {!detect.running && detect.manifest && (
          <p className="text-xs text-muted-foreground">
            Покрытие масок: {((detect.manifest.meanMaskCoverage ?? 0) * 100).toFixed(2)}% · треков: {detect.tracks.length}
          </p>
        )}
      </StageSection>

      <StageSection n={4} title="Inpaint" active={activeStage === 4}>
        <ToggleGroup
          variant="outline"
          size="sm"
          value={[inpaintMode]}
          onValueChange={(v) => {
            const next = v.at(-1)
            if (next === "tracks" || next === "masks" || next === "prompt") setInpaintMode(next)
          }}
        >
          <ToggleGroupItem value="tracks">По трекам</ToggleGroupItem>
          <ToggleGroupItem value="masks">По маскам</ToggleGroupItem>
          <ToggleGroupItem value="prompt">По промпту</ToggleGroupItem>
        </ToggleGroup>
        {inpaintMode === "masks" && (
          <p className="text-xs text-muted-foreground">
            Маски применяются ко всем кадрам — только для неподвижных объектов.
          </p>
        )}
        <div className="flex items-center gap-2 text-xs">
          <Label className="w-20 shrink-0">Инпейнтер</Label>
          <Select value={params.run.inpainter} onValueChange={(v) => { if (v) set({ run: { ...params.run, inpainter: v } }) }} disabled={noSource}>
            <SelectTrigger size="sm" className="flex-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {INPAINTER_OPTIONS.map((o) => (
                <SelectItem key={o} value={o}>{o}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <ParamSlider
          label="Dilate масок, px"
          value={params.run.mask_dilate_px}
          min={0}
          max={15}
          disabled={noSource}
          onChange={(v) => set({ run: { ...params.run, mask_dilate_px: v } })}
        />
        {params.run.inpainter === "opencv-telea" && (
          <ParamSlider
            label="TELEA radius"
            value={params.run.telea_radius}
            min={1}
            max={30}
            disabled={noSource}
            onChange={(v) => set({ run: { ...params.run, telea_radius: v } })}
          />
        )}
        <div className="flex items-center gap-2 text-xs">
          <Label className="flex-1">Проверять leftover</Label>
          <Switch
            checked={params.run.verify}
            onCheckedChange={(checked) => set({ run: { ...params.run, verify: checked } })}
            disabled={noSource}
          />
        </div>
        <Button size="sm" variant="ghost" className="self-start" onClick={() => setAdvancedOpen((o) => !o)}>
          {advancedOpen ? "Скрыть параметры" : "Все параметры"}
        </Button>
        {advancedOpen && (
          <AdvancedFields
            params={params}
            onParamsChange={onParamsChange}
            disabled={noSource}
          />
        )}
        <Button
          size="sm"
          disabled={
            noSource ||
            inpaint.running ||
            (inpaintMode === "tracks" && detect.tracks.length === 0) ||
            (inpaintMode === "masks" && !(masks && masks.length > 0)) ||
            (inpaintMode === "prompt" && !params.prompt.trim())
          }
          onClick={runInpaint}
        >
          Запустить
        </Button>
        {inpaint.running && (
          <div className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">
              {Math.round(inpaint.progress.fraction * 100)}% {inpaint.progress.eta && `· ETA ${inpaint.progress.eta}`}
            </span>
            <Button size="xs" variant="outline" disabled={!inpaint.jobId} onClick={() => inpaint.jobId && void cancelJob(inpaint.jobId)}>
              Стоп
            </Button>
          </div>
        )}
        {inpaint.error && <p className="text-xs text-destructive">{inpaint.error}</p>}
      </StageSection>

      <StageSection n={5} title="Результат" active={activeStage === 5}>
        {resultJob?.state === "COMPLETED" ? (
          <>
            <p className="text-xs text-muted-foreground">Готово — сравнение открывается в просмотрщике.</p>
            <div className="flex gap-2">
              {onOpenResult && (
                <Button size="sm" variant="outline" onClick={onOpenResult}>
                  Открыть сравнение
                </Button>
              )}
              <Button
                size="sm"
                disabled={!resultJob.output_url}
                render={resultJob.output_url ? <a href={resultJob.output_url} download /> : undefined}
              >
                Скачать
              </Button>
            </div>
          </>
        ) : (
          <p className="text-xs text-muted-foreground">Результата ещё нет — запустите inpaint.</p>
        )}
        <div className="flex flex-col gap-1 text-xs">
          <Label>Форматы вывода</Label>
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {OUTPUT_FORMATS.map((f) => (
              <label key={f} className="flex items-center gap-1.5">
                <Switch
                  size="sm"
                  checked={params.run.formats.includes(f)}
                  disabled={noSource}
                  onCheckedChange={(checked) =>
                    set({
                      run: {
                        ...params.run,
                        formats: checked
                          ? [...params.run.formats, f]
                          : params.run.formats.filter((x) => x !== f),
                      },
                    })
                  }
                />
                {f}
              </label>
            ))}
          </div>
        </div>
        <PresetsPopover params={params} onParamsChange={onParamsChange} disabled={noSource} />
        <Button size="sm" variant="outline" disabled={noSource} onClick={() => onParamsChange(resetParams())}>
          Сбросить к дефолтам
        </Button>
      </StageSection>
    </aside>
  )
}

