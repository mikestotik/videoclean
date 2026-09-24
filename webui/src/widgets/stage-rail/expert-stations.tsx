import { cancelJob } from "@/entities/job"
import { Button } from "@/shared/ui/button"
import { Input } from "@/shared/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/shared/ui/select"
import { Switch } from "@/shared/ui/switch"
import { useEventsOptional } from "@/shared/events"
import {
  ADVANCED_META,
  DEVICE_META,
  FORMAT_META,
  INPAINTER_META,
  MODE_HINTS,
  RUN_PARAM_META,
} from "./param-meta"
import {
  DEVICE_OPTIONS,
  INPAINTER_OPTIONS,
  OUTPUT_FORMATS,
  type EditorParams,
} from "./params"
import { TargetsEditor } from "./targets-editor"
import {
  BackendSelectors,
  FieldLabel,
  LlmChip,
  ParamSlider,
  type OptionsShape,
} from "./stage-parts"

const CEILING_HINT = "пусто = вся машина"

const PROPAINTER_KEYS = [
  "propainter_mask_dilation",
  "propainter_ref_stride",
  "propainter_neighbor_length",
  "propainter_subvideo_length",
  "propainter_raft_iter",
] as const

type Props = {
  params: EditorParams
  onChange: (next: EditorParams) => void
  disabled?: boolean
  findDisabled?: boolean
  onFind: () => void
  detectRunning?: boolean
  detectJobId?: string | null
  detectProgress?: { fraction: number; eta: string }
  onOpenConfig?: () => void
}

function Station({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="flex flex-col gap-2 border-t border-border/60 pt-3">
      <h3 className="text-sm font-medium">{title}</h3>
      {children}
    </section>
  )
}

function CeilingInput({
  label,
  value,
  disabled,
  onChange,
}: {
  label: string
  value: string
  disabled?: boolean
  onChange: (value: string) => void
}) {
  return (
    <div className="flex items-center gap-2 text-xs">
      <FieldLabel hint={CEILING_HINT} className="w-[7.5rem] shrink-0">
        {label}
      </FieldLabel>
      <Input
        value={value}
        disabled={disabled}
        inputMode="numeric"
        className="h-7 flex-1 text-xs"
        onChange={(e) => {
          const next = e.target.value.trim()
          if (next !== "" && !/^\d+$/.test(next)) return
          onChange(next)
        }}
      />
    </div>
  )
}

export function ExpertStations({
  params,
  onChange,
  disabled,
  findDisabled,
  onFind,
  detectRunning,
  detectJobId,
  detectProgress,
  onOpenConfig,
}: Props) {
  const events = useEventsOptional()
  const opts = (events?.snapshot?.options as OptionsShape | undefined) ?? null
  const catalog = opts?.models?.inpainter ?? []
  const inpainters = opts?.inpainters?.length ? opts.inpainters : [...INPAINTER_OPTIONS]
  const modelChoices = catalog.filter((m) => m.backend === params.run.inpainter)
  const selectedModel =
    modelChoices.find((m) => m.id === params.run.inpainter_model) ||
    modelChoices.find((m) => m.model_ref === params.run.inpainter_model) ||
    modelChoices[0]

  const setRun = (patch: Partial<EditorParams["run"]>) =>
    onChange({ ...params, run: { ...params.run, ...patch } })
  const setAdv = (key: string, value: string) =>
    onChange({ ...params, advanced: { ...params.advanced, [key]: value } })

  const llava = /llava/i.test(params.run.llm_model || "")

  return (
    <div className="flex flex-col gap-3">
      <Station title="Детектор">
        <BackendSelectors
          detector={params.run.detector}
          detectorModel={params.run.detector_model}
          segmenter={params.run.segmenter}
          segmenterModel={params.run.segmenter_model}
          onChange={(patch) => setRun(patch)}
          disabled={disabled}
          detectorOnly
        />
        <ParamSlider
          label={ADVANCED_META.detector_threshold.label}
          hint={ADVANCED_META.detector_threshold.hint}
          value={Number(params.advanced.detector_threshold || 0.15)}
          min={ADVANCED_META.detector_threshold.min ?? 0.05}
          max={ADVANCED_META.detector_threshold.max ?? 0.5}
          step={ADVANCED_META.detector_threshold.step ?? 0.01}
          disabled={disabled}
          onChange={(v) => setAdv("detector_threshold", String(v))}
        />
        <ParamSlider
          label={ADVANCED_META.detector_keyframes.label}
          hint={ADVANCED_META.detector_keyframes.hint}
          value={Number(params.advanced.detector_keyframes || 0)}
          min={0}
          max={ADVANCED_META.detector_keyframes.max ?? 40}
          step={1}
          disabled={disabled}
          formatValue={(v) => (v <= 0 ? "авто" : String(v))}
          onChange={(v) => setAdv("detector_keyframes", v <= 0 ? "" : String(v))}
        />
        <ParamSlider
          label={ADVANCED_META.detector_nms_iou.label}
          hint={ADVANCED_META.detector_nms_iou.hint}
          value={Number(params.advanced.detector_nms_iou || 0.3)}
          min={ADVANCED_META.detector_nms_iou.min ?? 0.1}
          max={ADVANCED_META.detector_nms_iou.max ?? 0.7}
          step={ADVANCED_META.detector_nms_iou.step ?? 0.05}
          disabled={disabled}
          onChange={(v) => setAdv("detector_nms_iou", String(v))}
        />
        <ParamSlider
          label={ADVANCED_META.detector_max_box_area.label}
          hint={ADVANCED_META.detector_max_box_area.hint}
          value={Number(params.advanced.detector_max_box_area || 0.45)}
          min={ADVANCED_META.detector_max_box_area.min ?? 0.05}
          max={ADVANCED_META.detector_max_box_area.max ?? 0.7}
          step={ADVANCED_META.detector_max_box_area.step ?? 0.01}
          disabled={disabled}
          onChange={(v) => setAdv("detector_max_box_area", String(v))}
        />
        <div className="flex flex-wrap items-center gap-2">
          <Button size="sm" disabled={findDisabled || disabled} onClick={onFind}>
            Найти рамки
          </Button>
          {detectRunning && (
            <>
              <span className="text-xs text-muted-foreground">
                {Math.round((detectProgress?.fraction ?? 0) * 100)}%
                {detectProgress?.eta ? ` · ETA ${detectProgress.eta}` : ""}
              </span>
              <Button
                size="xs"
                variant="outline"
                disabled={!detectJobId}
                onClick={() => detectJobId && void cancelJob(detectJobId)}
              >
                Стоп
              </Button>
            </>
          )}
        </div>
      </Station>

      <Station title="Трекинг">
        <ParamSlider
          label={ADVANCED_META.tracker_min_score.label}
          hint={ADVANCED_META.tracker_min_score.hint}
          value={Number(params.advanced.tracker_min_score || 0.55)}
          min={ADVANCED_META.tracker_min_score.min ?? 0.2}
          max={ADVANCED_META.tracker_min_score.max ?? 0.9}
          step={ADVANCED_META.tracker_min_score.step ?? 0.05}
          disabled={disabled}
          onChange={(v) => setAdv("tracker_min_score", String(v))}
        />
        <ParamSlider
          label={ADVANCED_META.tracker_max_template_area.label}
          hint={ADVANCED_META.tracker_max_template_area.hint}
          value={Number(params.advanced.tracker_max_template_area || 0.12)}
          min={ADVANCED_META.tracker_max_template_area.min ?? 0.02}
          max={ADVANCED_META.tracker_max_template_area.max ?? 0.4}
          step={ADVANCED_META.tracker_max_template_area.step ?? 0.01}
          disabled={disabled}
          onChange={(v) => setAdv("tracker_max_template_area", String(v))}
        />
      </Station>

      <Station title="Сегментация">
        <BackendSelectors
          detector={params.run.detector}
          detectorModel={params.run.detector_model}
          segmenter={params.run.segmenter}
          segmenterModel={params.run.segmenter_model}
          onChange={(patch) => setRun(patch)}
          disabled={disabled}
          segmenterOnly
        />
        <ParamSlider
          label={RUN_PARAM_META.mask_dilate_px.label}
          hint={RUN_PARAM_META.mask_dilate_px.hint}
          value={params.run.mask_dilate_px}
          min={RUN_PARAM_META.mask_dilate_px.min ?? 0}
          max={RUN_PARAM_META.mask_dilate_px.max ?? 15}
          step={1}
          disabled={disabled}
          onChange={(v) => setRun({ mask_dilate_px: v })}
        />
      </Station>

      <Station title="Заливка">
        <div className="flex items-center gap-2 text-xs">
          <FieldLabel className="w-[7.5rem] shrink-0" hint="Чем заполнять вырезанные области.">
            Инпейнтер
          </FieldLabel>
          <Select
            value={params.run.inpainter}
            disabled={disabled}
            onValueChange={(v) => {
              if (!v) return
              const nextModels = catalog.filter((m) => m.backend === v)
              const next = nextModels.find((m) => m.state === "ready") || nextModels[0]
              setRun({
                inpainter: v,
                inpainter_model: next ? modelRef(next.backend, next.model_ref, next.title) : "",
              })
            }}
          >
            <SelectTrigger size="sm" className="flex-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {inpainters.map((o) => (
                <SelectItem key={o} value={o}>
                  {INPAINTER_META[o]?.label ?? o}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <FieldLabel className="w-[7.5rem] shrink-0" hint="Веса выбранного инпейнтера.">
            Модель
          </FieldLabel>
          <Select
            value={selectedModel?.id ?? ""}
            disabled={disabled || modelChoices.length === 0}
            onValueChange={(id) => {
              if (!id) return
              const m = modelChoices.find((x) => x.id === id)
              if (m) setRun({ inpainter_model: modelRef(m.backend, m.model_ref, m.title) })
            }}
          >
            <SelectTrigger size="sm" className="flex-1">
              <SelectValue placeholder="Модель">
                {selectedModel ? selectedModel.title : null}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {modelChoices.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  {m.title}
                  {m.state && m.state !== "ready" ? ` · ${m.state}` : ""}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {PROPAINTER_KEYS.map((key) => {
          const meta = ADVANCED_META[key]
          return (
            <ParamSlider
              key={key}
              label={meta.label}
              hint={meta.hint}
              value={Number(params.advanced[key] || meta.min || 0)}
              min={meta.min ?? 0}
              max={meta.max ?? 100}
              step={meta.step ?? 1}
              disabled={disabled}
              onChange={(v) => setAdv(key, String(v))}
            />
          )
        })}
        <ParamSlider
          label={ADVANCED_META.inpaint_chunk_overlap.label}
          hint={ADVANCED_META.inpaint_chunk_overlap.hint}
          value={Number(params.advanced.inpaint_chunk_overlap || 0)}
          min={ADVANCED_META.inpaint_chunk_overlap.min ?? 0}
          max={ADVANCED_META.inpaint_chunk_overlap.max ?? 32}
          step={1}
          disabled={disabled}
          onChange={(v) => setAdv("inpaint_chunk_overlap", String(v))}
        />
        <CeilingInput
          label="Сторона заливки"
          value={params.run.inpaint_max_side}
          disabled={disabled}
          onChange={(v) => setRun({ inpaint_max_side: v })}
        />
      </Station>

      <Station title="Проверка">
        <div className="flex items-center gap-2">
          <FieldLabel className="flex-1" hint={RUN_PARAM_META.verify.hint}>
            {RUN_PARAM_META.verify.label}
          </FieldLabel>
          <Switch
            checked={params.run.verify}
            disabled={disabled}
            onCheckedChange={(checked) => setRun({ verify: checked })}
          />
        </div>
        <ParamSlider
          label={ADVANCED_META.verify_max_passes.label}
          hint={ADVANCED_META.verify_max_passes.hint}
          value={Number(params.advanced.verify_max_passes || 0)}
          min={ADVANCED_META.verify_max_passes.min ?? 0}
          max={ADVANCED_META.verify_max_passes.max ?? 3}
          step={1}
          disabled={disabled}
          onChange={(v) => setAdv("verify_max_passes", String(v))}
        />
        <ParamSlider
          label={RUN_PARAM_META.verify_max_coverage.label}
          hint={RUN_PARAM_META.verify_max_coverage.hint}
          value={params.run.verify_max_coverage}
          min={RUN_PARAM_META.verify_max_coverage.min ?? 0.01}
          max={RUN_PARAM_META.verify_max_coverage.max ?? 0.5}
          step={RUN_PARAM_META.verify_max_coverage.step ?? 0.01}
          disabled={disabled}
          onChange={(v) => setRun({ verify_max_coverage: v })}
        />
        <ParamSlider
          label={RUN_PARAM_META.min_mask_coverage.label}
          hint={RUN_PARAM_META.min_mask_coverage.hint}
          value={params.run.min_mask_coverage}
          min={RUN_PARAM_META.min_mask_coverage.min ?? 0}
          max={RUN_PARAM_META.min_mask_coverage.max ?? 0.02}
          step={RUN_PARAM_META.min_mask_coverage.step ?? 0.0001}
          disabled={disabled}
          onChange={(v) => setRun({ min_mask_coverage: v })}
        />
        <div className="flex items-center gap-2">
          <FieldLabel className="flex-1" hint="Второй проход детектора по остатку. Выключен, пока не включите.">
            Повторный поиск
          </FieldLabel>
          <Switch
            checked={params.run.verify_redetect}
            disabled={disabled}
            onCheckedChange={(checked) => setRun({ verify_redetect: checked })}
          />
        </div>
      </Station>

      <Station title="Разбор фразы">
        <LlmChip
          value={params.run.llm_model}
          disabled={disabled}
          onOpenConfig={onOpenConfig}
          onChange={(patch) => setRun(patch)}
        />
        <ParamSlider
          label={ADVANCED_META.prompt_frame_stride.label}
          hint={ADVANCED_META.prompt_frame_stride.hint}
          value={Number(params.advanced.prompt_frame_stride || 0)}
          min={ADVANCED_META.prompt_frame_stride.min ?? 0}
          max={ADVANCED_META.prompt_frame_stride.max ?? 20}
          step={1}
          disabled={disabled}
          onChange={(v) => setAdv("prompt_frame_stride", String(v))}
        />
        <ParamSlider
          label={ADVANCED_META.prompt_frame_max.label}
          hint={ADVANCED_META.prompt_frame_max.hint}
          value={Number(params.advanced.prompt_frame_max || 8)}
          min={ADVANCED_META.prompt_frame_max.min ?? 1}
          max={ADVANCED_META.prompt_frame_max.max ?? 24}
          step={1}
          disabled={disabled}
          onChange={(v) => setAdv("prompt_frame_max", String(v))}
        />
        <ParamSlider
          label={ADVANCED_META.vision_batch.label}
          hint={ADVANCED_META.vision_batch.hint}
          value={Math.min(Number(params.advanced.vision_batch || 2), llava ? 2 : 8)}
          min={ADVANCED_META.vision_batch.min ?? 1}
          max={llava ? 2 : (ADVANCED_META.vision_batch.max ?? 8)}
          step={1}
          disabled={disabled}
          onChange={(v) => setAdv("vision_batch", String(llava ? Math.min(v, 2) : v))}
        />
        <ParamSlider
          label={ADVANCED_META.parse_chunk_frames.label}
          hint={ADVANCED_META.parse_chunk_frames.hint}
          value={Number(params.advanced.parse_chunk_frames || 0)}
          min={ADVANCED_META.parse_chunk_frames.min ?? 0}
          max={ADVANCED_META.parse_chunk_frames.max ?? 300}
          step={ADVANCED_META.parse_chunk_frames.step ?? 10}
          disabled={disabled}
          onChange={(v) => setAdv("parse_chunk_frames", String(v))}
        />
        <TargetsEditor
          targets={params.targets}
          disabled={disabled}
          onChange={(targets) => onChange({ ...params, targets })}
        />
      </Station>

      <Station title="Устройство">
        <div className="flex items-center gap-2 text-xs">
          <FieldLabel className="w-[7.5rem] shrink-0" hint={DEVICE_META[params.run.device || "auto"]?.hint}>
            Устройство
          </FieldLabel>
          <Select
            value={params.run.device || "auto"}
            disabled={disabled}
            onValueChange={(v) => {
              if (v) setRun({ device: v })
            }}
          >
            <SelectTrigger size="sm" className="flex-1">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {DEVICE_OPTIONS.map((d) => (
                <SelectItem key={d} value={d}>
                  {DEVICE_META[d]?.label ?? d}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <CeilingInput
          label="Потолок VRAM"
          value={params.run.max_vram_mb}
          disabled={disabled}
          onChange={(v) => setRun({ max_vram_mb: v })}
        />
        <CeilingInput
          label="Потоки CPU"
          value={params.run.cpu_threads}
          disabled={disabled}
          onChange={(v) => setRun({ cpu_threads: v })}
        />
        <ParamSlider
          label={ADVANCED_META.inpaint_workers.label}
          hint={ADVANCED_META.inpaint_workers.hint}
          value={Number(params.advanced.inpaint_workers || 0)}
          min={0}
          max={ADVANCED_META.inpaint_workers.max ?? 16}
          step={1}
          disabled={disabled}
          formatValue={(v) => (v <= 0 ? "авто" : String(v))}
          onChange={(v) => setAdv("inpaint_workers", String(v))}
        />
        <div className="flex items-center gap-2">
          <FieldLabel className="flex-1" hint={RUN_PARAM_META.keep_workdir.hint}>
            {RUN_PARAM_META.keep_workdir.label}
          </FieldLabel>
          <Switch
            checked={params.run.keep_workdir}
            disabled={disabled}
            onCheckedChange={(checked) => setRun({ keep_workdir: checked })}
          />
        </div>
        <div className="flex flex-col gap-1.5 text-xs">
          <FieldLabel hint="Форматы, которые job кладёт рядом с MP4.">Форматы</FieldLabel>
          {OUTPUT_FORMATS.map((f) => (
            <label key={f} className="flex items-center gap-2">
              <Switch
                size="sm"
                checked={params.run.formats.includes(f)}
                disabled={disabled}
                onCheckedChange={(checked) => {
                  const next = checked
                    ? [...params.run.formats, f]
                    : params.run.formats.filter((x) => x !== f)
                  setRun({ formats: next.length ? next : ["mp4"] })
                }}
              />
              <span>{FORMAT_META[f]?.label ?? f}</span>
            </label>
          ))}
        </div>
      </Station>

      <Station title="Осмотр">
        <ParamSlider
          label="Каждый N-й"
          hint={MODE_HINTS.detectStride}
          value={params.detect.stride}
          min={1}
          max={30}
          step={1}
          disabled={disabled}
          onChange={(v) => onChange({ ...params, detect: { ...params.detect, stride: v, all: false } })}
        />
      </Station>
    </div>
  )
}

function modelRef(backend: string, modelRefValue: string, title: string): string {
  if (backend === "lama") return "big-lama"
  if (modelRefValue.startsWith("http://") || modelRefValue.startsWith("https://")) return title
  return modelRefValue
}
