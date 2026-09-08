import type { TargetRow } from "@/entities/targets"

export type StageTarget = TargetRow & {
  enabled: boolean
  source: "auto" | "manual"
}

export type InpaintMode = "tracks" | "masks" | "prompt"

export type EditorParams = {
  prompt: string
  targets: StageTarget[]
  detect: { mode: "targets" | "prompt"; all: boolean; stride: number }
  run: {
    inpainter: string
    mask_dilate_px: number
    telea_radius: number
    verify: boolean
    formats: string[]
    keep_workdir: boolean
    min_mask_coverage: number
    verify_max_coverage: number
    llm_base_url: string
    llm_api_key: string
    llm_model: string
  }
  advanced: Record<string, string>
}

export const INPAINTER_OPTIONS = ["opencv-telea", "lama", "propainter"] as const

export const OUTPUT_FORMATS = ["mp4", "mov", "mkv", "webm"] as const

export const ADVANCED_DEFAULTS: Record<string, string> = {
  detector_threshold: "0.15",
  detector_keyframes: "",
  detector_nms_iou: "0.3",
  detector_max_box_area: "0.25",
  tracker_min_score: "0.55",
  tracker_max_template_area: "0.12",
  prompt_frame_stride: "4",
  prompt_frame_max: "8",
  parse_chunk_frames: "0",
  vision_batch: "2",
  propainter_mask_dilation: "4",
  propainter_ref_stride: "10",
  propainter_neighbor_length: "10",
  propainter_subvideo_length: "80",
  propainter_raft_iter: "20",
}

export const DEFAULT_PARAMS: EditorParams = {
  prompt: "",
  targets: [],
  detect: { mode: "targets", all: true, stride: 8 },
  run: {
    inpainter: "opencv-telea",
    mask_dilate_px: 3,
    telea_radius: 9,
    verify: true,
    formats: ["mp4"],
    keep_workdir: false,
    min_mask_coverage: 0.0004,
    verify_max_coverage: 0.12,
    llm_base_url: "",
    llm_api_key: "",
    llm_model: "",
  },
  advanced: { ...ADVANCED_DEFAULTS },
}

export function resetParams(): EditorParams {
  return { ...DEFAULT_PARAMS, targets: [], advanced: { ...ADVANCED_DEFAULTS } }
}

export function autoStride(frameCount: number): number {
  return Math.max(1, Math.round(frameCount / 400))
}

export function enabledTargets(params: EditorParams): TargetRow[] {
  return params.targets
    .filter((t) => t.enabled && t.query.trim())
    .map(({ kind, query, where }) => ({ kind, query, where }))
}

export function toRunParams(params: EditorParams): Record<string, string | number | boolean> {
  const out: Record<string, string | number | boolean> = {
    inpainter: params.run.inpainter,
    mask_dilate_px: params.run.mask_dilate_px,
    telea_radius: params.run.telea_radius,
    verify: params.run.verify ? "1" : "0",
    keep_workdir: params.run.keep_workdir ? "1" : "0",
    min_mask_coverage: params.run.min_mask_coverage,
    verify_max_coverage: params.run.verify_max_coverage,
    llm_base_url: params.run.llm_base_url,
    llm_api_key: params.run.llm_api_key,
    llm_model: params.run.llm_model,
    formats: params.run.formats.join(","),
    ...params.advanced,
  }
  return out
}

export function presetSnapshot(params: EditorParams): Record<string, unknown> {
  return { detect: params.detect, run: params.run, advanced: params.advanced }
}

export function applyPreset(params: EditorParams, payload: Record<string, unknown>): EditorParams {
  const detect = isRecord(payload.detect) ? (payload.detect as EditorParams["detect"]) : null
  const run = isRecord(payload.run) ? (payload.run as EditorParams["run"]) : null
  const advanced = isRecord(payload.advanced) ? (payload.advanced as Record<string, string>) : null
  return {
    ...params,
    detect: detect ? { ...params.detect, ...detect, mode: detect.mode === "prompt" ? "prompt" : "targets" } : params.detect,
    run: run ? { ...params.run, ...pickRun(run) } : params.run,
    advanced: advanced ? { ...ADVANCED_DEFAULTS, ...advanced } : params.advanced,
  }
}

function pickRun(run: Record<string, unknown>): Partial<EditorParams["run"]> {
  const out: Partial<EditorParams["run"]> = {}
  if (typeof run.inpainter === "string") out.inpainter = run.inpainter
  if (typeof run.mask_dilate_px === "number") out.mask_dilate_px = run.mask_dilate_px
  if (typeof run.telea_radius === "number") out.telea_radius = run.telea_radius
  if (typeof run.verify === "boolean") out.verify = run.verify
  if (Array.isArray(run.formats)) out.formats = run.formats.map(String)
  if (typeof run.keep_workdir === "boolean") out.keep_workdir = run.keep_workdir
  if (typeof run.min_mask_coverage === "number") out.min_mask_coverage = run.min_mask_coverage
  if (typeof run.verify_max_coverage === "number") out.verify_max_coverage = run.verify_max_coverage
  if (typeof run.llm_base_url === "string") out.llm_base_url = run.llm_base_url
  if (typeof run.llm_api_key === "string") out.llm_api_key = run.llm_api_key
  if (typeof run.llm_model === "string") out.llm_model = run.llm_model
  return out
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}
