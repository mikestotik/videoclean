import type { TargetRow } from "@/entities/targets"

export type StageTarget = TargetRow & {
  enabled: boolean
  source: "auto" | "manual"
}

export type InpaintMode = "tracks" | "masks" | "prompt"

export type MaskPolicy = "static" | "propagate"

export type EditorParams = {
  prompt: string
  targets: StageTarget[]
  detect: { mode: "targets" | "prompt"; all: boolean; stride: number }
  maskPolicy: MaskPolicy
  run: {
    profile: "custom" | "fast" | "balanced" | "quality"
    inpainter: string
    inpainter_model: string
    mask_dilate_px: number
    verify: boolean
    formats: string[]
    keep_workdir: boolean
    min_mask_coverage: number
    verify_max_coverage: number
    llm_base_url: string
    llm_api_key: string
    llm_model: string
    detector: string
    detector_model: string
    segmenter: string
    segmenter_model: string
    device: string
  }
  advanced: Record<string, string>
}

export type PipelineProfileId = EditorParams["run"]["profile"]

export const INPAINTER_OPTIONS = ["lama", "propainter"] as const

export const DEVICE_OPTIONS = ["cpu", "cuda", "mps"] as const

export const OUTPUT_FORMATS = [
  "mp4",
  "mov",
  "mkv",
  "webm",
  "hls-fmp4",
  "hls-ts",
  "dash",
] as const

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
  verify_max_passes: "1",
  inpaint_workers: "0",
  inpaint_chunk_overlap: "8",
}

/** Client-side mirror of server profile_defaults (device-aware). */
export function builtInProfileDefaults(
  id: Exclude<PipelineProfileId, "custom">,
  device: string,
): { run: Partial<EditorParams["run"]>; advanced: Record<string, string> } {
  const cuda = (device || "cpu").toLowerCase() === "cuda"
  if (id === "fast") {
    return {
      run: {
        profile: "fast",
        verify: false,
        mask_dilate_px: 2,
        segmenter: "sam2",
        inpainter: "lama",
      },
      advanced: {
        detector_keyframes: "6",
        verify_max_passes: "0",
        inpaint_workers: "0",
        inpaint_chunk_overlap: "0",
      },
    }
  }
  if (id === "balanced") {
    return {
      run: {
        profile: "balanced",
        verify: true,
        mask_dilate_px: 3,
        segmenter: "sam2",
        inpainter: "lama",
      },
      advanced: {
        detector_keyframes: "10",
        verify_max_passes: "1",
        inpaint_workers: "0",
        inpaint_chunk_overlap: "8",
      },
    }
  }
  return {
    run: {
      profile: "quality",
      verify: true,
      mask_dilate_px: 5,
      segmenter: cuda ? "sam2-video" : "sam2",
      inpainter: cuda ? "propainter" : "lama",
    },
    advanced: {
      detector_keyframes: cuda ? "16" : "12",
      verify_max_passes: "2",
      inpaint_workers: cuda ? "1" : "0",
      inpaint_chunk_overlap: "12",
      propainter_subvideo_length: "80",
    },
  }
}

export function applyBuiltInProfile(params: EditorParams, id: PipelineProfileId): EditorParams {
  if (id === "custom") {
    return { ...params, run: { ...params.run, profile: "custom" } }
  }
  const patch = builtInProfileDefaults(id, params.run.device)
  return {
    ...params,
    run: { ...params.run, ...patch.run, profile: id },
    advanced: { ...params.advanced, ...patch.advanced },
  }
}

export const DEFAULT_PARAMS: EditorParams = {
  prompt: "",
  targets: [],
  detect: { mode: "targets", all: true, stride: 8 },
  maskPolicy: "static",
  run: {
    profile: "balanced",
    inpainter: "lama",
    inpainter_model: "",
    mask_dilate_px: 3,
    verify: true,
    formats: ["mp4"],
    keep_workdir: false,
    min_mask_coverage: 0.0004,
    verify_max_coverage: 0.12,
    llm_base_url: "",
    llm_api_key: "",
    llm_model: "",
    detector: "",
    detector_model: "",
    segmenter: "sam2",
    segmenter_model: "",
    device: "",
  },
  advanced: {
    ...ADVANCED_DEFAULTS,
    detector_keyframes: "10",
    verify_max_passes: "1",
    inpaint_chunk_overlap: "8",
  },
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
    profile: params.run.profile,
    inpainter: params.run.inpainter,
    inpainter_model: params.run.inpainter_model,
    mask_dilate_px: params.run.mask_dilate_px,
    verify: params.run.verify ? "1" : "0",
    keep_workdir: params.run.keep_workdir ? "1" : "0",
    min_mask_coverage: params.run.min_mask_coverage,
    verify_max_coverage: params.run.verify_max_coverage,
    llm_base_url: params.run.llm_base_url,
    llm_api_key: params.run.llm_api_key,
    llm_model: params.run.llm_model,
    detector: params.run.detector,
    detector_model: params.run.detector_model,
    segmenter: params.run.segmenter,
    segmenter_model: params.run.segmenter_model,
    device: params.run.device,
    mask_policy: params.maskPolicy,
    ...params.advanced,
  }
  return out
}

/** Params that affect detect preview (segmenter is forced to sam2 on the server). */
export function toDetectParams(params: EditorParams): Record<string, string | number | boolean> {
  const run = toRunParams(params)
  const {
    segmenter: _seg,
    segmenter_model: _segModel,
    inpainter: _inp,
    verify: _verify,
    mask_policy: _policy,
    propainter_mask_dilation: _pmd,
    propainter_ref_stride: _prs,
    propainter_neighbor_length: _pnl,
    propainter_subvideo_length: _psl,
    propainter_raft_iter: _pri,
    ...detectParams
  } = run
  return detectParams
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
  if (
    run.profile === "custom" ||
    run.profile === "fast" ||
    run.profile === "balanced" ||
    run.profile === "quality"
  ) {
    out.profile = run.profile
  }
  if (typeof run.inpainter === "string") out.inpainter = run.inpainter
  if (typeof run.inpainter_model === "string") out.inpainter_model = run.inpainter_model
  if (typeof run.mask_dilate_px === "number") out.mask_dilate_px = run.mask_dilate_px
  if (typeof run.verify === "boolean") out.verify = run.verify
  if (Array.isArray(run.formats)) out.formats = run.formats.map(String)
  if (typeof run.keep_workdir === "boolean") out.keep_workdir = run.keep_workdir
  if (typeof run.min_mask_coverage === "number") out.min_mask_coverage = run.min_mask_coverage
  if (typeof run.verify_max_coverage === "number") out.verify_max_coverage = run.verify_max_coverage
  if (typeof run.llm_base_url === "string") out.llm_base_url = run.llm_base_url
  if (typeof run.llm_api_key === "string") out.llm_api_key = run.llm_api_key
  if (typeof run.llm_model === "string") out.llm_model = run.llm_model
  if (typeof run.detector_model === "string") out.detector_model = run.detector_model
  if (typeof run.segmenter_model === "string") out.segmenter_model = run.segmenter_model
  if (typeof run.detector === "string") out.detector = run.detector
  if (typeof run.detector_model === "string") out.detector_model = run.detector_model
  if (typeof run.segmenter === "string") out.segmenter = run.segmenter
  if (typeof run.segmenter_model === "string") out.segmenter_model = run.segmenter_model
  if (typeof run.device === "string") out.device = run.device
  return out
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}
