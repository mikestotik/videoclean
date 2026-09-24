import type { TargetRow } from "@/entities/targets"
import { paramLabel } from "./param-meta"

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
    verify_redetect: boolean
    max_vram_mb: string
    cpu_threads: string
    inpaint_max_side: string
    webm_crf: number
    segment_seconds: number
  }
  advanced: Record<string, string>
}

export type PipelineProfileId = EditorParams["run"]["profile"]

export type BuiltinProfileId = Exclude<PipelineProfileId, "custom">

export type Scenario =
  | { kind: "builtin"; id: BuiltinProfileId; resolvedDevice: string }
  | { kind: "preset"; id: string; payload: Record<string, unknown>; resolvedDevice: string }
  | { kind: "custom"; resolvedDevice: string }

export const INPAINTER_OPTIONS = ["lama", "propainter"] as const

export const DEVICE_OPTIONS = ["auto", "cpu", "cuda", "mps"] as const

export const OUTPUT_FORMATS = [
  "mp4",
  "mov",
  "mkv",
  "webm",
  "hls-fmp4",
  "hls-ts",
  "dash",
] as const

export const PROFILE_KEYS = [
  "verify",
  "verify_max_passes",
  "detector_keyframes",
  "mask_dilate_px",
  "segmenter",
  "inpainter",
  "inpaint_workers",
  "inpaint_chunk_overlap",
  "propainter_subvideo_length",
] as const

export type ProfileKey = (typeof PROFILE_KEYS)[number]

const PRESET_SKIP = new Set([
  "prompt",
  "userPrompt",
  "stride",
  "detect",
  "run",
  "advanced",
  "masks",
  "masks_override",
  "tracks",
  "tracks_override",
  "targets",
  "targets_override",
  "webhook_url",
  "webhook_secret",
  "source_id",
  "kind",
  "indices",
  "start",
  "count",
  "all",
  "input_path",
  "output_path",
  "llm_api_key",
  "video",
])

export const ADVANCED_DEFAULTS: Record<string, string> = {
  detector_threshold: "0.15",
  detector_keyframes: "",
  detector_nms_iou: "0.3",
  detector_max_box_area: "0.45",
  tracker_min_score: "0.55",
  tracker_max_template_area: "0.12",
  prompt_frame_stride: "4",
  prompt_frame_max: "8",
  parse_chunk_frames: "0",
  vision_batch: "2",
  select_relax: "1",
  propainter_mask_dilation: "4",
  propainter_ref_stride: "10",
  propainter_neighbor_length: "10",
  propainter_subvideo_length: "80",
  propainter_raft_iter: "20",
  verify_max_passes: "1",
  inpaint_workers: "0",
  inpaint_chunk_overlap: "8",
}

type Json = string | number | boolean | null | string[]

const SAVE_KEYS = [
  "profile",
  "device",
  "detector",
  "detector_model",
  "detector_threshold",
  "segmenter",
  "segmenter_model",
  "inpainter",
  "inpainter_model",
  "verify",
  "mask_dilate_px",
  "keep_workdir",
  "min_mask_coverage",
  "verify_max_coverage",
  "formats",
  "webm_crf",
  "segment_seconds",
  "llm_model",
  "llm_base_url",
  "detector_keyframes",
  "detector_nms_iou",
  "detector_max_box_area",
  "tracker_min_score",
  "tracker_max_template_area",
  "prompt_frame_stride",
  "prompt_frame_max",
  "parse_chunk_frames",
  "vision_batch",
  "select_relax",
  "propainter_mask_dilation",
  "propainter_ref_stride",
  "propainter_neighbor_length",
  "propainter_subvideo_length",
  "propainter_raft_iter",
  "verify_max_passes",
  "inpaint_workers",
  "inpaint_chunk_overlap",
  "verify_redetect",
  "max_vram_mb",
  "cpu_threads",
  "inpaint_max_side",
  "mask_policy",
] as const

type SaveKey = (typeof SAVE_KEYS)[number]

/** cpu | cuda | mps. Never "". `auto` and empty fall through to cpu for the mirror only. */
export function recipeDevice(device: string): "cpu" | "cuda" | "mps" {
  const d = (device || "").trim().toLowerCase()
  if (d === "cuda" || d === "mps" || d === "cpu") return d
  return "cpu"
}

/** Client-side mirror of server profile_defaults after the engine change (device-aware). */
export function builtInProfileDefaults(
  id: BuiltinProfileId,
  device: string,
): { run: Partial<EditorParams["run"]>; advanced: Record<string, string> } {
  const cuda = recipeDevice(device) === "cuda"
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
      inpaint_workers: "0",
      inpaint_chunk_overlap: "12",
      ...(cuda ? { propainter_subvideo_length: "80" } : {}),
    },
  }
}

const BASE_PARAMS: EditorParams = {
  prompt: "",
  targets: [],
  detect: { mode: "targets", all: false, stride: 8 },
  maskPolicy: "propagate",
  run: {
    profile: "custom",
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
    device: "auto",
    verify_redetect: false,
    max_vram_mb: "",
    cpu_threads: "",
    inpaint_max_side: "",
    webm_crf: 32,
    segment_seconds: 6,
  },
  advanced: { ...ADVANCED_DEFAULTS },
}

export function cloneParams(params: EditorParams): EditorParams {
  return {
    ...params,
    targets: params.targets.map((t) => ({ ...t })),
    detect: { ...params.detect },
    run: { ...params.run, formats: [...params.run.formats] },
    advanced: { ...params.advanced },
  }
}

export function pipelineBaselineParams(): EditorParams {
  return cloneParams(BASE_PARAMS)
}

export function applyBuiltInProfile(
  params: EditorParams,
  id: PipelineProfileId,
  resolvedDevice = "cpu",
): EditorParams {
  if (id === "custom") {
    return { ...params, run: { ...params.run, profile: "custom" } }
  }
  const patch = builtInProfileDefaults(id, resolvedDevice)
  const advanced = { ...params.advanced, ...patch.advanced }
  if (!patch.advanced.propainter_subvideo_length) {
    advanced.propainter_subvideo_length = ADVANCED_DEFAULTS.propainter_subvideo_length ?? "80"
  }
  if (!patch.advanced.inpaint_workers) advanced.inpaint_workers = "0"
  return {
    ...params,
    run: { ...params.run, ...patch.run, profile: id },
    advanced,
  }
}

export function cleanBuiltinParams(id: BuiltinProfileId, resolvedDevice: string): EditorParams {
  return applyBuiltInProfile(pipelineBaselineParams(), id, resolvedDevice)
}

export const DEFAULT_PARAMS: EditorParams = cleanBuiltinParams("balanced", "cpu")

export function resetParams(): EditorParams {
  return cloneParams(DEFAULT_PARAMS)
}

export function autoStride(frameCount: number): number {
  return Math.max(1, Math.round(frameCount / 400))
}

export function enabledTargets(params: EditorParams): TargetRow[] {
  return params.targets
    .filter((t) => t.enabled && t.query.trim())
    .map(({ kind, query, where }) => ({ kind, query, where }))
}

export function hasEnabledManualTarget(params: EditorParams): boolean {
  return params.targets.some((t) => t.enabled && t.source === "manual" && t.query.trim().length > 0)
}

export function effectiveRecipeDevice(params: EditorParams, resolvedDevice: string): "cpu" | "cuda" | "mps" {
  const pin = (params.run.device || "auto").trim().toLowerCase()
  if (pin === "cpu" || pin === "cuda" || pin === "mps") return pin
  return recipeDevice(resolvedDevice)
}

function advNum(params: EditorParams, key: string, fallback: number): number {
  const raw = params.advanced[key]
  if (raw === undefined || raw.trim() === "") return fallback
  const n = Number(raw)
  return Number.isFinite(n) ? n : fallback
}

function advOptionalNum(params: EditorParams, key: string): number | null {
  const raw = params.advanced[key]
  if (raw === undefined || raw.trim() === "") return null
  const n = Number(raw)
  return Number.isFinite(n) ? n : null
}

function ceiling(raw: string): number | null {
  const t = raw.trim()
  if (!t) return null
  const n = Number(t)
  return Number.isFinite(n) ? n : null
}

function savableState(params: EditorParams): Record<SaveKey, Json> {
  return {
    profile: params.run.profile,
    device: (params.run.device || "auto").trim().toLowerCase() || "auto",
    detector: params.run.detector,
    detector_model: params.run.detector_model,
    detector_threshold: advNum(params, "detector_threshold", 0.15),
    segmenter: params.run.segmenter,
    segmenter_model: params.run.segmenter_model,
    inpainter: params.run.inpainter,
    inpainter_model: params.run.inpainter_model,
    verify: params.run.verify,
    mask_dilate_px: params.run.mask_dilate_px,
    keep_workdir: params.run.keep_workdir,
    min_mask_coverage: params.run.min_mask_coverage,
    verify_max_coverage: params.run.verify_max_coverage,
    formats: [...params.run.formats],
    webm_crf: params.run.webm_crf,
    segment_seconds: params.run.segment_seconds,
    llm_model: params.run.llm_model,
    llm_base_url: params.run.llm_base_url,
    detector_keyframes: advOptionalNum(params, "detector_keyframes"),
    detector_nms_iou: advNum(params, "detector_nms_iou", 0.3),
    detector_max_box_area: advNum(params, "detector_max_box_area", 0.45),
    tracker_min_score: advNum(params, "tracker_min_score", 0.55),
    tracker_max_template_area: advNum(params, "tracker_max_template_area", 0.12),
    prompt_frame_stride: advNum(params, "prompt_frame_stride", 4),
    prompt_frame_max: advNum(params, "prompt_frame_max", 8),
    parse_chunk_frames: advNum(params, "parse_chunk_frames", 0),
    vision_batch: advNum(params, "vision_batch", 2),
    select_relax: (params.advanced.select_relax ?? "1") !== "0",
    propainter_mask_dilation: advNum(params, "propainter_mask_dilation", 4),
    propainter_ref_stride: advNum(params, "propainter_ref_stride", 10),
    propainter_neighbor_length: advNum(params, "propainter_neighbor_length", 10),
    propainter_subvideo_length: advNum(params, "propainter_subvideo_length", 80),
    propainter_raft_iter: advNum(params, "propainter_raft_iter", 20),
    verify_max_passes: advNum(params, "verify_max_passes", 1),
    inpaint_workers: advNum(params, "inpaint_workers", 0),
    inpaint_chunk_overlap: advNum(params, "inpaint_chunk_overlap", 8),
    verify_redetect: params.run.verify_redetect,
    max_vram_mb: ceiling(params.run.max_vram_mb),
    cpu_threads: ceiling(params.run.cpu_threads),
    inpaint_max_side: ceiling(params.run.inpaint_max_side),
    mask_policy: params.maskPolicy,
  }
}

function sameJson(a: Json, b: Json): boolean {
  if (a === b) return true
  if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) < 1e-6
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((v, i) => v === b[i])
  }
  return false
}

function diffState(params: EditorParams, baseline: EditorParams): Record<string, Json> {
  const a = savableState(params)
  const b = savableState(baseline)
  const out: Record<string, Json> = {}
  for (const key of SAVE_KEYS) {
    if (!sameJson(a[key], b[key])) out[key] = a[key]
  }
  return out
}

export function profileKeysMatch(params: EditorParams, id: BuiltinProfileId, resolvedDevice: string): boolean {
  const clean = cleanBuiltinParams(id, resolvedDevice)
  const a = savableState(params)
  const b = savableState(clean)
  return PROFILE_KEYS.every((key) => sameJson(a[key], b[key]))
}

export function inferScenario(params: EditorParams, resolvedDevice: string): Scenario {
  const device = effectiveRecipeDevice(params, resolvedDevice)
  const profile = params.run.profile
  if (
    (profile === "fast" || profile === "balanced" || profile === "quality") &&
    profileKeysMatch(params, profile, device)
  ) {
    return { kind: "builtin", id: profile, resolvedDevice: device }
  }
  return { kind: "custom", resolvedDevice: device }
}

function inferResolvedDevice(params: EditorParams): string {
  return effectiveRecipeDevice(params, "cpu")
}

function writeWire(out: Record<string, string | number | boolean>, diff: Record<string, Json>) {
  for (const [key, value] of Object.entries(diff)) {
    if (value === null || value === "") continue
    if (typeof value === "boolean") {
      out[key] = value ? "1" : "0"
      continue
    }
    if (Array.isArray(value)) {
      if (value.length === 0) continue
      out[key] = value.join(",")
      continue
    }
    out[key] = value
  }
}

/** Fields the job form should actually send. Built-in recipe keys stay off the wire. */
export function explicitFields(
  params: EditorParams,
  scenario?: Scenario,
): Record<string, string | number | boolean> {
  const resolved = scenario?.resolvedDevice ?? inferResolvedDevice(params)
  const chosen = scenario ?? inferScenario(params, resolved)
  const out: Record<string, string | number | boolean> = {}
  if (chosen.kind === "preset") {
    out.preset = chosen.id
    const base = applyPreset(pipelineBaselineParams(), chosen.payload, resolved)
    const diff = diffState(params, base)
    delete diff.profile
    writeWire(out, diff)
    return out
  }
  if (chosen.kind === "builtin") {
    out.profile = chosen.id
    const diff = diffState(params, cleanBuiltinParams(chosen.id, chosen.resolvedDevice))
    delete diff.profile
    for (const key of PROFILE_KEYS) delete diff[key]
    writeWire(out, diff)
    return out
  }
  out.profile = "custom"
  const diff = diffState(params, pipelineBaselineParams())
  delete diff.profile
  writeWire(out, diff)
  return out
}

export function presetPayload(params: EditorParams, resolvedDevice: string): Record<string, unknown> {
  const scenario = inferScenario(params, resolvedDevice)
  if (scenario.kind === "builtin") {
    const diff = diffState(params, cleanBuiltinParams(scenario.id, scenario.resolvedDevice))
    delete diff.profile
    for (const key of PROFILE_KEYS) delete diff[key]
    return { profile: scenario.id, ...diff }
  }
  const diff = diffState(params, pipelineBaselineParams())
  delete diff.profile
  return { profile: "custom", ...diff }
}

export function samePresetPayload(
  params: EditorParams,
  payload: Record<string, unknown>,
  resolvedDevice: string,
): boolean {
  const applied = applyPreset(pipelineBaselineParams(), payload, resolvedDevice)
  const left = presetPayload(params, resolvedDevice)
  const right = presetPayload(applied, resolvedDevice)
  return stable(left) === stable(right)
}

function stable(value: Record<string, unknown>): string {
  const keys = Object.keys(value).sort()
  const out: Record<string, unknown> = {}
  for (const key of keys) out[key] = value[key]
  return JSON.stringify(out)
}

export function divergentLabels(params: EditorParams, baseline: EditorParams): string[] {
  const diff = diffState(params, baseline)
  delete diff.profile
  return Object.keys(diff).map((key) => paramLabel(key))
}

export function structuralDivergence(params: EditorParams, resolvedDevice: string): boolean {
  const pin = (params.run.device || "auto").trim().toLowerCase()
  const ceilings =
    params.run.max_vram_mb.trim() !== "" ||
    params.run.cpu_threads.trim() !== "" ||
    params.run.inpaint_max_side.trim() !== ""
  if (pin !== "auto" && pin !== "") return true
  if (ceilings) return true
  const profile = params.run.profile
  if (profile !== "fast" && profile !== "balanced" && profile !== "quality") return true
  return !profileKeysMatch(params, profile, effectiveRecipeDevice(params, resolvedDevice))
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value)
}

export function flattenPresetPayload(payload: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  const run = isRecord(payload.run) ? payload.run : null
  const advanced = isRecord(payload.advanced) ? payload.advanced : null
  if (run) {
    for (const [key, value] of Object.entries(run)) out[key] = value
  }
  if (advanced) {
    for (const [key, value] of Object.entries(advanced)) out[key] = value
  }
  for (const [key, value] of Object.entries(payload)) {
    if (key === "detect" || key === "run" || key === "advanced") continue
    out[key] = value
  }
  for (const key of PRESET_SKIP) delete out[key]
  return out
}

function asBool(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") return value
  if (typeof value === "number") return value !== 0
  if (typeof value === "string") {
    const t = value.trim().toLowerCase()
    if (!t) return fallback
    return t === "1" || t === "true" || t === "yes" || t === "on"
  }
  return fallback
}

function asNum(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string" && value.trim() !== "") {
    const n = Number(value)
    if (Number.isFinite(n)) return n
  }
  return fallback
}

function asOptionalNumString(value: unknown): string | undefined {
  if (value === null || value === undefined || value === "") return ""
  if (typeof value === "number" && Number.isFinite(value)) return String(value)
  if (typeof value === "string") return value.trim()
  return undefined
}

function asString(value: unknown, fallback = ""): string {
  if (typeof value === "string") return value
  if (typeof value === "number" && Number.isFinite(value)) return String(value)
  return fallback
}

function asFormats(value: unknown, fallback: string[]): string[] {
  if (Array.isArray(value)) {
    const list = value.map((v) => String(v).trim()).filter(Boolean)
    return list.length ? list : fallback
  }
  if (typeof value === "string" && value.trim()) {
    const list = value.split(",").map((v) => v.trim()).filter(Boolean)
    return list.length ? list : fallback
  }
  return fallback
}

function isProfile(value: unknown): value is PipelineProfileId {
  return value === "custom" || value === "fast" || value === "balanced" || value === "quality"
}

/** Flat preset payload. Nested `{detect, run, advanced}` is lifted, then dropped. */
export function applyPreset(
  params: EditorParams,
  payload: Record<string, unknown>,
  resolvedDevice = "cpu",
): EditorParams {
  const flat = flattenPresetPayload(payload)
  let next = pipelineBaselineParams()
  next = {
    ...next,
    prompt: params.prompt,
    targets: params.targets.map((t) => ({ ...t })),
    detect: { ...next.detect, stride: params.detect.stride },
  }
  const profile = flat.profile
  if (profile === "fast" || profile === "balanced" || profile === "quality") {
    const pinned =
      typeof flat.device === "string" && ["cpu", "cuda", "mps"].includes(flat.device.trim().toLowerCase())
        ? flat.device
        : resolvedDevice
    next = applyBuiltInProfile(next, profile, pinned)
  } else if (profile === "custom" || profile === undefined) {
    next = { ...next, run: { ...next.run, profile: "custom" } }
  }
  const run = { ...next.run }
  const advanced = { ...next.advanced }
  if (isProfile(flat.profile)) run.profile = flat.profile
  if (typeof flat.device === "string") run.device = flat.device.trim().toLowerCase() || "auto"
  else run.device = "auto"
  if (typeof flat.inpainter === "string") run.inpainter = flat.inpainter
  if (typeof flat.inpainter_model === "string") run.inpainter_model = flat.inpainter_model
  if (flat.mask_dilate_px !== undefined) run.mask_dilate_px = asNum(flat.mask_dilate_px, run.mask_dilate_px)
  if (flat.verify !== undefined) run.verify = asBool(flat.verify, run.verify)
  if (flat.formats !== undefined) run.formats = asFormats(flat.formats, run.formats)
  if (flat.keep_workdir !== undefined) run.keep_workdir = asBool(flat.keep_workdir, run.keep_workdir)
  if (flat.min_mask_coverage !== undefined) run.min_mask_coverage = asNum(flat.min_mask_coverage, run.min_mask_coverage)
  if (flat.verify_max_coverage !== undefined) {
    run.verify_max_coverage = asNum(flat.verify_max_coverage, run.verify_max_coverage)
  }
  if (typeof flat.llm_base_url === "string") run.llm_base_url = flat.llm_base_url
  if (typeof flat.llm_model === "string") run.llm_model = flat.llm_model
  if (typeof flat.detector === "string") run.detector = flat.detector
  if (typeof flat.detector_model === "string") run.detector_model = flat.detector_model
  if (typeof flat.segmenter === "string") run.segmenter = flat.segmenter
  if (typeof flat.segmenter_model === "string") run.segmenter_model = flat.segmenter_model
  if (flat.verify_redetect !== undefined) run.verify_redetect = asBool(flat.verify_redetect, false)
  if (flat.max_vram_mb !== undefined) run.max_vram_mb = asOptionalNumString(flat.max_vram_mb) ?? ""
  if (flat.cpu_threads !== undefined) run.cpu_threads = asOptionalNumString(flat.cpu_threads) ?? ""
  if (flat.inpaint_max_side !== undefined) run.inpaint_max_side = asOptionalNumString(flat.inpaint_max_side) ?? ""
  if (flat.webm_crf !== undefined) run.webm_crf = asNum(flat.webm_crf, run.webm_crf)
  if (flat.segment_seconds !== undefined) run.segment_seconds = asNum(flat.segment_seconds, run.segment_seconds)
  if (flat.mask_policy === "static" || flat.mask_policy === "propagate") {
    next = { ...next, maskPolicy: flat.mask_policy }
  }
  for (const key of Object.keys(ADVANCED_DEFAULTS)) {
    if (flat[key] === undefined) continue
    const text = asOptionalNumString(flat[key])
    if (text === undefined) advanced[key] = asString(flat[key])
    else advanced[key] = text
  }
  if (flat.detector_threshold !== undefined) advanced.detector_threshold = String(asNum(flat.detector_threshold, 0.15))
  if (flat.select_relax !== undefined) advanced.select_relax = asBool(flat.select_relax, true) ? "1" : "0"
  return { ...next, run, advanced }
}

export function toRunParams(params: EditorParams): Record<string, string | number | boolean> {
  return explicitFields(params)
}

/** Params that affect detect preview. Stride stays a separate inspection field. */
export function toDetectParams(
  params: EditorParams,
  scenario?: Scenario,
): Record<string, string | number | boolean> {
  const fields = explicitFields(params, scenario)
  delete fields.mask_policy
  return fields
}

export function presetSnapshot(params: EditorParams, resolvedDevice = "cpu"): Record<string, unknown> {
  return presetPayload(params, resolvedDevice)
}
