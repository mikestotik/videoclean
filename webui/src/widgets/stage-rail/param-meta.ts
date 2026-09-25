/** Human labels + short tooltip hints for pipeline params (Russian UI). */

export type ParamMeta = {
  label: string
  hint: string
  min?: number
  max?: number
  step?: number
}

export const INPAINTER_META: Record<string, { label: string; hint: string }> = {
  lama: {
    label: "LaMa (нейросеть)",
    hint: "Покадровая нейросеть. Работает на CPU и GPU. Нет учёта соседних кадров — на видео возможен flicker.",
  },
  propainter: {
    label: "ProPainter (видео)",
    hint: "Учитывает движение между кадрами. Лучшее качество, нужен CUDA.",
  },
}

export const DEVICE_META: Record<string, { label: string; hint: string }> = {
  auto: { label: "Авто", hint: "cuda, иначе mps, иначе cpu." },
  cpu: { label: "CPU", hint: "Медленнее, но работает везде." },
  cuda: { label: "CUDA (NVIDIA)", hint: "Видеокарта NVIDIA — быстрее нейросети и нужен для ProPainter." },
  mps: { label: "MPS (Apple)", hint: "Ускорение на чипах Apple Silicon." },
}

export const FORMAT_META: Record<
  string,
  { label: string; hint: string; kind: "file" | "package" }
> = {
  mp4: { label: "MP4", hint: "Обычный файл для плееров и мессенджеров.", kind: "file" },
  mov: { label: "MOV", hint: "Контейнер QuickTime, удобен для Final Cut / Premiere.", kind: "file" },
  mkv: { label: "MKV", hint: "Гибкий контейнер, часто для архива.", kind: "file" },
  webm: { label: "WebM", hint: "Для веба (VP9/Opus).", kind: "file" },
  "hls-fmp4": {
    label: "HLS (fMP4)",
    hint: "Пакет HLS с сегментами fMP4 и master.m3u8 — для стриминга и HTML5-плееров.",
    kind: "package",
  },
  "hls-ts": {
    label: "HLS (TS)",
    hint: "Классический HLS с MPEG-TS сегментами и master.m3u8.",
    kind: "package",
  },
  dash: {
    label: "DASH",
    hint: "MPEG-DASH пакет с manifest.mpd.",
    kind: "package",
  },
}

export const RUN_PARAM_META = {
  mask_dilate_px: {
    label: "Расширение маски",
    hint: "На сколько пикселей расширить маску сегментации до заливки. Края текста остаются — увеличьте; фон пропадает — уменьшите. У ProPainter есть отдельный запас маски в «Заливке».",
    min: 0,
    max: 15,
    step: 1,
  },
  min_mask_coverage: {
    label: "Мин. покрытие маски",
    hint: "Если средняя маска меньше этой доли кадра — прогон считается пустым и падает. Страховка от «ничего не удалил».",
    min: 0,
    max: 0.02,
    step: 0.0001,
  },
  verify_max_coverage: {
    label: "Потолок остатка",
    hint: "Если остаток занимает больше этой доли кадра, повторная заливка пропускается.",
    min: 0.01,
    max: 0.5,
    step: 0.01,
  },
  verify: {
    label: "Проверять остатки",
    hint: "После заливки ищет остатки (детектор + residual). Расширяет маску и заливает только проблемные участки. Чуть дольше, но чище.",
  },

  keep_workdir: {
    label: "Сохранить рабочие файлы",
    hint: "Оставить кадры/маски/промежуточные файлы джобы — удобно разбирать качество.",
  },
} as const satisfies Record<string, ParamMeta | { label: string; hint: string }>

export const ADVANCED_META: Record<string, ParamMeta> = {
  detector_threshold: {
    label: "Порог детектора",
    hint: "Минимальная уверенность бокса. Ничего не находит — снизьте (0.10–0.12). Много мусора — поднимите.",
    min: 0.05,
    max: 0.5,
    step: 0.01,
  },
  detector_keyframes: {
    label: "Ключевые кадры",
    hint: "На скольких кадрах крутится детектор; между ними — трекинг. Пусто = авто (~2/с, 8–24). Для динамики лучше ≥16–24.",
    min: 0,
    max: 40,
    step: 1,
  },
  detector_nms_iou: {
    label: "Схлопывание рамок",
    hint: "Почти одинаковые рамки схлопываются по пересечению (IoU). Много дублей одной цели — ниже; разные объекты слиплись — выше.",
    min: 0.1,
    max: 0.7,
    step: 0.05,
  },
  detector_max_box_area: {
    label: "Макс. размер рамки",
    hint: "Рамки больше этой доли кадра отбрасываются. Обычно 45%. Для широких титров — 50–55%.",
    min: 0.05,
    max: 0.7,
    step: 0.01,
  },
  select_relax: {
    label: "Мягкий отбор треков",
    hint: "Если where/ordinal ничего не нашли — ослабить фильтр. Выключите для строгого совпадения (меньше чужих объектов).",
  },
  tracker_min_score: {
    label: "Порог трекинга",
    hint: "Насколько уверенно шаблон цепляется между ключевыми кадрами. Дыры — снизьте; боксы плывут — поднимите.",
    min: 0.2,
    max: 0.9,
    step: 0.05,
  },
  tracker_max_template_area: {
    label: "Макс. размер шаблона",
    hint: "Слишком крупные кропы не трекаются (цепляются за фон). Обычно не трогать.",
    min: 0.02,
    max: 0.4,
    step: 0.01,
  },
  prompt_frame_stride: {
    label: "Шаг кадров разбора",
    hint: "Каждый N-й кадр уходит в модель при разборе фразы. 0 — только текст, без картинок.",
    min: 0,
    max: 20,
    step: 1,
  },
  prompt_frame_max: {
    label: "Макс. кадров разбора",
    hint: "Сколько кадров максимум отправить в модель. Больше — тяжелее запрос.",
    min: 1,
    max: 24,
    step: 1,
  },
  parse_chunk_frames: {
    label: "Кусок разбора",
    hint: "0 — один разбор на весь ролик. Больше нуля — разбор кусками по столько кадров, когда объекты появляются и исчезают.",
    min: 0,
    max: 300,
    step: 10,
  },
  vision_batch: {
    label: "Кадров в запросе",
    hint: "Сколько кадров уходит в одном запросе к модели. Для llava-phi3 держите 2.",
    min: 1,
    max: 8,
    step: 1,
  },
  propainter_mask_dilation: {
    label: "Запас маски ProPainter",
    hint: "На сколько пикселей расширить маску уже внутри ProPainter. Это отдельно от «Расширения маски» в сегментации.",
    min: 0,
    max: 20,
    step: 1,
  },
  propainter_ref_stride: {
    label: "Шаг опорных кадров",
    hint: "Как часто ProPainter берёт кадр-опору на весь ролик.",
    min: 1,
    max: 30,
    step: 1,
  },
  propainter_neighbor_length: {
    label: "Соседние кадры",
    hint: "Сколько соседних кадров ProPainter смотрит по времени.",
    min: 1,
    max: 30,
    step: 1,
  },
  propainter_subvideo_length: {
    label: "Длина фрагмента",
    hint: "На сколько кадров режется расчёт движения.",
    min: 20,
    max: 160,
    step: 10,
  },
  propainter_raft_iter: {
    label: "Точность движения",
    hint: "Итерации оценки движения в ProPainter. Больше — точнее и дольше.",
    min: 5,
    max: 40,
    step: 1,
  },
  verify_max_passes: {
    label: "Повторные проходы",
    hint: "Сколько раз искать остатки и заливать их снова. 0 — не перезаливать. 2 — максимум для сценария «Качество».",
    min: 0,
    max: 3,
    step: 1,
  },
  inpaint_workers: {
    label: "Параллель заливки",
    hint: "Сколько кадров LaMa считает сразу. 0 — авто (на CPU по ядрам, на GPU обычно 1). ProPainter всегда один.",
    min: 0,
    max: 16,
    step: 1,
  },
  inpaint_chunk_overlap: {
    label: "Перекрытие фрагментов",
    hint: "Сколько кадров соседние куски заливки делят между собой, чтобы на стыке не было шва.",
    min: 0,
    max: 32,
    step: 1,
  },
}

export const PARAM_LABELS: Record<string, string> = {
  device: "Устройство",
  detector: "Детектор",
  detector_model: "Модель детектора",
  segmenter: "Сегментация",
  segmenter_model: "Модель SAM",
  inpainter: "Инпейнтер",
  inpainter_model: "Модель заливки",
  formats: "Форматы",
  mask_policy: "Мазки",
  max_vram_mb: "Потолок VRAM",
  cpu_threads: "Потоки CPU",
  inpaint_max_side: "Сторона заливки",
  verify_redetect: "Повторный поиск",
  llm_model: "Модель разбора",
  llm_base_url: "Адрес модели разбора",
  webm_crf: "WebM CRF",
  segment_seconds: "Сегмент",
}

export function paramLabel(key: string): string {
  if (PARAM_LABELS[key]) return PARAM_LABELS[key]
  const run = RUN_PARAM_META[key as keyof typeof RUN_PARAM_META]
  if (run) return run.label
  return ADVANCED_META[key]?.label ?? key
}

/** Which expert station owns a saved field. `mask_policy` lives above the stations. */
export const PARAM_STATION: Record<string, string> = {
  llm_model: "phrase",
  llm_base_url: "phrase",
  prompt_frame_stride: "phrase",
  prompt_frame_max: "phrase",
  vision_batch: "phrase",
  parse_chunk_frames: "phrase",
  detector: "detect",
  detector_model: "detect",
  detector_threshold: "detect",
  detector_keyframes: "detect",
  detector_nms_iou: "detect",
  detector_max_box_area: "detect",
  tracker_min_score: "detect",
  tracker_max_template_area: "detect",
  select_relax: "detect",
  segmenter: "segment",
  segmenter_model: "segment",
  mask_dilate_px: "segment",
  inpainter: "fill",
  inpainter_model: "fill",
  propainter_mask_dilation: "fill",
  propainter_ref_stride: "fill",
  propainter_neighbor_length: "fill",
  propainter_subvideo_length: "fill",
  propainter_raft_iter: "fill",
  inpaint_chunk_overlap: "fill",
  inpaint_max_side: "fill",
  verify: "verify",
  verify_max_passes: "verify",
  verify_max_coverage: "verify",
  min_mask_coverage: "verify",
  verify_redetect: "verify",
  device: "device",
  max_vram_mb: "device",
  cpu_threads: "device",
  inpaint_workers: "fill",
  keep_workdir: "device",
  formats: "output",
  webm_crf: "output",
  segment_seconds: "output",
  mask_policy: "masks",
}

export function stationForParam(key: string): string | null {
  return PARAM_STATION[key] ?? null
}

/** DOM anchor when two saved keys share one control. */
export function paramAnchor(key: string): string {
  if (key === "llm_base_url") return "llm_model"
  return key
}

export function formatShare(value: number): string {
  const pct = value * 100
  const digits = pct >= 10 ? 0 : pct >= 1 ? 1 : 2
  return `${pct.toFixed(digits)}%`
}

export const MODE_HINTS = {
  inpaintTracks: "Заливка по найденным трекам (после «Маски»). Обычно лучший путь.",
  inpaintMasks: "Заливка по нарисованным вручную маскам на источнике.",
  inpaintPrompt: "Полный прогон от промпта/целей без готовых треков — дольше.",
  maskStatic: "Одна и та же маска на все кадры (плитка).",
  maskPropagate: "Рамки-якоря → протянуть по времени → сегментация.",
  detectAll: "Детекция по всему ролику — нужно для удаления по трекам.",
  detectStride: "Только для кнопки «Найти рамки». На «Обработать» и сохранённый сценарий не влияет.",
  boxHold: "Правка рамки копируется вперёд до следующего ключа.",
  boxFrame: "Правка только на текущем кадре.",
} as const
