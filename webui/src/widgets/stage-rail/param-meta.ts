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
    hint: "На сколько пикселей расширить вырез. Если остаются края текста — увеличьте; если «съедает» фон — уменьшите.",
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
    label: "Потолок leftover",
    hint: "При проверке leftover: если остаток больше — повторная заливка пропускается.",
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
    hint: "На скольких кадрах реально крутится детектор; между ними — трекинг. Пусто = авто. Дыры по кадрам — увеличьте.",
    min: 0,
    max: 40,
    step: 1,
  },
  detector_nms_iou: {
    label: "NMS IoU",
    hint: "Схлопывание почти одинаковых боксов. Много дублей — ниже; разные объекты слиплись — выше.",
    min: 0.1,
    max: 0.7,
    step: 0.05,
  },
  detector_max_box_area: {
    label: "Макс. площадь бокса",
    hint: "Боксы больше этой доли кадра отбрасываются (защита от «нашёл полкадра»). Широкие титры — поднимите.",
    min: 0.05,
    max: 0.6,
    step: 0.01,
  },
  tracker_min_score: {
    label: "Порог трекинга",
    hint: "Насколько уверенно шаблон цепляется между ключевыми кадрами. Дыры — снизьте; боксы плывут — поднимите.",
    min: 0.2,
    max: 0.9,
    step: 0.05,
  },
  tracker_max_template_area: {
    label: "Макс. шаблон трекинга",
    hint: "Слишком крупные кропы не трекаются (цепляются за фон). Обычно не трогать.",
    min: 0.02,
    max: 0.4,
    step: 0.01,
  },
  prompt_frame_stride: {
    label: "Шаг кадров для LLM",
    hint: "Каждый N-й кадр уходит в vision-модель при разборе промпта. 0 — только текст, без картинок.",
    min: 0,
    max: 20,
    step: 1,
  },
  prompt_frame_max: {
    label: "Макс. кадров для LLM",
    hint: "Сколько кадров максимум отправить в vision. Больше — тяжелее для Ollama.",
    min: 1,
    max: 24,
    step: 1,
  },
  parse_chunk_frames: {
    label: "Чанк парсера",
    hint: "0 — один разбор на весь ролик. N>0 — разбор кусками по N кадров (когда объекты появляются и исчезают).",
    min: 0,
    max: 300,
    step: 10,
  },
  vision_batch: {
    label: "Пакет vision",
    hint: "Сколько кадров в одном запросе к vision-LLM. Для llava-phi3 держите 2.",
    min: 1,
    max: 8,
    step: 1,
  },
  propainter_mask_dilation: {
    label: "ProPainter: dilate",
    hint: "Расширение маски перед заливкой ProPainter.",
    min: 0,
    max: 20,
    step: 1,
  },
  propainter_ref_stride: {
    label: "ProPainter: ref stride",
    hint: "Шаг глобальных опорных кадров.",
    min: 1,
    max: 30,
    step: 1,
  },
  propainter_neighbor_length: {
    label: "ProPainter: neighbors",
    hint: "Сколько соседних кадров учитывать по времени.",
    min: 1,
    max: 30,
    step: 1,
  },
  propainter_subvideo_length: {
    label: "ProPainter: subvideo",
    hint: "Длина чанка для расчёта optical flow.",
    min: 20,
    max: 160,
    step: 10,
  },
  propainter_raft_iter: {
    label: "ProPainter: RAFT",
    hint: "Итерации оценки движения. Больше — точнее и дольше.",
    min: 5,
    max: 40,
    step: 1,
  },
  verify_max_passes: {
    label: "Verify: проходы",
    hint: "Сколько раз искать остатки и локально перезаливать. 0 — не перезаливать. 2 — максимум для профиля «Качество».",
    min: 0,
    max: 3,
    step: 1,
  },
  inpaint_workers: {
    label: "Потоки инпейнта",
    hint: "Параллельные кадры для LaMa. 0 — авто (на CPU по ядрам, на GPU обычно 1). ProPainter всегда 1.",
    min: 0,
    max: 16,
    step: 1,
  },
  inpaint_chunk_overlap: {
    label: "Overlap чанков",
    hint: "Перекрытие кадров при нарезке видео-инпейнта и локальном verify — чтобы не было швов.",
    min: 0,
    max: 32,
    step: 1,
  },
}

export const MODE_HINTS = {
  inpaintTracks: "Заливка по найденным трекам (после «Маски»). Обычно лучший путь.",
  inpaintMasks: "Заливка по нарисованным вручную маскам на источнике.",
  inpaintPrompt: "Полный прогон от промпта/целей без готовых треков — дольше.",
  maskStatic: "Одна и та же маска на все кадры (плитка).",
  maskPropagate: "Рамки-якоря → протянуть по времени → сегментация.",
  detectAll: "Детекция по всему ролику — нужно для удаления по трекам.",
  detectStride: "Только каждый N-й кадр — быстрее, для осмотра, не для удаления.",
  boxHold: "Правка рамки копируется вперёд до следующего ключа.",
  boxFrame: "Правка только на текущем кадре.",
} as const
