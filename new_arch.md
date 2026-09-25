## User

IDEA-Research/grounding-dino-tiny должен на CUDA или на процессоре работать? я сейчас на сервере сделал запуск и походу он работал на процессоре

## Tools

- Read: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/preferences.md
- Read: /home/user/.grok/installed-plugins/superpowers-21e2a56d/skills/using-superpowers/SKILL.md
- Search: grounding.?dino|GroundingDino|grounding_dino
- Search: dino-tiny|IDEA-Research

## Assistant

Сейчас посмотрю, как в коде выбирается устройство для Grounding DINO.

## Tools

- Read: /home/user/Projects/videoclean/videoclean/adapters/detectors/grounding_dino.py
- Search: device|resolve_device|auto.*cuda|cuda.*available
- Search: def resolve_device|device.*=.*auto|pick_device|normalize_device
- Read: /home/user/Projects/videoclean/videoclean/application/budget.py (2-101)
- Read: /home/user/Projects/videoclean/videoclean/composition.py (191-230)
- Search: device|cuda|cpu

## Assistant

**На GPU-сервере он должен идти на CUDA.** Сама модель `IDEA-Research/grounding-dino-tiny` умеет и CPU, и CUDA — это обычный Transformers-вес. В videoclean устройство задаёт `--device` / поле `device` в job, и детектор просто делает `_model.to(self.device)`.

Как выбирается устройство:

- `auto` (дефолт) → `cuda`, если `torch.cuda.is_available()`, иначе `mps`, иначе `cpu`
- явный `cpu` / `cuda` / `mps` — как написали

На CUDA forward идёт через `torch.autocast(..., bfloat16)`. На CPU — без autocast, fp32, заметно медленнее.

Если у тебя на сервере реально крутился CPU, типичные причины:

1. В форме/CLI/рецепте явно стояло `device=cpu`
2. `device=auto`, но у процесса `torch.cuda.is_available() == False` (битый/невидимый CUDA, другой Python, драйвер, init-ошибка) — тогда auto тихо уходит на CPU
3. Смотришь не тот процесс (другой venv / системный Python без CUDA-сборки torch)

Быстрая проверка на том же интерпретаторе, что крутит сервер:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
uv run videoclean doctor --device cuda --detector grounding-dino
```

В логе/статусе детекта должно быть что-то вроде `ready (... on cuda)`. Если doctor/job пишут `on cpu` при живой карте — проблема в torch/CUDA окружении или в явном `device=cpu`, а не в самой tiny-модели.

## User

Fri Sep 25 2026 15:20:31 GMT+0300 (Moscow Standard Time) ==========
Fri Sep 25 2026 15:20:31 GMT+0300 (Moscow Standard Time) == CUDA ==
Fri Sep 25 2026 15:20:31 GMT+0300 (Moscow Standard Time) ==========
Fri Sep 25 2026 15:20:31 GMT+0300 (Moscow Standard Time) CUDA Version 12.8.1
Fri Sep 25 2026 15:20:31 GMT+0300 (Moscow Standard Time) Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
Fri Sep 25 2026 15:20:31 GMT+0300 (Moscow Standard Time) This container image and its contents are governed by the NVIDIA Deep Learning Container License.
Fri Sep 25 2026 15:20:31 GMT+0300 (Moscow Standard Time) By pulling and using the container, you accept the terms and conditions of this license:
Fri Sep 25 2026 15:20:31 GMT+0300 (Moscow Standard Time) https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
Fri Sep 25 2026 15:20:31 GMT+0300 (Moscow Standard Time) A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
Fri Sep 25 2026 15:20:31 GMT+0300 (Moscow Standard Time) fetching main
Fri Sep 25 2026 15:20:34 GMT+0300 (Moscow Standard Time) From https://github.com/mikestotik/videoclean
Fri Sep 25 2026 15:20:34 GMT+0300 (Moscow Standard Time)  * branch            main       -> FETCH_HEAD
Fri Sep 25 2026 15:20:34 GMT+0300 (Moscow Standard Time)    0725745..8cf076c  main       -> origin/main
Fri Sep 25 2026 15:20:35 GMT+0300 (Moscow Standard Time) HEAD is now at 8cf076c ci(runpod): add job/step timeouts so hung deploys die
Fri Sep 25 2026 15:20:35 GMT+0300 (Moscow Standard Time) Resolved 94 packages in 16ms
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time) Uninstalled 6 packages in 1.31s
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time) Installed 14 packages in 369ms
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  - antlr4-python3-runtime==4.9.3
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  - hydra-core==1.3.7
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  - iopath==0.1.10
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-cublas-cu12==12.8.4.1
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-cuda-cupti-cu12==12.8.90
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-cuda-nvrtc-cu12==12.8.93
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-cuda-runtime-cu12==12.8.90
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-cudnn-cu12==9.10.2.21
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-cufft-cu12==11.3.3.83
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-cufile-cu12==1.13.1.3
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-curand-cu12==10.3.9.90
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-cusolver-cu12==11.7.3.90
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-cusparse-cu12==12.5.8.93
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-cusparselt-cu12==0.7.1
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-nccl-cu12==2.27.3
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-nvjitlink-cu12==12.8.93
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  + nvidia-nvtx-cu12==12.8.90
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  - omegaconf==2.3.1
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  - portalocker==4.4.0
Fri Sep 25 2026 15:20:38 GMT+0300 (Moscow Standard Time)  - sam-2==1.0 (from git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4)
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) purging venv CUDA/torch packages (use image builds):
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-cublas-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-cuda-cupti-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-cuda-nvrtc-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-cuda-runtime-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-cudnn-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-cufft-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-cufile-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-curand-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-cusolver-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-cusparse-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-cusparselt-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-nccl-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-nvjitlink-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) nvidia-nvtx-cu12
Fri Sep 25 2026 15:20:39 GMT+0300 (Moscow Standard Time) Using Python 3.12.3 environment at: /workspace/.venv
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time) Uninstalled 14 packages in 1.35s
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-cublas-cu12==12.8.4.1
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-cuda-cupti-cu12==12.8.90
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-cuda-nvrtc-cu12==12.8.93
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-cuda-runtime-cu12==12.8.90
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-cudnn-cu12==9.10.2.21
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-cufft-cu12==11.3.3.83
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-cufile-cu12==1.13.1.3
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-curand-cu12==10.3.9.90
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-cusolver-cu12==11.7.3.90
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-cusparse-cu12==12.5.8.93
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-cusparselt-cu12==0.7.1
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-nccl-cu12==2.27.3
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-nvjitlink-cu12==12.8.93
Fri Sep 25 2026 15:20:40 GMT+0300 (Moscow Standard Time)  - nvidia-nvtx-cu12==12.8.90
Fri Sep 25 2026 15:20:41 GMT+0300 (Moscow Standard Time) Using Python 3.12.3 environment at: /workspace/.venv
Fri Sep 25 2026 15:20:41 GMT+0300 (Moscow Standard Time) Resolved 1 package in 287ms
Fri Sep 25 2026 15:20:41 GMT+0300 (Moscow Standard Time) Installed 1 package in 359ms
Fri Sep 25 2026 15:20:41 GMT+0300 (Moscow Standard Time)  + sam-2==1.0 (from git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4)
Fri Sep 25 2026 15:20:42 GMT+0300 (Moscow Standard Time) Using Python 3.12.3 environment at: /workspace/.venv
Fri Sep 25 2026 15:20:42 GMT+0300 (Moscow Standard Time) Resolved 10 packages in 368ms
Fri Sep 25 2026 15:20:43 GMT+0300 (Moscow Standard Time) Installed 5 packages in 578ms
Fri Sep 25 2026 15:20:43 GMT+0300 (Moscow Standard Time)  + antlr4-python3-runtime==4.9.3
Fri Sep 25 2026 15:20:43 GMT+0300 (Moscow Standard Time)  + hydra-core==1.3.7
Fri Sep 25 2026 15:20:43 GMT+0300 (Moscow Standard Time)  + iopath==0.1.10
Fri Sep 25 2026 15:20:43 GMT+0300 (Moscow Standard Time)  + omegaconf==2.3.1
Fri Sep 25 2026 15:20:43 GMT+0300 (Moscow Standard Time)  + portalocker==4.4.0
Fri Sep 25 2026 15:20:48 GMT+0300 (Moscow Standard Time) torch 2.8.0+cu128 cuda=True file=/usr/local/lib/python3.12/dist-packages/torch/__init__.py
Fri Sep 25 2026 15:20:49 GMT+0300 (Moscow Standard Time) === videoclean doctor --device cuda ===
Fri Sep 25 2026 15:20:58 GMT+0300 (Moscow Standard Time) Budget  device=cuda requested=auto vram_budget=23238213632 ceiling=None
Fri Sep 25 2026 15:20:58 GMT+0300 (Moscow Standard Time) cpu_threads=48/48 nvenc=False
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time) This machine
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   Python     3.12.3
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   FFmpeg     /usr/bin/ffmpeg
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   ffprobe    /usr/bin/ffprobe
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   OpenCV     4.11.0
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   torch      2.8.0+cu128   cuda=yes  mps=no
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time) This command (resolved config)
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   device           cuda   (ML adapters only)  available
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   detector         grounding-dino
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   detector-model   IDEA-Research/grounding-dino-tiny
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   detector-thr     0.15
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   segmenter        sam2
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   segmenter-model  facebook/sam2.1-hiera-tiny
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   inpainter        lama
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   inpainter-model  camenduru/ProPainter
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   prompt-parser    llm
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   verify           True
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   llm              auto  model=—  url=—
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   vision-frames    stride=4  max=8  batch=2
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   detector-tune    keyframes=default  nms-iou=0.3  max-box-area=0.45
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   tracker-tune     min-score=0.55  max-template-area=0.12
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   propainter-tune  dilate=4  ref-stride=10  neighbor=10  subvideo=80  raft=20
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   prompt-templates builtin
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   format           mp4
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   media            ffmpeg   (always, not a flag)
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time) Adapters that will actually run
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   ffmpeg           ready
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   grounding-dino   ready (weights=disk; loads on first detect, device=cuda)
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   sam2             ready (runtime=transformers.Sam2Model, weights=disk, 
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time) device=cuda)
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   lama             unavailable: big-lama.pt not found at 
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time) /workspace/data/weights/lama/big-lama.pt. Pass --download-models once, or: curl 
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time) -L 
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time) https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/bi
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time) g-lama.pt -o /workspace/data/weights/lama/big-lama.pt  (Uses --device. CPU 
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time) works; CUDA/MPS faster.)
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time)   llm              unavailable: llm unreachable at 
Fri Sep 25 2026 15:22:02 GMT+0300 (Moscow Standard Time) http://127.0.0.1:11434/v1/models: [Errno 111] Connection refused
Fri Sep 25 2026 15:22:03 GMT+0300 (Moscow Standard Time) === videoclean serve (background) port=7860 ===
Fri Sep 25 2026 15:22:03 GMT+0300 (Moscow Standard Time) started pid=580 log=/workspace/run/serve.log
Fri Sep 25 2026 15:22:03 GMT+0300 (Moscow Standard Time) UI: https://cpy65b5zj24xab-7860.proxy.runpod.net
Fri Sep 25 2026 15:22:03 GMT+0300 (Moscow Standard Time) holding container; serve log → /workspace/run/serve.log
Fri Sep 25 2026 15:22:16 GMT+0300 (Moscow Standard Time) INFO:     Started server process [580]
Fri Sep 25 2026 15:22:16 GMT+0300 (Moscow Standard Time) INFO:     Waiting for application startup.
Fri Sep 25 2026 15:22:16 GMT+0300 (Moscow Standard Time) INFO:     Application startup complete.
Fri Sep 25 2026 15:22:16 GMT+0300 (Moscow Standard Time) INFO:     Uvicorn running on http://0.0.0.0:7860 (Press CTRL+C to quit)
Fri Sep 25 2026 15:22:24 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.66:24674 - "GET / HTTP/1.1" 200 OK
Fri Sep 25 2026 15:22:26 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:26190 - "GET / HTTP/1.1" 200 OK
Fri Sep 25 2026 15:22:27 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:11820 - "GET /assets/index-mJvYNWSk.js HTTP/1.1" 200 OK
Fri Sep 25 2026 15:22:27 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:28614 - "GET /assets/index-CSQXOix7.css HTTP/1.1" 200 OK
Fri Sep 25 2026 15:22:27 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:11820 - "GET /api/events HTTP/1.1" 200 OK
Fri Sep 25 2026 15:22:27 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:26190 - "GET /assets/inter-latin-wght-normal-Dx4kXJAl.woff2 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:22:28 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.33:27162 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:22:28 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:54270 - "GET /assets/inter-cyrillic-wght-normal-DqGufNeO.woff2 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:22:28 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.33:27162 - "GET /vite.svg HTTP/1.1" 404 Not Found
Fri Sep 25 2026 15:22:40 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.33:22180 - "GET /assets/jetbrains-mono-latin-wght-normal-B9CIFXIH.woff2 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:22:40 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:62440 - "GET /assets/jetbrains-mono-cyrillic-wght-normal-D73BlboJ.woff2 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:22:49 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:30902 - "POST /api/models/download HTTP/1.1" 200 OK
Fri Sep 25 2026 15:23:05 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:18048 - "POST /api/models/download HTTP/1.1" 200 OK
Fri Sep 25 2026 15:23:18 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.33:21460 - "POST /api/models/download HTTP/1.1" 200 OK
Fri Sep 25 2026 15:23:34 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:59692 - "POST /api/models/download HTTP/1.1" 200 OK
Fri Sep 25 2026 15:24:12 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:45016 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:25:40 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:47290 - "POST /api/sources HTTP/1.1" 201 Created
Fri Sep 25 2026 15:25:52 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:45608 - "POST /api/sources/s_20260925_122539_09a5f794/downscale HTTP/1.1" 201 Created
Fri Sep 25 2026 15:25:53 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:31552 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:25:53 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.33:19672 - "GET /api/sources/s_20260925_122548_e8fbd5c5/annotations HTTP/1.1" 200 OK
Fri Sep 25 2026 15:25:53 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:62348 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 200 OK
Fri Sep 25 2026 15:25:53 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:62356 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:25:56 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:62364 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:25:56 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:62380 - "GET /api/sources/s_20260925_122539_09a5f794/video HTTP/1.1" 200 OK
Fri Sep 25 2026 15:25:58 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.33:19686 - "GET /vite.svg HTTP/1.1" 404 Not Found
Fri Sep 25 2026 15:26:45 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.66:60930 - "GET / HTTP/1.1" 200 OK
Fri Sep 25 2026 15:26:47 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:36108 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:26:47 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:36114 - "GET /api/sources/s_20260925_122539_09a5f794/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:26:47 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:36104 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:26:47 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:54898 - "GET /api/sources/s_20260925_122548_e8fbd5c5 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:26:47 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:36108 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:26:47 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:36108 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:26:47 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:36118 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:26:47 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:36130 - "GET /api/sources/s_20260925_122548_e8fbd5c5/annotations HTTP/1.1" 200 OK
Fri Sep 25 2026 15:28:12 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:48262 - "POST /api/jobs HTTP/1.1" 201 Created
Fri Sep 25 2026 15:29:55 GMT+0300 (Moscow Standard Time) Pretrained flow completion model has loaded...
Fri Sep 25 2026 15:29:56 GMT+0300 (Moscow Standard Time) Pretrained ProPainter has loaded...
Fri Sep 25 2026 15:29:56 GMT+0300 (Moscow Standard Time) Network [InpaintGenerator] was created. Total number of parameters: 39.4 million. To see the architecture, do print(network).
Fri Sep 25 2026 15:32:59 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.33:44468 - "POST /api/jobs/20260925_122811_a74ebc3a/cancel HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:39 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.33:52958 - "GET /v/s_20260925_122548_e8fbd5c5 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:40 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:34950 - "GET /assets/index-CSQXOix7.css HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:40 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:12070 - "GET /assets/index-mJvYNWSk.js HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:40 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:12070 - "GET /api/events HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:40 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:34950 - "GET /assets/inter-cyrillic-wght-normal-DqGufNeO.woff2 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:40 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.33:52958 - "GET /assets/inter-latin-wght-normal-Dx4kXJAl.woff2 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:40 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:43754 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:40 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:43752 - "GET /api/sources/s_20260925_122548_e8fbd5c5 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:41 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:43752 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:41 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:34950 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:33:41 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:34954 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:33:41 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:18298 - "GET /assets/jetbrains-mono-latin-wght-normal-B9CIFXIH.woff2 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:41 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:18300 - "GET /api/sources/s_20260925_122548_e8fbd5c5/annotations HTTP/1.1" 200 OK
Fri Sep 25 2026 15:33:41 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:34954 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:33:41 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:34950 - "GET /api/sources/s_20260925_122539_09a5f794/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:33:44 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:34950 - "GET /api/sources/s_20260925_122539_09a5f794/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:34:10 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:51096 - "DELETE /api/jobs/20260925_122811_a74ebc3a HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:16 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.33:47252 - "DELETE /api/jobs/20260925_122811_a74ebc3a HTTP/1.1" 404 Not Found
Fri Sep 25 2026 15:34:19 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:54486 - "GET /v/s_20260925_122548_e8fbd5c5 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:19 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:63074 - "GET /assets/index-mJvYNWSk.js HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:19 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:41378 - "GET /assets/index-CSQXOix7.css HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:19 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:63074 - "GET /api/events HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:54490 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:41394 - "GET /assets/inter-cyrillic-wght-normal-DqGufNeO.woff2 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:54492 - "GET /assets/inter-latin-wght-normal-Dx4kXJAl.woff2 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:41378 - "GET /api/sources/s_20260925_122548_e8fbd5c5 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:54492 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:41394 - "GET /assets/jetbrains-mono-latin-wght-normal-B9CIFXIH.woff2 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:41378 - "GET /api/sources/s_20260925_122548_e8fbd5c5/annotations HTTP/1.1" 200 OK
Fri Sep 25 2026 15:34:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:41424 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:34:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:41410 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:34:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:41426 - "GET /api/sources/s_20260925_122539_09a5f794/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:34:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:41394 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:36:32 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:24328 - "POST /api/presets HTTP/1.1" 201 Created
Fri Sep 25 2026 15:36:32 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.34:39924 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:37:14 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:14034 - "POST /api/jobs HTTP/1.1" 201 Created
Fri Sep 25 2026 15:37:27 GMT+0300 (Moscow Standard Time) Pretrained flow completion model has loaded...
Fri Sep 25 2026 15:37:27 GMT+0300 (Moscow Standard Time) Pretrained ProPainter has loaded...
Fri Sep 25 2026 15:37:27 GMT+0300 (Moscow Standard Time) Network [InpaintGenerator] was created. Total number of parameters: 39.4 million. To see the architecture, do print(network).
Fri Sep 25 2026 15:42:42 GMT+0300 (Moscow Standard Time) /workspace/.venv/lib/python3.12/site-packages/sam2/sam2_video_predictor.py:786: UserWarning: cannot import name '_C' from 'sam2' (/workspace/.venv/lib/python3.12/site-packages/sam2/__init__.py)
Fri Sep 25 2026 15:42:42 GMT+0300 (Moscow Standard Time) Skipping the post-processing step due to the error above. You can still use SAM 2 and it's OK to ignore the error above, although some post-processing functionality may be limited (which doesn't affect the results in most cases; see https://github.com/facebookresearch/sam2/blob/main/INSTALL.md).
Fri Sep 25 2026 15:42:42 GMT+0300 (Moscow Standard Time)   pred_masks_gpu = fill_holes_in_mask_scores(
Fri Sep 25 2026 15:44:26 GMT+0300 (Moscow Standard Time) 
propagate in video:   0%|          | 0/672 [00:00<?, ?it/s]
propagate in video:   0%|          | 2/672 [00:01<05:47,  1.93it/s]
propagate in video:   0%|          | 3/672 [00:02<07:59,  1.39it/s]
propagate in video:   1%|          | 4/672 [00:02<06:33,  1.70it/s]
propagate in video:   1%|          | 5/672 [00:03<08:09,  1.36it/s]
propagate in video:   1%|          | 6/672 [00:04<08:58,  1.24it/s]
propagate in video:   1%|          | 7/672 [00:04<07:50,  1.41it/s]
propagate in video:   1%|          | 8/672 [00:05<07:08,  1.55it/s]
propagate in video:   1%|▏         | 9/672 [00:06<09:52,  1.12it/s]
propagate in video:   1%|▏         | 10/672 [00:07<10:11,  1.08it/s]
propagate in video:   2%|▏         | 11/672 [00:08<09:02,  1.22it/s]
propagate in video:   2%|▏         | 12/672 [00:08<07:41,  1.43it/s]
propagate in video:   2%|▏         | 13/672 [00:09<07:00,  1.57it/s]
propagate in video:   2%|▏         | 14/672 [00:09<06:35,  1.66it/s]
propagate in video:   2%|▏         | 15/672 [00:10<06:06,  1.79it/s]
propagate in video:   2%|▏         | 16/672 [00:10<05:54,  1.85it/s]
propagate in video:   3%|▎         | 17/672 [00:11<06:10,  1.77it/s]
propagate in video:   3%|▎         | 18/672 [00:11<05:47,  1.88it/s]
propagate in video:   3%|▎         | 19/672 [00:12<05:48,  1.87it/s]
propagate in video:   3%|▎         | 20/672 [00:12<05:41,  1.91it/s]
propagate in video:   3%|▎         | 21/672 [00:14<07:39,  1.42it/s]
propagate in video:   3%|▎         | 22/672 [00:15<08:48,  1.23it/s]
propagate in video:   3%|▎         | 23/672 [00:16<10:47,  1.00it/s]
propagate in video:   4%|▎         | 24/672 [00:17<10:06,  1.07it/s]
propagate in video:   4%|▎         | 25/672 [00:18<10:02,  1.07it/s]
propagate in video:   4%|▍         | 26/672 [00:18<08:37,  1.25it/s]
propagate in video:   4%|▍         | 27/672 [00:19<09:22,  1.15it/s]
propagate in video:   4%|▍         | 28/672 [00:20<08:17,  1.29it/s]
propagate in video:   4%|▍         | 29/672 [00:20<07:02,  1.52it/s]
propagate in video:   4%|▍         | 30/672 [00:21<07:50,  1.36it/s]
propagate in video:   5%|▍         | 31/672 [00:22<08:39,  1.23it/s]
propagate in video:   5%|▍         | 32/672 [00:23<07:23,  1.44it/s]
propagate in video:   5%|▍         | 33/672 [00:23<07:57,  1.34it/s]
propagate in video:   5%|▌         | 34/672 [00:25<09:09,  1.16it/s]
propagate in video:   5%|▌         | 35/672 [00:25<07:48,  1.36it/s]
propagate in video:   5%|▌         | 36/672 [00:26<10:07,  1.05it/s]
propagate in video:   6%|▌         | 37/672 [00:28<10:28,  1.01it/s]
propagate in video:   6%|▌         | 38/672 [00:28<08:50,  1.20it/s]
propagate in video:   6%|▌         | 39/672 [00:30<11:18,  1.07s/it]
propagate in video:   6%|▌         | 40/672 [00:31<11:08,  1.06s/it]
propagate in video:   6%|▌         | 41/672 [00:32<10:49,  1.03s/it]
propagate in video:   6%|▋         | 42/672 [00:32<09:08,  1.15it/s]
propagate in video:   6%|▋         | 43/672 [00:33<09:52,  1.06it/s]
propagate in video:   7%|▋         | 44/672 [00:34<08:25,  1.24it/s]
propagate in video:   7%|▋         | 45/672 [00:35<08:29,  1.23it/s]
propagate in video:   7%|▋         | 46/672 [00:36<09:35,  1.09it/s]
propagate in video:   7%|▋         | 47/672 [00:36<08:14,  1.26it/s]
propagate in video:   7%|▋         | 48/672 [00:37<08:07,  1.28it/s]
propagate in video:   7%|▋         | 49/672 [00:37<07:23,  1.40it/s]
propagate in video:   7%|▋         | 50/672 [00:39<09:20,  1.11it/s]
propagate in video:   8%|▊         | 51/672 [00:40<09:31,  1.09it/s]
propagate in video:   8%|▊         | 52/672 [00:41<10:48,  1.05s/it]
propagate in video:   8%|▊         | 53/672 [00:42<09:11,  1.12it/s]
propagate in video:   8%|▊         | 54/672 [00:43<09:17,  1.11it/s]
propagate in video:   8%|▊         | 55/672 [00:43<07:49,  1.31it/s]
propagate in video:   8%|▊         | 56/672 [00:44<08:30,  1.21it/s]
propagate in video:   8%|▊         | 57/672 [00:45<07:44,  1.32it/s]
propagate in video:   9%|▊         | 58/672 [00:45<06:46,  1.51it/s]
propagate in video:   9%|▉         | 59/672 [00:46<08:13,  1.24it/s]
propagate in video:   9%|▉         | 60/672 [00:47<07:21,  1.39it/s]
propagate in video:   9%|▉         | 61/672 [00:48<08:00,  1.27it/s]
propagate in video:   9%|▉         | 62/672 [00:49<08:30,  1.20it/s]
propagate in video:   9%|▉         | 63/672 [00:49<07:29,  1.36it/s]
propagate in video:  10%|▉         | 64/672 [00:50<07:55,  1.28it/s]INFO:     100.64.1.36:22910 - "GET /assets/jetbrains-mono-cyrillic-wght-normal-D73BlboJ.woff2 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:44:34 GMT+0300 (Moscow Standard Time) 
propagate in video:  10%|▉         | 65/672 [00:51<08:20,  1.21it/s]
propagate in video:  10%|▉         | 66/672 [00:51<07:15,  1.39it/s]
propagate in video:  10%|▉         | 67/672 [00:52<06:34,  1.53it/s]
propagate in video:  10%|█         | 68/672 [00:53<07:37,  1.32it/s]
propagate in video:  10%|█         | 69/672 [00:53<06:54,  1.45it/s]
propagate in video:  10%|█         | 70/672 [00:54<06:00,  1.67it/s]
propagate in video:  11%|█         | 71/672 [00:55<06:55,  1.45it/s]
propagate in video:  11%|█         | 72/672 [00:56<07:49,  1.28it/s]
propagate in video:  11%|█         | 73/672 [00:56<06:52,  1.45it/s]
propagate in video:  11%|█         | 74/672 [00:57<06:06,  1.63it/s]
propagate in video:  11%|█         | 75/672 [00:57<05:44,  1.73it/s]
propagate in video:  11%|█▏        | 76/672 [00:59<08:21,  1.19it/s]INFO:     100.64.1.32:15906 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:44:34 GMT+0300 (Moscow Standard Time) 
propagate in video:  11%|█▏        | 77/672 [00:59<07:28,  1.33it/s]INFO:     100.64.1.35:63872 - "GET /api/sources/s_20260925_122548_e8fbd5c5 HTTP/1.1" 200 OK
Fri Sep 25 2026 15:44:34 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:63896 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:44:34 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:63884 - "GET /api/sources/s_20260925_122539_09a5f794/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:44:35 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:15906 - "GET /api/presets HTTP/1.1" 200 OK
Fri Sep 25 2026 15:44:35 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:63872 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:44:35 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.36:45546 - "GET /api/sources/s_20260925_122548_e8fbd5c5/annotations HTTP/1.1" 200 OK
Fri Sep 25 2026 15:44:35 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.35:63900 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:46:19 GMT+0300 (Moscow Standard Time) 
propagate in video:  12%|█▏        | 78/672 [01:00<07:36,  1.30it/s]
propagate in video:  12%|█▏        | 79/672 [01:01<09:01,  1.10it/s]
propagate in video:  12%|█▏        | 80/672 [01:02<07:53,  1.25it/s]
propagate in video:  12%|█▏        | 81/672 [01:03<08:12,  1.20it/s]
propagate in video:  12%|█▏        | 82/672 [01:03<07:16,  1.35it/s]
propagate in video:  12%|█▏        | 83/672 [01:04<07:37,  1.29it/s]
propagate in video:  12%|█▎        | 84/672 [01:05<07:56,  1.24it/s]
propagate in video:  13%|█▎        | 85/672 [01:06<09:48,  1.00s/it]
propagate in video:  13%|█▎        | 86/672 [01:08<10:55,  1.12s/it]
propagate in video:  13%|█▎        | 87/672 [01:09<09:57,  1.02s/it]
propagate in video:  13%|█▎        | 88/672 [01:09<09:44,  1.00s/it]
propagate in video:  13%|█▎        | 89/672 [01:10<08:15,  1.18it/s]
propagate in video:  13%|█▎        | 90/672 [01:11<07:22,  1.32it/s]
propagate in video:  14%|█▎        | 91/672 [01:11<06:35,  1.47it/s]
propagate in video:  14%|█▎        | 92/672 [01:12<07:34,  1.28it/s]
propagate in video:  14%|█▍        | 93/672 [01:14<10:08,  1.05s/it]
propagate in video:  14%|█▍        | 94/672 [01:15<11:06,  1.15s/it]
propagate in video:  14%|█▍        | 95/672 [01:16<09:10,  1.05it/s]
propagate in video:  14%|█▍        | 96/672 [01:16<07:47,  1.23it/s]
propagate in video:  15%|█▍        | 98/672 [01:17<06:00,  1.59it/s]
propagate in video:  15%|█▍        | 99/672 [01:18<06:23,  1.49it/s]
propagate in video:  15%|█▍        | 100/672 [01:18<05:43,  1.67it/s]
propagate in video:  15%|█▌        | 101/672 [01:19<05:13,  1.82it/s]
propagate in video:  15%|█▌        | 102/672 [01:19<04:51,  1.95it/s]
propagate in video:  15%|█▌        | 103/672 [01:20<05:20,  1.78it/s]
propagate in video:  15%|█▌        | 104/672 [01:20<04:53,  1.94it/s]
propagate in video:  16%|█▌        | 105/672 [01:20<04:33,  2.08it/s]
propagate in video:  16%|█▌        | 106/672 [01:21<04:30,  2.09it/s]
propagate in video:  16%|█▌        | 107/672 [01:21<04:26,  2.12it/s]
propagate in video:  16%|█▌        | 108/672 [01:23<07:09,  1.31it/s]
propagate in video:  16%|█▌        | 109/672 [01:24<07:46,  1.21it/s]
propagate in video:  16%|█▋        | 110/672 [01:24<06:50,  1.37it/s]
propagate in video:  17%|█▋        | 111/672 [01:25<07:40,  1.22it/s]
propagate in video:  17%|█▋        | 112/672 [01:27<08:53,  1.05it/s]
propagate in video:  17%|█▋        | 113/672 [01:27<08:47,  1.06it/s]
propagate in video:  17%|█▋        | 114/672 [01:29<10:51,  1.17s/it]
propagate in video:  17%|█▋        | 115/672 [01:31<12:04,  1.30s/it]
propagate in video:  17%|█▋        | 116/672 [01:31<09:37,  1.04s/it]
propagate in video:  17%|█▋        | 117/672 [01:32<08:00,  1.16it/s]
propagate in video:  18%|█▊        | 118/672 [01:32<06:50,  1.35it/s]
propagate in video:  18%|█▊        | 119/672 [01:33<08:21,  1.10it/s]
propagate in video:  18%|█▊        | 120/672 [01:34<07:14,  1.27it/s]
propagate in video:  18%|█▊        | 121/672 [01:34<06:07,  1.50it/s]
propagate in video:  18%|█▊        | 122/672 [01:36<08:02,  1.14it/s]
propagate in video:  18%|█▊        | 123/672 [01:37<07:52,  1.16it/s]
propagate in video:  18%|█▊        | 124/672 [01:37<06:33,  1.39it/s]
propagate in video:  19%|█▊        | 125/672 [01:37<05:41,  1.60it/s]
propagate in video:  19%|█▉        | 126/672 [01:38<05:02,  1.81it/s]
propagate in video:  19%|█▉        | 127/672 [01:39<06:54,  1.32it/s]
propagate in video:  19%|█▉        | 128/672 [01:40<08:42,  1.04it/s]
propagate in video:  19%|█▉        | 129/672 [01:41<07:17,  1.24it/s]
propagate in video:  19%|█▉        | 130/672 [01:41<06:10,  1.46it/s]
propagate in video:  19%|█▉        | 131/672 [01:42<06:29,  1.39it/s]
propagate in video:  20%|█▉        | 132/672 [01:42<05:38,  1.60it/s]
propagate in video:  20%|█▉        | 133/672 [01:43<05:13,  1.72it/s]
propagate in video:  20%|█▉        | 134/672 [01:43<04:59,  1.80it/s]
propagate in video:  20%|██        | 135/672 [01:44<04:53,  1.83it/s]
propagate in video:  20%|██        | 136/672 [01:45<05:29,  1.63it/s]
propagate in video:  20%|██        | 137/672 [01:46<07:38,  1.17it/s]
propagate in video:  21%|██        | 138/672 [01:47<06:40,  1.33it/s]
propagate in video:  21%|██        | 139/672 [01:47<05:40,  1.57it/s]
propagate in video:  21%|██        | 140/672 [01:48<06:44,  1.32it/s]
propagate in video:  21%|██        | 141/672 [01:49<06:08,  1.44it/s]
propagate in video:  21%|██        | 142/672 [01:49<05:37,  1.57it/s]
propagate in video:  21%|██▏       | 143/672 [01:50<05:13,  1.69it/s]
propagate in video:  21%|██▏       | 144/672 [01:50<04:52,  1.80it/s]
propagate in video:  22%|██▏       | 145/672 [01:51<04:39,  1.89it/s]
propagate in video:  22%|██▏       | 146/672 [01:51<05:07,  1.71it/s]
propagate in video:  22%|██▏       | 147/672 [01:52<06:24,  1.36it/s]
propagate in video:  22%|██▏       | 148/672 [01:53<06:50,  1.28it/s]
propagate in video:  22%|██▏       | 149/672 [01:54<06:06,  1.43it/s]
propagate in video:  22%|██▏       | 150/672 [01:55<07:38,  1.14it/s]
propagate in video:  22%|██▏       | 151/672 [01:55<06:37,  1.31it/s]
propagate in video:  23%|██▎       | 152/672 [01:56<06:34,  1.32it/s]
propagate in video:  23%|██▎       | 153/672 [01:57<07:20,  1.18it/s]
propagate in video:  23%|██▎       | 154/672 [01:59<09:01,  1.04s/it]
propagate in video:  23%|██▎       | 155/672 [02:00<09:53,  1.15s/it]
propagate in video:  23%|██▎       | 156/672 [02:01<09:04,  1.05s/it]
propagate in video:  23%|██▎       | 157/672 [02:02<09:57,  1.16s/it]
propagate in video:  24%|██▎       | 158/672 [02:03<08:26,  1.02it/s]
propagate in video:  24%|██▎       | 159/672 [02:04<08:10,  1.04it/s]
propagate in video:  24%|██▍       | 160/672 [02:05<08:01,  1.06it/s]
propagate in video:  24%|██▍       | 161/672 [02:06<09:40,  1.14s/it]
propagate in video:  24%|██▍       | 162/672 [02:08<10:33,  1.24s/it]
propagate in video:  24%|██▍       | 163/672 [02:09<10:17,  1.21s/it]
propagate in video:  24%|██▍       | 164/672 [02:10<10:44,  1.27s/it]
propagate in video:  25%|██▍       | 165/672 [02:11<08:51,  1.05s/it]
propagate in video:  25%|██▍       | 166/672 [02:11<07:18,  1.15it/s]
propagate in video:  25%|██▍       | 167/672 [02:12<06:10,  1.36it/s]
propagate in video:  25%|██▌       | 168/672 [02:12<05:19,  1.58it/s]
propagate in video:  25%|██▌       | 169/672 [02:13<05:52,  1.43it/s]
propagate in video:  25%|██▌       | 170/672 [02:14<06:37,  1.26it/s]
propagate in video:  25%|██▌       | 171/672 [02:15<05:41,  1.47it/s]
propagate in video:  26%|██▌       | 172/672 [02:15<05:56,  1.40it/s]
propagate in video:  26%|██▌       | 173/672 [02:17<07:40,  1.08it/s]
propagate in video:  26%|██▌       | 174/672 [02:18<08:33,  1.03s/it]
propagate in video:  26%|██▌       | 175/672 [02:19<08:07,  1.02it/s]
propagate in video:  26%|██▌       | 176/672 [02:19<06:54,  1.20it/s]
propagate in video:  26%|██▋       | 177/672 [02:21<08:42,  1.06s/it]
propagate in video:  26%|██▋       | 178/672 [02:22<09:38,  1.17s/it]
propagate in video:  27%|██▋       | 179/672 [02:24<09:44,  1.19s/it]
propagate in video:  27%|██▋       | 180/672 [02:24<08:06,  1.01it/s]
propagate in video:  27%|██▋       | 181/672 [02:26<09:15,  1.13s/it]
propagate in video:  27%|██▋       | 182/672 [02:26<08:02,  1.01it/s]
propagate in video:  27%|██▋       | 183/672 [02:27<06:32,  1.25it/s]
propagate in video:  27%|██▋       | 184/672 [02:28<08:09,  1.00s/it]
propagate in video:  28%|██▊       | 185/672 [02:30<09:39,  1.19s/it]
propagate in video:  28%|██▊       | 186/672 [02:30<07:50,  1.03it/s]
propagate in video:  28%|██▊       | 187/672 [02:31<06:35,  1.23it/s]
propagate in video:  28%|██▊       | 188/672 [02:31<05:31,  1.46it/s]
propagate in video:  28%|██▊       | 189/672 [02:31<04:49,  1.67it/s]
propagate in video:  28%|██▊       | 190/672 [02:32<04:23,  1.83it/s]
propagate in video:  28%|██▊       | 191/672 [02:33<05:13,  1.53it/s]
propagate in video:  29%|██▊       | 192/672 [02:33<04:45,  1.68it/s]
propagate in video:  29%|██▉       | 194/672 [02:34<04:29,  1.77it/s]
propagate in video:  29%|██▉       | 195/672 [02:35<04:16,  1.86it/s]
propagate in video:  29%|██▉       | 196/672 [02:36<05:02,  1.57it/s]
propagate in video:  29%|██▉       | 197/672 [02:36<04:30,  1.76it/s]
propagate in video:  29%|██▉       | 198/672 [02:37<05:12,  1.51it/s]
propagate in video:  30%|██▉       | 199/672 [02:37<04:58,  1.58it/s]
propagate in video:  30%|██▉       | 200/672 [02:39<06:09,  1.28it/s]
propagate in video:  30%|██▉       | 201/672 [02:40<06:41,  1.17it/s]
propagate in video:  30%|███       | 202/672 [02:40<06:34,  1.19it/s]
propagate in video:  30%|███       | 203/672 [02:41<06:20,  1.23it/s]
propagate in video:  30%|███       | 204/672 [02:42<05:35,  1.40it/s]
propagate in video:  31%|███       | 205/672 [02:43<05:52,  1.33it/s]
propagate in video:  31%|███       | 206/672 [02:44<06:50,  1.13it/s]INFO:     100.64.1.32:49190 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:46:20 GMT+0300 (Moscow Standard Time) 
propagate in video:  31%|███       | 207/672 [02:44<05:44,  1.35it/s]
propagate in video:  31%|███       | 208/672 [02:45<04:57,  1.56it/s]INFO:     100.64.1.32:49190 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:46:20 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:49190 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:46:21 GMT+0300 (Moscow Standard Time) 
propagate in video:  31%|███       | 209/672 [02:46<06:08,  1.26it/s]
propagate in video:  31%|███▏      | 210/672 [02:46<05:16,  1.46it/s]INFO:     100.64.1.32:49200 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:46:21 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:49216 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:46:22 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:49222 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:46:22 GMT+0300 (Moscow Standard Time) INFO:     100.64.1.32:49222 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:46:22 GMT+0300 (Moscow Standard Time) 
propagate in video:  31%|███▏      | 211/672 [02:47<05:55,  1.30it/s]INFO:     100.64.1.32:49222 - "GET /api/sources/s_20260925_122548_e8fbd5c5/video HTTP/1.1" 206 Partial Content
Fri Sep 25 2026 15:49:14 GMT+0300 (Moscow Standard Time) 
propagate in video:  32%|███▏      | 212/672 [02:48<06:50,  1.12it/s]
propagate in video:  32%|███▏      | 213/672 [02:50<08:23,  1.10s/it]
propagate in video:  32%|███▏      | 214/672 [02:51<08:15,  1.08s/it]
propagate in video:  32%|███▏      | 215/672 [02:51<06:56,  1.10it/s]
propagate in video:  32%|███▏      | 216/672 [02:52<06:48,  1.12it/s]
propagate in video:  32%|███▏      | 217/672 [02:53<05:48,  1.30it/s]
propagate in video:  32%|███▏      | 218/672 [02:54<07:11,  1.05it/s]
propagate in video:  33%|███▎      | 219/672 [02:55<07:03,  1.07it/s]
propagate in video:  33%|███▎      | 220/672 [02:55<05:51,  1.29it/s]
propagate in video:  33%|███▎      | 221/672 [02:56<05:53,  1.27it/s]
propagate in video:  33%|███▎      | 222/672 [02:57<05:12,  1.44it/s]
propagate in video:  33%|███▎      | 223/672 [02:58<06:40,  1.12it/s]
propagate in video:  33%|███▎      | 224/672 [02:58<05:44,  1.30it/s]
propagate in video:  33%|███▎      | 225/672 [02:59<05:07,  1.45it/s]
propagate in video:  34%|███▎      | 226/672 [03:00<05:23,  1.38it/s]
propagate in video:  34%|███▍      | 227/672 [03:01<05:47,  1.28it/s]
propagate in video:  34%|███▍      | 228/672 [03:01<04:56,  1.50it/s]
propagate in video:  34%|███▍      | 229/672 [03:02<05:48,  1.27it/s]
propagate in video:  34%|███▍      | 230/672 [03:03<05:10,  1.43it/s]
propagate in video:  34%|███▍      | 231/672 [03:03<04:46,  1.54it/s]
propagate in video:  35%|███▍      | 232/672 [03:04<05:16,  1.39it/s]
propagate in video:  35%|███▍      | 233/672 [03:05<05:27,  1.34it/s]
propagate in video:  35%|███▍      | 234/672 [03:05<04:57,  1.47it/s]
propagate in video:  35%|███▍      | 235/672 [03:06<04:31,  1.61it/s]
propagate in video:  35%|███▌      | 236/672 [03:07<06:14,  1.16it/s]
propagate in video:  35%|███▌      | 237/672 [03:09<06:58,  1.04it/s]
propagate in video:  35%|███▌      | 238/672 [03:10<07:26,  1.03s/it]
propagate in video:  36%|███▌      | 239/672 [03:10<06:13,  1.16it/s]
propagate in video:  36%|███▌      | 240/672 [03:11<06:31,  1.10it/s]
propagate in video:  36%|███▌      | 241/672 [03:12<05:51,  1.23it/s]
propagate in video:  36%|███▌      | 242/672 [03:12<05:22,  1.33it/s]
propagate in video:  36%|███▌      | 243/672 [03:13<04:39,  1.53it/s]
propagate in video:  36%|███▋      | 244/672 [03:13<04:06,  1.74it/s]
propagate in video:  36%|███▋      | 245/672 [03:15<06:02,  1.18it/s]
propagate in video:  37%|███▋      | 246/672 [03:16<07:13,  1.02s/it]
propagate in video:  37%|███▋      | 247/672 [03:18<08:37,  1.22s/it]
propagate in video:  37%|███▋      | 248/672 [03:19<08:23,  1.19s/it]
propagate in video:  37%|███▋      | 249/672 [03:19<06:59,  1.01it/s]
propagate in video:  37%|███▋      | 250/672 [03:20<05:57,  1.18it/s]
propagate in video:  37%|███▋      | 251/672 [03:20<05:03,  1.39it/s]
propagate in video:  38%|███▊      | 252/672 [03:21<04:23,  1.60it/s]
propagate in video:  38%|███▊      | 253/672 [03:22<04:58,  1.40it/s]
propagate in video:  38%|███▊      | 254/672 [03:23<05:20,  1.30it/s]
propagate in video:  38%|███▊      | 255/672 [03:23<04:43,  1.47it/s]
propagate in video:  38%|███▊      | 256/672 [03:24<04:10,  1.66it/s]
propagate in video:  38%|███▊      | 257/672 [03:24<03:44,  1.85it/s]
propagate in video:  38%|███▊      | 258/672 [03:24<03:32,  1.95it/s]
propagate in video:  39%|███▊      | 259/672 [03:25<04:48,  1.43it/s]
propagate in video:  39%|███▊      | 260/672 [03:26<05:16,  1.30it/s]
propagate in video:  39%|███▉      | 261/672 [03:27<04:38,  1.48it/s]
propagate in video:  39%|███▉      | 262/672 [03:27<04:06,  1.66it/s]
propagate in video:  39%|███▉      | 263/672 [03:28<03:54,  1.75it/s]
propagate in video:  39%|███▉      | 264/672 [03:28<03:31,  1.93it/s]
propagate in video:  39%|███▉      | 265/672 [03:29<04:33,  1.49it/s]
propagate in video:  40%|███▉      | 266/672 [03:30<05:09,  1.31it/s]
propagate in video:  40%|███▉      | 267/672 [03:31<04:35,  1.47it/s]
propagate in video:  40%|███▉      | 268/672 [03:32<05:37,  1.20it/s]
propagate in video:  40%|████      | 269/672 [03:32<05:04,  1.32it/s]
propagate in video:  40%|████      | 270/672 [03:33<05:33,  1.21it/s]
propagate in video:  40%|████      | 271/672 [03:34<04:57,  1.35it/s]
propagate in video:  40%|████      | 272/672 [03:34<04:16,  1.56it/s]
propagate in video:  41%|████      | 273/672 [03:36<05:18,  1.25it/s]
propagate in video:  41%|████      | 274/672 [03:36<04:34,  1.45it/s]
propagate in video:  41%|████      | 275/672 [03:37<05:22,  1.23it/s]
propagate in video:  41%|████      | 276/672 [03:38<04:45,  1.39it/s]
propagate in video:  41%|████      | 277/672 [03:38<04:06,  1.60it/s]
propagate in video:  41%|████▏     | 278/672 [03:39<05:37,  1.17it/s]
propagate in video:  42%|████▏     | 279/672 [03:40<05:52,  1.11it/s]
propagate in video:  42%|████▏     | 280/672 [03:41<04:54,  1.33it/s]
propagate in video:  42%|████▏     | 281/672 [03:41<04:20,  1.50it/s]
propagate in video:  42%|████▏     | 282/672 [03:42<04:04,  1.60it/s]
propagate in video:  42%|████▏     | 283/672 [03:42<03:39,  1.77it/s]
propagate in video:  42%|████▏     | 284/672 [03:43<03:30,  1.84it/s]
propagate in video:  42%|████▏     | 285/672 [03:43<03:12,  2.02it/s]
propagate in video:  43%|████▎     | 286/672 [03:44<04:23,  1.47it/s]
propagate in video:  43%|████▎     | 287/672 [03:45<04:05,  1.57it/s]
propagate in video:  43%|████▎     | 288/672 [03:48<08:56,  1.40s/it]
propagate in video:  43%|████▎     | 290/672 [03:48<05:36,  1.13it/s]
propagate in video:  43%|████▎     | 291/672 [03:49<04:53,  1.30it/s]
propagate in video:  43%|████▎     | 292/672 [03:50<04:57,  1.28it/s]
propagate in video:  44%|████▎     | 293/672 [03:51<06:15,  1.01it/s]
propagate in video:  44%|████▍     | 294/672 [03:52<06:19,  1.00s/it]
propagate in video:  44%|████▍     | 295/672 [03:54<07:01,  1.12s/it]
propagate in video:  44%|████▍     | 296/672 [03:55<06:22,  1.02s/it]
propagate in video:  44%|████▍     | 297/672 [03:56<06:52,  1.10s/it]
propagate in video:  44%|████▍     | 298/672 [03:57<06:50,  1.10s/it]
propagate in video:  44%|████▍     | 299/672 [03:57<05:46,  1.08it/s]
propagate in video:  45%|████▍     | 300/672 [03:58<05:40,  1.09it/s]
propagate in video:  45%|████▍     | 301/672 [03:59<04:53,  1.26it/s]
propagate in video:  45%|████▍     | 302/672 [04:00<05:19,  1.16it/s]
propagate in video:  45%|████▌     | 303/672 [04:00<04:40,  1.31it/s]
propagate in video:  45%|████▌     | 304/672 [04:01<04:52,  1.26it/s]
propagate in video:  45%|████▌     | 305/672 [04:03<05:45,  1.06it/s]
propagate in video:  46%|████▌     | 306/672 [04:04<06:07,  1.00s/it]
propagate in video:  46%|████▌     | 307/672 [04:04<05:16,  1.15it/s]
propagate in video:  46%|████▌     | 308/672 [04:05<05:46,  1.05it/s]
propagate in video:  46%|████▌     | 309/672 [04:07<07:01,  1.16s/it]
propagate in video:  46%|████▌     | 310/672 [04:08<06:44,  1.12s/it]
propagate in video:  46%|████▋     | 311/672 [04:09<05:34,  1.08it/s]
propagate in video:  46%|████▋     | 312/672 [04:09<04:43,  1.27it/s]
propagate in video:  47%|████▋     | 313/672 [04:09<04:04,  1.47it/s]
propagate in video:  47%|████▋     | 314/672 [04:10<03:33,  1.68it/s]
propagate in video:  47%|████▋     | 315/672 [04:10<03:20,  1.78it/s]
propagate in video:  47%|████▋     | 316/672 [04:12<05:12,  1.14it/s]
propagate in video:  47%|████▋     | 317/672 [04:13<05:35,  1.06it/s]
propagate in video:  47%|████▋     | 318/672 [04:13<04:44,  1.24it/s]
propagate in video:  47%|████▋     | 319/672 [04:14<04:53,  1.20it/s]
propagate in video:  48%|████▊     | 320/672 [04:16<05:27,  1.07it/s]
propagate in video:  48%|████▊     | 321/672 [04:16<04:39,  1.26it/s]
propagate in video:  48%|████▊     | 322/672 [04:16<03:58,  1.47it/s]
propagate in video:  48%|████▊     | 323/672 [04:17<03:36,  1.61it/s]
propagate in video:  48%|████▊     | 324/672 [04:17<03:11,  1.81it/s]
propagate in video:  48%|████▊     | 325/672 [04:18<03:39,  1.58it/s]
propagate in video:  49%|████▊     | 326/672 [04:20<05:19,  1.08it/s]
propagate in video:  49%|████▊     | 327/672 [04:20<05:00,  1.15it/s]
propagate in video:  49%|████▉     | 328/672 [04:21<04:35,  1.25it/s]
propagate in video:  49%|████▉     | 329/672 [04:22<04:13,  1.36it/s]
propagate in video:  49%|████▉     | 330/672 [04:23<04:52,  1.17it/s]
propagate in video:  49%|████▉     | 331/672 [04:24<04:33,  1.25it/s]
propagate in video:  49%|████▉     | 332/672 [04:24<04:09,  1.36it/s]
propagate in video:  50%|████▉     | 333/672 [04:25<04:10,  1.35it/s]
propagate in video:  50%|████▉     | 334/672 [04:25<03:45,  1.50it/s]
propagate in video:  50%|████▉     | 335/672 [04:27<05:04,  1.10it/s]
propagate in video:  50%|█████     | 336/672 [04:27<04:39,  1.20it/s]
propagate in video:  50%|█████     | 337/672 [04:28<03:57,  1.41it/s]
propagate in video:  50%|█████     | 338/672 [04:29<04:05,  1.36it/s]
propagate in video:  50%|█████     | 339/672 [04:30<04:43,  1.18it/s]
propagate in video:  51%|█████     | 340/672 [04:30<04:15,  1.30it/s]
propagate in video:  51%|█████     | 341/672 [04:31<03:59,  1.38it/s]
propagate in video:  51%|█████     | 342/672 [04:32<04:27,  1.23it/s]
propagate in video:  51%|█████     | 343/672 [04:33<04:02,  1.36it/s]
propagate in video:  51%|█████     | 344/672 [04:33<03:44,  1.46it/s]
propagate in video:  51%|█████▏    | 345/672 [04:34<03:35,  1.52it/s]
propagate in video:  51%|█████▏    | 346/672 [04:35<04:04,  1.33it/s]
propagate in video:  52%|█████▏    | 347/672 [04:35<03:30,  1.54it/s]
propagate in video:  52%|█████▏    | 348/672 [04:35<03:04,  1.75it/s]
propagate in video:  52%|█████▏    | 349/672 [04:36<02:50,  1.89it/s]
propagate in video:  52%|█████▏    | 350/672 [04:37<03:16,  1.64it/s]
propagate in video:  52%|█████▏    | 351/672 [04:38<03:32,  1.51it/s]
propagate in video:  52%|█████▏    | 352/672 [04:39<04:13,  1.26it/s]
propagate in video:  53%|█████▎    | 353/672 [04:39<04:09,  1.28it/s]
propagate in video:  53%|█████▎    | 354/672 [04:40<03:41,  1.43it/s]
propagate in video:  53%|█████▎    | 355/672 [04:40<03:21,  1.57it/s]
propagate in video:  53%|█████▎    | 356/672 [04:41<03:01,  1.74it/s]
propagate in video:  53%|█████▎    | 357/672 [04:41<02:56,  1.79it/s]
propagate in video:  53%|█████▎    | 358/672 [04:42<02:39,  1.97it/s]
propagate in video:  53%|█████▎    | 359/672 [04:42<02:29,  2.10it/s]
propagate in video:  54%|█████▎    | 360/672 [04:43<03:01,  1.72it/s]
propagate in video:  54%|█████▎    | 361/672 [04:44<03:49,  1.36it/s]
propagate in video:  54%|█████▍    | 362/672 [04:45<03:25,  1.51it/s]
propagate in video:  54%|█████▍    | 363/672 [04:45<03:48,  1.35it/s]
propagate in video:  54%|█████▍    | 364/672 [04:46<03:15,  1.58it/s]
propagate in video:  54%|█████▍    | 365/672 [04:46<02:59,  1.71it/s]
propagate in video:  54%|█████▍    | 366/672 [04:47<03:25,  1.49it/s]
propagate in video:  55%|█████▍    | 367/672 [04:49<04:32,  1.12it/s]
propagate in video:  55%|█████▍    | 368/672 [04:49<04:31,  1.12it/s]
propagate in video:  55%|█████▍    | 369/672 [04:51<04:44,  1.07it/s]
propagate in video:  55%|█████▌    | 370/672 [04:51<04:00,  1.26it/s]
propagate in video:  55%|█████▌    | 371/672 [04:54<07:10,  1.43s/it]
propagate in video:  55%|█████▌    | 372/672 [04:55<06:59,  1.40s/it]
propagate in video:  56%|█████▌    | 373/672 [04:57<07:08,  1.43s/it]
propagate in video:  56%|█████▌    | 374/672 [04:58<06:42,  1.35s/it]
propagate in video:  56%|█████▌    | 375/672 [04:59<06:05,  1.23s/it]
propagate in video:  56%|█████▌    | 376/672 [05:01<06:44,  1.37s/it]
propagate in video:  56%|█████▌    | 377/672 [05:02<06:11,  1.26s/it]
propagate in video:  56%|█████▋    | 378/672 [05:02<05:44,  1.17s/it]
propagate in video:  56%|█████▋    | 379/672 [05:03<04:56,  1.01s/it]
propagate in video:  57%|█████▋    | 380/672 [05:04<05:07,  1.05s/it]
propagate in video:  57%|█████▋    | 381/672 [05:06<05:55,  1.22s/it]
propagate in video:  57%|█████▋    | 382/672 [05:07<05:19,  1.10s/it]
propagate in video:  57%|█████▋    | 383/672 [05:08<05:01,  1.04s/it]
propagate in video:  57%|█████▋    | 384/672 [05:08<04:11,  1.14it/s]
propagate in video:  57%|█████▋    | 385/672 [05:10<05:37,  1.18s/it]
propagate in video:  57%|█████▋    | 386/672 [05:11<05:00,  1.05s/it]
propagate in video:  58%|█████▊    | 387/672 [05:11<04:19,  1.10it/s]
propagate in video:  58%|█████▊    | 388/672 [05:12<04:13,  1.12it/s]
propagate in video:  58%|█████▊    | 389/672 [05:13<04:32,  1.04it/s]
propagate in video:  58%|█████▊    | 390/672 [05:14<04:45,  1.01s/it]
propagate in video:  58%|█████▊    | 391/672 [05:15<04:00,  1.17it/s]
propagate in video:  58%|█████▊    | 392/672 [05:16<04:13,  1.11it/s]
propagate in video:  58%|█████▊    | 393/672 [05:17<05:04,  1.09s/it]
propagate in video:  59%|█████▊    | 394/672 [05:19<05:08,  1.11s/it]
propagate in video:  59%|█████▉    | 395/672 [05:20<05:03,  1.10s/it]
propagate in video:  59%|█████▉    | 396/672 [05:20<04:07,  1.12it/s]
propagate in video:  59%|█████▉    | 397/672 [05:21<04:41,  1.02s/it]
propagate in video:  59%|█████▉    | 398/672 [05:22<04:34,  1.00s/it]
propagate in video:  59%|█████▉    | 399/672 [05:23<03:50,  1.18it/s]
propagate in video:  60%|█████▉    | 400/672 [05:23<03:15,  1.39it/s]
propagate in video:  60%|█████▉    | 401/672 [05:24<03:22,  1.34it/s]
propagate in video:  60%|█████▉    | 402/672 [05:25<03:41,  1.22it/s]
propagate in video:  60%|█████▉    | 403/672 [05:26<04:15,  1.05it/s]
propagate in video:  60%|██████    | 404/672 [05:27<04:21,  1.02it/s]
propagate in video:  60%|██████    | 405/672 [05:29<04:48,  1.08s/it]
propagate in video:  60%|██████    | 406/672 [05:29<03:59,  1.11it/s]
propagate in video:  61%|██████    | 407/672 [05:30<04:31,  1.02s/it]
propagate in video:  61%|██████    | 408/672 [05:31<04:13,  1.04it/s]
propagate in video:  61%|██████    | 409/672 [05:32<04:17,  1.02it/s]
propagate in video:  61%|██████    | 410/672 [05:33<03:44,  1.16it/s]
propagate in video:  61%|██████    | 411/672 [05:34<04:24,  1.01s/it]
propagate in video:  61%|██████▏   | 412/672 [05:35<03:44,  1.16it/s]
propagate in video:  61%|██████▏   | 413/672 [05:35<03:17,  1.31it/s]
propagate in video:  62%|██████▏   | 414/672 [05:36<03:32,  1.21it/s]
propagate in video:  62%|██████▏   | 415/672 [05:37<03:07,  1.37it/s]
propagate in video:  62%|██████▏   | 416/672 [05:38<03:11,  1.34it/s]
propagate in video:  62%|█████�

## Assistant

По этому логу **Grounding DINO шёл на CUDA**, не на CPU.

Доказательства из бута:

- `torch 2.8.0+cu128 cuda=True`
- doctor: `device=cuda`, `Budget device=cuda … vram_budget≈23 GB`
- адаптер: `grounding-dino ready (… device=cuda)`
- `sam2 … device=cuda`

То, что выглядит «как CPU», в логе — **не детектор**. После `POST /api/jobs` видно:

1. загрузка ProPainter (`Pretrained flow/ProPainter has loaded`)
2. потом долгое **`propagate in video: … 672`** у **sam2-video** (~1–2 кадра/с)

На 672 кадрах это десятки минут даже на GPU. Tiny DINO отрабатывает только ключевые кадры и в этом логе почти не виден — он уже прошёл до ProPainter/SAM.

Предупреждение `cannot import name '_C' from 'sam2'` — нет нативного CUDA-расширения для post-processing дыр в масках; SAM2 пишет, что шаг пропускается. На устройство самого forward это не переключает пайплайн на CPU.

**Итог:** окружение и конфиг — CUDA; узкое место в этом прогоне — `sam2-video` propagate (и дальше inpaint), а не `grounding-dino-tiny` на процессоре.

Если нужно подтвердить по живому job: в UI/отчёте стадии смотри `device` у джоба и тайминги `detect` vs `segment`/`inpaint`. `detect` на tiny+ключевых кадрах на 4090 обычно секунды–десятки секунд; минуты+ почти всегда segment/inpaint.

## User

ок, разберись как ускорить процесс, все очень долго ана карте и процессоре ресурсы простаивают, сейчас загружена карта только на 5 гб а процессок в пике на 30 процентов

## Assistant

Разберусь с узкими местами пайплайна и почему GPU/CPU почти простаивают. Сначала посмотрю спеки/код стадий и как сейчас устроены workers/батчи.

## Tools

- Read: /home/user/.grok/installed-plugins/superpowers-21e2a56d/skills/brainstorming/SKILL.md
- Read: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/videoclean-inpainting.md
- Read: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/videoclean-stage-rail.md (1-65)
- Tool: Plan mode entered

## Assistant

Это скорее исследование узких мест с последующим планом ускорения. Смотрю, где пайплайн сериализуется и почему GPU почти пустой.

## Tools

- Search: inpaint_workers|resolve_workers|propagate|offload|batch_size|vram
- Read: /home/user/Projects/videoclean/docs/superpowers/specs/2026-09-24-runtime-and-editor.md (2-121)
- Tool: Ask: Что важнее ускорить в первую очередь на этом RunPod-прогоне?
- Read: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py
- Read: /home/user/Projects/videoclean/videoclean/application/inpaint_runtime.py
- Search: offload_video_to_cpu|offload_state_to_cpu|init_state|FrameStore|weights_cache|plan_holes|inpaint_masked|resolve_workers
- Execute: cd /home/user/Projects/videoclean && git log --oneline -20 && git status -sb && git rev-parse HEAD && git rev-parse origin/main 2>/dev/null; ls docs/superpowers/specs/ 2>/dev/null | head -20 (Check git state and specs present)
- Read: /home/user/Projects/videoclean/videoclean/application/profiles.py (2-121)
- Search: offload|storage_device|sam2-video|inpaint_workers|PROFILE
- Read: /home/user/Projects/videoclean/videoclean/adapters/inpainters/propainter.py (2-121)
- Search: offload|vram_peak|timings|segment
- Tool: Ask: Какой сегментатор был в том медленном job (в UI или пресете)?
- Read: /home/user/Projects/videoclean/videoclean/adapters/inpainters/lama.py (89-188)
- Read: /home/user/Projects/videoclean/videoclean/application/hole_policy.py (2-181)
- Search: offload|storage_device|budget|vram
- Search: offload_video_to_cpu|offload_state
- Tool: Ask: Готовность к компромиссу по качеству ради скорости на quality/sam2-video?
- Search: vram_budget|Budget|make_segmenter|Sam2Video
- Search: vram_budget|budget
- Execute: python3 - <<'PY'
# Estimate SAM2 video tensor memory for 672 frames
n=672
image_size=1024
images = n*3*image_size*image_size*4
print(f"images float32 NCHW: {images/1024**3:.2f} GiB")
# rough feature cache - SAM2 often keeps per-frame embeddings; hiera-tiny is smaller
# image encoder features at stride 16: 64x64 x C(~256) 
feat = n*256*64*64*2  # bf16
print(f"rough per-frame feats bf16: {feat/1024**3:.2f} GiB")
print(f"images+feats rough: {(images+feat)/1024**3:.2f} GiB")
# with 24GB and ~23GB budget after reserve, 672@1024 may be tight with propainter weights too
# but during segment, propainter isn't loaded yet if sequential
PY (Estimate SAM2 video VRAM for 672 frames)
- Read: /home/user/Projects/videoclean/videoclean/composition.py (231-310)
- Execute: python3 - <<'PY'
import inspect, textwrap
# try find sam2 in common places
import sys
candidates=[]
for p in ['/workspace/.venv/lib/python3.12/site-packages/sam2','']:
    pass
import importlib.util
spec=importlib.util.find_spec('sam2')
print('sam2', spec)
PY
# search repo for comments about offload
rg -n "offload_video|offload_state|storage_device" /home/user/Projects/videoclean -g '*.md' -g '*.py' | head -40
 (Find SAM2 offload docs and references)
- Read: /home/user/Projects/videoclean/videoclean/application/use_cases/run_cleanup.py (561-610)
- Search: segmenter\.|self\.segmenter|_segment
- Read: /home/user/Projects/videoclean/videoclean/adapters/inpainters/propainter.py (121-220)
- Search: _ensure|_prepare|loaded_now|lazy
- Search: def _ensure|loaded_now|\.to\(
- Tool: Ask: В план ускорения quality-пути: кроме GPU-residency для sam2-video, включать ли отложенную загрузку ProPainter (сейчас _prepare грузит его до сегментации и отъедает VRAM)?
- Search: 120|SLA|sla_|timings|budget
- Search: frame_count|30fps|1080|SLA
- ListDir: tests/fixtures/sla
- Tool: Ask: Цель 2–3 мин на 1‑минутное видео: какой рецепт качества допустим? (честно: quality=sam2-video+ProPainter на длинном ролике сам по себе может не уложиться даже после ускорения)

## Assistant

Учту цель `sam2-video+ProPainter` и параллельно разберу ошибку `_raft_flows`.

## Tools

- Search: _raft_flows|raft_flows|def _raft
- Read: /home/user/Projects/videoclean/videoclean/adapters/inpainters/propainter.py (186-335)
- Execute: cd /home/user/Projects/videoclean && git log -5 --oneline -- videoclean/adapters/inpainters/propainter.py && git blame -L 210,230 videoclean/adapters/inpainters/propainter.py 2>/dev/null | head -30; wc -l videoclean/adapters/inpainters/propainter.py; grep -n "def " videoclean/adapters/inpainters/propainter.py (Inspect propainter methods and history)

## Assistant

Нашёл критический баг в ProPainter и причину простоя GPU. Пишу план.

## Tools

- Read: /home/user/Projects/videoclean/videoclean/adapters/inpainters/propainter.py (331-390)
- Edit: /home/user/.grok/sessions/%2Fhome%2Fuser%2FProjects%2Fvideoclean/01a0d898-b995-7a41-a066-8d46e5b1646c/plan.md
- Tool: Plan mode exited

## Assistant

План одобрен — начинаю с фикса ProPainter, затем lazy load и GPU-residency для sam2-video.

## Tools

- Read: /home/user/.grok/installed-plugins/superpowers-21e2a56d/skills/test-driven-development/SKILL.md
- Read: /home/user/Projects/videoclean/tests/test_propainter_pair.py
- Read: /home/user/Projects/videoclean/videoclean/adapters/inpainters/propainter.py (241-390)
- Edit: /home/user/Projects/videoclean/tests/test_propainter_pair.py
- Execute: cd /home/user/Projects/videoclean && uv run pytest -q tests/test_propainter_pair.py::test_propainter_keeps_raft_helpers_on_the_class 2>&1 (Verify new ProPainter test fails (RED))
- Execute: cd /home/user/Projects/videoclean && python3 <<'PY'
from pathlib import Path
path = Path('videoclean/adapters/inpainters/propainter.py')
text = path.read_text()
# Extract the misplaced methods (currently nested under _pad_pair)
start = text.index('    def _raft_flows(self, frames, video_length):')
end = text.index('\n\ndef find_vendor()')
methods = text[start:end]
# Dedent one level (4 spaces) — they were class-indented but nested wrongly under _pad_pair
# Actually they have 4-space indent which is correct for class methods. The problem is they're
# after _pad_pair's returns. We need:
# 1) close class after _run
# 2) _pad_pair alone
# 3) wait - methods need to be ON the class, so BEFORE _pad_pair, after _run

# Better approach: cut methods out of _pad_pair, insert after _run (before _pad_pair)
pad_start = text.index('\ndef _pad_pair(')
run_end = text.index('        return out[:keep]\n\n\ndef _pad_pair(')
# Actually find exact
marker = '        return out[:keep]\n'
idx = text.index(marker)
after_run = idx + len(marker)
# Current structure from after_run:
# \ndef _pad_pair(...):\n ... return ...\n\n    def _raft_flows...

# New structure:
#     def _raft_flows... (class methods still indented)
#         return out[:keep]
#
# def _pad_pair(...):
#     ... only the real body ...
# def find_vendor...

pad_header_end = text.index('    return [frames[0], frames[0]], [masks[0], masks[0]], 1\n', after_run) + len('    return [frames[0], frames[0]], [masks[0], masks[0]], 1\n')
# methods start after blank line
assert text[pad_header_end:pad_header_end+2] == '\n '
# methods from first "    def _raft" to before find_vendor
methods_block = text[start:end]  # includes leading spaces of def
# _pad_pair function only:
pad_only = text[text.index('\ndef _pad_pair('):start].rstrip() + '\n'

new_text = (
    text[:after_run]
    + '\n'
    + methods_block
    + '\n'
    + pad_only
    + '\n'
    + text[end:]  # starts with \n\ndef find_vendor
)
path.write_text(new_text)
print('rewrote', path)
# sanity
ns = {}
exec(compile(path.read_text(), str(path), 'exec'), {'__name__': 'x'})
# can't exec easily due to imports
print('bytes', path.stat().st_size)
# check structure with ast
import ast
tree = ast.parse(path.read_text())
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='ProPainterInpainter']
assert classes
methods = {n.name for n in classes[0].body if isinstance(n, ast.FunctionDef)}
print('class methods sample:', sorted(m for m in methods if m.startswith('_'))[:20])
for need in ('_raft_flows','_complete_flow','_propagate','_transformer','_run'):
    print(need, need in methods)
funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
print('_pad_pair module', '_pad_pair' in funcs)
# ensure nested funcs inside _pad_pair are gone
pad = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='_pad_pair'][0]
nested = [n.name for n in ast.walk(pad) if isinstance(n, ast.FunctionDef) and n is not pad]
print('nested in _pad_pair', nested)
PY (Restore ProPainter methods onto the class via AST rewrite)
- Execute: cd /home/user/Projects/videoclean && python3 <<'PY'
from pathlib import Path
import ast
path = Path('videoclean/adapters/inpainters/propainter.py')
tree = ast.parse(path.read_text())
classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='ProPainterInpainter']
methods = {n.name for n in classes[0].body if isinstance(n, ast.FunctionDef)}
for need in ('_raft_flows','_complete_flow','_propagate','_transformer','_run','_ensure'):
    print(need, need in methods)
funcs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
print('_pad_pair module', '_pad_pair' in funcs)
pad = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='_pad_pair'][0]
nested = [n.name for n in ast.walk(pad) if isinstance(n, ast.FunctionDef) and n is not pad]
print('nested in _pad_pair', nested)
# show around boundary
lines = path.read_text().splitlines()
for i,l in enumerate(lines[230:270], start=231):
    print(f'{i}:{l}')
PY
uv run pytest -q tests/test_propainter_pair.py 2>&1
 (Verify AST structure and run ProPainter tests)
- Read: /home/user/Projects/videoclean/videoclean/application/use_cases/run_cleanup.py (551-650)
- Search: _prepare_runtime|ensure\(|inpainter
- Read: /home/user/Projects/videoclean/videoclean/application/budget.py (69-148)
- Read: /home/user/Projects/videoclean/videoclean/application/hole_policy.py (181-280)
- Search: lama|propainter|family
- Search: LamaInpainter|propainter.*lama|dual|_lama

## Assistant

Пишу тест для выбора GPU/CPU storage у sam2-video, потом правки runtime.

## Tools

- Edit: /home/user/Projects/videoclean/tests/test_sam2_video_storage.py
- Execute: cd /home/user/Projects/videoclean && uv run pytest -q tests/test_sam2_video_storage.py 2>&1 (RED: sam2 storage choice tests should fail)
- Edit: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py
- Edit: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py
- Edit: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py
- Edit: /home/user/Projects/videoclean/videoclean/application/use_cases/run_cleanup.py
- Read: /home/user/Projects/videoclean/videoclean/application/use_cases/run_cleanup.py (661-740)
- Search: _paint_holes|inpaint_masked|_ready_inpainter
- Read: /home/user/Projects/videoclean/videoclean/application/use_cases/run_cleanup.py (721-770)
- Search: _hole_report|segmenter|_budget_report|report\.json
- Read: /home/user/Projects/videoclean/videoclean/application/use_cases/run_cleanup.py (481-520)
- Edit: /home/user/Projects/videoclean/videoclean/application/use_cases/run_cleanup.py
- Read: /home/user/Projects/videoclean/videoclean/application/use_cases/run_cleanup.py (758-797)
- Edit: /home/user/Projects/videoclean/videoclean/application/use_cases/run_cleanup.py
- Execute: cd /home/user/Projects/videoclean && uv run pytest -q tests/test_propainter_pair.py tests/test_sam2_video_storage.py tests/test_hole_policy.py tests/test_adapters_status.py 2>&1 (Run targeted tests for ProPainter and sam2 changes)
- Execute: cd /home/user/Projects/videoclean && uv run pytest -q 2>&1 (Run full pytest suite)
- Read: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/videoclean-git.md (1-13)
- Edit: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/observations/_inbox/2026-09-25-quality-throughput.md

## Assistant

Сделано три связанных фикса под твой quality-путь. Все **268** тестов зелёные. Коммит не создавал — скажи, если пушить на `main`.

### Что было не так

1. **ProPainter падал** с `'… has no attribute '_raft_flows'`  
   После `_pad_pair` методы `_raft_flows` / `_complete_flow` / `_propagate` / `_transformer` оказались **внутри** функции (мёртвый код), а не на классе.

2. **GPU почти пустой (~5 ГБ)**  
   `sam2-video` всегда ставил `offload_video_to_cpu=True` и `storage_device=cpu` → каждый кадр `propagate` ходил по PCIe (~1 it/s).

3. **ProPainter грузился до сегментации**  
   `_prepare_runtime` делал `_ensure` у inpainter сразу, веса заливки отъедали VRAM у SAM2 ещё до `propagate`.

### Что изменилось

| Место | Изменение |
|--------|-----------|
| `propainter.py` | Методы снова на классе |
| `sam2_video.py` | `choose_sam2_storage`: CUDA, если тензор кадров + запас влезают в budget |
| `run_cleanup.py` | Inpainter lazy до `_paint_holes`; budget обновляется после загрузки; в отчёте `sam2OffloadVideo` / `sam2StorageDevice` |

### Что ждать на поде после деплоя

- Короткий/средний клип (как твои ~672 кадра): **GPU-path**, VRAM заметно выше, `propagate` быстрее.
- Полная **минута (~1800 кадров @1024)**: один float32-тензор ≈ **21 ГиБ** — на 24 ГБ после reserve часто **всё ещё offload**. Для жёстких 2–3 мин на quality нужен следующий шаг: chunked `sam2-video` или меньший `image_size`.

### Как проверить

После обновления кода на поде — quality на том же ролике и смотри:

- нет ошибки `_raft_flows`;
- в `nvidia-smi` во время segment VRAM растёт;
- в `report.json` → `budget.sam2OffloadVideo: false` (если влезло) и тайминги `segment` / `inpaint`.

Могу сразу сделать коммит и пуш в `main`, либо следующим заходом набросать chunked propagate под минутные ролики.

## User

комить и пуш

## Tools

- Execute: cd /home/user/Projects/videoclean && git status -sb && git diff --stat && git log -5 --oneline && git rev-parse --abbrev-ref HEAD && git status -u --short (Inspect git status, diff, and recent commits)
- Execute: cd /home/user/Projects/videoclean && git diff webui/src/widgets/editor-viewer/index.tsx && git diff videoclean/ tests/test_propainter_pair.py tests/test_sam2_video_storage.py | head -5; ls tests/test_sam2_video_storage.py (Check unrelated webui diff vs our changes)

## Assistant

В коммит возьму только фиксы throughput; правку подсказки в `editor-viewer` не трогаю.

## Tools

- Execute: cd /home/user/Projects/videoclean && git add \
  tests/test_propainter_pair.py \
  tests/test_sam2_video_storage.py \
  videoclean/adapters/inpainters/propainter.py \
  videoclean/adapters/segmenters/sam2_video.py \
  videoclean/application/use_cases/run_cleanup.py && \
git -c user.name=mikestotik -c user.email=mikestotik@gmail.com commit -m "$(cat <<'EOF'
fix: restore ProPainter methods and keep sam2-video on GPU when it fits

Move _raft_flows and related helpers back onto ProPainterInpainter after
_pad_pair swallowed them. Choose CUDA storage for sam2-video from the VRAM
budget, and load the inpainter only before paint so segment keeps the card.
EOF
)" && git status -sb && git push origin main && git status -sb && git log -1 --format='%h %an <%ae> %s' (Commit throughput fixes and push to main)

## Assistant

Запушил в `main`: **`5f90cb8`** — `fix: restore ProPainter methods and keep sam2-video on GPU when it fits`.

В коммите: ProPainter, sam2-video storage, lazy inpainter, тесты.  
Локально осталась незакоммиченная правка подсказки в `webui/src/widgets/editor-viewer/index.tsx` — к этому фиксу она не относится.

На поде нужен pull/redeploy, чтобы подхватить коммит.

## User

Run set -euo pipefail
TCP SSH not reachable, falling back to proxy
using RUNPOD_SSH_PROXY_SUFFIX → cpy65b5zj24xab-6441174d
using proxy SSH: cpy65b5zj24xab-6441174d@ssh.runpod.io
Warning: Permanently added 'ssh.runpod.io' (RSA) to the list of known hosts.

container not found
Connection to ssh.runpod.io closed.

Traceback (most recent call last):
  File "/home/runner/work/videoclean/videoclean/scripts/runpod_ssh_exec.py", line 356, in <module>
    raise SystemExit(main())
                     ^^^^^^
  File "/home/runner/work/videoclean/videoclean/scripts/runpod_ssh_exec.py", line 351, in main
    ssh_session(ssh_argv, remote_script, upload=upload, timeout=args.timeout)
  File "/home/runner/work/videoclean/videoclean/scripts/runpod_ssh_exec.py", line 66, in ssh_session
    _wait_prompt(fd, buf)
  File "/home/runner/work/videoclean/videoclean/scripts/runpod_ssh_exec.py", line 45, in _wait_prompt
    raise RuntimeError("timed out waiting for shell prompt")
RuntimeError: timed out waiting for shell prompt выдает на гитхабе

## Assistant

Разберу падение GitHub Actions на SSH в RunPod: сначала скрипт и воркфлоу, потом причину `container not found` / timeout prompt.

## Tools

- Read: /home/user/Projects/videoclean/scripts/runpod_ssh_exec.py (1-120)
- Search: container not found|proxy|RUNPOD_SSH|wait_prompt|ssh_session
- ListDir: .github/workflows
- Read: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/videoclean-runpod.md
- Read: /home/user/Projects/videoclean/scripts/runpod_ssh_exec.py (301-357)
- Read: /home/user/Projects/videoclean/.github/workflows/runpod-deploy.yml (1-120)
- Read: /home/user/Projects/videoclean/docs/RUNPOD.md (1-80)
- Tool: Ask: Под на RunPod сейчас какой?

## Assistant

Под Running, а proxy отвечает `container not found` — часто это протухший `RUNPOD_SSH_PROXY_SUFFIX`. Проверю резолв username и сделаю быстрый fail + retry.

## Tools

- Read: /home/user/Projects/videoclean/scripts/runpod_ssh_exec.py (121-320)
- Search: RUNPOD_SSH_PROXY|proxy|find-running|ssh
- Execute: cd /home/user/Projects/videoclean && git log -5 --oneline -- scripts/runpod_ssh_exec.py .github/workflows/runpod-deploy.yml && ls tests/*runpod* 2>/dev/null; rg -n "container not found|_wait_prompt|PROXY_SUFFIX" tests scripts --glob '*.py' 2>/dev/null | head -20 (Find existing tests and PROXY_SUFFIX usage)

## Assistant

Причина: proxy SSH отвечает `container not found` (часто протухший suffix), а скрипт 45 секунд ждёт prompt. Сделаю fail-fast и retry через live-резолв username.

## Tools

- Execute: cd /home/user/Projects/videoclean && ls tests/test_*ssh* tests/test_*runpod* 2>/dev/null; head -5 scripts/runpod_ssh_exec.py (Check for existing SSH exec tests)
- Tool: Ask: Как чинить деплой?
- Edit: /home/user/Projects/videoclean/tests/test_runpod_ssh_exec.py
- Edit: /home/user/Projects/videoclean/tests/test_runpod_ssh_exec.py
- Execute: cd /home/user/Projects/videoclean && uv run pytest -q tests/test_runpod_ssh_exec.py 2>&1 (RED: run new runpod ssh exec tests)
- Edit: /home/user/Projects/videoclean/scripts/runpod_ssh_exec.py
- Edit: /home/user/Projects/videoclean/scripts/runpod_ssh_exec.py
- Edit: /home/user/Projects/videoclean/docs/RUNPOD.md
- Execute: cd /home/user/Projects/videoclean && git add scripts/runpod_ssh_exec.py tests/test_runpod_ssh_exec.py docs/RUNPOD.md && git -c user.name=mikestotik -c user.email=mikestotik@gmail.com commit -m "$(cat <<'EOF'
fix(runpod): fail fast on dead SSH proxy and retry live usernames

Detect "container not found" instead of hanging on the shell prompt.
Probe runpodctl/API proxy users before the stale Actions suffix so deploy
survives a recycled Connect username.
EOF
)" && git push origin main && git status -sb && git log -1 --oneline (Commit and push RunPod SSH proxy fix)

## Assistant

**Что случилось:** это не «таймаут prompt» как первопричина. Proxy SSH ответил **`container not found`** и закрыл сессию; скрипт 45 секунд ждал `root@…#` и упал уже на этом.

Типичные причины при «под Running»:
- протухший **`RUNPOD_SSH_PROXY_SUFFIX`** (`6441174d` в логе) после рестарта/пересоздания контейнера;
- TCP SSH недоступен → всегда идём в proxy;
- контейнер в API ещё «RUNNING», а SSH-proxy mapping уже мёртвый.

**Что запушено:** `434b297` — fail-fast на `container not found`, probe live username (`runpodctl`/API) **раньше** suffix, понятная ошибка вместо висения.

Перезапусти workflow **RunPod deploy → deploy**. Если снова упадёт — открой Connect у пода и обнови GitHub Variable `RUNPOD_SSH_PROXY_SUFFIX` на хвост из:

`ssh cpy65b5zj24xab-XXXXXXXX@ssh.runpod.io`

## User

sam2-video failed: OutOfMemoryError: CUDA out of memory. Tried to allocate 56.00 MiB. GPU 0 has a total capacity of 23.53 GiB of which 39.69 MiB is free. Including non-PyTorch memory, this process has 23.48 GiB memory in use. Of the allocat

## Assistant

OOM из‑за того, что кадры и state SAM2 заняли почти всю карту — не осталось места на forward. Подкручу политику residency и добавлю fallback на CPU при OOM.

## Tools

- Read: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py (1-220)
- Read: /home/user/Projects/videoclean/tests/test_sam2_video_storage.py
- Read: /home/user/.grok/installed-plugins/superpowers-21e2a56d/skills/systematic-debugging/SKILL.md

## Assistant

Корневая причина: при `images + 3 ГиБ ≤ budget` кадры и state держались на GPU; кэш фич на длинном ролике доедал остаток (23.48 ГиБ занято, 40 МиБ свободно). Сделаю три уровня residency и OOM-fallback.

## Tools

- Edit: /home/user/Projects/videoclean/tests/test_sam2_video_storage.py
- Execute: cd /home/user/Projects/videoclean && uv run pytest -q tests/test_sam2_video_storage.py 2>&1 (RED: updated sam2 storage tests)
- Edit: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py
- Edit: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py
- Edit: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py
- Execute: cd /home/user/Projects/videoclean && git add videoclean/adapters/segmenters/sam2_video.py tests/test_sam2_video_storage.py && git -c user.name=mikestotik -c user.email=mikestotik@gmail.com commit -m "$(cat <<'EOF'
fix(sam2-video): hybrid GPU storage and OOM offload retry

Keep frames on CUDA with state on CPU when a full-GPU cushion would not
fit, and retry fully offloaded after CUDA OOM so long quality clips stop
dying with 40 MiB free.
EOF
)" && git push origin main && git log -1 --oneline && git status -sb (Commit and push sam2-video OOM fix)

## Assistant

**Почему OOM:** прошлый фикс клал на GPU и кадры, и state. На длинном ролике кэш фич раздувался → **23.48 ГиБ занято, ~40 МиБ свободно**, падение на аллокации 56 МиБ.

**Что сделано** (`c9018ed`, уже в `main`):

1. **Три режима**
   - полный GPU — только если кадры + **8 ГиБ** запаса влезают;
   - **hybrid** — кадры на CUDA, state на CPU (основной путь для длинных клипов);
   - полный offload — если даже кадры не влезают.

2. **OOM → retry** с полным offload (job не умирает сразу).

После деплоя на под в отчёте смотри `budget.sam2OffloadVideo` / `sam2StorageDevice`. На минутном ролике чаще будет hybrid: `offloadVideo=false`, state на CPU — быстрее чистого CPU, без заполнения всей карты.

## User

и захвати мои изменения когда будешь комитить и пушить

## Tools

- Execute: cd /home/user/Projects/videoclean && git status -sb && git diff && git log -3 --oneline (Inspect uncommitted user changes for commit)
- Execute: cd /home/user/Projects/videoclean && git add webui/src/widgets/editor-viewer/index.tsx && git -c user.name=mikestotik -c user.email=mikestotik@gmail.com commit -m "$(cat <<'EOF'
fix(webui): shorten annotate viewer hint text

Drop the phrase/brush sentence so the shortcut line stays readable.
EOF
)" && git push origin main && git status -sb && git log -1 --oneline (Commit and push user editor-viewer hint change)

## Assistant

Закоммитил и запушил твою правку: **`58fa32d`** — укороченная подсказка в annotate-режиме `editor-viewer`.

Рабочее дерево чистое, `main` совпадает с `origin/main`.

## User

sam2-video failed: OutOfMemoryError: CUDA out of memory. Tried to allocate 24.00 MiB. GPU 0 has a total capacity of 23.53 GiB of which 3.69 MiB is free. Including non-PyTorch memory, this process has 23.51 GiB memory in use. Of the allocate

## Assistant

Снова OOM на sam2-video — проверю, срабатывает ли retry с полным offload и хватает ли headroom в текущем коде.

## Tools

- Read: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/videoclean-pipeline.md
- Read: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/preferences.md
- Search: choose_sam2_storage|OutOfMemory|offload|empty_cache|retry
- Execute: cd /home/user/Projects/videoclean && git log --oneline -8 && git status -sb && git diff --stat HEAD && ls webui/src/widgets/editor-viewer 2>/dev/null | head -5; rg -n "annotate|подсказ|hint" webui/src/widgets/editor-viewer -g '*.tsx' 2>/dev/null | head -20 (Check git state and recent commits)

## Assistant

Код с hybrid/OOM-retry уже на main — разберу, почему исключение всё ещё пробивается наружу.

## Tools

- Read: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py
- Read: /home/user/Projects/videoclean/tests/test_sam2_video_storage.py
- Search: vram_budget|make_segmenter|Sam2Video|sam2-video|lazy|inpainter
- Execute: cd /home/user/Projects/videoclean && git show c9018ed --stat && git log -1 --format='%H %ci %s' c9018ed; # check if user's local changes exist uncommitted
git status; git diff --name-only; git diff HEAD -- webui/ | head -80 (Inspect OOM fix commit and local diffs)

## Assistant

Похоже, retry не освобождает VRAM после первого падения — проверю budget и как чистится state.

## Tools

- Search: vram_budget|mem_get_info|memory_allocated|resolve_budget|free
- Search: vram_budget_bytes|resolve_budget|sam2Offload|last_storage
- Execute: cd /home/user/Projects/videoclean && python -c "import torch; print(getattr(torch.cuda,'OutOfMemoryError',None)); print(getattr(torch,'OutOfMemoryError',None)); import inspect; print(issubclass(torch.cuda.OutOfMemoryError, RuntimeError) if hasattr(torch.cuda,'OutOfMemoryError') else 'n/a')" 2>/dev/null; rg -n "offload_state|cached_features|inference_state" --glob '!**/node_modules/**' -g '*.py' | head -40; find . -path '*sam2*video*predictor*' 2>/dev/null | head -10; uv run python -c "import sam2; import os; print(os.path.dirname(sam2.__file__))" 2>/dev/null (Locate SAM2 predictor and OOM types)
- Read: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/videoclean-inpainting.md
- Read: /home/user/Projects/videoclean/videoclean/application/use_cases/run_cleanup.py (541-640)
- Read: /home/user/Projects/videoclean/videoclean/application/budget.py
- Search: _prepare_runtime|_segment|detect|empty_cache|unload|to\(\"cpu\"\)
- Execute: cd /home/user/Projects/videoclean && uv run python -c "import sam2,os; print(os.path.dirname(sam2.__file__))" 2>&1; find /home/user -path '*sam2*video_predictor*' 2>/dev/null | head -5; ls ~/.cache/uv 2>/dev/null | head -3 (Find installed sam2 package path)

## Assistant

Причина: после OOM traceback держит CUDA-тензоры, а hybrid headroom 2 GiB слишком мал — retry падает на той же карте. Исправляю освобождение памяти, live-budget и порог hybrid.

## Tools

- Read: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/videoclean-git.md
- Execute: cd /home/user/Projects/videoclean && uv run pytest -q tests/test_sam2_video_storage.py 2>&1 (Run existing sam2 storage tests)
- Edit: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py
- Edit: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py
- Edit: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py

## Assistant

Добавлю явный сброс `state` при падении propagate — так retry не упрётся в те же тензоры.

## Tools

- Edit: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py

## Assistant

Обновляю тесты под новый hybrid headroom и добавляю проверки reclaim.

## Tools

- Edit: /home/user/Projects/videoclean/tests/test_sam2_video_storage.py
- Execute: cd /home/user/Projects/videoclean && uv run pytest -q tests/test_sam2_video_storage.py 2>&1 (Run updated sam2 storage tests)
- Execute: cd /home/user/Projects/videoclean && uv run pytest -q tests/test_sam2_video_storage.py tests/test_budget.py tests/test_propainter_pair.py 2>&1 (Run related regression tests)
- Read: /home/user/Projects/videoclean/videoclean/adapters/segmenters/sam2_video.py (2-181)
- Edit: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/videoclean-pipeline.md
- Execute: cd /home/user/Projects/videoclean && git status -sb && git diff --stat && git log -3 --oneline (Review status before commit)
- Execute: cd /home/user/Projects/videoclean && git add tests/test_sam2_video_storage.py videoclean/adapters/segmenters/sam2_video.py && git -c user.name=mikestotik -c user.email=mikestotik@gmail.com commit -m "$(cat <<'EOF'
fix(sam2-video): reclaim CUDA on OOM and raise hybrid headroom

The previous retry left inference_state tensors pinned by the exception
traceback, so empty_cache alone still showed ~23 GiB in use and the
offload retry died allocating 24 MiB. Clear state/traceback, probe live
free VRAM, and require 6 GiB hybrid headroom before keeping frames on GPU.
EOF
)" && git status -sb && git push origin main (Commit OOM fix and push to main)
- Edit: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/observations/_inbox/2026-09-25-sam2-oom-reclaim.md

## Assistant

Починил и запушил: **`33e44e6`** на `main`.

**Почему снова падало.** Первый OOM оставлял `inference_state` на GPU — traceback исключения держал тензоры живыми, `empty_cache` почти ничего не освобождал. Retry с offload стартовал при ~23.5 GiB занятых и снова умирал на аллокации ~24 MiB. Плюс hybrid headroom в 2 GiB позволял класть на GPU слишком длинные клипы.

**Что сделано:**
- при OOM сбрасываем traceback/`state`, `gc` + `empty_cache`, затем полный CPU offload;
- бюджет берём по **живому** free VRAM, а не только по значению из prepare;
- hybrid headroom поднят до **6 GiB** (длинные клипы чаще уходят в полный offload).

Локальных незакоммиченных правок не было — в коммит вошёл только этот фикс.

Чтобы на RunPod подтянулось: pull/restart пода (или дождись deploy). После рестарта тот же клип не должен умирать с этим OOM; в логе при нехватке VRAM может мелькнуть `OOM — retry offload`, и проход продолжится.

## User

/home/user/Desktop/logs.txt как бы нам ускориться? в 10 раз ато и более? 30 минут 20 секундное видео обрабатывал ну это же не дело и обработал вообще дерьмо полное, думаю нужно из других элементов пайплайн собирать, какие у нас есть варианты вообще?

## Assistant

Сначала разберу лог и текущие варианты пайплайна — откуда ушли 30 минут на 20 секунд и что можно собрать иначе.

## Tools

- Read: /home/user/Desktop/logs.txt (1-200)
- Read: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/preferences.md
- Read: /home/user/.grok/memory-v2/workspaces/videoclean-3a52d3e3/topics/videoclean-pipeline.md
- Tool: Ask: Как ответить на разбор лога и варианты ускорения?

## Assistant

В логе ищу тайминги джобы и полный список доступных детекторов/сегментеров/инпейнтеров.

## Tools

- Search: propagate|sam2|ProPainter|OOM|offload|job|stage|detect|inpaint|budget|seconds|elapsed|it/s|failed|error|quality|balanced|report|vram|duration|frame
- Execute: wc -l /home/user/Desktop/logs.txt; tail -n 150 /home/user/Desktop/logs.txt (Count log lines and show ending)
- Search: preset|quality|balanced|fast|segmenter|inpainter|detector|make_
- ListDir: videoclean/adapters
- Search: fast|balanced|quality|PROFILE|sam2-video|propainter|lama
- Read: /home/user/Projects/videoclean/docs/MODELS.md (1-63)
- Search: "fast"|"balanced"|"quality"|profile|segmenter|inpainter
- Execute: cd /home/user/Projects/videoclean && rg -n "def apply_profile|PROFILE|fast|balanced|quality" videoclean --glob '*.py' | head -60; ls docs/PARAMS.md 2>/dev/null; rg -n "^## |fast|balanced|quality|sam2|propainter|lama|detector" docs/PARAMS.md docs/MODELS.md 2>/dev/null | head -80 (Find profiles and model docs)
- Read: /home/user/Projects/videoclean/videoclean/application/profiles.py (1-130)
- Tool: Ask: Куда целимся дальше по пайплайну?

## Assistant

Понял: нужны другие архитектуры пайплайна в целом, не перестановки внутри текущего приложения. Соберу карту подходов, которые реально дают ×10 и лучше по качеству.

## Tools

- WebSearch: fast video object removal inpainting pipeline 2024 2025 SAM2 ProPainter alternatives
- WebSearch: realtime video inpainting logo removal architecture DiffuEraser MiniMax Remover ProPainter alternative

## Assistant

По логу это не «чуть медленно», а архитектурный потолок. На `20260925_140804` (~20 с, 672 кадра) ушло ~32 мин wall-clock: один только `sam2-video propagate` ~6–7 мин при ~1.7 it/s, потом ещё ProPainter. Схема detect → SAM2-video → ProPainter физически не даст ×10 на длинном клипе: оба этапа линейны по числу кадров и тяжёлые.

Ниже — **другие архитектуры пайплайна** (не перестановки внутри текущего стека).

---

### 1. Узкий delogo / watermark pipeline  
**Идея:** логотип/текст — почти статичный ROI. Нет open-vocab детекта на всём ролике.

`сцена-кат → фиксированный/слабо двигающийся бокс → crop окна → короткий video-inpaint → composite обратно`

Примеры мысли: shot-aware windowed ProPainter-delogo, ffmpeg delogo для совсем плоских кейсов, OCR→бокс для титров.

| | |
|---|---|
| Скорость | часто ×10…×50 на логотипах |
| Качество | отлично на статике; слабо на «убери человека» |
| Когда брать | продукт про водяные знаки / титры / corner logo |

---

### 2. Unified track+remove (одна модель)  
**Идея:** не «сначала маски на все кадры, потом заливка», а joint модель: маска на 1-м кадре → автоудаление по ролику.

Ориентиры: **MiniMax-Remover**, **RT-Remover** (в бенчах заявляют десятки FPS за счёт joint track+remove в diffusion transformer), **ROSE** / side-effect-aware removers.

| | |
|---|---|
| Скорость | единственный класс, где ×10…realtime реалистичен |
| Качество | лучше ProPainter на больших дырах и «эффектах» (тень, отражение) |
| Цена | новые веса, другой runtime, VRAM, лицензии |

Это главный кандидат, если цель — общее «убери объект», а не только лого.

---

### 3. Diffusion video inpaint (маска снаружи)  
**Идея:** маски отдельно (клик/SAM2 на keyframes), заливка — video diffusion.

Ориентиры: **DiffuEraser**, **FloED**, **EraserDiT**, **VideoPainter**, VIP.

| | |
|---|---|
| Скорость | обычно минуты, не секунды; EraserDiT и distilled варианты ближе к «быстро» |
| Качество | сильнее классического flow+transformer на сложных дырах |
| Риск | галлюцинации, другой UX, тяжелее деплой |

Имеет смысл как **замена ProPainter**, если маски уже есть дёшево.

---

### 4. Keyframe paint + temporal propagate  
**Идея:** залить 1–N ключевых кадров сильным image-inpaint (FLUX Fill / SDXL inpaint), дальше I2V / latent blend / optical-flow warp.

| | |
|---|---|
| Скорость | средняя; выигрыш на длинных роликах |
| Качество | нестабильно на быстром движении |
| Когда | прототип «красиво на постере», не broadcast-delogo |

---

### 5. Специализация по типу цели  
Разные задачи → разные пайплайны, один «universal quality» убивает и скорость, и результат.

| Задача | Архитектура |
|---|---|
| Corner logo / watermark | (1) windowed delogo |
| Бегущая строка / титры | OCR → track boxes → лёгкий inpaint |
| Человек / объект в кадре | (2) unified remover или (3) diffusion + редкие маски |
| «Убери всё лишнее по тексту» | VL/детектор только на keyframes → (2)/(3), без покадрового SAM2-video |

Плохой результат у вас скорее от неверной семьи модели под задачу + маски/заливка не сошлись, а не только от «мало GPU».

---

### 6. Продуктовый слой без своего ML  
Runway / Adobe / CapCut object remove — готовые чёрные ящики. Имеет смысл как бенчмарк качества/скорости, от которого отталкивать свой стек.

---

## Что реально даст ×10

На вашем прогоне ×10 ≈ **~3 минуты на 20 с**. Это почти недостижимо, пока в горячем пути остаются полный **SAM2-video на все кадры** и тяжёлый **ProPainter на весь клип**.

Практичная развилка:

1. **Продукт = логотипы/текст** → архитектура (1), ROI-only. Самый быстрый путь к ×10 и «не дерьмо».  
2. **Продукт = произвольные объекты** → архитектура (2), unified remover; маски/детект только для seed на 1–нескольких кадрах.  
3. **Нужна «киношность» дыры** → (3) DiffuEraser / EraserDiT / MiniMax как inpaint-ядро, сегментация максимально дешёвая.

---

## Честная рекомендация

Сначала зафиксировать **одну** продуктовую задачу (logo vs object vs text). Под неё выбрать одну из семей (1) или (2). Universal «DINO + SAM2-video + ProPainter» оставить как медленный research-path, не как дефолт.

Если скажешь, какая задача главная (лого / титры / люди / «всё по промпту»), накидаю конкретный целевой пайплайн из 4–6 блоков и что брать open-source первым прототипом.