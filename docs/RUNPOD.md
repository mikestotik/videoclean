# RunPod (быстрый CUDA-dev loop)

Деплой **без** сборки CUDA-образа: база `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, код и venv на Network Volume, обновления через GitHub Actions + SSH/rsync.

## Один раз руками

1. **Network Volume** в том же DC, где берёшь RTX 4090 (например `EU-RO-1`, ~50 GB).
2. **SSH-ключ** для CI:
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/runpod_github_actions -C "github-actions-videoclean" -N ""
   cat ~/.ssh/runpod_github_actions.pub
   ```
   Публичный ключ → [RunPod Settings → SSH Public Keys](https://www.console.runpod.io/user/settings).
3. **GitHub Secrets** (Settings → Secrets → Actions):
   | Secret | Значение |
   |---|---|
   | `RUNPOD_API_KEY` | API key из RunPod |
   | `RUNPOD_NETWORK_VOLUME_ID` | id volume (как в S3/API) |
   | `RUNPOD_SSH_PRIVATE_KEY` | весь файл `~/.ssh/runpod_github_actions` включая `BEGIN/END` |
4. **Variables** (Settings → Secrets and variables → Actions → Variables):
   | Variable | Значение |
   |---|---|
   | `RUNPOD_SSH_PROXY_SUFFIX` | если API/v2 403: хвост из Connect `ssh <podId>-XXXXXXXX@ssh.runpod.io` → `XXXXXXXX` |
   | `RUNPOD_POD_NAME` | опционально, дефолт `videoclean-dev` |
   | `RUNPOD_DATA_CENTER` | опционально, дефолт `EU-RO-1` |

   Username для proxy Actions берёт так: `RUNPOD_SSH_PROXY_SUFFIX` → `runpodctl ssh info` → API v2/GraphQL.

## Автообновление после коммита

Пока pod **RUNNING**: каждый push в `main` → Actions:

1. проверяет, что `webui` собирается в CI;
2. по SSH (`ssh.runpod.io`) на поде: `git pull` → `bun run build` → restart `serve`.

`static_dist` в git не хранится — UI собирается на volume (bun кэшируется). Руками после коммита ничего делать не нужно.

Если pod выключен: push **не** поднимает GPU (экономия). Сначала Actions → **start**, дальше снова автодеплой с пушей.

## Как пользоваться

Actions → **RunPod deploy** → Run workflow:

| action | Что делает |
|---|---|
| **start** | Создаёт pod (4090 + volume), ждёт SSH, синкает код, поднимает UI |
| **deploy** | Синкает код на уже RUNNING pod и перезапускает serve |
| **stop** | **Terminate** pod (volume остаётся; GPU больше не биллится) |

На каждый **push в `main`** (пути library/server/webui/scripts): если pod RUNNING — hot deploy; если нет — workflow зелёный с пометкой «pod offline».

UI после старта: `https://<pod-id>-7860.proxy.runpod.net`

## Volume layout

```
/workspace/videoclean/   # checkout репо
/workspace/.venv/        # uv env (--system-site-packages → torch из образа)
/workspace/uv-cache/
/workspace/hf/           # HF_HOME
/workspace/data/         # jobs / uploads
/workspace/run/          # pid + serve.log
```

## Локальные скрипты на поде

- `scripts/runpod-bootstrap.sh` — clone + start (удобно для manual template)
- `scripts/runpod-start.sh` — cold start (sync deps, background serve, hold container)
- `scripts/runpod-restart.sh` — hot restart после rsync
- `scripts/runpod-api.sh` — REST-хелпер для CI

Torch **не** переустанавливается (`uv sync --no-install-package torch/torchvision`).

## Troubleshooting

- **`container not found` / timeout waiting for shell prompt** — proxy username устарел или контейнер ещё не готов. Actions сначала пробует live user из `runpodctl`/API, потом `RUNPOD_SSH_PROXY_SUFFIX`. Открой Connect у RUNNING-пода и обнови Variable `RUNPOD_SSH_PROXY_SUFFIX` на хвост из `ssh <podId>-XXXXXXXX@ssh.runpod.io`. Если под только что стартовал — подожди минуту и rerun workflow.
- **SSH просит пароль** — в Settings попал не тот ключ (нужна строка `ssh-ed25519 AAAA...`, не fingerprint). Ключ, добавленный после старта пода, на живой pod сам не попадёт — нужен новый pod.
- **Volume не монтируется** — DC volume ≠ DC GPU. Для volume в `eu-ro-1` стартуй pod в `EU-RO-1`.
- **Нет UI / пустой static** — один раз прогони `deploy` или `start` из Actions (CI кладёт `server/static_dist`).
- **CUDA doctor: модели unavailable** — нормально на первом старте. Веса качай из Config в UI; кэш на volume (`/workspace/hf`, `/workspace/data/weights`).
- **torch 2.14 / cu130 вместо 2.8+cu128** — в venv попал чужой torch (обычно через `sam2` deps). На поде:
  ```bash
  rm -rf /workspace/.venv
  bash /workspace/videoclean/scripts/runpod-restart.sh
  ```
  Скрипты теперь ставят sam2 с `--no-deps` и чистят venv от torch/nvidia-*.
- **Pod сразу умер после deploy** — смотри `/workspace/run/serve.log` по SSH.
