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
4. Опционально **Variables**: `RUNPOD_POD_NAME` (дефолт `videoclean-dev`), `RUNPOD_DATA_CENTER` (дефолт `EU-RO-1`).

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

- **SSH просит пароль** — в Settings попал не тот ключ (нужна строка `ssh-ed25519 AAAA...`, не fingerprint). Ключ, добавленный после старта пода, на живой pod сам не попадёт — нужен новый pod.
- **Volume не монтируется** — DC volume ≠ DC GPU. Для volume в `eu-ro-1` стартуй pod в `EU-RO-1`.
- **Нет UI / пустой static** — один раз прогони `deploy` или `start` из Actions (CI кладёт `server/static_dist`).
- **CUDA doctor ругается** — веса ещё не скачаны: открой Config в UI и догрузи модели; кэш останется на volume.
- **Pod сразу умер после deploy** — смотри `/workspace/run/serve.log` по SSH.
