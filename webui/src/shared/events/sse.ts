import { apiUrl } from "@/shared/api/client"
import type { PollSnapshot } from "./types"

export type SseHandlers = {
  onSnapshot?: (data: PollSnapshot) => void
  onJobs?: (jobs: PollSnapshot["jobs"]) => void
  onSources?: (sources: NonNullable<PollSnapshot["sources"]>) => void
  onDownloads?: (data: Pick<PollSnapshot, "downloads" | "models" | "options">) => void
  onMeta?: (data: Pick<PollSnapshot, "models" | "doctor" | "options" | "ollama" | "providers" | "device">) => void
  onStatus?: (status: "connecting" | "live" | "reconnecting") => void
  onError?: (err: unknown) => void
}

/** Fetch-based SSE (supports split-origin; EventSource cannot set custom headers). */
export function connectEvents(signal: AbortSignal, handlers: SseHandlers): void {
  let attempt = 0

  const run = async () => {
    while (!signal.aborted) {
      handlers.onStatus?.(attempt === 0 ? "connecting" : "reconnecting")
      try {
        const res = await fetch(apiUrl("/api/events"), {
          signal,
          headers: { Accept: "text/event-stream" },
        })
        if (!res.ok || !res.body) {
          throw new Error(`SSE ${res.status} ${res.statusText}`)
        }
        attempt = 0
        handlers.onStatus?.("live")
        await readStream(res.body, signal, handlers)
      } catch (err) {
        if (signal.aborted) return
        handlers.onError?.(err)
      }
      attempt += 1
      const wait = Math.min(10_000, 500 * 2 ** Math.min(attempt, 4))
      await sleep(wait, signal)
    }
  }

  void run()
}

async function readStream(body: ReadableStream<Uint8Array>, signal: AbortSignal, handlers: SseHandlers) {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buf = ""
  try {
    while (!signal.aborted) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const parts = buf.split("\n\n")
      buf = parts.pop() ?? ""
      for (const chunk of parts) {
        dispatchChunk(chunk, handlers)
      }
    }
  } finally {
    reader.releaseLock()
  }
}

function dispatchChunk(chunk: string, handlers: SseHandlers) {
  const lines = chunk.split("\n")
  let event = "message"
  const dataLines: string[] = []
  for (const line of lines) {
    if (!line || line.startsWith(":")) continue
    if (line.startsWith("event:")) {
      event = line.slice(6).trim()
      continue
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart())
    }
  }
  if (!dataLines.length) return
  let parsed: unknown
  try {
    parsed = JSON.parse(dataLines.join("\n"))
  } catch {
    return
  }
  const data = parsed as PollSnapshot
  if (event === "snapshot") handlers.onSnapshot?.(data)
  else if (event === "jobs" && data.jobs) handlers.onJobs?.(data.jobs)
  else if (event === "sources" && data.sources) handlers.onSources?.(data.sources)
  else if (event === "downloads") handlers.onDownloads?.(data)
  else if (event === "meta") handlers.onMeta?.(data)
}

function sleep(ms: number, signal: AbortSignal): Promise<void> {
  return new Promise((resolve) => {
    if (signal.aborted) {
      resolve()
      return
    }
    const t = setTimeout(resolve, ms)
    signal.addEventListener(
      "abort",
      () => {
        clearTimeout(t)
        resolve()
      },
      { once: true },
    )
  })
}
