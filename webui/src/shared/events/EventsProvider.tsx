import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react"
import type { Job } from "@/entities/job"
import type { Source } from "@/entities/source"
import { publishJobs } from "./jobBus"
import { connectEvents } from "./sse"
import type { EventsStatus, PollSnapshot } from "./types"

type EventsContextValue = {
  status: EventsStatus
  snapshot: PollSnapshot | null
  jobs: Job[]
  sources: Source[]
}

const EventsContext = createContext<EventsContextValue | null>(null)

export function EventsProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<EventsStatus>("connecting")
  const [snapshot, setSnapshot] = useState<PollSnapshot | null>(null)
  const [jobs, setJobs] = useState<Job[]>([])
  const [sources, setSources] = useState<Source[]>([])

  useEffect(() => {
    const ac = new AbortController()
    connectEvents(ac.signal, {
      onStatus: setStatus,
      onSnapshot: (data) => {
        setSnapshot(data)
        const nextJobs = data.jobs ?? []
        setJobs(nextJobs)
        publishJobs(nextJobs)
        setSources(data.sources ?? [])
      },
      onJobs: (next) => {
        setJobs(next)
        publishJobs(next)
      },
      onSources: setSources,
      onDownloads: (data) => {
        setSnapshot((prev) =>
          prev
            ? {
                ...prev,
                downloads: data.downloads ?? prev.downloads,
                models: (data.models as PollSnapshot["models"]) ?? prev.models,
                options: data.options ?? prev.options,
              }
            : prev,
        )
      },
      onMeta: (data) => {
        setSnapshot((prev) =>
          prev
            ? {
                ...prev,
                models: (data.models as PollSnapshot["models"]) ?? prev.models,
                doctor: data.doctor ?? prev.doctor,
                options: data.options ?? prev.options,
                ollama: data.ollama ?? prev.ollama,
                providers: data.providers ?? prev.providers,
                device: data.device ?? prev.device,
              }
            : {
                jobs: [],
                models: (data.models as PollSnapshot["models"]) ?? {},
                doctor: data.doctor ?? {},
                ollama: data.ollama ?? { ok: false, base_url: "", models: [] },
                options: data.options,
                providers: data.providers,
                device: data.device,
              },
        )
      },
    })
    return () => ac.abort()
  }, [])

  const value = useMemo(
    () => ({ status, snapshot, jobs, sources }),
    [status, snapshot, jobs, sources],
  )

  return <EventsContext.Provider value={value}>{children}</EventsContext.Provider>
}

export function useEvents(): EventsContextValue {
  const ctx = useContext(EventsContext)
  if (!ctx) throw new Error("useEvents must be used within EventsProvider")
  return ctx
}

/** Optional: null outside provider (for leaf widgets in tests). */
export function useEventsOptional(): EventsContextValue | null {
  return useContext(EventsContext)
}
