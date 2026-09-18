import type { Job } from "@/entities/job/types"

type Listener = (jobs: Job[]) => void

let latest: Job[] = []
const listeners = new Set<Listener>()

/** Called from EventsProvider whenever the jobs list updates. */
export function publishJobs(jobs: Job[]): void {
  latest = jobs
  for (const listener of listeners) listener(jobs)
}

export function getCachedJobs(): Job[] {
  return latest
}

export function subscribeJobs(listener: Listener): () => void {
  listeners.add(listener)
  listener(latest)
  return () => {
    listeners.delete(listener)
  }
}

export function findCachedJob(id: string): Job | undefined {
  return latest.find((j) => j.id === id)
}
