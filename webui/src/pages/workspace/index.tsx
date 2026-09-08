import { useState } from "react"
import { JobsSidebar } from "@widgets/jobs-sidebar"
import { RunForm } from "@widgets/run-form"
import type { Job } from "@/entities/job"

export function WorkspacePage() {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)

  return (
    <div className="flex gap-4">
      <div className="flex min-w-0 flex-1 flex-col gap-4">
        <RunForm onSubmitted={() => setRefreshKey((k) => k + 1)} />
      </div>
      <JobsSidebar selectedId={selectedId} onSelect={(j: Job) => setSelectedId(j.id)} refreshKey={refreshKey} />
    </div>
  )
}
