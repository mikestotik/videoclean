import { useEffect, useState } from "react"
import { JobsSidebar } from "@widgets/jobs-sidebar"
import { RunForm } from "@widgets/run-form"
import { useFrameSelection } from "@features/frame-selection"
import { Filmstrip } from "@widgets/filmstrip"
import { Viewer } from "@widgets/viewer"
import { probeJob, type Job, type MediaProbe } from "@/entities/job"

export function WorkspacePage() {
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [probe, setProbe] = useState<MediaProbe | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const selection = useFrameSelection()

  useEffect(() => {
    if (!selectedJob) return
    let alive = true
    probeJob(selectedJob.id)
      .then((p) => {
        if (alive) setProbe(p)
      })
      .catch(() => {
        if (alive) setProbe(null)
      })
    return () => {
      alive = false
    }
  }, [selectedJob])

  return (
    <div className="flex gap-4">
      <div className="flex min-w-0 flex-1 flex-col gap-4">
        {selectedJob?.input_url && probe && (
          <>
            <Viewer src={selectedJob.input_url} fps={probe.fps} current={selection.current} />
            <Filmstrip
              src={selectedJob.input_url}
              fps={probe.fps}
              frameCount={probe.frame_count}
              thumbnails={48}
              selected={selection.selected}
              current={selection.current}
              onToggle={selection.toggle}
              onCurrent={(idx) => selection.setCurrent(idx)}
            />
          </>
        )}
        {!selectedJob && (
          <p className="text-sm text-muted-foreground">
            Выберите джоб справа, чтобы открыть редактор кадров.
          </p>
        )}
        <RunForm onSubmitted={() => setRefreshKey((k) => k + 1)} />
      </div>
      <JobsSidebar
        selectedId={selectedJob?.id ?? null}
        onSelect={(j: Job) => {
          setProbe(null)
          setSelectedJob(j)
        }}
        refreshKey={refreshKey}
      />
    </div>
  )
}
