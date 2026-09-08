import { useEffect, useState } from "react"
import { JobsSidebar } from "@widgets/jobs-sidebar"
import { RunForm } from "@widgets/run-form"
import { TargetTable } from "@widgets/target-table"
import { useFrameSelection } from "@features/frame-selection"
import { usePreviewRun } from "@features/preview-run"
import { Filmstrip } from "@widgets/filmstrip"
import { PreviewGrid } from "@widgets/preview-grid"
import { Viewer } from "@widgets/viewer"
import { Button } from "@/shared/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card"
import { Textarea } from "@/shared/ui/textarea"
import { probeJob, type Job, type MediaProbe } from "@/entities/job"
import type { TargetRow } from "@/entities/targets"

export function WorkspacePage() {
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [probe, setProbe] = useState<MediaProbe | null>(null)
  const [refreshKey, setRefreshKey] = useState(0)
  const [previewPrompt, setPreviewPrompt] = useState("")
  const [targets, setTargets] = useState<TargetRow[]>([])
  const selection = useFrameSelection()
  const preview = usePreviewRun(selectedJob, selection.selected, targets)

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
        {selectedJob && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Превью на выделенных кадрах ({selection.selected.size})
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <Textarea
                value={previewPrompt}
                onChange={(e) => setPreviewPrompt(e.target.value)}
                placeholder="Промпт для разбора (mode=parse)"
              />
              <div className="flex gap-2">
                <Button size="sm" disabled={preview.running} onClick={() => preview.run("parse", previewPrompt)}>
                  Разбор + маски
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  disabled={preview.running}
                  onClick={() => preview.run("detect", "")}
                >
                  Маски по таргетам
                </Button>
              </div>
              {preview.error && <p className="text-sm text-destructive">{preview.error}</p>}
              {preview.running && <p className="text-sm text-muted-foreground">Считаю…</p>}
            </CardContent>
          </Card>
        )}
        {preview.jobId && preview.manifest && (
          <PreviewGrid jobId={preview.jobId} manifest={preview.manifest} />
        )}
        <TargetTable rows={targets} onChange={setTargets} />
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
