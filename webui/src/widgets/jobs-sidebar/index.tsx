import { useEffect, useState } from "react"
import { cancelJob, deleteJob, listJobs, retryJob, type Job } from "@/entities/job"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card"
import { ScrollArea } from "@/shared/ui/scroll-area"

type Props = {
  selectedId: string | null
  onSelect: (job: Job) => void
  refreshKey: number
}

export function JobsSidebar({ selectedId, onSelect, refreshKey }: Props) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [error, setError] = useState("")

  useEffect(() => {
    let alive = true
    const tick = () => {
      listJobs()
        .then((j) => {
          if (alive) {
            setJobs(j)
            setError("")
          }
        })
        .catch((e) => {
          if (alive) setError(e instanceof Error ? e.message : String(e))
        })
    }
    tick()
    const t = setInterval(tick, 3000)
    return () => {
      alive = false
      clearInterval(t)
    }
  }, [refreshKey])

  const action = async (fn: (id: string) => Promise<unknown>, id: string) => {
    try {
      await fn(id)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <Card className="w-80 shrink-0">
      <CardHeader>
        <CardTitle className="text-base">Джобы</CardTitle>
      </CardHeader>
      <CardContent>
        {error && <p className="mb-2 text-sm text-destructive">{error}</p>}
        <ScrollArea className="h-[60vh]">
          <div className="flex flex-col gap-2 pr-2">
            {jobs.map((j) => (
              <button
                key={j.id}
                className={`flex flex-col items-start gap-1 rounded-md border p-2 text-left text-xs ${
                  j.id === selectedId ? "border-primary" : ""
                }`}
                onClick={() => onSelect(j)}
              >
                <span className="flex items-center gap-2">
                  <span className="font-mono">{j.id.slice(0, 8)}</span>
                  <Badge variant={j.state === "COMPLETED" ? "default" : j.state === "FAILED" ? "destructive" : "secondary"}>
                    {j.kind} · {j.state}
                  </Badge>
                </span>
                {j.stage && (
                  <span className="text-muted-foreground">
                    {j.stage} {Math.round(j.fraction * 100)}% {j.eta && `· ETA ${j.eta}`}
                  </span>
                )}
                <span className="flex gap-1">
                  {j.state === "RUNNING" || j.state === "QUEUED" ? (
                    <Button
                      size="xs"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation()
                        action(cancelJob, j.id)
                      }}
                    >
                      Стоп
                    </Button>
                  ) : null}
                  {j.state === "FAILED" ? (
                    <Button
                      size="xs"
                      variant="outline"
                      onClick={(e) => {
                        e.stopPropagation()
                        action(retryJob, j.id)
                      }}
                    >
                      Повтор
                    </Button>
                  ) : null}
                  {j.has_output && (
                    <a href={j.output_url ?? "#"} download onClick={(e) => e.stopPropagation()}>
                      <Button size="xs" variant="outline">Скачать</Button>
                    </a>
                  )}
                  <Button
                    size="xs"
                    variant="destructive"
                    onClick={(e) => {
                      e.stopPropagation()
                      action(deleteJob, j.id)
                    }}
                  >
                    Удалить
                  </Button>
                </span>
                {j.error && <p className="text-destructive">{j.error.slice(0, 200)}</p>}
              </button>
            ))}
            {jobs.length === 0 && <p className="text-xs text-muted-foreground">Пока пусто.</p>}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  )
}
