import { useCallback, useEffect, useState } from "react"
import { api } from "@/shared/api/client"
import { Badge } from "@/shared/ui/badge"
import { Button } from "@/shared/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card"
import { ScrollArea } from "@/shared/ui/scroll-area"

type ModelInfo = {
  id: string
  title: string
  kind: string
  state?: string
  progress?: number
  downloadable?: boolean
}
type PollData = {
  models: Record<string, ModelInfo[]>
  doctor: Record<string, unknown>
  ollama: { ok: boolean; base_url: string; models: string[] }
}

export function ConfigPage() {
  const [data, setData] = useState<PollData | null>(null)
  const [busy, setBusy] = useState("")
  const [error, setError] = useState("")

  const refresh = useCallback(() => {
    api<PollData>("/api/poll")
      .then((d) => {
        setData(d)
        setError("")
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)))
  }, [])
  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [refresh])

  const download = async (id: string) => {
    setBusy(id)
    setError("")
    try {
      await api("/api/models/download", { method: "POST", body: JSON.stringify({ id }) })
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy("")
    }
  }

  const cancelDownloads = async () => {
    setBusy("cancel")
    setError("")
    try {
      await api("/api/models/cancel", { method: "POST" })
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy("")
    }
  }

  if (!data) return <p className="text-sm text-muted-foreground">{error || "Загрузка…"}</p>

  return (
    <ScrollArea className="h-[calc(100svh-8rem)]">
      <div className="flex flex-col gap-4 pr-2">
        {error && <p className="text-sm text-destructive">{error}</p>}
        {Object.entries(data.models).map(([kind, models]) => (
          <Card key={kind}>
            <CardHeader>
              <CardTitle className="text-base">{kind}</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2 text-sm">
              {models.map((m) => (
                <div key={m.id} className="flex items-center gap-2">
                  <span className="font-mono">{m.title}</span>
                  {m.state === "downloading" ? (
                    <Badge variant="secondary">
                      {m.state} {Math.round((m.progress ?? 0) * 100)}%
                    </Badge>
                  ) : (
                    m.state && <Badge variant="secondary">{m.state}</Badge>
                  )}
                  {m.state === "downloading" && (
                    <Button size="xs" variant="outline" disabled={busy === "cancel"} onClick={() => void cancelDownloads()}>
                      {busy === "cancel" ? "…" : "Отменить"}
                    </Button>
                  )}
                  <Button
                    size="xs"
                    variant="outline"
                    disabled={busy === m.id || m.state === "downloading" || m.downloadable === false}
                    onClick={() => void download(m.id)}
                  >
                    {busy === m.id ? "…" : "Download"}
                  </Button>
                </div>
              ))}
            </CardContent>
          </Card>
        ))}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Doctor</CardTitle>
          </CardHeader>
          <CardContent>
            <pre className="text-xs whitespace-pre-wrap">{JSON.stringify(data.doctor, null, 2)}</pre>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Ollama</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            {data.ollama.base_url} — {data.ollama.ok ? "ok" : "недоступен"} · моделей:{" "}
            {data.ollama.models.length}
          </CardContent>
        </Card>
      </div>
    </ScrollArea>
  )
}
