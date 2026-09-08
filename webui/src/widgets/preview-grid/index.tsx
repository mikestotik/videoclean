import { useState } from "react"
import { previewArtifactUrl, type PreviewManifest } from "@/entities/preview"
import { Badge } from "@/shared/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/shared/ui/card"
import { Dialog, DialogContent, DialogTitle, DialogTrigger } from "@/shared/ui/dialog"
import { Tabs, TabsList, TabsTrigger } from "@/shared/ui/tabs"

type Layer = "raw" | "boxes" | "mask"

type Props = { jobId: string; manifest: PreviewManifest }

export function PreviewGrid({ jobId, manifest }: Props) {
  const [layer, setLayer] = useState<Layer>("mask")

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="text-base">Превью масок</CardTitle>
        <Badge variant="secondary">покрытие ~{Math.round((manifest.meanMaskCoverage ?? 0) * 100)}%</Badge>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        <Tabs value={layer} onValueChange={(v) => setLayer(v as Layer)}>
          <TabsList>
            <TabsTrigger value="raw">raw</TabsTrigger>
            <TabsTrigger value="boxes">boxes</TabsTrigger>
            <TabsTrigger value="mask">mask</TabsTrigger>
          </TabsList>
        </Tabs>
        <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
          {manifest.frames.map((f) => (
            <Dialog key={f.index}>
              <DialogTrigger
                render={
                  <button className="relative overflow-hidden rounded-md border">
                    <img
                      src={previewArtifactUrl(jobId, f.artifacts[layer])}
                      alt={`frame ${f.index}`}
                      className="w-full"
                    />
                    <span className="absolute top-1 left-1 rounded bg-background/80 px-1 text-[10px]">
                      #{f.index} · {(f.maskCoverage * 100).toFixed(1)}%
                    </span>
                  </button>
                }
              />
              <DialogContent className="max-w-4xl">
                <DialogTitle>Кадр #{f.index}</DialogTitle>
                <img
                  src={previewArtifactUrl(jobId, f.artifacts[layer])}
                  alt={`frame ${f.index} full`}
                  className="max-h-[80vh] w-full object-contain"
                />
              </DialogContent>
            </Dialog>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
