import { useRef } from "react"
import { cn } from "@/shared/lib/utils"
import type { DetectBox } from "@features/detect-run"

export type OverlayTrack = {
  id: number
  label: string
  box: DetectBox | null
  enabled: boolean
}

const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"] as const
type Handle = (typeof HANDLES)[number]

function clampBox(box: DetectBox, w: number, h: number): DetectBox {
  let [x1, y1, x2, y2] = box
  x1 = Math.max(0, Math.min(w - 4, x1))
  y1 = Math.max(0, Math.min(h - 4, y1))
  x2 = Math.max(x1 + 4, Math.min(w, x2))
  y2 = Math.max(y1 + 4, Math.min(h, y2))
  return [x1, y1, x2, y2]
}

function applyHandle(start: DetectBox, handle: Handle, x: number, y: number): DetectBox {
  let [x1, y1, x2, y2] = start
  if (handle.includes("w")) x1 = x
  if (handle.includes("e")) x2 = x
  if (handle.includes("n")) y1 = y
  if (handle.includes("s")) y2 = y
  if (x2 < x1) [x1, x2] = [x2, x1]
  if (y2 < y1) [y1, y2] = [y2, y1]
  return [x1, y1, x2, y2]
}

function handleClass(handle: Handle): string {
  const pos: Record<Handle, string> = {
    nw: "left-0 top-0 -translate-x-1/2 -translate-y-1/2 cursor-nwse-resize",
    n: "left-1/2 top-0 -translate-x-1/2 -translate-y-1/2 cursor-ns-resize",
    ne: "right-0 top-0 translate-x-1/2 -translate-y-1/2 cursor-nesw-resize",
    e: "right-0 top-1/2 translate-x-1/2 -translate-y-1/2 cursor-ew-resize",
    se: "bottom-0 right-0 translate-x-1/2 translate-y-1/2 cursor-nwse-resize",
    s: "bottom-0 left-1/2 -translate-x-1/2 translate-y-1/2 cursor-ns-resize",
    sw: "bottom-0 left-0 -translate-x-1/2 translate-y-1/2 cursor-nesw-resize",
    w: "left-0 top-1/2 -translate-x-1/2 -translate-y-1/2 cursor-ew-resize",
  }
  return pos[handle]
}

type Props = {
  tracks: OverlayTrack[]
  videoWidth: number
  videoHeight: number
  selectedId: number | null
  onSelect: (id: number) => void
  onResize?: (id: number, box: DetectBox) => void
}

export function DetectBoxes({ tracks, videoWidth, videoHeight, selectedId, onSelect, onResize }: Props) {
  const dragRef = useRef<{
    id: number
    handle: Handle
    start: DetectBox
  } | null>(null)
  const stageRef = useRef<HTMLDivElement>(null)

  if (videoWidth <= 0 || videoHeight <= 0) return null

  const toVideo = (clientX: number, clientY: number): [number, number] => {
    const rect = stageRef.current?.getBoundingClientRect()
    if (!rect || rect.width === 0 || rect.height === 0) return [0, 0]
    return [
      ((clientX - rect.left) / rect.width) * videoWidth,
      ((clientY - rect.top) / rect.height) * videoHeight,
    ]
  }

  const onHandleMove = (e: React.PointerEvent) => {
    const drag = dragRef.current
    if (!drag || !onResize) return
    const [x, y] = toVideo(e.clientX, e.clientY)
    onResize(drag.id, clampBox(applyHandle(drag.start, drag.handle, x, y), videoWidth, videoHeight))
  }

  return (
    <div ref={stageRef} className="pointer-events-none absolute inset-0">
      {tracks.map((t) => {
        const box = t.box
        if (!box || box.length < 4 || !t.enabled) return null
        const selected = t.id === selectedId
        return (
          <div
            key={t.id}
            className={cn(
              "pointer-events-auto absolute border",
              selected ? "border-mask" : "border-mask/60",
            )}
            style={{
              left: `${(box[0] / videoWidth) * 100}%`,
              top: `${(box[1] / videoHeight) * 100}%`,
              width: `${((box[2] - box[0]) / videoWidth) * 100}%`,
              height: `${((box[3] - box[1]) / videoHeight) * 100}%`,
            }}
            onPointerDown={(e) => {
              e.stopPropagation()
              onSelect(t.id)
            }}
          >
            {selected &&
              onResize &&
              HANDLES.map((handle) => (
                <span
                  key={handle}
                  className={cn("absolute z-10 size-2 rounded-sm bg-mask", handleClass(handle))}
                  onPointerDown={(e) => {
                    e.stopPropagation()
                    e.currentTarget.setPointerCapture(e.pointerId)
                    dragRef.current = { id: t.id, handle, start: box }
                    onSelect(t.id)
                  }}
                  onPointerMove={onHandleMove}
                  onPointerUp={() => {
                    dragRef.current = null
                  }}
                />
              ))}
          </div>
        )
      })}
    </div>
  )
}
