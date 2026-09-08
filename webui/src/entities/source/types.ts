export type SourceProbe = {
  fps: number
  duration_s: number
  width: number
  height: number
  frame_count: number
  has_audio: boolean
}

export type Source = {
  id: string
  name: string
  createdAt: string
  probe: SourceProbe
  video_url: string
  annotations_url: string
}
