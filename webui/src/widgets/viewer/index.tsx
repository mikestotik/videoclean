type Props = { src: string; fps: number; current: number | null }

export function Viewer({ src, fps, current }: Props) {
  return (
    <video
      key={src}
      src={src}
      controls
      className="max-h-[50vh] w-full rounded-md border bg-black"
      ref={(el) => {
        if (el && current !== null && fps > 0) el.currentTime = current / fps
      }}
    />
  )
}
