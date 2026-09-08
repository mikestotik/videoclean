type Props = {
  url: string | null
  opacity: number
}

export function MaskOverlay({ url, opacity }: Props) {
  if (!url) return null
  return (
    <img
      src={url}
      alt="mask"
      className="pointer-events-none absolute inset-0 h-full w-full object-fill"
      style={{ opacity }}
    />
  )
}
