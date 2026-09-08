import { cn } from "@/shared/lib/utils"

export function Timecode({ children, className }: { children: string; className?: string }) {
  return <span className={cn("font-mono text-sm tabular-nums tracking-tight", className)}>{children}</span>
}
