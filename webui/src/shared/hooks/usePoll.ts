import { useEffect, useRef } from "react"

export function usePoll(fn: () => void | Promise<void>, ms: number, paused = false): void {
  const fnRef = useRef(fn)

  useEffect(() => {
    fnRef.current = fn
    if (paused) return

    const tick = () => {
      void fnRef.current()
    }

    tick()
    const id = setInterval(tick, ms)
    return () => clearInterval(id)
  }, [fn, ms, paused])
}
