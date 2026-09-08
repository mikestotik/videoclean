import { useState } from "react"
import { rangeSelect, toggleSelect } from "@/entities/frame"

export function useFrameSelection() {
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [current, setCurrent] = useState<number | null>(null)

  const toggle = (idx: number, additive: boolean, range: boolean) => {
    if (range) {
      setSelected((sel) => rangeSelect(sel, idx))
      setCurrent(idx)
      return
    }
    if (additive) setSelected((sel) => toggleSelect(sel, idx))
    else setSelected(new Set([idx]))
    setCurrent(idx)
  }

  const clear = () => {
    setSelected(new Set())
    setCurrent(null)
  }

  return { selected, current, toggle, setCurrent, clear }
}
