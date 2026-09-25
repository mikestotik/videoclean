import { PARAM_STATION, paramLabel } from "./param-meta"

const jargon = /leftover|NMS|Verify:|ref stride|subvideo|\bdilate\b|RAFT|Overlap/i

for (const key of Object.keys(PARAM_STATION)) {
  const label = paramLabel(key)
  if (label === key) throw new Error(`${key} has no label`)
  if (jargon.test(label)) throw new Error(`${key} label is jargon: ${label}`)
}
