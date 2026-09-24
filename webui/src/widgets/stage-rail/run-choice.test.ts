import { chooseRun } from "./run-choice"

function same(actual: unknown, expected: unknown) {
  const left = JSON.stringify(actual)
  const right = JSON.stringify(expected)
  if (left !== right) throw new Error(`${left} !== ${right}`)
}

same(chooseRun({ strokes: true, tracksFull: true, manualTargets: true }), {
  path: "masks",
  maskPolicy: "propagate",
})
same(chooseRun({ strokes: false, tracksFull: true, useTracks: true }), { path: "tracks" })
same(chooseRun({ strokes: false, tracksFull: false, manualTargets: true }), { path: "targets" })
same(chooseRun({}), { path: "prompt" })
