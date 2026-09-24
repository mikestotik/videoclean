import type { MaskPolicy } from "./params"

export type RunPath = "masks" | "tracks" | "targets" | "prompt"

export type RunChoice = {
  path: RunPath
  maskPolicy?: MaskPolicy
}

export type RunChoiceInput = {
  strokes?: boolean
  tracksFull?: boolean
  useTracks?: boolean
  manualTargets?: boolean
  maskPolicy?: MaskPolicy
}

/** Strokes, then full-length tracks, then a manual target, else the user phrase. */
export function chooseRun(input: RunChoiceInput = {}): RunChoice {
  if (input.strokes) {
    return {
      path: "masks",
      maskPolicy: input.maskPolicy === "static" ? "static" : "propagate",
    }
  }
  if (input.tracksFull && input.useTracks) return { path: "tracks" }
  if (input.manualTargets) return { path: "targets" }
  return { path: "prompt" }
}

export function pendingRunStatus(choice: RunChoice): string {
  if (choice.path === "masks") {
    return choice.maskPolicy === "static" ? "Мазки · на весь ролик" : "Мазки · по движению"
  }
  if (choice.path === "tracks") return "По найденным рамкам"
  if (choice.path === "targets") return "Поиск по всем кадрам → заливка"
  return "Разбор фразы → поиск по всем кадрам → заливка"
}
