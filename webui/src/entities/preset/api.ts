import { api } from "@/shared/api/client"
import type { Preset } from "./types"

export const listPresets = () => api<Preset[]>("/api/presets")

export const savePreset = (name: string, payload: Record<string, unknown>) =>
  api<Preset>("/api/presets", {
    method: "POST",
    body: JSON.stringify({ name, payload }),
  })

export const deletePreset = (id: string) =>
  api<{ ok: boolean }>(`/api/presets/${id}`, { method: "DELETE" })
