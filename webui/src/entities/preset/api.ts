import { api } from "@/shared/api/client"
import type { Preset } from "./types"

export const listPresets = () => api<Preset[]>("/api/presets")

export const savePreset = (name: string, payload: Record<string, unknown>, replace?: boolean) =>
  api<Preset>("/api/presets", {
    method: "POST",
    body: JSON.stringify(replace ? { name, payload, replace } : { name, payload }),
  })

export const updatePreset = (id: string, body: { name?: string; payload?: Record<string, unknown> }) =>
  api<Preset>(`/api/presets/${id}`, {
    method: "PUT",
    body: JSON.stringify(body),
  })

export const deletePreset = (id: string) =>
  api<{ ok: boolean }>(`/api/presets/${id}`, { method: "DELETE" })
