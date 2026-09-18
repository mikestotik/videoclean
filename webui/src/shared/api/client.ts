/** API origin for split deploy. Empty = same-origin / vite proxy. */
export function apiBase(): string {
  const raw = String(import.meta.env.VITE_API_BASE_URL ?? "").trim()
  return raw.replace(/\/$/, "")
}

/** Join VITE_API_BASE_URL with a path or pass through absolute http(s) URLs. */
export function apiUrl(path: string): string {
  if (!path) return path
  if (/^https?:\/\//i.test(path) || path.startsWith("blob:") || path.startsWith("data:")) {
    return path
  }
  const base = apiBase()
  const p = path.startsWith("/") ? path : `/${path}`
  return base ? `${base}${p}` : p
}

export class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(apiUrl(path), {
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init?.headers,
    },
  })
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`
    try {
      const data = await res.json()
      if (data?.detail) message = String(data.detail)
    } catch {
      // keep default message
    }
    throw new ApiError(res.status, message)
  }
  return (await res.json()) as T
}
