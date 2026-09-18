import { useCallback, useEffect, useState } from "react"

export type AppRoute =
  | { page: "workspace"; sourceId: string | null }
  | { page: "settings" }

const LAST_WORKSPACE_KEY = "videoclean:lastWorkspace"

export function parsePath(pathname: string): AppRoute {
  const path = pathname.replace(/\/+$/, "") || "/"
  if (path === "/settings" || path === "/config") return { page: "settings" }
  const match = path.match(/^\/v\/([^/]+)$/)
  if (match) {
    try {
      return { page: "workspace", sourceId: decodeURIComponent(match[1]) }
    } catch {
      return { page: "workspace", sourceId: match[1] }
    }
  }
  return { page: "workspace", sourceId: null }
}

export function toPath(route: AppRoute): string {
  if (route.page === "settings") return "/settings"
  if (route.sourceId) return `/v/${encodeURIComponent(route.sourceId)}`
  return "/"
}

export function rememberWorkspacePath(path: string) {
  try {
    sessionStorage.setItem(LAST_WORKSPACE_KEY, path)
  } catch {
    /* ignore quota / private mode */
  }
}

export function lastWorkspacePath(): string {
  try {
    return sessionStorage.getItem(LAST_WORKSPACE_KEY) || "/"
  } catch {
    return "/"
  }
}

export function useAppRoute() {
  const [route, setRoute] = useState<AppRoute>(() => parsePath(window.location.pathname))

  useEffect(() => {
    const onPop = () => setRoute(parsePath(window.location.pathname))
    window.addEventListener("popstate", onPop)
    return () => window.removeEventListener("popstate", onPop)
  }, [])

  useEffect(() => {
    if (route.page === "workspace") rememberWorkspacePath(toPath(route))
  }, [route])

  const navigate = useCallback((next: AppRoute, replace = false) => {
    const path = toPath(next)
    const current = window.location.pathname.replace(/\/+$/, "") || "/"
    if (path !== current) {
      if (replace) window.history.replaceState(null, "", path)
      else window.history.pushState(null, "", path)
    }
    setRoute(next)
  }, [])

  return { route, navigate }
}
