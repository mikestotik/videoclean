import { useCallback, useEffect } from "react"
import { Clapperboard, Settings2 } from "lucide-react"
import { ConfigPage } from "@pages/config"
import { WorkspacePage } from "@pages/workspace"
import { EventsProvider } from "@/shared/events"
import {
  lastWorkspacePath,
  parsePath,
  useAppRoute,
  type AppRoute,
} from "@/shared/lib/route"
import { cn } from "@/shared/lib/utils"

const NAV = [
  { id: "workspace", label: "Редактор", icon: Clapperboard },
  { id: "settings", label: "Система", icon: Settings2 },
] as const

function navRoute(id: (typeof NAV)[number]["id"]): AppRoute {
  if (id === "settings") return { page: "settings" }
  return parsePath(lastWorkspacePath())
}

export function App() {
  const { route, navigate } = useAppRoute()

  useEffect(() => {
    const openConfig = () => navigate({ page: "settings" })
    window.addEventListener("videoclean:open-config", openConfig)
    return () => window.removeEventListener("videoclean:open-config", openConfig)
  }, [navigate])

  const onRouteSourceIdChange = useCallback(
    (id: string | null, replace?: boolean) => {
      navigate({ page: "workspace", sourceId: id }, Boolean(replace))
    },
    [navigate],
  )

  const activeNav = route.page === "settings" ? "settings" : "workspace"

  return (
    <EventsProvider>
    <div className="flex h-svh flex-col overflow-hidden bg-background">
      <header className="flex h-11 shrink-0 items-center gap-3 border-b border-border/80 px-3">
        <div className="flex items-center gap-2">
          <span
            className="flex size-6 items-center justify-center rounded-md bg-primary/15 text-primary"
            aria-hidden
          >
            <Clapperboard className="size-3.5" />
          </span>
          <span className="text-sm font-semibold tracking-tight">VideoClean</span>
        </div>

        <nav className="flex items-center gap-0.5 rounded-md bg-muted/60 p-0.5" aria-label="Разделы">
          {NAV.map(({ id, label, icon: Icon }) => {
            const active = activeNav === id
            return (
              <button
                key={id}
                type="button"
                onClick={() => navigate(navRoute(id))}
                className={cn(
                  "inline-flex h-7 items-center gap-1.5 rounded px-2.5 text-xs font-medium transition-colors",
                  active
                    ? "bg-card text-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground",
                )}
                aria-current={active ? "page" : undefined}
              >
                <Icon className="size-3.5 opacity-70" />
                {label}
              </button>
            )
          })}
        </nav>
      </header>
      <main className="min-h-0 flex-1">
        {route.page === "workspace" ? (
          <WorkspacePage
            routeSourceId={route.sourceId}
            onRouteSourceIdChange={onRouteSourceIdChange}
          />
        ) : (
          <ConfigPage />
        )}
      </main>
    </div>
    </EventsProvider>
  )
}

export default App
