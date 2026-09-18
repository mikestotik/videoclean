import { useEffect, useState } from "react"
import { Clapperboard, Settings2 } from "lucide-react"
import { ConfigPage } from "@pages/config"
import { WorkspacePage } from "@pages/workspace"
import { cn } from "@/shared/lib/utils"

const NAV = [
  { id: "workspace", label: "Редактор", icon: Clapperboard },
  { id: "config", label: "Система", icon: Settings2 },
] as const

export function App() {
  const [tab, setTab] = useState<(typeof NAV)[number]["id"]>("workspace")
  useEffect(() => {
    const openConfig = () => setTab("config")
    window.addEventListener("videoclean:open-config", openConfig)
    return () => window.removeEventListener("videoclean:open-config", openConfig)
  }, [])

  return (
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
            const active = tab === id
            return (
              <button
                key={id}
                type="button"
                onClick={() => setTab(id)}
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
      <main className="min-h-0 flex-1">{tab === "workspace" ? <WorkspacePage /> : <ConfigPage />}</main>
    </div>
  )
}

export default App
