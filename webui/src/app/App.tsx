import { useState } from "react"
import { ConfigPage } from "@pages/config"
import { WorkspacePage } from "@pages/workspace"
import { Tabs, TabsList, TabsTrigger } from "@/shared/ui/tabs"

export function App() {
  const [tab, setTab] = useState("workspace")
  return (
    <div className="flex h-svh flex-col overflow-hidden">
      <header className="flex items-center gap-4 border-b border-border px-4 py-2">
        <h1 className="text-sm font-medium">videoclean</h1>
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList>
            <TabsTrigger value="workspace">Редактор</TabsTrigger>
            <TabsTrigger value="config">Конфиг</TabsTrigger>
          </TabsList>
        </Tabs>
      </header>
      <main className="min-h-0 flex-1">{tab === "workspace" ? <WorkspacePage /> : <ConfigPage />}</main>
    </div>
  )
}

export default App
