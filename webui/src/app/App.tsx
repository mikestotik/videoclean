import { useState } from "react"
import { ConfigPage } from "@pages/config"
import { WorkspacePage } from "@pages/workspace"
import { Tabs, TabsList, TabsTrigger } from "@/shared/ui/tabs"

export function App() {
  const [tab, setTab] = useState("workspace")
  return (
    <div className="mx-auto flex min-h-svh max-w-6xl flex-col gap-4 p-4">
      <header className="flex items-center gap-4">
        <h1 className="text-lg font-medium">videoclean</h1>
        <Tabs value={tab} onValueChange={setTab}>
          <TabsList>
            <TabsTrigger value="workspace">Рабочая</TabsTrigger>
            <TabsTrigger value="config">Конфиг</TabsTrigger>
          </TabsList>
        </Tabs>
      </header>
      <main className="flex-1">{tab === "workspace" ? <WorkspacePage /> : <ConfigPage />}</main>
    </div>
  )
}

export default App
