import { StrictMode } from "react"
import { createRoot } from "react-dom/client"

import "./index.css"
import App from "@app/App"
import { ThemeProvider } from "@/shared/ui/theme-provider"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ThemeProvider>
      <App />
    </ThemeProvider>
  </StrictMode>
)
