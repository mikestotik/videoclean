import path from "path"
import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

const r = (p: string) => path.resolve(__dirname, p)

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@app": r("./src/app"),
      "@pages": r("./src/pages"),
      "@widgets": r("./src/widgets"),
      "@features": r("./src/features"),
      "@entities": r("./src/entities"),
      "@shared": r("./src/shared"),
      "@": r("./src"),
    },
  },
  server: {
    proxy: {
      "/api": "http://127.0.0.1:7860",
      "/health": "http://127.0.0.1:7860",
    },
  },
  build: { outDir: "../server/static_dist", emptyOutDir: true },
})
