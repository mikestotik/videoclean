/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Browser-reachable API origin for split deploy. Empty = same-origin / vite proxy. */
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
