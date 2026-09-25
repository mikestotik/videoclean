import { targetsFromParsedPrompt } from "./params"

function same(actual: unknown, expected: unknown) {
  const left = JSON.stringify(actual)
  const right = JSON.stringify(expected)
  if (left !== right) throw new Error(`${left} !== ${right}`)
}

const rows = targetsFromParsedPrompt("Remove: red logo (top-right), burned-in subtitle")
same(
  rows.map((row) => ({ query: row.query, where: row.where })),
  [
    { query: "red logo", where: "top-right" },
    { query: "burned-in subtitle", where: null },
  ],
)
same(targetsFromParsedPrompt("").length, 0)
same(targetsFromParsedPrompt("one line\ntwo").map((row) => row.query), ["one line", "two"])
