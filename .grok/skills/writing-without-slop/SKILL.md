---
name: writing-without-slop
description: >
  Use when writing or editing user-facing prose: chat replies, README, VISION,
  DECISIONS, design docs, PR descriptions, comments that explain, or any markdown.
  Use when text has an AI accent, sand-in-the-eyes readability, unedited model
  draft, em-dash advertising cadence, not-X-but-Y, cliffhanger headings, filler
  (важно отметить, let's dive in), slogans, always-three lists, or vague
  "research shows". Use when the user runs /writing-without-slop.
---

# Writing without slop

Source: ToxaBes, «Невыносимая слопность бытия», Habr, https://habr.com/ru/articles/1063010/

Using a model is fine. Shipping the unedited draft is the failure. The reader tires of **recognizable form**, not of grammar. Frequency of a device is the tell; one instance is ordinary writing.

Applies to Russian and English. Same contract for chat and for files.

## Output contract

Every reply and every markdown page is this, in order:

1. The fact, decision, or action.
2. The supporting detail the reader needs (names, numbers, identifiers, limits).
3. What you did not verify, if anything.

Ordinary punctuation: period, comma, colon, parentheses. Headings name the object (`PolicyHook`, `needs_commit`, `compile()`). A paragraph is 2–5 sentences. A section ends on a fact, constraint, or example.

## Before / after

```
# before
Harnesys — это не просто runtime, а мощный инструмент, который меняет правила игры.

Давайте будем честны: переносимость играет ключевую роль.

Важно отметить: это не про код, а про данные. Хост сохраняет снимок — и только так.
```

```
# after
Harnesys компилирует JSON-определение агента в IR. Host сохраняет snapshot до `needs_commit`.

Определение переносится как данные. Runtime состояние не хранит.
```

## Lexical markers (delete or replace)

RU: честный разбор, давайте будем честны, без воды, без купюр, откровенно говоря, спойлер, давайте погрузимся, играет ключевую/решающую роль, меняет правила игры, важно отметить, стоит подчеркнуть, уникальный, инновационный, революционный, непревзойденный, мощный инструмент, несомненно, безусловно, гармонично, комплексный подход, системно, является [оценкой] вместо глагола действия.

EN: delve, underscore, boast, meticulous, commendable, showcase, intricate, tapestry, let's dive in, deep dive, game-changer, plays a crucial/vital role, it's important to note, landscape, testament to, leverage, robust, seamless.

A human may use any of these once. A draft that stacks them is still a draft.

## Ten structural patterns

Edit for **rate**, not a total ban.

| # | Pattern | What to write instead |
|---|---|---|
| 1 | Em dash (`—` / `--`) as the default comma, colon, or parenthesis; space before ` —` | Keep at most one real break of thought per several paragraphs. Elsewhere: comma, colon, parentheses, or two sentences. |
| 2 | Antithesis `не X, а Y` / `it's not X, it's Y` / `дело не в X` | State Y. At most one contrast per document, at the strongest point, never as the first sentence of a section. |
| 3 | One-line paragraph for fake drama (`А потом всё пошло не так.`) | One-sentence paragraphs only at a real turn. Otherwise join with the neighbor. |
| 4 | Cliffhanger headings (`Трещина в фундаменте`, `The hidden cost`) | Heading = noun the section is about. Mix lengths. Check headings as a list, apart from the body. |
| 5 | Filler (`по сути`, `грубо говоря`, `казалось бы`, `таким образом`, `в итоге`, `furthermore`) | Delete if the sentence still means the same. Keep only a real caveat or a real transition. |
| 6 | Aphoristic closer after every block | Close with the next fact, a limit, a question that the next section answers, or an identifier. |
| 7 | Forced symmetry and rule of three (`инновационный, трансформирующий, прорывной`) | List as many items as exist (2, 4, 5). Break parallel syntax once: one short item, one long. |
| 8 | Concreteness decay: names and numbers in the first third, slogans in the last | Reread the last third alone. Put the same density of ids, numbers, and examples there. |
| 9 | Vague authority (`исследования показывают`, `эксперты отмечают`, `стало поворотным моментом`) | Name the source, file, identifier, or measurement. If you cannot, cut the sentence. Empty authority is worse than silence. |
| 10 | Pseudo-sincerity (`без воды`, `честно говоря`, `you are a helpful assistant` tone) | Write the content. Do not advertise honesty, lack of ads, or a personal chat. |

Copy-paste formatting (bold on every term, italic on every definition) is cheap to fix: mark identifiers with backticks, leave the rest roman.

## Three passes on a document

Chat replies: one pass of the self-check below.

Markdown / docs / long answers:

1. **Rhythm.** Count `—`, `не X, а Y`, one-liners, matching heading shapes. Break the cadence.
2. **Concreteness.** Last third still has names, numbers, schema fields, commands.
3. **Glue and closers.** Cut filler. Replace half the slogans with the next fact.

## Self-check (before send or commit)

- First sentence is a claim or an action, not a negation and not a trust pitch.
- `—` count: extra dashes became commas, colons, or new sentences.
- Contrast constructions: 0 or 1 in the whole piece.
- Headings: each contains a concrete noun from the project.
- Filler words: deleting them does not change the argument.
- Last third: still specific.
- No "research shows" without a citation.

## Rationalizations

| Excuse | Reality |
|---|---|
| "The dash is correct punctuation." | Legal once. Advertising cadence at every clause is the accent. |
| "Contrast makes the point sharper." | After the second `не X, а Y` the reader tracks the template, not the point. |
| "Short paragraphs improve scanability." | In a spec there is no plot. Fake beats feel like tempo manipulation. |
| "A memorable closer helps." | In this repo the memorable unit is `needs_commit`, not a proverb. |
| "Three bullets look complete." | Completeness is the items that exist. Padding to three is the rule-of-three tell. |
| "I already sound human enough." | Fluency at sentence level plus template at section level is exactly the fatigue. |

## Out of scope

Protocol identifiers, error strings, JSON examples, and quotes stay verbatim. Do not "humanize" a field name.

Do not add a detector, a score, or a claim that the text was or was not written by a model. Edit the form. Leave the facts.
