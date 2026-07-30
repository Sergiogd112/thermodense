# Domain docs

Thermodense is a single-context repository.

## Before exploring

- Read the root `CONTEXT.md`.
- Read relevant architectural decision records under root `docs/adr/` when that directory exists.
- If an optional document or directory does not exist, proceed silently. `/grill-with-docs` creates domain documentation lazily when a term or durable architectural decision is resolved.

## Use canonical vocabulary

Use the scientific and data-product terminology defined in `CONTEXT.md` in issues, plans, code, tests, and prose. Do not drift to synonyms that the glossary explicitly marks as avoided.

If a required concept is absent, first reconsider whether existing vocabulary covers it. If it is a real domain-language gap, resolve it with `/grill-with-docs` rather than silently inventing a competing term.

## Respect architectural decisions

If proposed work conflicts with an existing ADR, identify that conflict explicitly and explain why the decision may need to be reopened rather than silently overriding it.
