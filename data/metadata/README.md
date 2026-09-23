# Verified-source workflow

`source_registry.json` is the audit trail for every source the prototype may use.
It records why the source is trusted, what it may be used for, and whether its
content can be stored locally.

## Rules for this project

1. A source must be from an official BIS domain before its status can be
   `verified_official`.
2. The team must add a registry record **before** downloading or processing a
   new source.
3. A public webpage is not automatically permission to redistribute its text,
   PDFs, or full Indian Standards. Keep raw documents out of Git unless their
   reuse status has been explicitly checked.
4. Indian Standards search and LIMS are live lookup sources. We may link users
   to them, but do not treat a cached result as permanent or current.
5. Each answerable RAG chunk will later contain `source_id`, `source_url`,
   `title`, `retrieved_at`, and, for PDFs, `page_number`.
6. If no approved source supports an answer, the app must state that it has
   insufficient verified information.

## Initial source-set boundary

The first corpus will cover only these topics:

- Product certification basics and process guidance
- Public product-specific guidance for one or two selected demo products
- How to locate laboratory services and testing information
- Hallmarking basics
- Indian Standard title/number discovery through the official BIS portal

The initial corpus is English-first. Hindi support will be tested later with the
same evidence and will not be claimed as complete until it passes evaluation.

## Seed corpus used in Milestone 3

`curated_facts.json` contains short, team-authored paraphrases of the approved
public pages. It is deliberately limited to guidance that is useful for the
demo, and each entry retains its official source URL and review date. It does
not reproduce Indian Standard text, a product manual, or a raw webpage.

## Review checklist for a new source

- Is the domain owned by BIS or another named official authority?
- Is the page or document publicly available?
- Does it clearly support a planned user question?
- Is its update date recorded when visible?
- Is its storage or redistribution status known?
- Has a team member recorded the source in `source_registry.json`?
