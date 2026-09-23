# ADR 0002: Cap a section at 512 tokens

- Date: 2026-09-23

## Decision

PDF and DOCX policies are still split on top-level numbered headings. A subsection such as `3.1` stays inside its parent section. A section whose estimated size is 512 tokens or less stays one chunk. A longer section is cut into windows of at most 512 tokens, and the next window starts 80 tokens before the previous cut. The cut moves back to the nearest sentence end when that window is still larger than the overlap.

Token count is estimated as the word count times 1.3. That factor converts English words to tokens. It is not the overlap. Eighty tokens is about 16% of the 512 cap, inside the usual 10–20% overlap range.

The first window keeps the id `{slug}:v{version}:section-{n}`. Later windows use the same id plus `-part-2`, `-part-3`, and so on. Every window keeps the original section number.

## Why

The sections in this corpus are short: about 20 to 193 words, with a median near 70. One chunk per section already holds a whole rule, and splitting those sections, or splitting at every subsection, would store many tiny pieces. 512 tokens is the usual default chunk size. The cap only matters when a future section is longer than that.

Overlap applies only after a split, so a rule that lands on the cut appears in both windows. A larger overlap would repeat most of a section that is only a little over the cap.

## What stays

- Ingest of the current corpus still stores 63 chunks, and none of those ids gain a `-part-` suffix.
- The Markdown parser for a single six-section policy file is unchanged.
- v1 and v2 remain separate because the version is still part of the chunk id.
