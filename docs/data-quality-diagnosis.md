# Part 6: Planted data-quality diagnosis

## What was planted (Part 1)

The Doofenshmirtz `corpus/` keeps **both** versions of some policies as live
ingest. For nuclear shelter, the rules disagree on purpose:

| Version | File | Shelter rule |
| --- | --- | --- |
| Preparedness Policy **v1.0** | `… Preparedness Policy v1.0.pdf` | Take cover under desks / conference table |
| Preparedness Policy **v2.0** | `… Preparedness Policy v2.0 ….docx` | Climb into the break-room industrial refrigerator |

A second planted conflict (tokens: **1,000,000** in Time & Usage v1.0 vs
**500,000** in v2.0) works the same way. This write-up uses the nuclear
question.

## Question that surfaces it

```shell
python -m mini_rag_lab ask "Where should employees shelter during a nuclear apocalypse?"
```

## Observed response (25 September 2026)

```json
{
  "answer": "Employees should get into the break room and climb into the industrial refrigerator. This is the current shelter location recommended during a nuclear apocalypse. Under Version 1.0, employees were instructed to take cover under their desks.",
  "citation": {
    "document": "Preparedness Policy",
    "version": "2.0",
    "section": "4. Nuclear Apocalypse Protocol — Updated"
  },
  "retrieved_chunks": [
    {
      "document": "Preparedness Policy",
      "version": "2.0",
      "section": "4. Nuclear Apocalypse Protocol — Updated",
      "distance": 0.18560266494750977,
      "rerank_score": 6.514695167541504
    },
    {
      "document": "Preparedness Policy",
      "version": "1.0",
      "section": "4. Nuclear Apocalypse Protocol",
      "distance": 0.17275933352000916,
      "rerank_score": 6.294947624206543
    },
    {
      "document": "Preparedness Policy",
      "version": "2.0",
      "section": "1. Purpose",
      "distance": 0.27869704903523074,
      "rerank_score": -0.4965466260910034
    }
  ],
  "retrieval_strategy": "hybrid"
}
```

## Two-question debug

### 1. Was it retrieval?

**No — retrieval behaved correctly.** Hybrid search + MiniLM rerank returned
the nuclear §4 chunks for **both** versions in the top three (v2 first, v1
second). The citation points at the expected current section. Nothing is
“missing” from the index for this question; the pipeline found the relevant
policy text.

`select_generation_context` deliberately keeps a version-conflict pair in
context when both versions are in the candidate pool, so generation can see
the disagreement. Soft preference + the system prompt then favor the higher
version for the citation.

### 2. Was it the source data?

**Yes.** The failure mode is **lineage / corpus hygiene**: v1 and v2 both sit
in `corpus/` and both were ingested into `policy_chunks` as authoritative.
The source texts conflict:

- **v1.0 §4:** shelter under desks / conference table (only sanctioned position
  in that version).
- **v2.0 §4:** shelter in the industrial refrigerator (updated protocol).

The model’s answer mirrors that conflict (states fridge as current, mentions
v1 desks). That is faithful generation from bad dual sources, not a bug in
RRF, keyword `ILIKE`, embeddings, or the generator refusing to follow
context.

## Conclusion

| Layer | Verdict |
| --- | --- |
| Retrieval | OK — both conflicting sections retrieved and ranked highly |
| Generation | OK — answered from retrieved context; cited v2.0 |
| Source data | **Root cause** — stale v1.0 left live next to v2.0 |

## What a real fix would be (not applied here)

Keep the planted defect for the lab demo. In production you would:

1. Archive or delete superseded policy files from the ingest corpus, **or**
2. Ingest only the current version (and optionally store history offline),
   **or**
3. Mark older rows non-retrievable and re-ingest.

Changing hybrid thresholds or prompts would paper over the symptom; it would
not fix the dual live sources.
