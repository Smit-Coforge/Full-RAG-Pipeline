# Local Ollama Model Comparison

## Purpose

This comparison selects the local Ollama models used by Mini RAG Lab after
moving model execution from Docker to native macOS Ollama.

The embedding and generation roles are evaluated separately because they have
different API capabilities and cannot be interchanged safely.

## Test environment

- Ollama ran natively on the Mac and was reached from the app container through
  `http://host.docker.internal:11434`.
- PostgreSQL/pgvector remained in Docker.
- `nomic-embed-text` produced the stored chunk vectors and question vectors.
- Each generation variant received the same nearest retrieved policy chunk and
  the same grounded prompt, JSON schema, temperature `0`, and seed `42`.
- Application guardrails were applied to the first answer of each question.
- Each variant was warmed up once, then every question was timed five times
  with `keep_alive=10m`. Reported latency is the Ollama chat call, not the
  surrounding retrieval code.
- The repeated-run benchmark was collected on September 20, 2026.

## Installed model metadata

| Model | Role capability | Parameters | Quantization | Disk size | Context | Vector width |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| `nomic-embed-text` | Embedding | 137M | F16 | 0.26 GiB | 2,048 | 768 |
| `mistral:7b` | Completion, tools | 7.2B | Q4_K_M | 4.07 GiB | 32,768 | Not an embedding API |
| `qwen3:8b` | Completion, tools, thinking | 8.2B | Q4_K_M | 4.87 GiB | 40,960 | Not an embedding API |

The `embedding_length` reported in generation-model metadata is an internal
model dimension. Neither Mistral nor Qwen advertises Ollama's `embedding`
capability, so neither replaces `nomic-embed-text`.

## Repeated generation results

30 samples per variant (6 questions × 5 repeats after warmup):

| Generation variant | Mean | Median | p95 | Max | Mean eval tokens | Tokens / s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mistral 7B | 1.980 s | 1.949 s | 2.261 s | 2.443 s | 52.5 | 27.5 |
| Qwen 3 8B, thinking off | 1.756 s | 1.821 s | 2.164 s | 2.305 s | 40.7 | 24.2 |
| Qwen 3 8B, thinking on | 13.080 s | 9.115 s | 29.945 s | 31.235 s | 297.3 | 23.0 |

Decode speed is similar across variants. Thinking-on is slower because it
emits hidden reasoning tokens before the JSON answer.




## Interpretation

### Mistral 7B

Mistral is slightly faster per token than Qwen and uses about 0.8 GiB less
disk. After a warm load it is consistent: p95 2.26 s, max 2.44 s. It generates
a little more answer text than Qwen thinking-off (53 vs 41 tokens).

### Qwen 3 8B with thinking off

This variant had the lowest mean, median, p95, and maximum latency. It also
emitted the fewest tokens and still supports thinking if a future, more
complex policy needs it.

### Qwen 3 8B with thinking on

Thinking mode produced no accuracy improvement on this six-section policy. It
generated about seven times as many tokens as thinking-off, including 696
tokens for the airfare question, so wall time scaled with token count rather
than with a slower decoder.

## Verdict

Use:

```text
EMBEDDING_MODEL=nomic-embed-text
EMBEDDING_DIMENSIONS=768
GENERATION_MODEL=qwen3:8b
GENERATION_THINKING=false
```

`nomic-embed-text` is the only tested model with the required embedding
capability. Qwen 3 8B with thinking disabled is the preferred generation model
because all candidates achieved the same final guarded correctness, while this
variant had the best measured latency and retained optional thinking support.

Mistral 7B remains a valid lower-disk fallback.

## Limitations

- This is a small acceptance benchmark, not a general model-quality benchmark.
- Latency depends on host hardware, current model residency, and other local
  workload.
- The timed result is the Ollama chat call with a fixed retrieved chunk, not
  the full CLI path.
- A larger or more ambiguous policy may change the model ranking.
