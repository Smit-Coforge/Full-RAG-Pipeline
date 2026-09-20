# Mini RAG Lab

A grounded employee expense-policy assistant built with Python,
PostgreSQL/pgvector, and Ollama. The application is CLI-only: it embeds
`policy.md`, retrieves the nearest sections, and answers from that evidence.

## Package layout

```text
src/mini_rag_lab/
├── cli.py            # migrate, ingest, ask, evaluate
├── config.py         # environment settings
├── runtime.py        # adapter and service wiring
├── migrations.py     # SQL migration runner
├── domain/           # models, policy parsing, ports
├── services/         # ingestion, retrieval, generation, query, evaluation
└── adapters/         # Ollama and PostgreSQL/pgvector
```

## Docker development environment

Prerequisites:

- Docker Desktop must be running.
- Ollama must be running natively on macOS.
- Install the local models from a Mac terminal:

```shell
ollama pull nomic-embed-text
ollama pull qwen3:8b
```

Set the path to your host-level Cursor skills before starting the services.

macOS or Linux:

```shell
export CURSOR_SKILLS_DIR="$HOME/.cursor/skills"
```

Windows PowerShell:

```powershell
$env:CURSOR_SKILLS_DIR = "$HOME\.cursor\skills"
```

Build and start the application and PostgreSQL/pgvector containers:

```shell
docker compose up --build --detach
```

In Cursor, select **Dev Containers: Attach to Running Container...**, choose
`mini-rag-lab-app-1`, and select `/usr/local/bin/python` as the Python
interpreter. The host skills are mounted read-only at `/root/.cursor/skills`.
Run **Developer: Reload Window** after attaching so Cursor discovers them.

Verify the package and skills mount from the container terminal:

```shell
python -m mini_rag_lab --help
curl http://host.docker.internal:11434/api/tags
ls /root/.cursor/skills
```

## Commands

Run these from the app container after the database is healthy:

```shell
python -m mini_rag_lab migrate
python -m mini_rag_lab ingest
python -m mini_rag_lab ask "How much can I spend on food each day?"
python -m mini_rag_lab evaluate
```

The selected local models and the latency comparison are documented in
`docs/model-comparison.md`.

Stop the services when finished:

```shell
docker compose down
```

The named PostgreSQL volume survives `docker compose down`. Avoid
`docker compose down --volumes` unless you intend to delete the database.
