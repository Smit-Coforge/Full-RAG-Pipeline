# Mini RAG Lab

A grounded employee expense-policy assistant built with Python, FastAPI,
PostgreSQL/pgvector, and Ollama.

## Docker development environment

Prerequisite: Docker Desktop must be running.

Set the path to your host-level Cursor skills before starting the services.

macOS or Linux:

```shell
export CURSOR_SKILLS_DIR="$HOME/.cursor/skills"
```

Windows PowerShell:

```powershell
$env:CURSOR_SKILLS_DIR = "$HOME\.cursor\skills"
```

Build and start the application, PostgreSQL/pgvector, and Ollama containers:

```shell
docker compose up --build --detach
```

In Cursor, select **Dev Containers: Attach to Running Container...**, choose
`mini-rag-lab-app-1`, and select `/usr/local/bin/python` as the Python
interpreter. The host skills are mounted read-only at `/root/.cursor/skills`.
Run **Developer: Reload Window** after attaching so Cursor discovers them.

Verify the package and skills mount from the container terminal:

```shell
python -c "import mini_rag_lab; mini_rag_lab.main()"
ls /root/.cursor/skills
```

Stop the services when finished:

```shell
docker compose down
```

The named PostgreSQL and Ollama volumes survive `docker compose down`. Avoid
`docker compose down --volumes` unless you intend to delete their data and
downloaded models.