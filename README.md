# Mini RAG Lab

A grounded employee expense-policy assistant built with Python, FastAPI,
PostgreSQL/pgvector, and Ollama.

## Docker development image

Prerequisite: Docker Desktop must be running.

Build the image:

```bash
docker build -t mini-rag-lab .
```

Start a development container with the repository mounted:

```bash
docker run --name mini-rag-dev --rm -d \
  -v "$PWD:/workspace" \
  mini-rag-lab
```

In Cursor, select **Dev Containers: Attach to Running Container...**, choose
`mini-rag-dev`, and select `/usr/local/bin/python` as the Python interpreter.

Verify the image from the container terminal:

```bash
python -c "import mini_rag_lab; mini_rag_lab.main()"
```

Stop the container when finished:

```bash
docker stop mini-rag-dev
```

PostgreSQL/pgvector and Ollama will be connected when their application
components are implemented.