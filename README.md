# mini-rag

Grounded expense-policy assistant. It chunks `policy.md`, embeds sections, stores them, retrieves by cosine distance, and generates cited answers. 

Default stack: Ollama (`nomic-embed-text` + `mistral:7b`) on your Mac, Postgres in Docker.

## Prerequisites
- Python 3.12+
- Docker
- [Ollama](https://ollama.com) on your Mac

Pull the models used by the defaults:
```bash
ollama pull nomic-embed-text
ollama pull mistral:7b
```

## 1. Start Ollama (Mac)
Open the Ollama app, or run
```bash
ollama serve
```

Verify:
```bash
curl -s http://localhost:11434/api/tags
```

## 2. Start Postgres + pgvector (Docker)
**First time** (creates the container and applies `sql/init.sql`):
```bash
docker run --name mini-rag-db \
  -e POSTGRES_USER=YOUR_USER \
  -e POSTGRES_PASSWORD=YOUR_PASSWORD \
  -e POSTGRES_DB=mini_rag \
  -p 5432:5432 \
  -v "$(pwd)/sql/init.sql:/docker-entrypoint-initdb.d/init.sql" \
  -d pgvector/pgvector:pg16
```

**Later sessions:**
```bash
docker start mini-rag-db
```

Verify the table exists:
```bash
docker exec -it mini-rag-db psql -U YOUR_USER -d mini_rag -c '\dt'
```

## 3. Install the Python app (Mac)
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment variables
| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_PROVIDER` | `ollama` | Which embedding adapter to build |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model id |
| `EMBEDDING_DIMENSION` | `768` | Expected embedding size |
| `GENERATION_PROVIDER` | `ollama` | Which generation adapter to build |
| `GENERATION_MODEL` | `mistral:7b` | Generation model id |
| `STORE_PROVIDER` | `pgvector` | Which vector store adapter to build |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL (use `host.docker.internal` from the app container) |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/mini_rag` | Postgres DSN for pgvector (use `host.docker.internal` from the app container) |
| `POLICY_PATH` | `policy.md` | Path to the policy markdown file |
| `RETRIEVE_K` | `3` | Number of nearest chunks to return |

Provider values are validated as enums in `config.py` (`EmbeddingProvider`, `GenerationProvider`, `StoreProvider`). Allowed values today: `ollama` (embed/generate), `pgvector` (store). Anything else fails at settings load.

## 4. Run on your Mac
From the repo root (venv active):
```bash
# Rebuild the index
python src/ingest.py

# Smoke-test retrieval
python src/retrieve.py

# Full evaluation (six questions) using one retrieved chunk
RETRIEVE_K=1 DATABASE_URL=postgresql://YOUR_USER:YOUR_PASSWORD@localhost:5432/mini_rag OLLAMA_HOST=http://localhost:11434 python src/run.py

# Or export them for the rest of the shell session:
export RETRIEVE_K=1
export DATABASE_URL=postgresql://YOUR_USER:YOUR_PASSWORD@localhost:5432/mini_rag
export OLLAMA_HOST=http://localhost:11434
python src/run.py
```

## 5. Run the app in Docker
Ollama stays on the Mac; Postgres stays in `mini-rag-db`. Override hosts so the container can reach them:
```bash
docker build -t mini-rag .
docker run --rm \
  -e DATABASE_URL=postgresql://YOUR_USER:YOUR_PASSWORD@host.docker.internal:5432/mini_rag \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  -e RETRIEVE_K=1 \
  mini-rag
```

That runs `src/run.py` (see `Dockerfile` `CMD`).

## Verify it works
### Ingest
```bash
python src/ingest.py
docker exec -it mini-rag-db psql -U YOUR_USER -d mini_rag \
  -c 'SELECT chunk_id, section_title FROM chunks ORDER BY section;'
```
Expect **6** rows (Meals → Submission Deadline).

### Evaluation
```bash
RETRIEVE_K=1 python src/run.py
```
| Question | Expected citation | Expected answer idea |
|---|---|---|
| How much can I spend on food each day? | `1. Meals` | up to $65 per day |
| Can I book first-class airfare? | `3. Airfare` | economy required; business needs VP approval |
| My hotel costs $250. What do I need? | `2. Hotels` | manager approval before booking |
| Do I need a receipt for a $20 taxi? | `5. Receipts` | no receipt under $25 |
| Can I claim a limousine upgrade? | `4. Ground Transportation` | luxury upgrades not reimbursable |
| Does the company reimburse gym memberships? | **no citation** (`null`) | policy does not answer |

Also confirm:
- `retrieved_chunks` has at most `RETRIEVE_K` items
- `distance` values are numbers and increase down the list
- supported answers include `citation` with document, version, and section

## Project layout
```text
src/
  parse_policy.py      # split policy.md into sections
  ingest.py            # embed + save chunks
  retrieve.py          # query embed + store.search
  generate.py          # grounded answer -> RagResponse
  run.py               # six-question evaluation CLI
  config.py            # env settings
  adapters.py          # build embedder / generator / store
  embeddings/          # embedding adapters
  generation/          # generation adapters
  store/               # vector store adapters (pgvector)
  response.py          # structured output models
sql/init.sql           # Postgres schema (DB container only)
Dockerfile             # app image -> python src/run.py
```

## Troubleshooting
| Symptom | Fix |
|---|---|
| Connection refused to Postgres | `docker start mini-rag-db` |
| Connection refused to Ollama | Start the Ollama app / `ollama serve` |
| `vector <=> double precision[]` | Search must cast with `%s::vector` |
| Empty retrieval | Re-run `python src/ingest.py` |
| Docker app can't reach DB/Ollama | Use `host.docker.internal` in `DATABASE_URL` and `OLLAMA_HOST` |


## Design choices

- **Ollama embeddings (`nomic-embed-text`)** — local, no API key, 768-d vectors. Uses the model's `search_query` / `search_document` prefixes so query and document embeddings match.
- **Ollama generation (`mistral:7b`)** — same local runtime as embeddings; small enough to run on a laptop while still following the grounded cite-or-refuse prompt.
- **pgvector** — Postgres stays the system of record; cosine distance (`<=>`) gives nearest-neighbor search without a separate vector DB. Schema lives in `sql/init.sql` and is applied when the DB container is first created.
