# Doc Pipeline -- RAGAnything Knowledge Platform

Shared, reusable GitHub Actions workflow that automatically detects the
languages in a repository, generates deterministic documentation from
the code AST, ingests it into the RAGAnything knowledge graph (Neo4j +
Qdrant), and publishes human-readable pages to Wiki.js.

## How It Works

1. **Language detection** -- checks for marker files (`go.mod`,
   `tsconfig.json`, `pyproject.toml`, `Chart.yaml`, `*.proto`) at root
   and in common subdirectories (`backend/`, `frontend/`, `src/`, `app/`,
   `api/`). Falls back to scanning for `.py` files if no marker found.
2. **Conditional tool installation** -- only installs doc generators
   for the detected languages.
3. **Doc generation** -- each generator writes output into `docs/<lang>/`.
   All tools are deterministic AST/compiler parsers (no LLM involved).
4. **Transform + RAGAnything ingestion** -- converts doc output into
   entity-rich text and POSTs to the RAGAnything API for knowledge graph
   construction.
5. **Wiki.js publishing** -- for changed packages, queries RAGAnything
   for graph context, calls gpt-5.4-mini to generate human-readable
   prose grounded in AST docs + graph context + source code, and
   publishes to Wiki.js via GraphQL.
6. **Feedback loop** -- ingests the published Wiki.js pages back into
   RAGAnything as derived documentation.

## Adding the Pipeline to a Repo

### Step 1: Set repo-level secrets

On the **free GitHub org plan**, org-level secrets do NOT pass to
reusable workflows. You must set secrets on each repo individually:

```bash
# Get the keys
OPENAI_KEY=$(kubectl get secret openai-api-key -n memory -o jsonpath='{.data.api-key}' | base64 -d)
WIKIJS_KEY=$(kubectl get secret wikijs-api-keys -n docs -o jsonpath='{.data.WIKIJS_API_KEY}' | base64 -d)

# Set on the repo
echo "$OPENAI_KEY" | gh secret set OPENAI_API_KEY --repo MareAnalytica/<repo-name>
echo "$WIKIJS_KEY" | gh secret set WIKIJS_API_KEY --repo MareAnalytica/<repo-name>
```

### Step 2: Add the caller workflow

Create `.github/workflows/docs.yml` in your repo:

```yaml
name: Documentation Pipeline

on:
  push:
    branches: [main]
  workflow_dispatch:
    inputs:
      full_generation:
        type: boolean
        default: false
        description: "Generate all pages (not just changed packages)"

jobs:
  docs:
    uses: MareAnalytica/doc-pipeline/.github/workflows/doc-pipeline.yml@main
    with:
      full_generation: ${{ inputs.full_generation || false }}
    secrets: inherit
```

### Step 3: Push to main

The pipeline triggers automatically on every push to main. On the first
push, it generates docs for the changed packages and ingests them.

## Triggering Full Site Generation

To generate Wiki.js pages for ALL packages (not just those changed in
the latest commit), include `[full-docs]` in your commit message:

```bash
git commit --allow-empty -m "[full-docs] Seed all documentation pages"
git push
```

**Why a commit message tag instead of workflow_dispatch?**

On the GitHub free org plan, `secrets: inherit` does not pass secrets
through for `workflow_dispatch` events -- only for `push` events. The
`[full-docs]` commit message tag triggers full generation via a normal
push event where secrets work correctly. This is a known GitHub
limitation on free plans.

The `workflow_dispatch` with `full_generation: true` input exists in
the workflow definition but will only work if you upgrade to a GitHub
Team or Enterprise plan, or if you set secrets directly on the
doc-pipeline repo and reference them explicitly.

## Detected Languages and Tools

| Marker file(s) | Subdirs checked | Tool | Output |
|---|---|---|---|
| `go.mod` | `.`, `backend/`, `api/`, `src/` | gomarkdoc | Markdown |
| `tsconfig.json` | `.`, `frontend/`, `src/` | TypeDoc + typedoc-plugin-markdown | JSON + Markdown |
| `pyproject.toml`, `setup.py`, `requirements.txt` | `.`, `backend/`, `api/`, `app/`, `src/` | pydoc-markdown | Markdown |
| `*.py` files (fallback) | scans up to 3 levels deep | pydoc-markdown | Markdown |
| `*.proto` | scans up to 3 levels deep | protoc-gen-doc | JSON |
| `Chart.yaml` | `.` only | helm-docs | Markdown |

Tools are only installed when their language is detected.

## Required Secrets

| Secret | Purpose |
|---|---|
| `OPENAI_API_KEY` | Entity extraction (gpt-5.4-mini) + Wiki.js prose generation |
| `WIKIJS_API_KEY` | Publishing pages to Wiki.js via GraphQL |

**Must be set as repo-level secrets** on each repo (free org plan
limitation). See Step 1 above.

## Runner Requirements

Runs on `arc-runner-set` (ARC self-hosted runners in the K8s cluster).
The workflow uses `actions/setup-go@v5`, `actions/setup-node@v4`, and
`actions/setup-python@v5` to install runtimes dynamically. No custom
runner image needed.

The runner must have network access to:
- `raganything.memory.svc.cluster.local:9621` (RAGAnything API)
- `wikijs.docs.svc.cluster.local:3000` (Wiki.js GraphQL)
- `https://api.openai.com` (OpenAI API)

## Scripts

| Script | Purpose |
|---|---|
| `scripts/transform.py` | Converts doc tool output into RAGAnything-friendly entity-rich text. 5 language transformers with namespace-qualified entity names. |
| `scripts/generate_wiki_pages.py` | Generates Wiki.js pages from AST docs + RAGAnything graph context + source code via gpt-5.4-mini. Supports `--full` mode for seeding all packages. |
| `scripts/ingest_wikijs_to_rag.py` | Feeds published Wiki.js pages back into RAGAnything as derived documentation. |
| `scripts/generate_llms_txt.py` | Generates `/llms.txt` and `/llms-full.txt` for AI tool consumption. |
| `scripts/detect-languages.sh` | Standalone language detection for local testing. |

## Local Testing

```bash
# Test language detection
cd /path/to/your/repo
bash /path/to/doc-pipeline/scripts/detect-languages.sh

# Test transform
mkdir -p docs/go
gomarkdoc ./internal/... > docs/go/packages.md 2>/dev/null
python3 /path/to/doc-pipeline/scripts/transform.py docs/ --repo my-repo --commit test
```

## Widget

The `widget/` directory contains a chat widget for docs.mareanalytica.com
that queries RAGAnything's streaming API. See `widget/README.md`.
