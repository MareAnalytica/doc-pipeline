# Doc Pipeline -- RAGAnything Knowledge Platform (Epic 8)

Shared, reusable GitHub Actions workflow that automatically detects the
languages in a repository, generates documentation with the appropriate
toolchain, and feeds the output into the RAGAnything knowledge graph and
Wiki.js.

## How It Works

1. **Language detection** -- the workflow checks for marker files
   (`go.mod`, `tsconfig.json`, `pyproject.toml`, `Chart.yaml`, `*.proto`)
   and sets boolean flags.
2. **Conditional tool installation** -- only the doc generators needed
   for the detected languages are installed.
3. **Doc generation** -- each generator writes Markdown (or JSON) into a
   `docs/<lang>/` directory inside the runner workspace.
4. **RAGAnything ingestion** (Epic 9, placeholder) -- generated docs are
   transformed and POSTed to the RAGAnything API.
5. **Wiki.js publishing** (Epic 10, placeholder) -- human-readable pages
   are pushed to Wiki.js via its GraphQL API.

## Adding the Pipeline to a Repo

Copy the caller workflow into your repository:

```
.github/workflows/docs.yml
```

With these contents:

```yaml
name: Documentation Pipeline

on:
  push:
    branches: [main]

jobs:
  docs:
    uses: MareAnalytica/doc-pipeline/.github/workflows/doc-pipeline.yml@main
    secrets:
      OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      WIKIJS_API_KEY: ${{ secrets.WIKIJS_API_KEY }}
```

That is all that is required. The reusable workflow handles everything
else.

## Detected Languages and Tools

| Marker file(s)                                    | Flag         | Tool installed                  |
| ------------------------------------------------- | ------------ | ------------------------------- |
| `go.mod`                                          | `has_go`     | `gomarkdoc`                     |
| `tsconfig.json`                                   | `has_ts`     | `typedoc`, `typedoc-plugin-markdown` |
| `pyproject.toml`, `setup.py`, or `requirements.txt` | `has_python` | `pydoc-markdown`                |
| Any `*.proto` within 3 levels                     | `has_proto`  | `protoc-gen-doc`                |
| `Chart.yaml`                                      | `has_helm`   | `helm-docs`                     |

Tools are only installed and run when their corresponding flag is
`true`, keeping CI time and resource usage minimal.

## Required Secrets

| Secret           | Purpose                                         |
| ---------------- | ----------------------------------------------- |
| `OPENAI_API_KEY` | Used by the RAGAnything transform/ingest step    |
| `WIKIJS_API_KEY`  | Used to publish pages to Wiki.js via GraphQL     |

Both must be set as repository or organization secrets in GitHub.

## Runner Requirements

The workflow runs on **self-hosted** runners provisioned by ARC
(Actions Runner Controller) in the Kubernetes cluster. Ensure the
runner image has the following available:

- Go toolchain (for `go install`)
- Node.js / npm
- Python 3 / pip
- `protoc` (Protocol Buffers compiler) if repos use `.proto` files
- Network access to `raganything.memory.svc.cluster.local:9621` and
  the Wiki.js GraphQL endpoint

## Local Testing

You can run language detection locally to verify what the pipeline
would detect for a given repo:

```bash
cd /path/to/your/repo
bash /path/to/doc-pipeline/scripts/detect-languages.sh
```

## Roadmap

- **Epic 9** -- Transform and ingest into RAGAnything (placeholder step
  in workflow).
- **Epic 10** -- Generate and publish Wiki.js pages (placeholder step
  in workflow).
