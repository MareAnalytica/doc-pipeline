#!/usr/bin/env python3
"""Generate human-readable Wiki.js pages from deterministic docs, source code,
and RAGAnything graph context.

This is Epic 10 of the doc pipeline. It runs as the final CI step after
deterministic doc generation (gomarkdoc, TypeDoc, pydoc-markdown, etc.) and
RAGAnything ingestion. For each package/module that changed in the current push,
it:

1. Reads the raw AST docs (from the docs/ directory)
2. Reads the actual source code (from the git checkout)
3. Queries RAGAnything for cross-repo graph context
4. Calls OpenAI gpt-5.4-mini to generate grounded prose
5. Publishes to Wiki.js via GraphQL under /{repo}/{package}

Usage:
    python scripts/generate_wiki_pages.py \\
      --repo justpay-backend \\
      --docs-dir docs/ \\
      --source-dir . \\
      --raganything-url http://raganything.memory.svc.cluster.local:9621 \\
      --wikijs-url http://wikijs.docs.svc.cluster.local:3000 \\
      --wikijs-api-key "$WIKIJS_API_KEY" \\
      --openai-api-key "$OPENAI_API_KEY"
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Git diff: detect changed packages
# ---------------------------------------------------------------------------

def get_changed_files() -> list[str]:
    """Get files changed in the current commit via git diff HEAD~1..HEAD.

    Falls back to listing all tracked files if HEAD~1 does not exist (e.g.
    initial commit).
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "HEAD~1", "HEAD"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Initial commit or shallow clone -- list all tracked files
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
        )
    output = result.stdout.strip()
    if not output:
        return []
    return output.split("\n")


def map_files_to_packages(changed_files: list[str]) -> list[str]:
    """Map changed file paths to their parent package/module directories.

    Handles Go, TypeScript, and Python path conventions:
      - Go:     internal/justpay/service/auth_service.go  -> internal/justpay/service
      - TS:     src/components/auth/login-form.tsx         -> src/components/auth
      - Python: src/app/routes/payment.py                 -> src/app/routes

    Filters out non-source files (docs, configs, CI files, etc.) so only
    meaningful code packages are returned.
    """
    # Extensions that represent actual source code
    source_extensions = {
        ".go", ".ts", ".tsx", ".js", ".jsx",
        ".py", ".proto", ".yaml", ".yml",
    }
    # Top-level directories/files to skip
    skip_prefixes = (
        ".", "docs/", "test/", "tests/", "vendor/",
        "node_modules/", "__pycache__/",
    )

    packages: set[str] = set()
    for f in changed_files:
        if not f or not f.strip():
            continue
        f = f.strip()

        # Skip non-source and config files
        if any(f.startswith(prefix) for prefix in skip_prefixes):
            continue

        # Must have a directory component
        parts = f.rsplit("/", 1)
        if len(parts) < 2:
            continue

        directory = parts[0]
        filename = parts[1]

        # Check extension
        _, ext = os.path.splitext(filename)
        if ext.lower() not in source_extensions:
            continue

        packages.add(directory)

    return sorted(packages)


# ---------------------------------------------------------------------------
# Read raw docs + source code
# ---------------------------------------------------------------------------

def read_docs_for_package(docs_dir: Path, package: str) -> str:
    """Read all doc files under docs_dir that relate to the given package.

    Searches all subdirectories (go/, ts/, python/, proto/, helm/) for files
    whose name or content matches the package path. Falls back to reading all
    docs if the package-specific match fails.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return ""

    # Normalize package name for matching
    package_parts = package.replace("/", ".").replace("-", "_").lower()
    package_last = package.rsplit("/", 1)[-1].lower().replace("-", "_")

    collected: list[str] = []

    for doc_file in sorted(docs_dir.rglob("*")):
        if not doc_file.is_file():
            continue
        if doc_file.suffix not in (".md", ".json", ".txt"):
            continue

        file_stem = doc_file.stem.lower().replace("-", "_")

        # Match by filename containing the package name
        if package_last in file_stem or package_parts in file_stem:
            try:
                content = doc_file.read_text(encoding="utf-8", errors="replace")
                collected.append(
                    f"--- {doc_file.relative_to(docs_dir)} ---\n{content}"
                )
            except Exception:
                continue

    # If no package-specific match, try to find docs in a language subdirectory
    # that contains the package name anywhere in the content
    if not collected:
        for doc_file in sorted(docs_dir.rglob("*.md")):
            if not doc_file.is_file():
                continue
            try:
                content = doc_file.read_text(encoding="utf-8", errors="replace")
                # Check if the package path appears in the content
                if package in content or package_last in content.lower():
                    collected.append(
                        f"--- {doc_file.relative_to(docs_dir)} ---\n{content}"
                    )
            except Exception:
                continue

    return "\n\n".join(collected)


def read_source_for_package(source_dir: Path, package: str) -> str:
    """Read source files from the package directory.

    Returns concatenated source code, capped at a reasonable size to fit
    within the LLM context window.
    """
    source_extensions = {
        ".go", ".ts", ".tsx", ".js", ".jsx", ".py", ".proto",
    }
    pkg_path = Path(source_dir) / package
    if not pkg_path.is_dir():
        return ""

    collected: list[str] = []
    total_size = 0
    max_total = 12000  # Characters -- leaves room in the prompt

    for src_file in sorted(pkg_path.iterdir()):
        if not src_file.is_file():
            continue
        if src_file.suffix not in source_extensions:
            continue
        try:
            content = src_file.read_text(encoding="utf-8", errors="replace")
            collected.append(f"--- {src_file.name} ---\n{content}")
            total_size += len(content)
            if total_size >= max_total:
                break
        except Exception:
            continue

    return "\n\n".join(collected)


# ---------------------------------------------------------------------------
# RAGAnything query
# ---------------------------------------------------------------------------

async def query_raganything(
    url: str, package_name: str, repo: str
) -> str:
    """Query RAGAnything for graph context about a package.

    Uses the /query endpoint with mode=mix (hybrid vector + graph search) to
    retrieve cross-repo relationships and entity context.

    Returns an empty string on failure so the pipeline can continue without
    graph context.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{url}/query",
                json={
                    "query": (
                        f"What entities and relationships involve {package_name} "
                        f"in {repo}? Include cross-repo dependencies."
                    ),
                    "mode": "mix",
                },
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json().get("response", "")
    except (httpx.HTTPError, httpx.TimeoutException, KeyError, ValueError) as exc:
        print(
            f"Warning: RAGAnything query failed for {package_name}: {exc}",
            file=sys.stderr,
        )
        return ""


# ---------------------------------------------------------------------------
# OpenAI doc generation
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a technical documentation writer. Write clear, accurate developer "
    "documentation. You have three sources of truth: deterministic API reference "
    "(every symbol listed), cross-repo relationship context (from a knowledge "
    "graph), and actual source code. Document what the code does, how it connects "
    "to other services, and what a developer needs to know. Do not add information "
    "not grounded in your inputs. Use markdown formatting with proper headings, "
    "code blocks, and lists."
)


def build_user_prompt(
    raw_docs: str,
    graph_context: str,
    source_code: str,
    package_name: str,
    repo: str,
) -> str:
    """Build the user prompt for the OpenAI doc generation call.

    Truncates each section to fit within gpt-5.4-mini's context window while
    preserving the most useful information.
    """
    sections = [
        f"Write developer documentation for `{package_name}` in `{repo}`.",
        "",
        "## API Reference (from AST, every symbol listed)",
        raw_docs[:4000] if raw_docs else "No API reference available for this package.",
        "",
        "## Cross-Repo Relationships (from knowledge graph)",
        graph_context[:2000] if graph_context else "No cross-repo context available.",
        "",
        "## Source Code",
        source_code[:4000] if source_code else "No source code available.",
    ]
    return "\n".join(sections)


async def generate_docs(
    openai_key: str,
    raw_docs: str,
    graph_context: str,
    source_code: str,
    package_name: str,
    repo: str,
) -> str:
    """Call gpt-5.4-mini to generate human-readable documentation.

    The prompt is grounded in three layers:
      - AST reference (complete, every symbol)
      - Graph context (cross-repo relationships)
      - Source code (actual implementation)

    Returns the generated markdown content, or an error message on failure.
    """
    user_prompt = build_user_prompt(
        raw_docs, graph_context, source_code, package_name, repo
    )

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={
                    "model": "gpt-5.4-mini",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 2000,
                    "temperature": 0.3,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
    except (httpx.HTTPError, httpx.TimeoutException, KeyError, IndexError) as exc:
        print(
            f"Error: OpenAI generation failed for {package_name}: {exc}",
            file=sys.stderr,
        )
        return ""


# ---------------------------------------------------------------------------
# Wiki.js GraphQL publishing
# ---------------------------------------------------------------------------

WIKIJS_QUERY_PAGE_BY_PATH = """
query ($path: String!) {
  pages {
    singleByPath(path: $path, locale: "en") {
      id
    }
  }
}
"""

WIKIJS_MUTATION_UPDATE_PAGE = """
mutation ($id: Int!, $content: String!, $title: String!) {
  pages {
    update(id: $id, content: $content, title: $title, isPublished: true) {
      responseResult {
        succeeded
        message
      }
    }
  }
}
"""

WIKIJS_MUTATION_CREATE_PAGE = """
mutation ($content: String!, $path: String!, $title: String!, $description: String!, $tags: [String]!) {
  pages {
    create(
      content: $content
      description: $description
      path: $path
      title: $title
      tags: $tags
      locale: "en"
      isPublished: true
      isPrivate: false
      editor: "markdown"
    ) {
      responseResult {
        succeeded
        message
      }
      page {
        id
      }
    }
  }
}
"""


async def publish_to_wikijs(
    url: str, api_key: str, path: str, title: str, content: str
) -> bool:
    """Create or update a Wiki.js page via GraphQL.

    First queries for an existing page at the given path. If found, updates it;
    otherwise creates a new page. Pages are at /{repo}/{package} paths.

    Returns True on success, False on failure.
    """
    full_path = path
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient() as client:
            # Check if page exists
            resp = await client.post(
                f"{url}/graphql",
                headers=headers,
                json={
                    "query": WIKIJS_QUERY_PAGE_BY_PATH,
                    "variables": {"path": full_path},
                },
                timeout=60.0,
            )
            resp.raise_for_status()
            data = resp.json()

            existing = (
                data.get("data", {})
                .get("pages", {})
                .get("singleByPath")
            )

            if existing and existing.get("id"):
                # Update existing page
                page_id = existing["id"]
                resp = await client.post(
                    f"{url}/graphql",
                    headers=headers,
                    json={
                        "query": WIKIJS_MUTATION_UPDATE_PAGE,
                        "variables": {
                            "id": page_id,
                            "content": content,
                            "title": title,
                        },
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                result = (
                    resp.json()
                    .get("data", {})
                    .get("pages", {})
                    .get("update", {})
                    .get("responseResult", {})
                )
                if not result.get("succeeded"):
                    print(
                        f"Warning: Wiki.js update failed for {full_path}: "
                        f"{result.get('message', 'unknown error')}",
                        file=sys.stderr,
                    )
                    return False
                return True
            else:
                # Create new page
                resp = await client.post(
                    f"{url}/graphql",
                    headers=headers,
                    json={
                        "query": WIKIJS_MUTATION_CREATE_PAGE,
                        "variables": {
                            "content": content,
                            "description": f"Auto-generated documentation for {title}",
                            "path": full_path,
                            "title": title,
                            "tags": ["auto-generated", "doc-pipeline"],
                        },
                    },
                    timeout=60.0,
                )
                resp.raise_for_status()
                result = (
                    resp.json()
                    .get("data", {})
                    .get("pages", {})
                    .get("create", {})
                    .get("responseResult", {})
                )
                if not result.get("succeeded"):
                    print(
                        f"Warning: Wiki.js create failed for {full_path}: "
                        f"{result.get('message', 'unknown error')}",
                        file=sys.stderr,
                    )
                    return False
                return True

    except Exception as exc:
        detail = str(exc)
        # Try to get response body for HTTP errors
        if hasattr(exc, 'response') and exc.response is not None:
            try:
                detail = f"{exc} | Response: {exc.response.text[:500]}"
            except Exception:
                pass
        print(
            f"Error: Wiki.js publish failed for {full_path}: {type(exc).__name__}: {detail}",
            file=sys.stderr,
        )
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    API keys can be provided via CLI flags or environment variables.  The CLI
    flag takes precedence; the env var is used as fallback.  This avoids the
    shell-expansion issue seen in GitHub Actions reusable workflows where
    ``$OPENAI_API_KEY`` in the ``run:`` block can resolve to an empty string
    even though the secret is available in the ``env:`` context.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate human-readable Wiki.js pages from deterministic docs, "
            "source code, and RAGAnything graph context."
        ),
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository name (e.g. justpay-backend)",
    )
    parser.add_argument(
        "--docs-dir",
        required=True,
        help="Path to the directory containing generated docs",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Path to the source code checkout",
    )
    parser.add_argument(
        "--raganything-url",
        required=True,
        help="RAGAnything API base URL",
    )
    parser.add_argument(
        "--wikijs-url",
        required=True,
        help="Wiki.js base URL",
    )
    parser.add_argument(
        "--wikijs-api-key",
        default="",
        help="Wiki.js API key for GraphQL mutations (fallback: WIKIJS_API_KEY env var)",
    )
    parser.add_argument(
        "--openai-api-key",
        default="",
        help="OpenAI API key for gpt-5.4-mini (fallback: OPENAI_API_KEY env var)",
    )
    args = parser.parse_args(argv)

    # Resolve API keys: CLI arg -> env var -> empty string
    args.openai_api_key = (
        args.openai_api_key.strip()
        or os.environ.get("OPENAI_API_KEY", "").strip()
    )
    args.wikijs_api_key = (
        args.wikijs_api_key.strip()
        or os.environ.get("WIKIJS_API_KEY", "").strip()
    )

    return args


async def run(args: argparse.Namespace) -> None:
    """Execute the wiki page generation pipeline."""

    # ---- Validate API keys up-front ----
    if not args.openai_api_key:
        print(
            "WARNING: No OpenAI API key provided (--openai-api-key or "
            "OPENAI_API_KEY env var). Skipping doc generation.",
            file=sys.stderr,
        )
        return

    if not args.wikijs_api_key:
        print(
            "WARNING: No Wiki.js API key provided (--wikijs-api-key or "
            "WIKIJS_API_KEY env var). Skipping Wiki.js publishing.",
            file=sys.stderr,
        )
        return

    changed_files = get_changed_files()
    if not changed_files:
        print("No changed files detected. Nothing to generate.")
        return

    packages = map_files_to_packages(changed_files)
    if not packages:
        print("No source packages changed. Nothing to generate.")
        return

    print(f"Detected {len(packages)} changed package(s): {', '.join(packages)}")

    docs_dir = Path(args.docs_dir)
    source_dir = Path(args.source_dir)
    published = 0
    skipped = 0

    for package in packages:
        raw_docs = read_docs_for_package(docs_dir, package)
        source = read_source_for_package(source_dir, package)

        if not raw_docs and not source:
            print(f"Skipping {package}: no docs or source found")
            skipped += 1
            continue

        # Query RAGAnything for cross-repo graph context
        graph_context = await query_raganything(
            args.raganything_url, package, args.repo
        )

        # Generate prose documentation via OpenAI
        content = await generate_docs(
            args.openai_api_key,
            raw_docs,
            graph_context,
            source,
            package,
            args.repo,
        )

        if not content:
            print(f"Skipping {package}: OpenAI returned empty content")
            skipped += 1
            continue

        # Publish to Wiki.js
        page_path = f"{args.repo}/{package.replace('/', '-')}"
        title = f"{args.repo}: {package}"

        success = await publish_to_wikijs(
            args.wikijs_url, args.wikijs_api_key, page_path, title, content
        )

        if success:
            print(f"Published: /{page_path}")
            published += 1
        else:
            skipped += 1

    print(f"\nDone. Published {published} page(s), skipped {skipped}.")


def main() -> None:
    """Entry point."""
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
