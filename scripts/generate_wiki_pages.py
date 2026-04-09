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
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("wiki-gen")

# Explicit httpx timeout object -- bare float timeouts can silently fail
# if the event loop is blocked by synchronous work in another coroutine.
HTTPX_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=60.0, pool=10.0)

# Global run timeout: abort the entire pipeline after this many seconds
GLOBAL_RUN_TIMEOUT = 45 * 60  # 45 minutes


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
# Full mode: scan docs/ directory for ALL packages
# ---------------------------------------------------------------------------

def scan_all_packages(docs_dir: Path) -> list[str]:
    """Scan the docs/ directory to discover ALL packages/modules.

    Used in --full mode to generate pages for every documented package,
    not just those that changed in the latest commit.

    Handles each language subdirectory differently:
      - go/packages.md: Split by package headers
      - ts/typedoc.json: Parse top-level modules
      - python/*.md: One module per file
      - proto/*.json: One group per service/message
      - helm/*.md: One page per chart
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        return []

    packages: list[str] = []

    # --- Go: split packages.md by package headers ---
    go_packages_file = docs_dir / "go" / "packages.md"
    if go_packages_file.is_file():
        content = go_packages_file.read_text(encoding="utf-8", errors="replace")
        for line in content.splitlines():
            # Match lines like "# package auth" or "## internal/justpay/service"
            match = re.match(r'^#{1,2}\s+(?:package\s+)?(.+)', line)
            if match:
                pkg_name = match.group(1).strip()
                # Skip generic headers that aren't package names
                if pkg_name and not pkg_name.lower().startswith("table of"):
                    packages.append(f"go/{pkg_name}")

    # --- TypeScript: parse typedoc.json for top-level modules ---
    ts_typedoc_file = docs_dir / "ts" / "typedoc.json"
    if ts_typedoc_file.is_file():
        try:
            data = json.loads(
                ts_typedoc_file.read_text(encoding="utf-8", errors="replace")
            )
            children = data.get("children", [])
            for child in children:
                name = child.get("name", "")
                if name:
                    packages.append(f"ts/{name}")
        except (json.JSONDecodeError, KeyError):
            pass

    # --- Python: one module per .md file ---
    # Keep the original underscore-based filename as the package key
    # (e.g., hydra_adapter_checkpoint, not hydra.adapter.checkpoint)
    python_dir = docs_dir / "python"
    if python_dir.is_dir():
        for md_file in sorted(python_dir.glob("*.md")):
            packages.append(f"python/{md_file.stem}")

    # --- Protobuf: one group per .json file ---
    proto_dir = docs_dir / "proto"
    if proto_dir.is_dir():
        for json_file in sorted(proto_dir.glob("*.json")):
            packages.append(f"proto/{json_file.stem}")

    # --- Helm: one page per .md file ---
    helm_dir = docs_dir / "helm"
    if helm_dir.is_dir():
        for md_file in sorted(helm_dir.glob("*.md")):
            packages.append(f"helm/{md_file.stem}")

    return sorted(set(packages))


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

    # Normalize package name for matching (handle both dots and underscores)
    package_parts = package.replace("/", ".").replace("-", "_").lower()
    package_last = package.rsplit("/", 1)[-1].lower().replace("-", "_")
    # Also create underscore-only variants for filename matching
    package_parts_underscored = package_parts.replace(".", "_")
    package_last_underscored = package_last.replace(".", "_")

    collected: list[str] = []

    for doc_file in sorted(docs_dir.rglob("*")):
        if not doc_file.is_file():
            continue
        if doc_file.suffix not in (".md", ".json", ".txt"):
            continue

        file_stem = doc_file.stem.lower().replace("-", "_")

        # Match by filename containing the package name (dots or underscores)
        if (package_last in file_stem or package_parts in file_stem
                or package_last_underscored in file_stem
                or package_parts_underscored in file_stem):
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

    Tries multiple path mappings since full-mode package names may not
    directly correspond to source tree paths:
      - Direct path: source_dir/package
      - Dots to slashes: python/hydra.adapter.checkpoint -> src/hydra_adapter/checkpoint.py
      - Underscored: python/hydra_adapter_checkpoint -> src/hydra_adapter/checkpoint.py
    """
    source_extensions = {
        ".go", ".ts", ".tsx", ".js", ".jsx", ".py", ".proto",
    }
    source_dir = Path(source_dir)
    max_total = 12000

    # Strip language prefix (python/, go/, ts/, etc.) for source lookup
    lang_prefix, _, pkg_name = package.partition("/")

    # Build candidate paths
    candidates: list[Path] = []

    # Direct path
    candidates.append(source_dir / package)

    # For Python: map underscored doc filename to source path
    # e.g., hydra_adapter_checkpoint -> src/hydra_adapter/checkpoint.py
    # Strategy: try progressively splitting from the right to find the directory
    if lang_prefix == "python" and pkg_name:
        # Try splitting at each underscore position to find dir/file.py
        parts = pkg_name.split("_")
        for split_idx in range(len(parts) - 1, 0, -1):
            dir_name = "_".join(parts[:split_idx])
            file_name = "_".join(parts[split_idx:])
            candidates.append(source_dir / dir_name / f"{file_name}.py")
            candidates.append(source_dir / "src" / dir_name / f"{file_name}.py")
            # Also try the dir_name as a package directory
            candidates.append(source_dir / dir_name)
            candidates.append(source_dir / "src" / dir_name)
        # Also try the whole name as a directory or file
        candidates.append(source_dir / f"{pkg_name}.py")
        candidates.append(source_dir / "src" / f"{pkg_name}.py")
        candidates.append(source_dir / pkg_name)
        candidates.append(source_dir / "src" / pkg_name)
        # Try nested: hydra_adapter_connectors_lsports -> hydra_adapter/connectors/lsports.py
        # by trying the first known directory then recursing
        for root in [source_dir, source_dir / "src"]:
            remaining = parts[:]
            current = root
            while remaining:
                next_part = remaining.pop(0)
                test_dir = current / next_part
                # Try accumulating underscore segments into a directory name
                for j in range(1, len(remaining) + 1):
                    accumulated = "_".join([next_part] + remaining[:j-1])
                    if (current / accumulated).is_dir():
                        current = current / accumulated
                        remaining = remaining[j-1:]
                        break
                else:
                    if test_dir.is_dir():
                        current = test_dir
                    elif (current / f"{next_part}.py").is_file():
                        candidates.append(current / f"{next_part}.py")
                        break
                    else:
                        # Try remaining as filename
                        rest = "_".join([next_part] + remaining)
                        candidates.append(current / f"{rest}.py")
                        break

    # For Go: try internal/ prefixed paths
    if lang_prefix == "go" and pkg_name:
        candidates.append(source_dir / "internal" / pkg_name)
        candidates.append(source_dir / "internal" / pkg_name.replace(".", "/"))
        candidates.append(source_dir / "cmd" / pkg_name)
        candidates.append(source_dir / "pkg" / pkg_name)

    # For TypeScript
    if lang_prefix == "ts" and pkg_name:
        candidates.append(source_dir / "src" / pkg_name)
        candidates.append(source_dir / "src" / pkg_name.replace(".", "/"))

    collected: list[str] = []
    total_size = 0

    for pkg_path in candidates:
        if pkg_path.is_dir():
            for src_file in sorted(pkg_path.iterdir()):
                if not src_file.is_file() or src_file.suffix not in source_extensions:
                    continue
                try:
                    content = src_file.read_text(encoding="utf-8", errors="replace")
                    collected.append(f"--- {src_file.name} ---\n{content}")
                    total_size += len(content)
                    if total_size >= max_total:
                        break
                except Exception:
                    continue
            if collected:
                break
        elif pkg_path.with_suffix(".py").is_file():
            try:
                content = pkg_path.with_suffix(".py").read_text(encoding="utf-8", errors="replace")
                collected.append(f"--- {pkg_path.with_suffix('.py').name} ---\n{content}")
            except Exception:
                pass
            if collected:
                break

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
    rag_timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
    try:
        log.info("  RAG query start: %s", package_name)
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=rag_timeout) as client:
            response = await client.post(
                f"{url}/query",
                json={
                    "query": (
                        f"What entities and relationships involve {package_name} "
                        f"in {repo}? Include cross-repo dependencies."
                    ),
                    "mode": "mix",
                },
            )
            response.raise_for_status()
            elapsed = time.monotonic() - t0
            log.info("  RAG query done: %s (%.1fs)", package_name, elapsed)
            return response.json().get("response", "")
    except (httpx.HTTPError, httpx.TimeoutException, KeyError, ValueError) as exc:
        elapsed = time.monotonic() - t0
        log.warning(
            "  RAG query failed for %s after %.1fs: %s", package_name, elapsed, exc
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
        log.info("  OpenAI call start: %s", package_name)
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={
                    "model": "gpt-5.4-mini",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_completion_tokens": 4000,
                    "temperature": 0.3,
                },
            )
            response.raise_for_status()
            data = response.json()
            elapsed = time.monotonic() - t0
            log.info("  OpenAI call done: %s (%.1fs)", package_name, elapsed)
            return data["choices"][0]["message"]["content"]
    except (httpx.HTTPError, httpx.TimeoutException, KeyError, IndexError) as exc:
        elapsed = time.monotonic() - t0
        log.error(
            "  OpenAI generation failed for %s after %.1fs: %s",
            package_name, elapsed, exc,
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
      content
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


def _content_hash(text: str) -> str:
    """Return a short SHA-256 hex digest for content comparison."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


async def publish_to_wikijs(
    url: str, api_key: str, path: str, title: str, content: str
) -> str:
    """Create or update a Wiki.js page via GraphQL.

    First queries for an existing page at the given path. If found, compares
    content hashes -- if unchanged, skips the update to avoid the expensive
    UPDATE mutation that can hang on large pages. If the content changed (or
    the page does not exist), it creates/updates accordingly.

    Returns "published", "unchanged", or "failed".
    """
    full_path = path
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=HTTPX_TIMEOUT) as client:
            # --- Step 1: Check if page exists ---
            log.info("  Wiki.js lookup: %s", full_path)
            t0 = time.monotonic()
            resp = await client.post(
                f"{url}/graphql",
                headers=headers,
                json={
                    "query": WIKIJS_QUERY_PAGE_BY_PATH,
                    "variables": {"path": full_path},
                },
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.monotonic() - t0
            log.info("  Wiki.js lookup done: %s (%.1fs)", full_path, elapsed)

            existing = (
                data.get("data", {})
                .get("pages", {})
                .get("singleByPath")
            )

            if existing and existing.get("id"):
                page_id = existing["id"]

                # --- Content hash comparison: skip update if unchanged ---
                existing_content = existing.get("content", "")
                if _content_hash(existing_content) == _content_hash(content):
                    log.info(
                        "  Wiki.js SKIP (unchanged): %s (page %d)",
                        full_path, page_id,
                    )
                    return "unchanged"

                # --- Step 2a: Update existing page ---
                log.info(
                    "  Wiki.js UPDATE start: %s (page %d, %d bytes)",
                    full_path, page_id, len(content),
                )
                t0 = time.monotonic()
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
                )
                resp.raise_for_status()
                elapsed = time.monotonic() - t0
                result = (
                    resp.json()
                    .get("data", {})
                    .get("pages", {})
                    .get("update", {})
                    .get("responseResult", {})
                )
                if not result.get("succeeded"):
                    log.warning(
                        "  Wiki.js UPDATE failed for %s after %.1fs: %s",
                        full_path, elapsed, result.get("message", "unknown error"),
                    )
                    return "failed"
                log.info("  Wiki.js UPDATE done: %s (%.1fs)", full_path, elapsed)
                return "published"
            else:
                # --- Step 2b: Create new page ---
                log.info(
                    "  Wiki.js CREATE start: %s (%d bytes)",
                    full_path, len(content),
                )
                t0 = time.monotonic()
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
                )
                resp.raise_for_status()
                elapsed = time.monotonic() - t0
                result = (
                    resp.json()
                    .get("data", {})
                    .get("pages", {})
                    .get("create", {})
                    .get("responseResult", {})
                )
                if not result.get("succeeded"):
                    log.warning(
                        "  Wiki.js CREATE failed for %s after %.1fs: %s",
                        full_path, elapsed, result.get("message", "unknown error"),
                    )
                    return "failed"
                log.info("  Wiki.js CREATE done: %s (%.1fs)", full_path, elapsed)
                return "published"

    except Exception as exc:
        detail = str(exc)
        # Try to get response body for HTTP errors
        if hasattr(exc, 'response') and exc.response is not None:
            try:
                detail = f"{exc} | Response: {exc.response.text[:500]}"
            except Exception:
                pass
        log.error(
            "  Wiki.js publish failed for %s: %s: %s",
            full_path, type(exc).__name__, detail,
        )
        return "failed"


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
    parser.add_argument(
        "--full",
        action="store_true",
        default=False,
        help=(
            "Generate Wiki.js pages for ALL packages in docs/, not just "
            "those changed in the latest commit."
        ),
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


async def _run_inner(args: argparse.Namespace) -> None:
    """Inner pipeline loop, called under a global timeout."""

    # ---- Discover packages: full mode scans docs/, incremental uses git diff ----
    if args.full:
        packages = scan_all_packages(Path(args.docs_dir))
        if not packages:
            log.info("Full mode: no packages found in docs/ directory. Nothing to generate.")
            return
        log.info("Full mode: found %d package(s) in docs/: %s", len(packages), ", ".join(packages))
    else:
        changed_files = get_changed_files()
        if not changed_files:
            log.info("No changed files detected. Nothing to generate.")
            return

        packages = map_files_to_packages(changed_files)
        if not packages:
            log.info("No source packages changed. Nothing to generate.")
            return

        log.info("Detected %d changed package(s): %s", len(packages), ", ".join(packages))

    docs_dir = Path(args.docs_dir)
    source_dir = Path(args.source_dir)
    published = 0
    skipped = 0
    unchanged = 0

    PACKAGE_TIMEOUT = 120  # seconds -- max time per package before moving on

    for idx, package in enumerate(packages, 1):
        log.info("[%d/%d] Processing %s ...", idx, len(packages), package)
        try:
            result = await asyncio.wait_for(
                _process_one_package(args, docs_dir, source_dir, package),
                timeout=PACKAGE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning("  TIMEOUT: >%ds, skipping %s", PACKAGE_TIMEOUT, package)
            skipped += 1
            continue
        except Exception as exc:
            log.error("  ERROR on %s: %s: %s", package, type(exc).__name__, exc)
            skipped += 1
            continue

        if result == "published":
            published += 1
        elif result == "unchanged":
            unchanged += 1
        else:
            skipped += 1

        # Rate limit in full mode to avoid OpenAI API throttling
        if args.full:
            await asyncio.sleep(2)

    log.info(
        "Done. Published %d page(s), unchanged %d, skipped %d.",
        published, unchanged, skipped,
    )


async def run(args: argparse.Namespace) -> None:
    """Execute the wiki page generation pipeline with a global timeout."""

    # ---- Validate API keys up-front ----
    if not args.openai_api_key:
        log.warning(
            "No OpenAI API key provided (--openai-api-key or "
            "OPENAI_API_KEY env var). Skipping doc generation.",
        )
        return

    if not args.wikijs_api_key:
        log.warning(
            "No Wiki.js API key provided (--wikijs-api-key or "
            "WIKIJS_API_KEY env var). Skipping Wiki.js publishing.",
        )
        return

    try:
        await asyncio.wait_for(_run_inner(args), timeout=GLOBAL_RUN_TIMEOUT)
    except asyncio.TimeoutError:
        log.error(
            "GLOBAL TIMEOUT: entire run exceeded %d minutes, aborting.",
            GLOBAL_RUN_TIMEOUT // 60,
        )


async def _process_one_package(
    args, docs_dir: Path, source_dir: Path, package: str
) -> str:
    """Process a single package: read docs+source, query RAG, generate prose, publish.

    Returns one of: "published", "unchanged", "skipped".
    """
    raw_docs = read_docs_for_package(docs_dir, package)
    source = read_source_for_package(source_dir, package)

    if not raw_docs and not source:
        log.info("  Skipping %s: no docs or source found", package)
        return "skipped"

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
        log.info("  Skipping %s: OpenAI returned empty content", package)
        return "skipped"

    # Publish to Wiki.js
    page_path = f"{args.repo}/{package.replace('/', '-')}"
    title = f"{args.repo}: {package}"

    result = await publish_to_wikijs(
        args.wikijs_url, args.wikijs_api_key, page_path, title, content
    )

    if result == "published":
        log.info("  Published: /%s", page_path)
    elif result == "unchanged":
        log.info("  Unchanged (skipped update): /%s", page_path)
    else:
        log.warning("  Failed to publish: /%s", page_path)
        result = "skipped"

    return result


def main() -> None:
    """Entry point."""
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
