#!/usr/bin/env python3
"""Ingest Wiki.js pages back into RAGAnything as derived documentation.

This creates a feedback loop: deterministic docs -> LLM prose -> Wiki.js ->
RAGAnything graph. By tagging ingested pages as derived content, the entity
extractor can distinguish primary (AST-generated) sources from secondary
(prose) sources in the knowledge graph.

Usage:
    python scripts/ingest_wikijs_to_rag.py \
      --wikijs-url http://wikijs.docs.svc.cluster.local:3000 \
      --wikijs-api-key "$WIKIJS_API_KEY" \
      --raganything-url http://raganything.memory.svc.cluster.local:9621
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Wiki.js GraphQL queries
# ---------------------------------------------------------------------------

WIKIJS_LIST_PAGES = """{
  pages {
    list(orderBy: PATH) {
      id
      path
      title
    }
  }
}"""

WIKIJS_GET_PAGE = """
query ($id: Int!) {
  pages {
    single(id: $id) {
      content
      path
      title
      updatedAt
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Wiki.js API helpers
# ---------------------------------------------------------------------------

def wikijs_graphql(url: str, api_key: str, query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query against the Wiki.js API.

    Returns the parsed JSON response. Raises on HTTP or parse errors.
    """
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{url}/graphql",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    resp = urllib.request.urlopen(req, timeout=60)
    return json.loads(resp.read().decode("utf-8"))


def list_pages(url: str, api_key: str) -> list[dict]:
    """List all Wiki.js pages ordered by path."""
    data = wikijs_graphql(url, api_key, WIKIJS_LIST_PAGES)
    pages = data.get("data", {}).get("pages", {}).get("list", [])
    return pages


def get_page_content(url: str, api_key: str, page_id: int) -> dict | None:
    """Fetch full content of a single Wiki.js page by ID."""
    data = wikijs_graphql(url, api_key, WIKIJS_GET_PAGE, {"id": page_id})
    page = data.get("data", {}).get("pages", {}).get("single")
    return page


# ---------------------------------------------------------------------------
# RAGAnything ingestion
# ---------------------------------------------------------------------------

def ingest_to_raganything(rag_url: str, title: str, content: str, path: str) -> bool:
    """Ingest a Wiki.js page into RAGAnything as derived documentation.

    The document text is prefixed with a metadata tag so the entity extractor
    knows this is derived (prose) documentation rather than primary (AST) docs.

    Returns True on success, False on failure.
    """
    api_doc = {
        "text": f"[Wiki.js Documentation - {title}]\n\n{content}",
        "file_source": f"wikijs/{path}",
    }
    body = json.dumps(api_doc).encode("utf-8")
    req = urllib.request.Request(
        f"{rag_url}/documents/text",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        result = json.loads(resp.read().decode("utf-8"))
        print(f"  Ingested: wikijs/{path} -> {result.get('status', '?')}")
        return True
    except (urllib.error.HTTPError, urllib.error.URLError, Exception) as exc:
        print(f"  FAILED: wikijs/{path} -> {exc}", file=sys.stderr)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    The Wiki.js API key can be provided via CLI flag or WIKIJS_API_KEY env var.
    RAGAnything does not require authentication.
    """
    parser = argparse.ArgumentParser(
        description="Ingest Wiki.js pages into RAGAnything as derived documentation.",
    )
    parser.add_argument(
        "--wikijs-url",
        required=True,
        help="Wiki.js base URL",
    )
    parser.add_argument(
        "--wikijs-api-key",
        default="",
        help="Wiki.js API key (fallback: WIKIJS_API_KEY env var)",
    )
    parser.add_argument(
        "--raganything-url",
        required=True,
        help="RAGAnything API base URL",
    )
    args = parser.parse_args(argv)

    # Resolve API key: CLI arg -> env var
    args.wikijs_api_key = (
        args.wikijs_api_key.strip()
        or os.environ.get("WIKIJS_API_KEY", "").strip()
    )

    return args


def main() -> None:
    """Entry point: list Wiki.js pages and ingest each into RAGAnything."""
    args = parse_args()

    if not args.wikijs_api_key:
        print(
            "ERROR: No Wiki.js API key provided (--wikijs-api-key or "
            "WIKIJS_API_KEY env var).",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. List all pages
    print(f"Fetching page list from {args.wikijs_url}...")
    try:
        pages = list_pages(args.wikijs_url, args.wikijs_api_key)
    except Exception as exc:
        print(f"ERROR: Failed to list Wiki.js pages: {exc}", file=sys.stderr)
        sys.exit(1)

    if not pages:
        print("No pages found in Wiki.js. Nothing to ingest.")
        return

    print(f"Found {len(pages)} page(s) in Wiki.js.")

    ingested = 0
    skipped = 0

    for page_entry in pages:
        page_id = page_entry.get("id")
        page_path = page_entry.get("path", "")
        page_title = page_entry.get("title", "")

        # Skip the home page -- it's navigation, not documentation
        if page_path in ("home", "", "/"):
            print(f"  Skipping home page: {page_path}")
            skipped += 1
            continue

        # 2. Fetch full content
        try:
            page = get_page_content(args.wikijs_url, args.wikijs_api_key, page_id)
        except Exception as exc:
            print(
                f"  WARNING: Failed to fetch page {page_id} ({page_path}): {exc}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        if not page or not page.get("content"):
            print(f"  Skipping empty page: {page_path}")
            skipped += 1
            continue

        # 3. Ingest into RAGAnything
        success = ingest_to_raganything(
            args.raganything_url,
            page.get("title", page_title),
            page["content"],
            page.get("path", page_path),
        )

        if success:
            ingested += 1
        else:
            skipped += 1

    print(f"\nDone. Ingested {ingested} page(s), skipped {skipped}.")


if __name__ == "__main__":
    main()
