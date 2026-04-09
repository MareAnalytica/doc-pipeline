#!/usr/bin/env python3
"""Generate /llms.txt and /llms-full.txt from Wiki.js content.

This is Epic 15 of the doc pipeline. It queries all published pages from
Wiki.js via GraphQL, organizes them by top-level path section, and produces
two files following the llms.txt standard (https://llmstxt.org/):

  /llms.txt       -- Index with page titles, URLs, and descriptions
  /llms-full.txt  -- Full content inlined under each page heading

The llms.txt standard makes documentation consumable by AI tools without
requiring a RAG pipeline. It uses a simple markdown format: H1 site name,
blockquote summary, H2 sections, and lists of pages with URLs.

Usage:
    python scripts/generate_llms_txt.py \\
      --wikijs-url http://wikijs.docs.svc.cluster.local:3000 \\
      --wikijs-api-key "$WIKIJS_API_KEY" \\
      --site-url https://docs.mareanalytica.com \\
      --output-dir /tmp/llms-output/
"""

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class WikiPage:
    """A single Wiki.js page with metadata and optional content."""
    id: int
    path: str
    title: str
    description: str
    updated_at: str
    content: str = ""


@dataclass
class Section:
    """A group of pages sharing a common top-level path prefix."""
    name: str
    display_name: str
    pages: list[WikiPage] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Wiki.js GraphQL queries
# ---------------------------------------------------------------------------

QUERY_LIST_PAGES = """
{
  pages {
    list(orderBy: PATH) {
      id
      path
      title
      description
      updatedAt
    }
  }
}
"""

QUERY_PAGE_CONTENT = """
query ($path: String!, $locale: String!) {
  pages {
    singleByPath(path: $path, locale: $locale) {
      content
    }
  }
}
"""


async def fetch_all_pages(
    client: httpx.AsyncClient,
    wikijs_url: str,
    api_key: str,
) -> list[WikiPage]:
    """List all pages from Wiki.js via GraphQL.

    Returns pages sorted by path. Raises on HTTP or GraphQL errors.
    """
    headers = {"Authorization": f"Bearer {api_key}"}

    resp = await client.post(
        f"{wikijs_url}/graphql",
        headers=headers,
        json={"query": QUERY_LIST_PAGES},
        timeout=30.0,
    )
    resp.raise_for_status()
    data = resp.json()

    # Check for GraphQL-level errors
    if "errors" in data:
        raise RuntimeError(
            f"Wiki.js GraphQL error: {data['errors']}"
        )

    raw_pages = (
        data.get("data", {})
        .get("pages", {})
        .get("list", [])
    )

    pages = []
    for p in raw_pages:
        pages.append(WikiPage(
            id=p["id"],
            path=p["path"],
            title=p["title"],
            description=p.get("description", "") or "",
            updated_at=p.get("updatedAt", ""),
        ))

    return pages


async def fetch_page_content(
    client: httpx.AsyncClient,
    wikijs_url: str,
    api_key: str,
    path: str,
    locale: str = "en",
) -> str:
    """Fetch a single page's markdown content from Wiki.js.

    Returns the content string, or an empty string on failure.
    """
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        resp = await client.post(
            f"{wikijs_url}/graphql",
            headers=headers,
            json={
                "query": QUERY_PAGE_CONTENT,
                "variables": {"path": path, "locale": locale},
            },
            timeout=30.0,
        )
        resp.raise_for_status()
        data = resp.json()

        page = (
            data.get("data", {})
            .get("pages", {})
            .get("singleByPath")
        )
        if page:
            return page.get("content", "") or ""
        return ""
    except (httpx.HTTPError, httpx.TimeoutException, KeyError, ValueError) as exc:
        print(
            f"Warning: Failed to fetch content for {path}: {exc}",
            file=sys.stderr,
        )
        return ""


# ---------------------------------------------------------------------------
# Page organization
# ---------------------------------------------------------------------------

def derive_section_display_name(section_key: str) -> str:
    """Convert a path-based section key into a human-readable display name.

    Examples:
        "auto/justpay-backend"  -> "JustPay Backend"
        "auto/hydra-brain"      -> "Hydra Brain"
        "guides"                -> "Guides"
        "home"                  -> "Home"
    """
    # Take the last meaningful segment
    parts = section_key.strip("/").split("/")

    # For "auto/<repo-name>", use the repo name
    if len(parts) >= 2 and parts[0] == "auto":
        raw = parts[1]
    else:
        raw = parts[-1]

    # Convert kebab-case / snake_case to title case
    return raw.replace("-", " ").replace("_", " ").title()


def organize_pages_into_sections(pages: list[WikiPage]) -> list[Section]:
    """Group pages by their top-level path prefix.

    For pages under /auto/<repo>/, the section key is "auto/<repo>".
    For other pages, the section key is the first path segment.
    Single-segment paths (like "home") become their own section.

    Returns sections sorted alphabetically by display name, with pages
    within each section sorted by path.
    """
    section_map: dict[str, list[WikiPage]] = defaultdict(list)

    for page in pages:
        path_parts = page.path.strip("/").split("/")

        if len(path_parts) == 0 or not path_parts[0]:
            section_key = "root"
        elif path_parts[0] == "auto" and len(path_parts) >= 2:
            # Group by auto/<repo-name>
            section_key = f"auto/{path_parts[1]}"
        elif len(path_parts) == 1:
            section_key = path_parts[0]
        else:
            section_key = path_parts[0]

        section_map[section_key].append(page)

    sections = []
    for key, section_pages in section_map.items():
        display_name = derive_section_display_name(key)
        # Sort pages within section by path
        section_pages.sort(key=lambda p: p.path)
        sections.append(Section(
            name=key,
            display_name=display_name,
            pages=section_pages,
        ))

    # Sort sections alphabetically by display name
    sections.sort(key=lambda s: s.display_name)

    return sections


# ---------------------------------------------------------------------------
# llms.txt generation
# ---------------------------------------------------------------------------

SITE_TITLE = "MareAnalytica Documentation"
SITE_DESCRIPTION = (
    "Technical documentation for MareAnalytica's infrastructure, services, "
    "and applications. Auto-generated from source code via deterministic AST "
    "parsing and knowledge graph context."
)


def generate_llms_txt(
    sections: list[Section],
    site_url: str,
) -> str:
    """Generate the /llms.txt index file content.

    Format follows the llms.txt standard:
      - H1: site name
      - Blockquote: summary
      - H2: section headers
      - Bullet lists with [title](url): description
    """
    lines: list[str] = []

    # Header
    lines.append(f"# {SITE_TITLE}")
    lines.append("")
    lines.append(f"> {SITE_DESCRIPTION}")
    lines.append("")

    site_url = site_url.rstrip("/")

    for section in sections:
        lines.append(f"## {section.display_name}")
        lines.append("")

        for page in section.pages:
            url = f"{site_url}/{page.path}"
            desc_suffix = f": {page.description}" if page.description else ""
            lines.append(f"- [{page.title}]({url}){desc_suffix}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def generate_llms_full_txt(
    sections: list[Section],
    site_url: str,
) -> str:
    """Generate the /llms-full.txt file with full page content inlined.

    Same structure as llms.txt but replaces bullet links with H3 headings
    followed by the full page content.
    """
    lines: list[str] = []

    # Header
    lines.append(f"# {SITE_TITLE}")
    lines.append("")
    lines.append(f"> {SITE_DESCRIPTION}")
    lines.append("")

    site_url = site_url.rstrip("/")

    for section in sections:
        lines.append(f"## {section.display_name}")
        lines.append("")

        for page in section.pages:
            url = f"{site_url}/{page.path}"
            lines.append(f"### {page.title}")
            lines.append("")
            lines.append(f"Source: {url}")
            lines.append("")

            content = page.content.strip()
            if content:
                lines.append(content)
            else:
                lines.append("*No content available.*")

            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate /llms.txt and /llms-full.txt from Wiki.js content. "
            "Follows the llms.txt standard for AI-consumable documentation."
        ),
    )
    parser.add_argument(
        "--wikijs-url",
        required=True,
        help="Wiki.js base URL (e.g. http://wikijs.docs.svc.cluster.local:3000)",
    )
    parser.add_argument(
        "--wikijs-api-key",
        required=True,
        help="Wiki.js API key for GraphQL queries",
    )
    parser.add_argument(
        "--site-url",
        default="https://docs.mareanalytica.com",
        help="Public site URL for page links (default: https://docs.mareanalytica.com)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write llms.txt and llms-full.txt",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent page content fetches (default: 5)",
    )
    return parser.parse_args(argv)


async def fetch_all_content(
    client: httpx.AsyncClient,
    wikijs_url: str,
    api_key: str,
    pages: list[WikiPage],
    concurrency: int = 5,
) -> None:
    """Fetch full content for all pages with bounded concurrency.

    Mutates each WikiPage in place, setting the content field.
    Uses a semaphore to avoid overwhelming the Wiki.js server.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def fetch_one(page: WikiPage) -> None:
        async with semaphore:
            page.content = await fetch_page_content(
                client, wikijs_url, api_key, page.path,
            )

    tasks = [fetch_one(page) for page in pages]
    await asyncio.gather(*tasks)


async def run(args: argparse.Namespace) -> None:
    """Execute the llms.txt generation pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start = time.monotonic()

    async with httpx.AsyncClient() as client:
        # Step 1: List all pages
        print(f"Querying Wiki.js at {args.wikijs_url} ...")
        pages = await fetch_all_pages(client, args.wikijs_url, args.wikijs_api_key)
        print(f"Found {len(pages)} page(s).")

        if not pages:
            print("No pages found. Writing empty files.")
            (output_dir / "llms.txt").write_text("")
            (output_dir / "llms-full.txt").write_text("")
            return

        # Step 2: Organize into sections
        sections = organize_pages_into_sections(pages)
        print(
            f"Organized into {len(sections)} section(s): "
            f"{', '.join(s.display_name for s in sections)}"
        )

        # Step 3: Generate llms.txt (index only -- no content fetch needed)
        llms_txt = generate_llms_txt(sections, args.site_url)
        llms_txt_path = output_dir / "llms.txt"
        llms_txt_path.write_text(llms_txt, encoding="utf-8")
        print(f"Wrote {llms_txt_path} ({len(llms_txt)} bytes)")

        # Step 4: Fetch all page content for llms-full.txt
        print(
            f"Fetching content for {len(pages)} page(s) "
            f"(concurrency={args.concurrency}) ..."
        )
        await fetch_all_content(
            client, args.wikijs_url, args.wikijs_api_key,
            pages, args.concurrency,
        )

        # Step 5: Generate llms-full.txt
        llms_full_txt = generate_llms_full_txt(sections, args.site_url)
        llms_full_txt_path = output_dir / "llms-full.txt"
        llms_full_txt_path.write_text(llms_full_txt, encoding="utf-8")
        print(f"Wrote {llms_full_txt_path} ({len(llms_full_txt)} bytes)")

    elapsed = time.monotonic() - start
    print(f"Done in {elapsed:.1f}s.")


def main() -> None:
    """Entry point."""
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
