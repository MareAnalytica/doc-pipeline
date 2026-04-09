"""Unit tests for the llms.txt generator script (Epic 15)."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

# Add scripts directory to path so we can import the module
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_llms_txt import (
    SITE_DESCRIPTION,
    SITE_TITLE,
    Section,
    WikiPage,
    derive_section_display_name,
    fetch_all_pages,
    fetch_page_content,
    generate_llms_full_txt,
    generate_llms_txt,
    organize_pages_into_sections,
    parse_args,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_page(
    id: int = 1,
    path: str = "auto/justpay-backend/service-auth",
    title: str = "AuthService",
    description: str = "Authentication service",
    content: str = "",
) -> WikiPage:
    """Create a WikiPage with sensible defaults."""
    return WikiPage(
        id=id,
        path=path,
        title=title,
        description=description,
        updated_at="2026-04-09T00:00:00Z",
        content=content,
    )


SAMPLE_PAGES = [
    make_page(
        id=1,
        path="auto/justpay-backend/service-auth-service",
        title="AuthService",
        description="Core authentication service handling login, registration, and MFA",
    ),
    make_page(
        id=2,
        path="auto/justpay-backend/service-payment",
        title="PaymentService",
        description="Payment processing and transaction management",
    ),
    make_page(
        id=3,
        path="auto/hydra-brain/payment-worker",
        title="PaymentWorker",
        description="Streaming payment event processor",
    ),
    make_page(
        id=4,
        path="auto/hydra-brain/data-pipeline",
        title="DataPipeline",
        description="ETL pipeline for market data",
    ),
    make_page(
        id=5,
        path="guides/getting-started",
        title="Getting Started",
        description="Quick start guide for new developers",
    ),
    make_page(
        id=6,
        path="home",
        title="Home",
        description="Documentation home page",
    ),
]


# ---------------------------------------------------------------------------
# Section display name derivation
# ---------------------------------------------------------------------------


class TestDeriveSectionDisplayName:
    """Verify path-to-display-name conversion."""

    def test_auto_repo_path(self):
        assert derive_section_display_name("auto/justpay-backend") == "Justpay Backend"

    def test_auto_repo_kebab(self):
        assert derive_section_display_name("auto/hydra-brain") == "Hydra Brain"

    def test_single_segment(self):
        assert derive_section_display_name("guides") == "Guides"

    def test_single_segment_home(self):
        assert derive_section_display_name("home") == "Home"

    def test_underscore_path(self):
        assert derive_section_display_name("auto/my_service") == "My Service"

    def test_strips_slashes(self):
        assert derive_section_display_name("/auto/justpay-backend/") == "Justpay Backend"


# ---------------------------------------------------------------------------
# Page organization into sections
# ---------------------------------------------------------------------------


class TestOrganizePagesIntoSections:
    """Verify grouping pages by top-level path prefix."""

    def test_groups_by_auto_repo(self):
        sections = organize_pages_into_sections(SAMPLE_PAGES)
        section_names = {s.name for s in sections}

        assert "auto/justpay-backend" in section_names
        assert "auto/hydra-brain" in section_names

    def test_groups_non_auto_by_first_segment(self):
        sections = organize_pages_into_sections(SAMPLE_PAGES)
        section_names = {s.name for s in sections}

        assert "guides" in section_names
        assert "home" in section_names

    def test_correct_page_count_per_section(self):
        sections = organize_pages_into_sections(SAMPLE_PAGES)
        section_by_name = {s.name: s for s in sections}

        assert len(section_by_name["auto/justpay-backend"].pages) == 2
        assert len(section_by_name["auto/hydra-brain"].pages) == 2
        assert len(section_by_name["guides"].pages) == 1
        assert len(section_by_name["home"].pages) == 1

    def test_sections_sorted_by_display_name(self):
        sections = organize_pages_into_sections(SAMPLE_PAGES)
        display_names = [s.display_name for s in sections]
        assert display_names == sorted(display_names)

    def test_pages_sorted_by_path_within_section(self):
        sections = organize_pages_into_sections(SAMPLE_PAGES)
        for section in sections:
            paths = [p.path for p in section.pages]
            assert paths == sorted(paths), (
                f"Pages in {section.name} not sorted: {paths}"
            )

    def test_empty_pages_list(self):
        sections = organize_pages_into_sections([])
        assert sections == []

    def test_single_page(self):
        pages = [make_page(path="auto/my-repo/my-page", title="My Page")]
        sections = organize_pages_into_sections(pages)
        assert len(sections) == 1
        assert sections[0].name == "auto/my-repo"
        assert len(sections[0].pages) == 1


# ---------------------------------------------------------------------------
# llms.txt format output
# ---------------------------------------------------------------------------


class TestGenerateLlmsTxt:
    """Verify the /llms.txt output follows the standard."""

    def setup_method(self):
        self.sections = organize_pages_into_sections(SAMPLE_PAGES)
        self.site_url = "https://docs.mareanalytica.com"
        self.output = generate_llms_txt(self.sections, self.site_url)

    def test_starts_with_h1(self):
        assert self.output.startswith(f"# {SITE_TITLE}\n")

    def test_has_blockquote_summary(self):
        assert f"> {SITE_DESCRIPTION}" in self.output

    def test_has_h2_section_headers(self):
        assert "## Justpay Backend" in self.output
        assert "## Hydra Brain" in self.output
        assert "## Guides" in self.output
        assert "## Home" in self.output

    def test_has_page_links_with_urls(self):
        assert (
            "- [AuthService](https://docs.mareanalytica.com/"
            "auto/justpay-backend/service-auth-service)"
        ) in self.output

    def test_has_descriptions_after_links(self):
        assert (
            ": Core authentication service handling login, registration, and MFA"
        ) in self.output

    def test_no_page_content_inlined(self):
        """llms.txt should not contain any page body content."""
        assert "### AuthService" not in self.output
        assert "Source:" not in self.output

    def test_trailing_newline(self):
        assert self.output.endswith("\n")

    def test_no_description_omits_suffix(self):
        """Pages with empty descriptions should not have a trailing colon."""
        pages = [make_page(description="")]
        sections = organize_pages_into_sections(pages)
        output = generate_llms_txt(sections, self.site_url)
        # Should end with the closing paren, not ":"
        assert ")\n" in output or output.strip().endswith(")")

    def test_site_url_trailing_slash_stripped(self):
        """Trailing slash on site URL should not cause double slashes."""
        sections = organize_pages_into_sections(SAMPLE_PAGES)
        output = generate_llms_txt(sections, "https://docs.mareanalytica.com/")
        assert "https://docs.mareanalytica.com//auto" not in output
        assert "https://docs.mareanalytica.com/auto" in output


# ---------------------------------------------------------------------------
# llms-full.txt format output
# ---------------------------------------------------------------------------


class TestGenerateLlmsFullTxt:
    """Verify the /llms-full.txt output includes full content."""

    def setup_method(self):
        self.pages = [
            make_page(
                id=1,
                path="auto/justpay-backend/service-auth",
                title="AuthService",
                description="Auth service",
                content="# AuthService\n\nHandles user login and registration.\n\n## Methods\n\n- `Login(email, password)` - Authenticate user",
            ),
            make_page(
                id=2,
                path="auto/justpay-backend/service-payment",
                title="PaymentService",
                description="Payment service",
                content="# PaymentService\n\nProcesses payments via Paystack.",
            ),
        ]
        self.sections = organize_pages_into_sections(self.pages)
        self.site_url = "https://docs.mareanalytica.com"
        self.output = generate_llms_full_txt(self.sections, self.site_url)

    def test_starts_with_h1(self):
        assert self.output.startswith(f"# {SITE_TITLE}\n")

    def test_has_blockquote_summary(self):
        assert f"> {SITE_DESCRIPTION}" in self.output

    def test_has_h2_section_headers(self):
        assert "## Justpay Backend" in self.output

    def test_has_h3_page_headings(self):
        assert "### AuthService" in self.output
        assert "### PaymentService" in self.output

    def test_has_source_url(self):
        assert (
            "Source: https://docs.mareanalytica.com/auto/justpay-backend/service-auth"
        ) in self.output

    def test_full_content_inlined(self):
        assert "Handles user login and registration." in self.output
        assert "Processes payments via Paystack." in self.output

    def test_no_bullet_links(self):
        """Full version uses H3 headings, not bullet links."""
        assert "- [AuthService]" not in self.output

    def test_empty_content_shows_placeholder(self):
        pages = [make_page(content="")]
        sections = organize_pages_into_sections(pages)
        output = generate_llms_full_txt(sections, self.site_url)
        assert "*No content available.*" in output

    def test_trailing_newline(self):
        assert self.output.endswith("\n")


# ---------------------------------------------------------------------------
# Wiki.js GraphQL fetch tests (mocked)
# ---------------------------------------------------------------------------


class TestFetchAllPages:
    """Verify page listing from Wiki.js GraphQL."""

    @pytest.mark.asyncio
    async def test_successful_fetch(self):
        mock_response = httpx.Response(
            200,
            json={
                "data": {
                    "pages": {
                        "list": [
                            {
                                "id": 1,
                                "path": "auto/justpay-backend/service-auth",
                                "title": "AuthService",
                                "description": "Auth service",
                                "updatedAt": "2026-04-09T00:00:00Z",
                            },
                            {
                                "id": 2,
                                "path": "auto/hydra-brain/worker",
                                "title": "Worker",
                                "description": None,
                                "updatedAt": "2026-04-09T00:00:00Z",
                            },
                        ]
                    }
                }
            },
            request=httpx.Request("POST", "http://wiki/graphql"),
        )

        with patch("generate_llms_txt.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            pages = await fetch_all_pages(instance, "http://wiki", "key123")
            assert len(pages) == 2
            assert pages[0].path == "auto/justpay-backend/service-auth"
            assert pages[1].description == ""  # None coalesced to ""

    @pytest.mark.asyncio
    async def test_graphql_error_raises(self):
        mock_response = httpx.Response(
            200,
            json={
                "errors": [{"message": "Unauthorized"}],
            },
            request=httpx.Request("POST", "http://wiki/graphql"),
        )

        with patch("generate_llms_txt.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            with pytest.raises(RuntimeError, match="GraphQL error"):
                await fetch_all_pages(instance, "http://wiki", "bad-key")

    @pytest.mark.asyncio
    async def test_empty_page_list(self):
        mock_response = httpx.Response(
            200,
            json={"data": {"pages": {"list": []}}},
            request=httpx.Request("POST", "http://wiki/graphql"),
        )

        with patch("generate_llms_txt.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            pages = await fetch_all_pages(instance, "http://wiki", "key123")
            assert pages == []


class TestFetchPageContent:
    """Verify single page content fetch from Wiki.js GraphQL."""

    @pytest.mark.asyncio
    async def test_successful_content_fetch(self):
        mock_response = httpx.Response(
            200,
            json={
                "data": {
                    "pages": {
                        "singleByPath": {
                            "content": "# AuthService\n\nHandles login."
                        }
                    }
                }
            },
            request=httpx.Request("POST", "http://wiki/graphql"),
        )

        with patch("generate_llms_txt.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            content = await fetch_page_content(
                instance, "http://wiki", "key123",
                "auto/justpay-backend/service-auth",
            )
            assert content == "# AuthService\n\nHandles login."

    @pytest.mark.asyncio
    async def test_missing_page_returns_empty(self):
        mock_response = httpx.Response(
            200,
            json={"data": {"pages": {"singleByPath": None}}},
            request=httpx.Request("POST", "http://wiki/graphql"),
        )

        with patch("generate_llms_txt.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            content = await fetch_page_content(
                instance, "http://wiki", "key123", "nonexistent",
            )
            assert content == ""

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self):
        with patch("generate_llms_txt.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.side_effect = httpx.TimeoutException("timed out")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            content = await fetch_page_content(
                instance, "http://wiki", "key123", "some/path",
            )
            assert content == ""


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------


class TestParseArgs:
    """Verify CLI argument parsing."""

    def test_required_args(self):
        args = parse_args([
            "--wikijs-url", "http://wiki:3000",
            "--wikijs-api-key", "key123",
            "--output-dir", "/tmp/out",
        ])
        assert args.wikijs_url == "http://wiki:3000"
        assert args.wikijs_api_key == "key123"
        assert args.output_dir == "/tmp/out"
        assert args.site_url == "https://docs.mareanalytica.com"  # default
        assert args.concurrency == 5  # default

    def test_custom_site_url(self):
        args = parse_args([
            "--wikijs-url", "http://wiki:3000",
            "--wikijs-api-key", "key123",
            "--site-url", "https://custom.example.com",
            "--output-dir", "/tmp/out",
        ])
        assert args.site_url == "https://custom.example.com"

    def test_custom_concurrency(self):
        args = parse_args([
            "--wikijs-url", "http://wiki:3000",
            "--wikijs-api-key", "key123",
            "--output-dir", "/tmp/out",
            "--concurrency", "10",
        ])
        assert args.concurrency == 10

    def test_missing_required_arg_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--wikijs-url", "http://wiki:3000"])
