"""Unit tests for the Wiki.js page generation script (Epic 10)."""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Add scripts directory to path so we can import the module
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from generate_wiki_pages import (
    SYSTEM_PROMPT,
    WIKIJS_MUTATION_CREATE_PAGE,
    WIKIJS_MUTATION_UPDATE_PAGE,
    WIKIJS_QUERY_PAGE_BY_PATH,
    build_user_prompt,
    map_files_to_packages,
    publish_to_wikijs,
    query_raganything,
    read_docs_for_package,
    read_source_for_package,
)


# ---------------------------------------------------------------------------
# map_files_to_packages tests
# ---------------------------------------------------------------------------


class TestMapFilesToPackages:
    """Verify file-to-package mapping across Go, TypeScript, and Python."""

    def test_go_files(self):
        files = [
            "internal/justpay/service/auth_service.go",
            "internal/justpay/service/payment_service.go",
            "internal/justpay/model/user.go",
        ]
        packages = map_files_to_packages(files)
        assert "internal/justpay/service" in packages
        assert "internal/justpay/model" in packages

    def test_typescript_files(self):
        files = [
            "src/components/auth/login-form.tsx",
            "src/components/auth/register-form.tsx",
            "src/lib/api/client.ts",
        ]
        packages = map_files_to_packages(files)
        assert "src/components/auth" in packages
        assert "src/lib/api" in packages

    def test_python_files(self):
        files = [
            "src/app/routes/payment.py",
            "src/app/routes/auth.py",
            "src/app/models/user.py",
        ]
        packages = map_files_to_packages(files)
        assert "src/app/routes" in packages
        assert "src/app/models" in packages

    def test_filters_non_source_files(self):
        files = [
            "README.md",
            "Dockerfile",
            ".github/workflows/ci.yml",
            "docs/api.md",
            "src/main.go",
        ]
        packages = map_files_to_packages(files)
        # README.md, Dockerfile have no directory or wrong extension
        # .github/ and docs/ are filtered by skip_prefixes
        assert "src" in packages
        assert len(packages) == 1

    def test_filters_dot_directories(self):
        files = [
            ".github/workflows/deploy.yml",
            ".vscode/settings.json",
        ]
        packages = map_files_to_packages(files)
        assert packages == []

    def test_filters_vendor_and_node_modules(self):
        files = [
            "vendor/github.com/pkg/errors/errors.go",
            "node_modules/express/index.js",
        ]
        packages = map_files_to_packages(files)
        assert packages == []

    def test_root_level_files_skipped(self):
        """Files without a directory component should be skipped."""
        files = ["main.go", "app.ts", "server.py"]
        packages = map_files_to_packages(files)
        assert packages == []

    def test_empty_and_blank_lines(self):
        files = ["", "  ", "src/main.go", ""]
        packages = map_files_to_packages(files)
        assert packages == ["src"]

    def test_proto_files(self):
        files = [
            "proto/api/v1/service.proto",
            "proto/api/v1/types.proto",
        ]
        packages = map_files_to_packages(files)
        assert "proto/api/v1" in packages

    def test_deduplication(self):
        """Multiple files in the same package should produce one entry."""
        files = [
            "internal/auth/handler.go",
            "internal/auth/middleware.go",
            "internal/auth/token.go",
        ]
        packages = map_files_to_packages(files)
        assert packages == ["internal/auth"]

    def test_sorted_output(self):
        files = [
            "src/z/main.go",
            "src/a/main.go",
            "src/m/main.go",
        ]
        packages = map_files_to_packages(files)
        assert packages == ["src/a", "src/m", "src/z"]

    def test_mixed_languages(self):
        """A monorepo with Go, TS, and Python files."""
        files = [
            "backend/internal/handler/auth.go",
            "frontend/src/pages/login.tsx",
            "scripts/deploy/run.py",
        ]
        packages = map_files_to_packages(files)
        assert "backend/internal/handler" in packages
        assert "frontend/src/pages" in packages
        assert "scripts/deploy" in packages


# ---------------------------------------------------------------------------
# OpenAI prompt construction tests
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    """Verify the OpenAI user prompt is correctly assembled."""

    def test_includes_all_sections(self):
        prompt = build_user_prompt(
            raw_docs="func Login() error",
            graph_context="AuthService depends on UserRepository",
            source_code="func Login() error { return nil }",
            package_name="internal/auth",
            repo="justpay-backend",
        )
        assert "internal/auth" in prompt
        assert "justpay-backend" in prompt
        assert "func Login() error" in prompt
        assert "AuthService depends on UserRepository" in prompt
        assert "func Login() error { return nil }" in prompt

    def test_section_headers_present(self):
        prompt = build_user_prompt(
            raw_docs="docs",
            graph_context="context",
            source_code="code",
            package_name="pkg",
            repo="repo",
        )
        assert "## API Reference" in prompt
        assert "## Cross-Repo Relationships" in prompt
        assert "## Source Code" in prompt

    def test_missing_docs_shows_placeholder(self):
        prompt = build_user_prompt(
            raw_docs="",
            graph_context="",
            source_code="",
            package_name="pkg",
            repo="repo",
        )
        assert "No API reference available" in prompt
        assert "No cross-repo context available" in prompt
        assert "No source code available" in prompt

    def test_truncation(self):
        """Long inputs should be truncated to fit the context window."""
        # Use characters that do not appear in any section headers or labels
        long_docs = "X" * 10000
        long_context = "Y" * 5000
        long_source = "Z" * 10000

        prompt = build_user_prompt(
            raw_docs=long_docs,
            graph_context=long_context,
            source_code=long_source,
            package_name="mypkg",
            repo="myrepo",
        )
        # raw_docs truncated to 4000
        assert prompt.count("X") == 4000
        # graph_context truncated to 2000
        assert prompt.count("Y") == 2000
        # source_code truncated to 4000
        assert prompt.count("Z") == 4000

    def test_system_prompt_content(self):
        """The system prompt should instruct grounded documentation."""
        assert "technical documentation writer" in SYSTEM_PROMPT
        assert "Do not add information not grounded" in SYSTEM_PROMPT
        assert "three sources of truth" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Wiki.js GraphQL construction tests
# ---------------------------------------------------------------------------


class TestWikiJsGraphQL:
    """Verify the GraphQL queries and mutations are well-formed."""

    def test_query_page_by_path_structure(self):
        assert "singleByPath" in WIKIJS_QUERY_PAGE_BY_PATH
        assert "$path: String!" in WIKIJS_QUERY_PAGE_BY_PATH
        assert 'locale: "en"' in WIKIJS_QUERY_PAGE_BY_PATH

    def test_create_mutation_structure(self):
        assert "pages" in WIKIJS_MUTATION_CREATE_PAGE
        assert "create(" in WIKIJS_MUTATION_CREATE_PAGE
        assert "$content: String!" in WIKIJS_MUTATION_CREATE_PAGE
        assert "$path: String!" in WIKIJS_MUTATION_CREATE_PAGE
        assert "$title: String!" in WIKIJS_MUTATION_CREATE_PAGE
        assert 'editor: "markdown"' in WIKIJS_MUTATION_CREATE_PAGE
        assert "isPublished: true" in WIKIJS_MUTATION_CREATE_PAGE
        assert "isPrivate: false" in WIKIJS_MUTATION_CREATE_PAGE
        assert "responseResult" in WIKIJS_MUTATION_CREATE_PAGE
        assert "succeeded" in WIKIJS_MUTATION_CREATE_PAGE
        assert "message" in WIKIJS_MUTATION_CREATE_PAGE

    def test_update_mutation_structure(self):
        assert "pages" in WIKIJS_MUTATION_UPDATE_PAGE
        assert "update(" in WIKIJS_MUTATION_UPDATE_PAGE
        assert "$id: Int!" in WIKIJS_MUTATION_UPDATE_PAGE
        assert "$content: String!" in WIKIJS_MUTATION_UPDATE_PAGE
        assert "$title: String!" in WIKIJS_MUTATION_UPDATE_PAGE
        assert "isPublished: true" in WIKIJS_MUTATION_UPDATE_PAGE
        assert "responseResult" in WIKIJS_MUTATION_UPDATE_PAGE


# ---------------------------------------------------------------------------
# RAGAnything query tests (mocked)
# ---------------------------------------------------------------------------


class TestQueryRaganything:
    """Verify RAGAnything query logic with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_successful_query(self):
        mock_response = httpx.Response(
            200,
            json={"response": "AuthService calls UserRepository.FindByEmail"},
            request=httpx.Request("POST", "http://test/query"),
        )

        with patch("generate_wiki_pages.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await query_raganything(
                "http://test", "internal/auth", "justpay-backend"
            )
            assert result == "AuthService calls UserRepository.FindByEmail"

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self):
        with patch("generate_wiki_pages.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.side_effect = httpx.TimeoutException("timed out")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await query_raganything(
                "http://test", "internal/auth", "justpay-backend"
            )
            assert result == ""

    @pytest.mark.asyncio
    async def test_http_error_returns_empty(self):
        mock_response = httpx.Response(
            500,
            json={"error": "internal"},
            request=httpx.Request("POST", "http://test/query"),
        )

        with patch("generate_wiki_pages.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.return_value = mock_response
            mock_response.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "500", request=mock_response.request, response=mock_response
                )
            )
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await query_raganything(
                "http://test", "internal/auth", "justpay-backend"
            )
            assert result == ""


# ---------------------------------------------------------------------------
# Wiki.js publish tests (mocked)
# ---------------------------------------------------------------------------


class TestPublishToWikijs:
    """Verify Wiki.js publishing logic with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_creates_new_page(self):
        """When page does not exist, it should create a new one."""
        # First call: query returns no existing page
        query_response = httpx.Response(
            200,
            json={"data": {"pages": {"singleByPath": None}}},
            request=httpx.Request("POST", "http://wiki/graphql"),
        )
        # Second call: create mutation succeeds
        create_response = httpx.Response(
            200,
            json={
                "data": {
                    "pages": {
                        "create": {
                            "responseResult": {
                                "succeeded": True,
                                "message": "ok",
                            },
                            "page": {"id": 42},
                        }
                    }
                }
            },
            request=httpx.Request("POST", "http://wiki/graphql"),
        )

        with patch("generate_wiki_pages.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.side_effect = [query_response, create_response]
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await publish_to_wikijs(
                "http://wiki", "key123", "justpay-backend/internal-auth",
                "justpay-backend: internal/auth", "# Auth\nDocs here",
            )
            assert result is True

            # Verify the create mutation was called with correct path
            create_call = instance.post.call_args_list[1]
            variables = create_call.kwargs.get("json", create_call[1].get("json", {}))
            assert variables["variables"]["path"] == "auto/justpay-backend/internal-auth"

    @pytest.mark.asyncio
    async def test_updates_existing_page(self):
        """When page exists, it should update it."""
        query_response = httpx.Response(
            200,
            json={"data": {"pages": {"singleByPath": {"id": 99}}}},
            request=httpx.Request("POST", "http://wiki/graphql"),
        )
        update_response = httpx.Response(
            200,
            json={
                "data": {
                    "pages": {
                        "update": {
                            "responseResult": {
                                "succeeded": True,
                                "message": "ok",
                            }
                        }
                    }
                }
            },
            request=httpx.Request("POST", "http://wiki/graphql"),
        )

        with patch("generate_wiki_pages.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.side_effect = [query_response, update_response]
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await publish_to_wikijs(
                "http://wiki", "key123", "justpay-backend/internal-auth",
                "justpay-backend: internal/auth", "# Auth\nUpdated docs",
            )
            assert result is True

            # Verify the update mutation was called with the page ID
            update_call = instance.post.call_args_list[1]
            variables = update_call.kwargs.get("json", update_call[1].get("json", {}))
            assert variables["variables"]["id"] == 99

    @pytest.mark.asyncio
    async def test_handles_connection_failure(self):
        with patch("generate_wiki_pages.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.side_effect = httpx.ConnectError("refused")
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await publish_to_wikijs(
                "http://wiki", "key123", "repo/pkg",
                "title", "content",
            )
            assert result is False

    @pytest.mark.asyncio
    async def test_handles_create_failure(self):
        """When the create mutation returns succeeded=false."""
        query_response = httpx.Response(
            200,
            json={"data": {"pages": {"singleByPath": None}}},
            request=httpx.Request("POST", "http://wiki/graphql"),
        )
        create_response = httpx.Response(
            200,
            json={
                "data": {
                    "pages": {
                        "create": {
                            "responseResult": {
                                "succeeded": False,
                                "message": "duplicate path",
                            },
                            "page": None,
                        }
                    }
                }
            },
            request=httpx.Request("POST", "http://wiki/graphql"),
        )

        with patch("generate_wiki_pages.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post.side_effect = [query_response, create_response]
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await publish_to_wikijs(
                "http://wiki", "key123", "repo/pkg",
                "title", "content",
            )
            assert result is False


# ---------------------------------------------------------------------------
# read_docs_for_package tests
# ---------------------------------------------------------------------------


class TestReadDocsForPackage:
    """Verify doc file discovery logic."""

    def test_reads_matching_doc_file(self, tmp_path):
        docs_dir = tmp_path / "docs" / "go"
        docs_dir.mkdir(parents=True)
        (docs_dir / "auth_service.md").write_text("## func Login\nHandles login")
        content = read_docs_for_package(tmp_path / "docs", "internal/auth_service")
        assert "func Login" in content

    def test_returns_empty_for_missing_dir(self, tmp_path):
        content = read_docs_for_package(tmp_path / "nonexistent", "internal/auth")
        assert content == ""

    def test_content_match_fallback(self, tmp_path):
        """When filename does not match, falls back to content search."""
        docs_dir = tmp_path / "docs" / "go"
        docs_dir.mkdir(parents=True)
        (docs_dir / "packages.md").write_text(
            "# Package auth\nfunc Login() error"
        )
        content = read_docs_for_package(tmp_path / "docs", "internal/auth")
        # Should match because "auth" appears in the content
        assert "func Login" in content


# ---------------------------------------------------------------------------
# read_source_for_package tests
# ---------------------------------------------------------------------------


class TestReadSourceForPackage:
    """Verify source file reading logic."""

    def test_reads_source_files(self, tmp_path):
        pkg_dir = tmp_path / "internal" / "auth"
        pkg_dir.mkdir(parents=True)
        (pkg_dir / "handler.go").write_text("package auth\nfunc Login() {}")
        (pkg_dir / "README.md").write_text("# Auth")  # Not a source file

        content = read_source_for_package(tmp_path, "internal/auth")
        assert "package auth" in content
        assert "# Auth" not in content

    def test_returns_empty_for_missing_dir(self, tmp_path):
        content = read_source_for_package(tmp_path, "nonexistent/pkg")
        assert content == ""

    def test_respects_size_limit(self, tmp_path):
        """Source reading should stop when the size cap is reached."""
        pkg_dir = tmp_path / "src" / "big"
        pkg_dir.mkdir(parents=True)
        # Create files that collectively exceed the 12000 char limit
        for i in range(5):
            (pkg_dir / f"file_{i:02d}.go").write_text("x" * 5000)

        content = read_source_for_package(tmp_path, "src/big")
        # Should have stopped before reading all 5 files (25000 chars)
        assert len(content) < 25000
