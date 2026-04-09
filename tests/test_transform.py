"""Unit tests for doc-pipeline transform script."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Add scripts directory to path so we can import the module
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from transform import (
    INFRASTRUCTURE_ENTITIES,
    _is_infrastructure,
    qualify_name,
    _first_mention,
    transform_gomarkdoc,
    transform_helm_docs,
    transform_protoc_json,
    transform_pydoc_markdown,
    transform_typedoc_json,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
REPO = "justpay-backend"
COMMIT = "abc123"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_content(results: list[dict]) -> str:
    """Extract the content string from a single-document result list."""
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    doc = results[0]
    assert "content" in doc
    assert "metadata" in doc
    return doc["content"]


def _get_metadata(results: list[dict]) -> dict:
    """Extract metadata from a single-document result list."""
    assert len(results) == 1
    return results[0]["metadata"]


# ---------------------------------------------------------------------------
# TypeDoc JSON transformer tests
# ---------------------------------------------------------------------------

class TestTypedocTransformer:
    """Tests for the TypeDoc JSON reflection model transformer."""

    @pytest.fixture
    def results(self):
        return transform_typedoc_json(
            FIXTURES / "typedoc.json", REPO, COMMIT
        )

    def test_produces_output(self, results):
        assert len(results) == 1

    def test_metadata(self, results):
        meta = _get_metadata(results)
        assert meta["repo"] == REPO
        assert meta["commit"] == COMMIT
        assert meta["language"] == "typescript"

    def test_module_exports(self, results):
        content = _get_content(results)
        assert "Module auth" in content
        assert "justpay-backend/auth.AuthService" in content

    def test_class_with_methods(self, results):
        content = _get_content(results)
        assert "Class justpay-backend/auth.AuthService:" in content
        assert "Method login on justpay-backend/auth.AuthService" in content
        assert "LoginRequest" in content
        assert "Promise<LoginResult>" in content

    def test_constructor(self, results):
        content = _get_content(results)
        assert "Constructor of justpay-backend/auth.AuthService" in content
        assert "ZitadelClient" in content
        assert "UserRepository" in content

    def test_interface_with_fields(self, results):
        content = _get_content(results)
        assert "Interface justpay-backend/auth.LoginResult" in content
        assert "status (string)" in content
        assert "session_id (string)" in content
        assert "access_token (string)" in content

    def test_enum(self, results):
        content = _get_content(results)
        assert "Enum justpay-backend/auth.AuthStatus" in content
        assert "AUTHENTICATED" in content
        assert "MFA_REQUIRED" in content
        assert "FAILED" in content

    def test_standalone_function(self, results):
        content = _get_content(results)
        assert "Function justpay-backend/auth.createAuthHandler" in content
        assert "AuthHandlerConfig" in content
        assert "Router" in content

    def test_type_alias(self, results):
        content = _get_content(results)
        assert "Type alias justpay-backend/auth.SessionData" in content
        assert "BaseSession" in content

    def test_doc_comments(self, results):
        content = _get_content(results)
        assert "Handles authentication and session management" in content
        assert "Authenticates a user with email and password" in content

    def test_method_return_types(self, results):
        content = _get_content(results)
        # register method should also return Promise<LoginResult>
        assert content.count("Promise<LoginResult>") >= 2


# ---------------------------------------------------------------------------
# gomarkdoc Markdown transformer tests
# ---------------------------------------------------------------------------

class TestGomarkdocTransformer:
    """Tests for the gomarkdoc markdown transformer."""

    @pytest.fixture
    def results(self):
        return transform_gomarkdoc(
            FIXTURES / "packages.md", REPO, COMMIT
        )

    def test_produces_output(self, results):
        assert len(results) == 1

    def test_metadata(self, results):
        meta = _get_metadata(results)
        assert meta["repo"] == REPO
        assert meta["language"] == "go"

    def test_package_name(self, results):
        content = _get_content(results)
        assert "Package service" in content

    def test_type_declaration(self, results):
        content = _get_content(results)
        assert "Type justpay-backend/service.PaymentService" in content
        assert "Type justpay-backend/service.TenancyService" in content

    def test_constructor_function(self, results):
        content = _get_content(results)
        assert "justpay-backend/service.NewPaymentService" in content

    def test_method_signatures(self, results):
        content = _get_content(results)
        assert "Method ProcessPayment on justpay-backend/service.PaymentService" in content
        assert "Method GetPaymentStatus on justpay-backend/service.PaymentService" in content

    def test_parameter_types(self, results):
        content = _get_content(results)
        # Check that parameter types are extracted
        assert "context.Context" in content or "Context" in content

    def test_return_types(self, results):
        content = _get_content(results)
        assert "PaymentResult" in content or "PaymentStatus" in content

    def test_doc_comments(self, results):
        content = _get_content(results)
        assert "core business logic" in content or "payment processing" in content

    def test_struct_fields(self, results):
        content = _get_content(results)
        # The struct fields should be extracted
        assert "sqlx.DB" in content or "redis.Client" in content or "Fields:" in content


# ---------------------------------------------------------------------------
# pydoc-markdown transformer tests
# ---------------------------------------------------------------------------

class TestPydocMarkdownTransformer:
    """Tests for the pydoc-markdown output transformer."""

    @pytest.fixture
    def results(self):
        return transform_pydoc_markdown(
            FIXTURES / "api_routes.md", REPO, COMMIT
        )

    def test_produces_output(self, results):
        assert len(results) == 1

    def test_metadata(self, results):
        meta = _get_metadata(results)
        assert meta["repo"] == REPO
        assert meta["language"] == "python"

    def test_module_header(self, results):
        content = _get_content(results)
        assert "Module api_routes" in content

    def test_class_detection(self, results):
        content = _get_content(results)
        assert "PaymentCreateRequest" in content or "PaymentResponse" in content

    def test_function_with_route(self, results):
        content = _get_content(results)
        # Should detect FastAPI route decorators
        assert "create_payment" in content or "get_payment" in content

    def test_type_annotations(self, results):
        content = _get_content(results)
        assert "PaymentResponse" in content

    def test_pydantic_model(self, results):
        content = _get_content(results)
        # BaseModel subclass should be noted
        assert "BaseModel" in content

    def test_doc_text(self, results):
        content = _get_content(results)
        # Doc comments should flow through
        assert "payment" in content.lower()


# ---------------------------------------------------------------------------
# protoc-gen-doc JSON transformer tests
# ---------------------------------------------------------------------------

class TestProtocTransformer:
    """Tests for the protoc-gen-doc JSON transformer."""

    @pytest.fixture
    def results(self):
        return transform_protoc_json(
            FIXTURES / "proto-docs.json", REPO, COMMIT
        )

    def test_produces_output(self, results):
        assert len(results) == 1

    def test_metadata(self, results):
        meta = _get_metadata(results)
        assert meta["repo"] == REPO
        assert meta["language"] == "protobuf"

    def test_service_declaration(self, results):
        content = _get_content(results)
        assert "Service justpay-backend/payment.v1.PaymentService" in content
        assert "payment.v1" in content

    def test_rpc_methods(self, results):
        content = _get_content(results)
        assert "RPC ProcessPayment on justpay-backend/payment.v1.PaymentService" in content
        assert "accepts justpay-backend/payment.v1.PaymentRequest" in content
        assert "returns justpay-backend/payment.v1.PaymentResponse" in content

    def test_streaming_rpc(self, results):
        content = _get_content(results)
        assert "StreamPaymentUpdates" in content
        assert "server streaming" in content

    def test_message_definition(self, results):
        content = _get_content(results)
        assert "Message justpay-backend/payment.v1.PaymentRequest" in content
        assert "amount (double)" in content
        assert "currency (string)" in content
        assert "merchant_id (string)" in content

    def test_enum_definition(self, results):
        content = _get_content(results)
        assert "Enum justpay-backend/payment.v1.PaymentStatus" in content
        assert "PENDING (1)" in content
        assert "COMPLETED (2)" in content
        assert "FAILED (3)" in content
        assert "REFUNDED (4)" in content

    def test_field_descriptions(self, results):
        content = _get_content(results)
        assert "ISO 4217 currency code" in content
        assert "Idempotency key" in content

    def test_service_description(self, results):
        content = _get_content(results)
        assert "payment processing" in content.lower() or "Handles payment" in content


# ---------------------------------------------------------------------------
# helm-docs Markdown transformer tests
# ---------------------------------------------------------------------------

class TestHelmDocsTransformer:
    """Tests for the helm-docs markdown transformer."""

    @pytest.fixture
    def results(self):
        return transform_helm_docs(
            FIXTURES / "helm-readme.md", REPO, COMMIT
        )

    def test_produces_output(self, results):
        assert len(results) == 1

    def test_metadata(self, results):
        meta = _get_metadata(results)
        assert meta["repo"] == REPO
        assert meta["language"] == "helm"

    def test_chart_header(self, results):
        content = _get_content(results)
        assert "Helm chart justpay-api" in content
        assert "version 1.4.0" in content

    def test_replica_count(self, results):
        content = _get_content(results)
        assert "replicaCount" in content
        assert "Number of pod replicas" in content

    def test_image_values(self, results):
        content = _get_content(results)
        assert "image.repository" in content
        assert "image.tag" in content

    def test_resource_limits(self, results):
        content = _get_content(results)
        assert "resources.limits.cpu" in content
        assert "500m" in content

    def test_service_config(self, results):
        content = _get_content(results)
        assert "service.port" in content
        assert "8080" in content

    def test_default_values(self, results):
        content = _get_content(results)
        assert "default" in content.lower()

    def test_types_present(self, results):
        content = _get_content(results)
        assert "int" in content
        assert "string" in content
        assert "bool" in content


# ---------------------------------------------------------------------------
# Namespace qualification tests
# ---------------------------------------------------------------------------

class TestNamespaceQualification:
    """Verify that code entities are namespace-qualified and infra entities are not."""

    def test_qualify_name_code_entity(self):
        assert qualify_name("AuthService", "justpay-backend", "service") == "justpay-backend/service.AuthService"

    def test_qualify_name_infra_entity(self):
        assert qualify_name("PostgreSQL", "justpay-backend", "service") == "PostgreSQL"

    def test_qualify_name_case_insensitive_infra(self):
        # Infrastructure lookup should be case-insensitive
        assert _is_infrastructure("postgresql")
        assert _is_infrastructure("REDIS")
        assert _is_infrastructure("Kafka")

    def test_qualify_name_no_package(self):
        assert qualify_name("Foo", "my-repo", "") == "my-repo/Foo"

    def test_first_mention_code_entity(self):
        result = _first_mention("AuthService", "repo", "svc")
        assert result == "repo/svc.AuthService (AuthService)"

    def test_first_mention_infra_entity(self):
        result = _first_mention("Redis", "repo", "svc")
        assert result == "Redis"

    def test_infrastructure_set_comprehensive(self):
        """Key infrastructure names should all be in the set."""
        for name in [
            "MinIO", "Qdrant", "PostgreSQL", "Neo4j", "Zitadel", "Kafka",
            "Redis", "Docker", "Kubernetes", "Traefik", "Prometheus",
            "Grafana", "Ollama", "OpenAI",
        ]:
            assert _is_infrastructure(name), f"{name} should be infrastructure"

    def test_code_entities_not_infrastructure(self):
        """Common code names should NOT match infrastructure."""
        for name in [
            "AuthService", "PaymentService", "Client", "Handler",
            "UserRepository", "LoginRequest", "Config",
        ]:
            assert not _is_infrastructure(name), f"{name} should NOT be infrastructure"

    def test_typedoc_entities_qualified(self):
        results = transform_typedoc_json(FIXTURES / "typedoc.json", REPO, COMMIT)
        content = _get_content(results)
        assert "justpay-backend/auth.AuthService" in content
        assert "justpay-backend/auth.LoginResult" in content
        assert "justpay-backend/auth.createAuthHandler" in content
        assert "justpay-backend/auth.SessionData" in content

    def test_typedoc_infra_not_qualified(self):
        """ZitadelClient contains 'Zitadel' but the full name 'ZitadelClient' is
        NOT in the infra set, so it should appear unqualified only in type
        references (it's not a declared entity in this module)."""
        results = transform_typedoc_json(FIXTURES / "typedoc.json", REPO, COMMIT)
        content = _get_content(results)
        # ZitadelClient appears in field/param types, not as a declared entity
        assert "ZitadelClient" in content

    def test_gomarkdoc_entities_qualified(self):
        results = transform_gomarkdoc(FIXTURES / "packages.md", REPO, COMMIT)
        content = _get_content(results)
        assert "justpay-backend/service.PaymentService" in content
        assert "justpay-backend/service.TenancyService" in content
        assert "justpay-backend/service.NewPaymentService" in content

    def test_gomarkdoc_method_uses_qualified_receiver(self):
        results = transform_gomarkdoc(FIXTURES / "packages.md", REPO, COMMIT)
        content = _get_content(results)
        assert "Method ProcessPayment on justpay-backend/service.PaymentService" in content
        assert "Method CreateLease on justpay-backend/service.TenancyService" in content

    def test_protoc_entities_qualified(self):
        results = transform_protoc_json(FIXTURES / "proto-docs.json", REPO, COMMIT)
        content = _get_content(results)
        assert "justpay-backend/payment.v1.PaymentService" in content
        assert "justpay-backend/payment.v1.PaymentRequest" in content
        assert "justpay-backend/payment.v1.PaymentStatus" in content

    def test_protoc_cross_references_consistent(self):
        """Within the same repo, cross-references should use the same qualified name."""
        results = transform_protoc_json(FIXTURES / "proto-docs.json", REPO, COMMIT)
        content = _get_content(results)
        # PaymentService should appear the same way in service declaration and RPC lines
        svc_qname = "justpay-backend/payment.v1.PaymentService"
        assert content.count(svc_qname) >= 4, (
            f"Expected {svc_qname} at least 4 times (1 service + 3 RPCs), "
            f"found {content.count(svc_qname)}"
        )

    def test_pydoc_entities_qualified(self):
        results = transform_pydoc_markdown(FIXTURES / "api_routes.md", REPO, COMMIT)
        content = _get_content(results)
        assert "justpay-backend/api_routes.PaymentRouter" in content
        assert "justpay-backend/api_routes.PaymentCreateRequest" in content
        assert "justpay-backend/api_routes.PaymentResponse" in content

    def test_helm_not_qualified(self):
        """Helm chart values are config keys, not code entities -- no qualification."""
        results = transform_helm_docs(FIXTURES / "helm-readme.md", REPO, COMMIT)
        content = _get_content(results)
        # Helm values should NOT be qualified
        assert "justpay-backend/" not in content

    def test_different_repos_different_qualified_names(self):
        """The same entity name in different repos gets different qualified names."""
        q1 = qualify_name("Client", "justpay-backend", "service")
        q2 = qualify_name("Client", "hydra-brain", "models")
        assert q1 == "justpay-backend/service.Client"
        assert q2 == "hydra-brain/models.Client"
        assert q1 != q2


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    """Test the transform.py script as a CLI tool."""

    @pytest.fixture
    def docs_dir(self, tmp_path):
        """Set up a temporary docs directory with all fixture types."""
        # TypeDoc
        ts_dir = tmp_path / "ts"
        ts_dir.mkdir()
        import shutil
        shutil.copy(FIXTURES / "typedoc.json", ts_dir / "typedoc.json")

        # gomarkdoc
        go_dir = tmp_path / "go"
        go_dir.mkdir()
        shutil.copy(FIXTURES / "packages.md", go_dir / "packages.md")

        # pydoc-markdown
        py_dir = tmp_path / "python"
        py_dir.mkdir()
        shutil.copy(FIXTURES / "api_routes.md", py_dir / "api_routes.md")

        # protoc-gen-doc
        proto_dir = tmp_path / "proto"
        proto_dir.mkdir()
        shutil.copy(FIXTURES / "proto-docs.json", proto_dir / "proto-docs.json")

        # helm-docs
        helm_dir = tmp_path / "helm"
        helm_dir.mkdir()
        shutil.copy(FIXTURES / "helm-readme.md", helm_dir / "README.md")

        return tmp_path

    def test_help(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "transform.py"), "--help"],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "Transform doc tool output" in result.stdout

    def test_full_pipeline(self, docs_dir):
        result = subprocess.run(
            [
                sys.executable, str(SCRIPTS_DIR / "transform.py"),
                str(docs_dir),
                "--repo", "justpay-backend",
                "--commit", "abc123",
            ],
            capture_output=True, text=True
        )
        assert result.returncode == 0

        # Parse JSONL output
        output_lines = [
            line for line in result.stdout.strip().split("\n") if line.strip()
        ]
        assert len(output_lines) >= 5, (
            f"Expected at least 5 JSONL lines (one per doc type), got {len(output_lines)}"
        )

        # Each line should be valid JSON with content + metadata
        for line in output_lines:
            doc = json.loads(line)
            assert "content" in doc, f"Missing 'content' key in: {line[:100]}"
            assert "metadata" in doc, f"Missing 'metadata' key in: {line[:100]}"
            assert doc["metadata"]["repo"] == "justpay-backend"
            assert doc["metadata"]["commit"] == "abc123"
            assert doc["metadata"]["language"] in (
                "typescript", "go", "python", "protobuf", "helm"
            )
            # Content should be substantial
            assert len(doc["content"]) > 50, (
                f"Content too short for {doc['metadata']['language']}: "
                f"{doc['content'][:100]}"
            )

    def test_empty_directory(self, tmp_path):
        result = subprocess.run(
            [
                sys.executable, str(SCRIPTS_DIR / "transform.py"),
                str(tmp_path),
                "--repo", "empty-repo",
            ],
            capture_output=True, text=True
        )
        # Should exit 0 with a warning, not crash
        assert result.returncode == 0
        assert result.stdout.strip() == ""

    def test_partial_directory(self, tmp_path):
        """Only some doc types present -- should still work."""
        go_dir = tmp_path / "go"
        go_dir.mkdir()
        import shutil
        shutil.copy(FIXTURES / "packages.md", go_dir / "packages.md")

        result = subprocess.run(
            [
                sys.executable, str(SCRIPTS_DIR / "transform.py"),
                str(tmp_path),
                "--repo", "partial-repo",
            ],
            capture_output=True, text=True
        )
        assert result.returncode == 0
        lines = [l for l in result.stdout.strip().split("\n") if l.strip()]
        assert len(lines) == 1
        doc = json.loads(lines[0])
        assert doc["metadata"]["language"] == "go"


# ---------------------------------------------------------------------------
# Output quality / entity density tests
# ---------------------------------------------------------------------------

class TestEntityDensity:
    """Verify that output text has high entity and relationship density."""

    def test_typedoc_entity_count(self):
        results = transform_typedoc_json(
            FIXTURES / "typedoc.json", REPO, COMMIT
        )
        content = _get_content(results)
        # Should mention all key entities (qualified form)
        entities = [
            "justpay-backend/auth.AuthService",
            "justpay-backend/auth.LoginResult",
            "justpay-backend/auth.LoginRequest",
            "justpay-backend/auth.AuthStatus",
            "justpay-backend/auth.createAuthHandler",
            "justpay-backend/auth.SessionData",
            "ZitadelClient",      # appears in field types (unqualified ref)
            "UserRepository",     # appears in field types (unqualified ref)
            "AuthHandlerConfig",  # appears in param types (unqualified ref)
        ]
        found = sum(1 for e in entities if e in content)
        assert found >= 7, (
            f"Only found {found}/{len(entities)} entities in TypeDoc output"
        )

    def test_protoc_relationship_sentences(self):
        results = transform_protoc_json(
            FIXTURES / "proto-docs.json", REPO, COMMIT
        )
        content = _get_content(results)
        # Relationships should be explicit: "accepts X and returns Y"
        assert content.count("accepts") >= 3
        assert content.count("returns") >= 3

    def test_helm_structured_values(self):
        results = transform_helm_docs(
            FIXTURES / "helm-readme.md", REPO, COMMIT
        )
        content = _get_content(results)
        # Every value row should appear
        assert content.count("Value ") >= 10

    def test_no_raw_json_in_output(self):
        """Entity-rich text should not contain raw JSON fragments."""
        results = transform_typedoc_json(
            FIXTURES / "typedoc.json", REPO, COMMIT
        )
        content = _get_content(results)
        # Should not have JSON syntax
        assert '{"' not in content
        assert '"kind"' not in content
        assert '"type"' not in content

    def test_no_markdown_headers_in_output(self):
        """Output should be prose, not markdown."""
        for fixture, transformer in [
            ("packages.md", transform_gomarkdoc),
            ("api_routes.md", transform_pydoc_markdown),
            ("helm-readme.md", transform_helm_docs),
        ]:
            results = transformer(FIXTURES / fixture, REPO, COMMIT)
            if results:
                content = _get_content(results)
                # No markdown headers
                assert "\n# " not in content, (
                    f"Found markdown header in {fixture} output"
                )
                assert "\n## " not in content, (
                    f"Found markdown header in {fixture} output"
                )
