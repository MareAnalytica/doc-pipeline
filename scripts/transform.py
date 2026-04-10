#!/usr/bin/env python3
"""Transform deterministic doc tool output into RAGAnything-friendly entity-rich text.

Takes structured output from gomarkdoc, TypeDoc, pydoc-markdown, protoc-gen-doc,
and helm-docs, then reshapes it into natural-language prose that maximizes entity
and relationship extraction by RAGAnything's LLM entity extractor.

Usage:
    python scripts/transform.py docs/ --repo justpay-backend --commit abc123

Output: JSONL (one JSON object per line) with 'content' and 'metadata' fields.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Infrastructure entities -- these merge naturally across repos and should
# NOT be namespace-qualified.  Everything else is a code entity and gets
# qualified as {repo}/{package}.{Name}.
# ---------------------------------------------------------------------------

INFRASTRUCTURE_ENTITIES: frozenset[str] = frozenset({
    # Databases
    "PostgreSQL", "MySQL", "MariaDB", "MongoDB", "CockroachDB", "SQLite",
    "DynamoDB", "Cassandra", "ScyllaDB", "ClickHouse", "TimescaleDB",
    "InfluxDB", "SurrealDB",
    # Caches / queues
    "Redis", "Memcached", "Valkey",
    "Kafka", "RabbitMQ", "NATS", "Pulsar",
    # Object / file storage
    "MinIO", "S3",
    # Graph / vector / search
    "Neo4j", "Qdrant", "Milvus", "Weaviate", "Elasticsearch", "OpenSearch",
    "Typesense", "MeiliSearch",
    # Identity / auth
    "Zitadel", "Keycloak", "Auth0", "Okta",
    # Orchestration / runtime
    "Docker", "Kubernetes", "Nomad", "Podman",
    # Reverse proxy / mesh
    "Traefik", "Nginx", "Envoy", "Istio", "Linkerd", "Caddy",
    # Observability
    "Prometheus", "Grafana", "Jaeger", "Loki", "Tempo", "OpenTelemetry",
    "Datadog", "Sentry",
    # AI / ML runtimes
    "Ollama", "OpenAI", "vLLM", "TensorRT",
    # CI / CD
    "GitHub", "GitLab", "Jenkins", "ArgoCD", "Flux",
    # Cloud providers
    "AWS", "GCP", "Azure",
    # Misc infrastructure
    "Terraform", "Vault", "Consul", "Etcd",
})

# Lower-cased version for case-insensitive lookups
_INFRA_LOWER: frozenset[str] = frozenset(n.lower() for n in INFRASTRUCTURE_ENTITIES)

# ---------------------------------------------------------------------------
# Language primitives -- never create entities or relationships for these
# ---------------------------------------------------------------------------

LANGUAGE_PRIMITIVES: frozenset[str] = frozenset({
    "error", "string", "int", "int64", "float32", "float64", "bool",
    "nil", "void", "null", "undefined", "any", "number", "byte",
    "rune", "readonly", "context", "boolean", "object",
    "Error", "String", "Context", "Promise", "Observable",
    "Record", "Partial", "Required", "Pick", "Omit",
    "unknown", "never", "void", "Array", "Map", "Set",
    "true", "false", "None", "self", "cls",
    "io.Reader", "io.Writer", "http.Handler", "http.Request",
    "http.ResponseWriter", "fmt.Stringer", "sync.Mutex",
    "time.Time", "time.Duration",
})

_PRIMITIVES_LOWER: frozenset[str] = frozenset(n.lower() for n in LANGUAGE_PRIMITIVES)


def _is_primitive(name: str) -> bool:
    """Return True if *name* is a language primitive that should be skipped."""
    return name.lower() in _PRIMITIVES_LOWER or name in LANGUAGE_PRIMITIVES


def _is_infrastructure(name: str) -> bool:
    """Return True if *name* is a well-known infrastructure entity."""
    return name.lower() in _INFRA_LOWER


def qualify_name(name: str, repo: str, package: str) -> str:
    """Return namespace-qualified entity name or bare name for infra entities.

    Code entities  -> ``{repo}/{package}.{Name}``
    Infrastructure -> ``{Name}`` (unchanged)
    """
    if _is_infrastructure(name):
        return name
    if package:
        return f"{repo}/{package}.{name}"
    return f"{repo}/{name}"


def _first_mention(name: str, repo: str, package: str) -> str:
    """Format for first mention: ``qualified (short)`` for code entities."""
    qname = qualify_name(name, repo, package)
    if qname == name:
        # Infrastructure -- no parenthetical needed
        return name
    return f"{qname} ({name})"


# ---------------------------------------------------------------------------
# Graph output helpers
# ---------------------------------------------------------------------------

def _classify_entity_type(name: str, ast_kind: str) -> str:
    """Determine entity_type from AST kind and naming conventions.

    ast_kind values: "class", "struct", "interface", "function", "method",
    "enum", "module", "package", "service", "message", "rpc", "type_alias",
    "variable", "chart", "property", "field".
    """
    if ast_kind in ("interface",):
        return "interface"
    if ast_kind in ("function", "method", "rpc"):
        return "function"
    if ast_kind in ("enum",):
        return "concept"
    if ast_kind in ("module", "package", "namespace"):
        return "module"
    if ast_kind in ("service",):
        return "service"
    if ast_kind in ("message",):
        return "model"
    if ast_kind in ("chart",):
        return "tool"
    if ast_kind in ("type_alias",):
        return "concept"
    # class/struct -- determine from naming convention
    if ast_kind in ("class", "struct"):
        upper = name.upper()
        if any(kw in upper for kw in ("SERVICE", "HANDLER", "CONTROLLER")):
            return "service"
        if any(kw in upper for kw in ("REPOSITORY", "STORE", "REPO")):
            return "repository"
        if any(kw in upper for kw in ("MODEL", "ENTITY", "SCHEMA")):
            return "model"
        return "service"
    # fallback
    return "service"


def _make_entity(entity_name: str, description: str, entity_type: str,
                 source_id: str, file_path: str) -> dict:
    """Build a canonical entity dict."""
    return {
        "entity_name": entity_name,
        "entity_data": {
            "description": description,
            "entity_type": entity_type,
            "source_id": source_id,
            "file_path": file_path,
        },
    }


def _make_relationship(source_entity: str, target_entity: str,
                       description: str, keywords: str,
                       source_id: str, weight: float = 1.0) -> dict:
    """Build a canonical relationship dict."""
    return {
        "source_entity": source_entity,
        "target_entity": target_entity,
        "relation_data": {
            "description": description,
            "keywords": keywords,
            "weight": weight,
            "source_id": source_id,
        },
    }


def _deduplicate_entities(entities: list[dict]) -> list[dict]:
    """Deduplicate entities by entity_name, keeping the first occurrence."""
    seen: set[str] = set()
    result: list[dict] = []
    for e in entities:
        name = e["entity_name"]
        if name not in seen:
            seen.add(name)
            result.append(e)
    return result


def _deduplicate_relationships(rels: list[dict]) -> list[dict]:
    """Deduplicate relationships by (source, target, keywords) tuple."""
    seen: set[tuple[str, str, str]] = set()
    result: list[dict] = []
    for r in rels:
        key = (r["source_entity"], r["target_entity"],
               r["relation_data"]["keywords"])
        if key not in seen:
            seen.add(key)
            result.append(r)
    return result


# ---------------------------------------------------------------------------
# TypeDoc JSON transformer
# ---------------------------------------------------------------------------

# TypeDoc reflection kind constants (subset we care about)
_TYPEDOC_KIND_MODULE = 2
_TYPEDOC_KIND_NAMESPACE = 4
_TYPEDOC_KIND_ENUM = 8
_TYPEDOC_KIND_ENUM_MEMBER = 16
_TYPEDOC_KIND_VARIABLE = 32
_TYPEDOC_KIND_FUNCTION = 64
_TYPEDOC_KIND_CLASS = 128
_TYPEDOC_KIND_INTERFACE = 256
_TYPEDOC_KIND_CONSTRUCTOR = 512
_TYPEDOC_KIND_PROPERTY = 1024
_TYPEDOC_KIND_METHOD = 2048
_TYPEDOC_KIND_TYPE_ALIAS = 4194304
_TYPEDOC_KIND_ACCESSOR = 262144

_TYPEDOC_KIND_NAMES = {
    _TYPEDOC_KIND_MODULE: "Module",
    _TYPEDOC_KIND_NAMESPACE: "Namespace",
    _TYPEDOC_KIND_ENUM: "Enum",
    _TYPEDOC_KIND_ENUM_MEMBER: "Enum member",
    _TYPEDOC_KIND_VARIABLE: "Variable",
    _TYPEDOC_KIND_FUNCTION: "Function",
    _TYPEDOC_KIND_CLASS: "Class",
    _TYPEDOC_KIND_INTERFACE: "Interface",
    _TYPEDOC_KIND_CONSTRUCTOR: "Constructor",
    _TYPEDOC_KIND_PROPERTY: "Property",
    _TYPEDOC_KIND_METHOD: "Method",
    _TYPEDOC_KIND_TYPE_ALIAS: "Type alias",
    _TYPEDOC_KIND_ACCESSOR: "Accessor",
}


def _typedoc_type_to_str(t: dict | None) -> str:
    """Recursively convert a TypeDoc type node into a readable string."""
    if t is None:
        return "unknown"
    kind = t.get("type", "")
    if kind == "intrinsic":
        return t.get("name", "unknown")
    if kind == "reference":
        name = t.get("name", "unknown")
        args = t.get("typeArguments", [])
        if args:
            inner = ", ".join(_typedoc_type_to_str(a) for a in args)
            return f"{name}<{inner}>"
        return name
    if kind == "array":
        elem = _typedoc_type_to_str(t.get("elementType"))
        return f"{elem}[]"
    if kind == "union":
        parts = [_typedoc_type_to_str(u) for u in t.get("types", [])]
        return " | ".join(parts)
    if kind == "intersection":
        parts = [_typedoc_type_to_str(u) for u in t.get("types", [])]
        return " & ".join(parts)
    if kind == "literal":
        val = t.get("value")
        if isinstance(val, str):
            return f'"{val}"'
        return str(val) if val is not None else "null"
    if kind == "tuple":
        elems = [_typedoc_type_to_str(e) for e in t.get("elements", [])]
        return f"[{', '.join(elems)}]"
    if kind == "reflection":
        decl = t.get("declaration", {})
        sigs = decl.get("signatures", [])
        if sigs:
            sig = sigs[0]
            params = sig.get("parameters", [])
            param_str = ", ".join(
                f"{p.get('name', '?')}: {_typedoc_type_to_str(p.get('type'))}"
                for p in params
            )
            ret = _typedoc_type_to_str(sig.get("type"))
            return f"({param_str}) => {ret}"
        children = decl.get("children", [])
        if children:
            fields = ", ".join(
                f"{c.get('name', '?')}: {_typedoc_type_to_str(c.get('type'))}"
                for c in children
            )
            return "{ " + fields + " }"
        return "object"
    if kind == "mapped":
        return "MappedType"
    if kind == "conditional":
        return "ConditionalType"
    if kind == "indexedAccess":
        obj = _typedoc_type_to_str(t.get("objectType"))
        idx = _typedoc_type_to_str(t.get("indexType"))
        return f"{obj}[{idx}]"
    if kind == "query":
        return f"typeof {_typedoc_type_to_str(t.get('queryType'))}"
    if kind == "predicate":
        return f"{t.get('name', '?')} is {_typedoc_type_to_str(t.get('targetType'))}"
    # Fallback
    return t.get("name", kind or "unknown")


def _typedoc_extract_params(sig: dict) -> str:
    """Extract parameter list from a TypeDoc signature as a readable string."""
    params = sig.get("parameters", [])
    if not params:
        return "no parameters"
    parts = []
    for p in params:
        name = p.get("name", "?")
        ptype = _typedoc_type_to_str(p.get("type"))
        parts.append(f"{name} ({ptype})")
    return ", ".join(parts)


def _typedoc_extract_return(sig: dict) -> str:
    """Extract return type from a TypeDoc signature."""
    return _typedoc_type_to_str(sig.get("type"))


def _typedoc_comment_text(node: dict) -> str:
    """Extract the summary text from a TypeDoc comment block."""
    comment = node.get("comment", {})
    summary = comment.get("summary", [])
    if not summary:
        # Older TypeDoc versions use 'shortText'
        short = comment.get("shortText", "")
        return short.strip()
    parts = []
    for part in summary:
        if part.get("kind") == "text":
            parts.append(part.get("text", ""))
        elif part.get("kind") == "code":
            parts.append(part.get("text", ""))
    return " ".join(parts).strip()


def _typedoc_walk_module(module: dict, repo: str, _seen: set[str] | None = None) -> list[str]:
    """Walk a TypeDoc module/namespace and produce entity-rich sentences."""
    if _seen is None:
        _seen = set()
    lines: list[str] = []
    children = module.get("children", [])
    if not children:
        return lines

    module_name = module.get("name", "unknown")
    kind = module.get("kind", 0)
    # For the project root (kind 1), don't use its name as the package --
    # it's just a container.  Real modules (kind 2) provide the package.
    is_root = kind == 1
    pkg = "" if is_root else module_name

    if kind in (_TYPEDOC_KIND_MODULE, _TYPEDOC_KIND_NAMESPACE, 1):
        child_names = [c.get("name", "") for c in children if c.get("name")]
        if child_names:
            if is_root:
                # Root children are modules, not code entities -- list bare names
                exports = ", ".join(child_names)
            else:
                qualified_names = [qualify_name(n, repo, pkg) for n in child_names]
                exports = ", ".join(qualified_names)
                _seen.update(qualified_names)
            lines.append(
                f"Module {module_name} in repo {repo} exports: {exports}."
            )

    def _mention(name: str) -> str:
        """Return first-mention or subsequent-mention form."""
        qname = qualify_name(name, repo, pkg)
        if qname not in _seen:
            _seen.add(qname)
            return _first_mention(name, repo, pkg)
        return qname

    for child in children:
        child_kind = child.get("kind", 0)
        child_name = child.get("name", "unknown")
        comment = _typedoc_comment_text(child)
        comment_sentence = f" {comment}" if comment else ""

        if child_kind == _TYPEDOC_KIND_CLASS:
            lines.append(f"Class {_mention(child_name)}:{comment_sentence}")
            _typedoc_walk_class_or_interface(child, lines, "Class", child_name, repo, pkg, _seen)

        elif child_kind == _TYPEDOC_KIND_INTERFACE:
            lines.append(f"Interface {_mention(child_name)}:{comment_sentence}")
            _typedoc_walk_class_or_interface(child, lines, "Interface", child_name, repo, pkg, _seen)

        elif child_kind == _TYPEDOC_KIND_FUNCTION:
            sigs = child.get("signatures", [])
            for sig in sigs:
                params = _typedoc_extract_params(sig)
                ret = _typedoc_extract_return(sig)
                sig_comment = _typedoc_comment_text(sig)
                desc = f" {sig_comment}" if sig_comment else comment_sentence
                lines.append(
                    f"Function {_mention(child_name)} accepts {params} and returns {ret}.{desc}"
                )

        elif child_kind == _TYPEDOC_KIND_ENUM:
            lines.append(f"Enum {_mention(child_name)}:{comment_sentence}")
            members = child.get("children", [])
            if members:
                vals = ", ".join(
                    f"{m.get('name', '?')} ({m.get('type', {}).get('value', '?')})"
                    if m.get("type", {}).get("value") is not None
                    else m.get("name", "?")
                    for m in members
                )
                lines.append(f"  Values: {vals}.")

        elif child_kind == _TYPEDOC_KIND_TYPE_ALIAS:
            t = _typedoc_type_to_str(child.get("type"))
            lines.append(
                f"Type alias {_mention(child_name)} is defined as {t}.{comment_sentence}"
            )

        elif child_kind == _TYPEDOC_KIND_VARIABLE:
            t = _typedoc_type_to_str(child.get("type"))
            lines.append(
                f"Variable {_mention(child_name)} has type {t}.{comment_sentence}"
            )

        elif child_kind in (_TYPEDOC_KIND_MODULE, _TYPEDOC_KIND_NAMESPACE):
            # Recurse into sub-modules / namespaces
            sub_lines = _typedoc_walk_module(child, repo, _seen)
            lines.extend(sub_lines)

    return lines


def _typedoc_walk_class_or_interface(
    node: dict, lines: list[str], kind_label: str, parent_name: str,
    repo: str = "", pkg: str = "", _seen: set[str] | None = None,
) -> None:
    """Walk members of a class or interface and append entity-rich lines."""
    if _seen is None:
        _seen = set()
    children = node.get("children", [])
    parent_qname = qualify_name(parent_name, repo, pkg)

    # Collect properties/fields
    props = [c for c in children if c.get("kind") in (_TYPEDOC_KIND_PROPERTY, _TYPEDOC_KIND_ACCESSOR)]
    if props:
        fields = []
        for p in props:
            pname = p.get("name", "?")
            ptype = _typedoc_type_to_str(p.get("type"))
            fields.append(f"{pname} ({ptype})")
        lines.append(f"  Fields: {', '.join(fields)}.")

    # Methods and constructors
    for child in children:
        ckind = child.get("kind", 0)
        cname = child.get("name", "?")

        if ckind == _TYPEDOC_KIND_CONSTRUCTOR:
            sigs = child.get("signatures", [])
            for sig in sigs:
                params = _typedoc_extract_params(sig)
                lines.append(
                    f"  Constructor of {parent_qname} accepts {params}."
                )

        elif ckind == _TYPEDOC_KIND_METHOD:
            sigs = child.get("signatures", [])
            for sig in sigs:
                params = _typedoc_extract_params(sig)
                ret = _typedoc_extract_return(sig)
                sig_comment = _typedoc_comment_text(sig)
                desc = f" {sig_comment}" if sig_comment else ""
                lines.append(
                    f"  Method {cname} on {parent_qname} accepts {params} and returns {ret}.{desc}"
                )


def transform_typedoc_json(path: Path, repo: str, commit: str) -> list[dict]:
    """Transform TypeDoc JSON reflection model into entity-rich text."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = _typedoc_walk_module(data, repo)
    if not lines:
        return []

    content = "\n".join(lines)
    return [
        {
            "content": content,
            "metadata": {
                "repo": repo,
                "commit": commit,
                "language": "typescript",
                "source_file": str(path),
            },
        }
    ]


# ---------------------------------------------------------------------------
# TypeDoc JSON graph extractor
# ---------------------------------------------------------------------------

def _make_source_id(repo: str, path: Path) -> str:
    """Build a source_id from repo and file path.

    Strips leading directories to produce a relative doc path like
    ``justpay-backend/docs/go/packages.md``.
    """
    p = str(path)
    # If the path contains a known doc-tool directory marker, use from there
    for marker in ("/ts/", "/go/", "/python/", "/proto/", "/helm/"):
        idx = p.find(marker)
        if idx != -1:
            # Include one parent directory for context (e.g. "docs/go/...")
            slash_before = p.rfind("/", 0, idx)
            if slash_before != -1:
                return f"{repo}/{p[slash_before + 1:]}"
            return f"{repo}/{p[idx + 1:]}"
    # Fallback: use just the filename
    return f"{repo}/{path.name}"


def extract_graph_typedoc_json(path: Path, repo: str, commit: str) -> dict:
    """Extract entity/relationship graph from TypeDoc JSON reflection model."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_id = _make_source_id(repo, path)
    file_path = str(path)
    entities: list[dict] = []
    relationships: list[dict] = []
    known_entities: set[str] = set()  # track entity names we've created

    def _walk_typedoc_graph(node: dict, parent_module: str = "") -> None:
        kind = node.get("kind", 0)
        name = node.get("name", "unknown")
        comment = _typedoc_comment_text(node)
        children = node.get("children", [])
        is_root = kind == 1

        # Determine the package context
        pkg = "" if is_root else (name if kind in (_TYPEDOC_KIND_MODULE, _TYPEDOC_KIND_NAMESPACE) else parent_module)

        # Module entity
        if kind in (_TYPEDOC_KIND_MODULE, _TYPEDOC_KIND_NAMESPACE) and not is_root:
            qname = qualify_name(name, repo, "")
            etype = _classify_entity_type(name, "module")
            entities.append(_make_entity(
                qname, comment or f"Module {name} in repo {repo}",
                etype, source_id, file_path,
            ))
            known_entities.add(qname)

        for child in children:
            child_kind = child.get("kind", 0)
            child_name = child.get("name", "unknown")
            child_comment = _typedoc_comment_text(child)

            if child_kind == _TYPEDOC_KIND_CLASS:
                qname = qualify_name(child_name, repo, pkg)
                etype = _classify_entity_type(child_name, "class")
                entities.append(_make_entity(
                    qname, child_comment or f"Class {child_name}",
                    etype, source_id, file_path,
                ))
                known_entities.add(qname)

                # module-exports-symbol
                if pkg:
                    mod_qname = qualify_name(pkg, repo, "")
                    relationships.append(_make_relationship(
                        mod_qname, qname,
                        f"Module {pkg} exports {child_name}",
                        "exports, module-exports-symbol",
                        source_id,
                    ))

                # Walk methods -- create relationships only
                for member in child.get("children", []):
                    mk = member.get("kind", 0)
                    mname = member.get("name", "?")
                    if mk == _TYPEDOC_KIND_METHOD:
                        relationships.append(_make_relationship(
                            qname, qname,
                            f"{child_name} has method {mname}",
                            "has-method, class-has-method",
                            source_id,
                        ))
                        # Check return types for relationships
                        for sig in member.get("signatures", []):
                            ret_type = sig.get("type")
                            if ret_type:
                                _extract_type_relationships(
                                    ret_type, qname, pkg, repo,
                                    f"{child_name}.{mname} returns",
                                    "returns, function-returns-type",
                                    source_id, relationships, known_entities,
                                )
                            # Check param types
                            for param in sig.get("parameters", []):
                                pt = param.get("type")
                                if pt:
                                    _extract_type_relationships(
                                        pt, qname, pkg, repo,
                                        f"{child_name}.{mname} accepts",
                                        "accepts, function-accepts-type",
                                        source_id, relationships, known_entities,
                                    )

            elif child_kind == _TYPEDOC_KIND_INTERFACE:
                qname = qualify_name(child_name, repo, pkg)
                entities.append(_make_entity(
                    qname, child_comment or f"Interface {child_name}",
                    "interface", source_id, file_path,
                ))
                known_entities.add(qname)

                if pkg:
                    mod_qname = qualify_name(pkg, repo, "")
                    relationships.append(_make_relationship(
                        mod_qname, qname,
                        f"Module {pkg} exports {child_name}",
                        "exports, module-exports-symbol",
                        source_id,
                    ))

                # Walk methods on interface -- relationship only
                for member in child.get("children", []):
                    mk = member.get("kind", 0)
                    mname = member.get("name", "?")
                    if mk == _TYPEDOC_KIND_METHOD:
                        relationships.append(_make_relationship(
                            qname, qname,
                            f"{child_name} has method {mname}",
                            "has-method, class-has-method",
                            source_id,
                        ))

            elif child_kind == _TYPEDOC_KIND_FUNCTION:
                qname = qualify_name(child_name, repo, pkg)
                entities.append(_make_entity(
                    qname, child_comment or f"Function {child_name}",
                    "function", source_id, file_path,
                ))
                known_entities.add(qname)

                if pkg:
                    mod_qname = qualify_name(pkg, repo, "")
                    relationships.append(_make_relationship(
                        mod_qname, qname,
                        f"Module {pkg} exports {child_name}",
                        "exports, module-exports-symbol",
                        source_id,
                    ))

                # Extract return type relationships
                for sig in child.get("signatures", []):
                    ret_type = sig.get("type")
                    if ret_type:
                        _extract_type_relationships(
                            ret_type, qname, pkg, repo,
                            f"{child_name} returns",
                            "returns, function-returns-type",
                            source_id, relationships, known_entities,
                        )
                    for param in sig.get("parameters", []):
                        pt = param.get("type")
                        if pt:
                            _extract_type_relationships(
                                pt, qname, pkg, repo,
                                f"{child_name} accepts",
                                "accepts, function-accepts-type",
                                source_id, relationships, known_entities,
                            )

            elif child_kind == _TYPEDOC_KIND_ENUM:
                qname = qualify_name(child_name, repo, pkg)
                entities.append(_make_entity(
                    qname, child_comment or f"Enum {child_name}",
                    "concept", source_id, file_path,
                ))
                known_entities.add(qname)

                if pkg:
                    mod_qname = qualify_name(pkg, repo, "")
                    relationships.append(_make_relationship(
                        mod_qname, qname,
                        f"Module {pkg} exports {child_name}",
                        "exports, module-exports-symbol",
                        source_id,
                    ))

            elif child_kind == _TYPEDOC_KIND_TYPE_ALIAS:
                # Only create entity if it's not a primitive wrapper
                type_node = child.get("type", {})
                type_str = _typedoc_type_to_str(type_node)
                if not _is_primitive(type_str):
                    qname = qualify_name(child_name, repo, pkg)
                    entities.append(_make_entity(
                        qname, child_comment or f"Type alias {child_name} = {type_str}",
                        "concept", source_id, file_path,
                    ))
                    known_entities.add(qname)

            elif child_kind in (_TYPEDOC_KIND_MODULE, _TYPEDOC_KIND_NAMESPACE):
                _walk_typedoc_graph(child, pkg)

    _walk_typedoc_graph(data)

    return {
        "entities": _deduplicate_entities(entities),
        "relationships": _deduplicate_relationships(relationships),
    }


def _extract_type_relationships(
    type_node: dict, source_qname: str, pkg: str, repo: str,
    desc_prefix: str, keywords: str, source_id: str,
    relationships: list[dict], known_entities: set[str],
) -> None:
    """Extract relationships from TypeDoc type references to known entities."""
    if type_node is None:
        return
    kind = type_node.get("type", "")
    if kind == "reference":
        ref_name = type_node.get("name", "")
        if ref_name and not _is_primitive(ref_name) and not _is_infrastructure(ref_name):
            target_qname = qualify_name(ref_name, repo, pkg)
            # Only create relationship if target is a known entity
            if target_qname in known_entities:
                relationships.append(_make_relationship(
                    source_qname, target_qname,
                    f"{desc_prefix} {ref_name}",
                    keywords, source_id,
                ))
        # Recurse into type arguments
        for arg in type_node.get("typeArguments", []):
            _extract_type_relationships(
                arg, source_qname, pkg, repo,
                desc_prefix, keywords, source_id,
                relationships, known_entities,
            )
    elif kind == "array":
        elem = type_node.get("elementType")
        if elem:
            _extract_type_relationships(
                elem, source_qname, pkg, repo,
                desc_prefix, keywords, source_id,
                relationships, known_entities,
            )
    elif kind in ("union", "intersection"):
        for sub in type_node.get("types", []):
            _extract_type_relationships(
                sub, source_qname, pkg, repo,
                desc_prefix, keywords, source_id,
                relationships, known_entities,
            )


# ---------------------------------------------------------------------------
# gomarkdoc Markdown transformer
# ---------------------------------------------------------------------------

def transform_gomarkdoc(path: Path, repo: str, commit: str) -> list[dict]:
    """Transform gomarkdoc markdown into entity-rich text.

    gomarkdoc outputs GitHub-flavored markdown with:
    - H1 for package name
    - H2 for types (## type Foo) and standalone functions (## func Bar)
    - H3 for methods (### func (f *Foo) Baz)
    - Code blocks with full Go signatures
    - Doc comments as paragraph text
    """
    text = path.read_text(encoding="utf-8")
    lines: list[str] = []
    seen: set[str] = set()

    current_package = ""
    current_type = ""

    def _mention(name: str) -> str:
        qname = qualify_name(name, repo, current_package)
        if qname not in seen:
            seen.add(qname)
            return _first_mention(name, repo, current_package)
        return qname

    # Regex patterns for gomarkdoc structure
    pkg_header = re.compile(r"^#\s+(?:package\s+)?(\S+)", re.IGNORECASE)
    type_header = re.compile(r"^##\s+type\s+(\w+)", re.IGNORECASE)
    func_header = re.compile(r"^##\s+func\s+(\w+)", re.IGNORECASE)
    method_header = re.compile(
        r"^###\s+func\s+\(?(\w+)\s+\*?(\w+)\)?\s+(\w+)", re.IGNORECASE
    )
    # Alternative method header: ### func (*Foo) Bar  or  ### func (Foo) Bar
    method_header_alt = re.compile(
        r"^###\s+func\s+\(\*?(\w+)\)\s+(\w+)", re.IGNORECASE
    )
    # Standalone func under a type: ### func NewFoo
    standalone_func_under_type = re.compile(r"^###\s+func\s+(\w+)", re.IGNORECASE)

    # Signature extraction from code blocks
    func_sig_re = re.compile(
        r"func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\(([^)]*)\)\s*(.*)"
    )

    in_code_block = False
    code_lines: list[str] = []
    pending_entity = ""
    pending_kind = ""  # "type", "func", "method"

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        # Track code blocks
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                # Process accumulated code
                code_text = "\n".join(code_lines)
                _gomarkdoc_process_code(
                    code_text, lines, current_package, current_type,
                    pending_entity, pending_kind, repo, seen
                )
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(stripped)
            continue

        # Package header
        m = pkg_header.match(stripped)
        if m:
            current_package = m.group(1)
            lines.append(f"Package {current_package} in repo {repo}:")
            continue

        # Type header
        m = type_header.match(stripped)
        if m:
            current_type = m.group(1)
            pending_entity = current_type
            pending_kind = "type"
            lines.append(f"Type {_mention(current_type)}:")
            continue

        # Standalone function header (## func Foo)
        m = func_header.match(stripped)
        if m:
            fname = m.group(1)
            pending_entity = fname
            pending_kind = "func"
            current_type = ""
            continue

        # Method header (### func (f *Foo) Bar)
        m = method_header.match(stripped)
        if m:
            receiver_type = m.group(2)
            method_name = m.group(3)
            current_type = receiver_type
            pending_entity = method_name
            pending_kind = "method"
            continue

        # Alternative method header
        m = method_header_alt.match(stripped)
        if m:
            receiver_type = m.group(1)
            method_name = m.group(2)
            current_type = receiver_type
            pending_entity = method_name
            pending_kind = "method"
            continue

        # Standalone func under type (### func NewFoo)
        m = standalone_func_under_type.match(stripped)
        if m:
            fname = m.group(1)
            pending_entity = fname
            pending_kind = "func"
            continue

        # Doc comment lines (non-empty, non-header text)
        if stripped and not stripped.startswith("#") and not stripped.startswith("|"):
            # Could be a doc comment -- append if we have a pending entity
            if pending_entity and not stripped.startswith("-"):
                clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)
                if clean and len(clean) > 3:
                    lines.append(f"  {clean}")

    if not lines:
        return []

    content = "\n".join(lines)
    return [
        {
            "content": content,
            "metadata": {
                "repo": repo,
                "commit": commit,
                "language": "go",
                "source_file": str(path),
            },
        }
    ]


def _gomarkdoc_process_code(
    code: str,
    lines: list[str],
    package: str,
    current_type: str,
    pending_entity: str,
    pending_kind: str,
    repo: str,
    seen: set[str] | None = None,
) -> None:
    """Extract function/type signatures from Go code blocks."""
    if seen is None:
        seen = set()

    def _mention(name: str) -> str:
        qname = qualify_name(name, repo, package)
        if qname not in seen:
            seen.add(qname)
            return _first_mention(name, repo, package)
        return qname

    # Type definitions: type Foo struct { ... }
    type_match = re.search(r"type\s+(\w+)\s+(struct|interface)", code)
    if type_match:
        name = type_match.group(1)
        kind = type_match.group(2)
        # Extract fields from struct
        if kind == "struct":
            fields = re.findall(r"(\w+)\s+([\w.*\[\]]+)", code)
            if fields:
                field_str = ", ".join(
                    f"{fname} ({ftype})" for fname, ftype in fields
                    if fname not in ("type", "struct", "interface", name)
                )
                if field_str:
                    lines.append(f"  Fields: {field_str}.")
        return

    # Function signatures
    func_matches = re.finditer(
        r"func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\(([^)]*)\)\s*(.*)", code
    )
    for fm in func_matches:
        receiver_var = fm.group(1)
        receiver_type = fm.group(2)
        func_name = fm.group(3)
        params_raw = fm.group(4).strip()
        returns_raw = fm.group(5).strip()

        # Parse parameters
        if params_raw:
            params = _go_parse_params(params_raw)
            param_str = f"accepts {params}"
        else:
            param_str = "accepts no parameters"

        # Parse return type
        if returns_raw:
            returns_raw = returns_raw.strip("() ")
            ret_str = f"returns {returns_raw}"
        else:
            ret_str = "returns nothing"

        if receiver_type:
            recv_qname = qualify_name(receiver_type, repo, package)
            lines.append(
                f"  Method {func_name} on {recv_qname} {param_str} and {ret_str}."
            )
        else:
            lines.append(f"Function {_mention(func_name)} {param_str} and {ret_str}.")


def _go_parse_params(params_raw: str) -> str:
    """Parse Go function parameters into readable form."""
    # Handle multi-line params
    params_raw = re.sub(r"\s+", " ", params_raw).strip()
    if not params_raw:
        return "no parameters"

    parts = []
    # Split on commas, but respect nested types
    depth = 0
    current = ""
    for ch in params_raw:
        if ch in ("(", "[", "{"):
            depth += 1
            current += ch
        elif ch in (")", "]", "}"):
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())

    parsed = []
    for part in parts:
        tokens = part.rsplit(None, 1)
        if len(tokens) == 2:
            parsed.append(f"{tokens[0].strip()} ({tokens[1].strip()})")
        else:
            parsed.append(part)

    return ", ".join(parsed)


# ---------------------------------------------------------------------------
# gomarkdoc graph extractor
# ---------------------------------------------------------------------------

def extract_graph_gomarkdoc(path: Path, repo: str, commit: str) -> dict:
    """Extract entity/relationship graph from gomarkdoc markdown."""
    text = path.read_text(encoding="utf-8")
    source_id = _make_source_id(repo, path)
    file_path = str(path)
    entities: list[dict] = []
    relationships: list[dict] = []
    known_entities: set[str] = set()

    current_package = ""
    current_type = ""

    # Regex patterns (same as prose transformer)
    pkg_header = re.compile(r"^#\s+(?:package\s+)?(\S+)", re.IGNORECASE)
    type_header = re.compile(r"^##\s+type\s+(\w+)", re.IGNORECASE)
    func_header = re.compile(r"^##\s+func\s+(\w+)", re.IGNORECASE)
    method_header = re.compile(
        r"^###\s+func\s+\(?(\w+)\s+\*?(\w+)\)?\s+(\w+)", re.IGNORECASE
    )
    method_header_alt = re.compile(
        r"^###\s+func\s+\(\*?(\w+)\)\s+(\w+)", re.IGNORECASE
    )
    standalone_func_under_type = re.compile(r"^###\s+func\s+(\w+)", re.IGNORECASE)

    in_code_block = False
    code_lines: list[str] = []
    pending_entity = ""
    pending_kind = ""

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        # Track code blocks
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "\n".join(code_lines)
                # Extract struct field references from code blocks
                _gomarkdoc_extract_graph_from_code(
                    code_text, current_package, current_type,
                    pending_entity, pending_kind, repo, source_id, file_path,
                    entities, relationships, known_entities,
                )
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(stripped)
            continue

        # Package header
        m = pkg_header.match(stripped)
        if m:
            current_package = m.group(1)
            pkg_qname = qualify_name(current_package, repo, "")
            entities.append(_make_entity(
                pkg_qname, f"Go package {current_package} in repo {repo}",
                "module", source_id, file_path,
            ))
            known_entities.add(pkg_qname)
            continue

        # Type header
        m = type_header.match(stripped)
        if m:
            current_type = m.group(1)
            pending_entity = current_type
            pending_kind = "type"
            continue

        # Standalone function header
        m = func_header.match(stripped)
        if m:
            fname = m.group(1)
            pending_entity = fname
            pending_kind = "func"
            current_type = ""
            # Create function entity
            qname = qualify_name(fname, repo, current_package)
            if not _is_primitive(fname):
                entities.append(_make_entity(
                    qname, f"Function {fname} in package {current_package}",
                    "function", source_id, file_path,
                ))
                known_entities.add(qname)
                # module-exports-symbol
                if current_package:
                    pkg_qname = qualify_name(current_package, repo, "")
                    relationships.append(_make_relationship(
                        pkg_qname, qname,
                        f"Package {current_package} exports {fname}",
                        "exports, module-exports-symbol",
                        source_id,
                    ))
            continue

        # Method header (### func (f *Foo) Bar)
        m = method_header.match(stripped)
        if m:
            receiver_type = m.group(2)
            method_name = m.group(3)
            current_type = receiver_type
            pending_entity = method_name
            pending_kind = "method"
            # class-has-method relationship
            parent_qname = qualify_name(receiver_type, repo, current_package)
            relationships.append(_make_relationship(
                parent_qname, parent_qname,
                f"{receiver_type} has method {method_name}",
                "has-method, class-has-method",
                source_id,
            ))
            continue

        # Alternative method header
        m = method_header_alt.match(stripped)
        if m:
            receiver_type = m.group(1)
            method_name = m.group(2)
            current_type = receiver_type
            pending_entity = method_name
            pending_kind = "method"
            parent_qname = qualify_name(receiver_type, repo, current_package)
            relationships.append(_make_relationship(
                parent_qname, parent_qname,
                f"{receiver_type} has method {method_name}",
                "has-method, class-has-method",
                source_id,
            ))
            continue

        # Standalone func under type
        m = standalone_func_under_type.match(stripped)
        if m:
            fname = m.group(1)
            pending_entity = fname
            pending_kind = "func"
            qname = qualify_name(fname, repo, current_package)
            if not _is_primitive(fname):
                entities.append(_make_entity(
                    qname, f"Function {fname} in package {current_package}",
                    "function", source_id, file_path,
                ))
                known_entities.add(qname)
            continue

    return {
        "entities": _deduplicate_entities(entities),
        "relationships": _deduplicate_relationships(relationships),
    }


def _gomarkdoc_extract_graph_from_code(
    code: str, package: str, current_type: str,
    pending_entity: str, pending_kind: str,
    repo: str, source_id: str, file_path: str,
    entities: list[dict], relationships: list[dict],
    known_entities: set[str],
) -> None:
    """Extract graph entities/relationships from Go code blocks."""
    # Type definitions: type Foo struct { ... } or type Foo interface { ... }
    type_match = re.search(r"type\s+(\w+)\s+(struct|interface)", code)
    if type_match:
        name = type_match.group(1)
        kind = type_match.group(2)
        qname = qualify_name(name, repo, package)
        if kind == "interface":
            etype = "interface"
        else:
            etype = _classify_entity_type(name, "struct")
        entities.append(_make_entity(
            qname, f"Go {kind} {name} in package {package}",
            etype, source_id, file_path,
        ))
        known_entities.add(qname)
        # module-exports-symbol
        if package:
            pkg_qname = qualify_name(package, repo, "")
            relationships.append(_make_relationship(
                pkg_qname, qname,
                f"Package {package} exports {name}",
                "exports, module-exports-symbol",
                source_id,
            ))
        # Extract struct fields for type references
        if kind == "struct":
            fields = re.findall(r"(\w+)\s+([\w.*\[\]]+)", code)
            for fname, ftype in fields:
                if fname in ("type", "struct", "interface", name):
                    continue
                # Clean the type and check if it's a reference
                clean_type = ftype.strip("*[]")
                if clean_type and not _is_primitive(clean_type):
                    target_qname = qualify_name(clean_type, repo, package)
                    if target_qname in known_entities:
                        relationships.append(_make_relationship(
                            qname, target_qname,
                            f"{name} references {clean_type} via field {fname}",
                            "references, field-type",
                            source_id,
                        ))
        return

    # Function signatures
    func_matches = re.finditer(
        r"func\s+(?:\((\w+)\s+\*?(\w+)\)\s+)?(\w+)\(([^)]*)\)\s*(.*)", code
    )
    for fm in func_matches:
        receiver_type = fm.group(2)
        func_name = fm.group(3)
        returns_raw = fm.group(5).strip().strip("() ")

        if receiver_type:
            # Method -- relationship only
            parent_qname = qualify_name(receiver_type, repo, package)
            relationships.append(_make_relationship(
                parent_qname, parent_qname,
                f"{receiver_type} has method {func_name}",
                "has-method, class-has-method",
                source_id,
            ))
            # Check return type for reference
            if returns_raw and not _is_primitive(returns_raw):
                clean_ret = returns_raw.strip("*[]").split(",")[0].strip()
                if clean_ret and not _is_primitive(clean_ret):
                    target_qname = qualify_name(clean_ret, repo, package)
                    if target_qname in known_entities:
                        relationships.append(_make_relationship(
                            parent_qname, target_qname,
                            f"{receiver_type}.{func_name} returns {clean_ret}",
                            "returns, function-returns-type",
                            source_id,
                        ))
        else:
            # Standalone function -- entity already created from header
            if func_name and not _is_primitive(func_name):
                qname = qualify_name(func_name, repo, package)
                if returns_raw and not _is_primitive(returns_raw):
                    clean_ret = returns_raw.strip("*[]").split(",")[0].strip()
                    if clean_ret and not _is_primitive(clean_ret):
                        target_qname = qualify_name(clean_ret, repo, package)
                        if target_qname in known_entities:
                            relationships.append(_make_relationship(
                                qname, target_qname,
                                f"{func_name} returns {clean_ret}",
                                "returns, function-returns-type",
                                source_id,
                            ))


# ---------------------------------------------------------------------------
# pydoc-markdown transformer
# ---------------------------------------------------------------------------

def transform_pydoc_markdown(path: Path, repo: str, commit: str) -> list[dict]:
    """Transform pydoc-markdown output into entity-rich text.

    pydoc-markdown emits GitHub-flavored markdown with:
    - H1/H2 for module names
    - H2/H3 for classes, functions
    - Inline code signatures with decorators
    - Docstrings as paragraph text
    """
    text = path.read_text(encoding="utf-8")
    lines: list[str] = []
    seen: set[str] = set()

    current_module = path.stem
    current_class = ""

    def _mention(name: str) -> str:
        qname = qualify_name(name, repo, current_module)
        if qname not in seen:
            seen.add(qname)
            return _first_mention(name, repo, current_module)
        return qname

    # Regex patterns
    module_header = re.compile(r"^#+\s+(?:module\s+)?(\S+)", re.IGNORECASE)
    class_header = re.compile(r"^#+\s+(?:class\s+)?(\w+)\s*(?:\(([^)]*)\))?")
    func_header = re.compile(r"^#+\s+(?:def\s+)?(\w+)")

    # Signature patterns in code blocks
    class_sig = re.compile(r"class\s+(\w+)\s*(?:\(([^)]*)\))?:")
    func_sig = re.compile(r"def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(.+?))?:")
    decorator_re = re.compile(r"@(\w+(?:\.\w+)*)\s*(?:\(([^)]*)\))?")

    in_code_block = False
    code_lines: list[str] = []
    pending_decorators: list[str] = []

    lines.append(f"Module {current_module} in repo {repo}:")

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        # Track code blocks
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "\n".join(code_lines)
                _pydoc_process_code(
                    code_text, lines, current_module, current_class,
                    pending_decorators, repo, seen
                )
                code_lines = []
                pending_decorators = []
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(stripped)
            # Capture decorators
            dm = decorator_re.match(stripped)
            if dm:
                dec_name = dm.group(1)
                dec_args = dm.group(2)
                if dec_args:
                    pending_decorators.append(f"@{dec_name}({dec_args})")
                else:
                    pending_decorators.append(f"@{dec_name}")
            continue

        # Headers -- detect class/function definitions
        if stripped.startswith("#"):
            # Check for class
            # Pattern: ## class Foo or ## Foo(BaseClass)
            cm = re.match(r"^#+\s+(?:class\s+)?(\w+)\s*(?:Objects)?$", stripped)
            if cm:
                name = cm.group(1)
                # Heuristic: capitalized name is likely a class
                if name[0].isupper():
                    current_class = name
                    lines.append(f"Class {_mention(name)}:")
                    continue

            fm = re.match(r"^#+\s+(?:def\s+)?(\w+)$", stripped)
            if fm:
                name = fm.group(1)
                if name[0].islower() or name.startswith("_"):
                    if current_class:
                        class_qname = qualify_name(current_class, repo, current_module)
                        lines.append(f"  Method {name} on {class_qname}:")
                    else:
                        lines.append(f"Function {_mention(name)}:")
                continue

        # Doc text (non-header, non-code paragraphs)
        if stripped and not stripped.startswith("#") and not stripped.startswith("|"):
            clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)
            # Filter out very short or noisy lines
            if clean and len(clean) > 5 and not clean.startswith("---"):
                lines.append(f"  {clean}")

    if len(lines) <= 1:
        return []

    content = "\n".join(lines)
    return [
        {
            "content": content,
            "metadata": {
                "repo": repo,
                "commit": commit,
                "language": "python",
                "source_file": str(path),
            },
        }
    ]


def _pydoc_process_code(
    code: str,
    lines: list[str],
    module: str,
    current_class: str,
    decorators: list[str],
    repo: str = "",
    seen: set[str] | None = None,
) -> None:
    """Extract class and function definitions from Python code blocks."""
    if seen is None:
        seen = set()

    def _mention(name: str) -> str:
        qname = qualify_name(name, repo, module)
        if qname not in seen:
            seen.add(qname)
            return _first_mention(name, repo, module)
        return qname

    # Class definitions
    cm = re.search(r"class\s+(\w+)\s*(?:\(([^)]*)\))?:", code)
    if cm:
        name = cm.group(1)
        bases = cm.group(2)
        name_q = qualify_name(name, repo, module)
        if bases:
            lines.append(f"  {name_q} inherits from {bases}.")
        return

    # Function/method definitions
    fm = re.search(r"def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(.+?))?:", code)
    if fm:
        func_name = fm.group(1)
        params_raw = fm.group(2).strip()
        ret_type = fm.group(3)

        # Parse parameters
        if params_raw:
            params = _python_parse_params(params_raw)
            param_str = f"accepts {params}"
        else:
            param_str = "accepts no parameters"

        ret_str = f"returns {ret_type.strip()}" if ret_type else "returns None"

        # Handle decorators for route info
        route_info = ""
        for dec in decorators:
            if any(verb in dec.lower() for verb in (".get", ".post", ".put", ".delete", ".patch")):
                route_info = f" (route: {dec})"
                break

        if current_class and func_name != "__init__":
            class_qname = qualify_name(current_class, repo, module)
            lines.append(
                f"  Method {func_name} on {class_qname} {param_str} and {ret_str}.{route_info}"
            )
        elif func_name == "__init__":
            lines.append(f"  Constructor {param_str}.")
        else:
            lines.append(
                f"Function {_mention(func_name)} {param_str} and {ret_str}.{route_info}"
            )


def _python_parse_params(params_raw: str) -> str:
    """Parse Python function parameters into readable form."""
    params_raw = re.sub(r"\s+", " ", params_raw).strip()
    parts = []
    depth = 0
    current = ""
    for ch in params_raw:
        if ch in ("(", "[", "{"):
            depth += 1
            current += ch
        elif ch in (")", "]", "}"):
            depth -= 1
            current += ch
        elif ch == "," and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())

    parsed = []
    for part in parts:
        part = part.strip()
        if part == "self" or part == "cls":
            continue
        # Handle default values: name: Type = default
        m = re.match(r"(\w+)\s*:\s*([^=]+?)(?:\s*=\s*(.+))?$", part)
        if m:
            pname = m.group(1)
            ptype = m.group(2).strip()
            default = m.group(3)
            entry = f"{pname} ({ptype})"
            if default:
                entry += f", default {default.strip()}"
            parsed.append(entry)
        elif "=" in part:
            name_val = part.split("=", 1)
            parsed.append(f"{name_val[0].strip()} (default {name_val[1].strip()})")
        elif part.startswith("*"):
            parsed.append(part)
        else:
            parsed.append(part)

    if not parsed:
        return "no parameters"
    return ", ".join(parsed)


# ---------------------------------------------------------------------------
# pydoc-markdown graph extractor
# ---------------------------------------------------------------------------

def extract_graph_pydoc_markdown(path: Path, repo: str, commit: str) -> dict:
    """Extract entity/relationship graph from pydoc-markdown output."""
    text = path.read_text(encoding="utf-8")
    source_id = _make_source_id(repo, path)
    file_path = str(path)
    entities: list[dict] = []
    relationships: list[dict] = []
    known_entities: set[str] = set()

    current_module = path.stem
    current_class = ""

    # Create module entity
    mod_qname = qualify_name(current_module, repo, "")
    entities.append(_make_entity(
        mod_qname, f"Python module {current_module} in repo {repo}",
        "module", source_id, file_path,
    ))
    known_entities.add(mod_qname)

    in_code_block = False
    code_lines: list[str] = []

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        # Track code blocks
        if stripped.startswith("```"):
            if in_code_block:
                in_code_block = False
                code_text = "\n".join(code_lines)
                _pydoc_extract_graph_from_code(
                    code_text, current_module, current_class,
                    repo, source_id, file_path,
                    entities, relationships, known_entities,
                )
                code_lines = []
            else:
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(stripped)
            continue

        # Headers
        if stripped.startswith("#"):
            cm = re.match(r"^#+\s+(?:class\s+)?(\w+)\s*(?:Objects)?$", stripped)
            if cm:
                name = cm.group(1)
                if name[0].isupper():
                    current_class = name
                    qname = qualify_name(name, repo, current_module)
                    etype = _classify_entity_type(name, "class")
                    entities.append(_make_entity(
                        qname, f"Class {name} in module {current_module}",
                        etype, source_id, file_path,
                    ))
                    known_entities.add(qname)
                    # module-exports-symbol
                    relationships.append(_make_relationship(
                        mod_qname, qname,
                        f"Module {current_module} exports {name}",
                        "exports, module-exports-symbol",
                        source_id,
                    ))
                    continue

            fm = re.match(r"^#+\s+(?:def\s+)?(\w+)$", stripped)
            if fm:
                name = fm.group(1)
                if name[0].islower() or name.startswith("_"):
                    if current_class:
                        # Method -- relationship only
                        class_qname = qualify_name(current_class, repo, current_module)
                        relationships.append(_make_relationship(
                            class_qname, class_qname,
                            f"{current_class} has method {name}",
                            "has-method, class-has-method",
                            source_id,
                        ))
                    else:
                        # Standalone function
                        qname = qualify_name(name, repo, current_module)
                        if not _is_primitive(name):
                            entities.append(_make_entity(
                                qname, f"Function {name} in module {current_module}",
                                "function", source_id, file_path,
                            ))
                            known_entities.add(qname)
                            relationships.append(_make_relationship(
                                mod_qname, qname,
                                f"Module {current_module} exports {name}",
                                "exports, module-exports-symbol",
                                source_id,
                            ))

    return {
        "entities": _deduplicate_entities(entities),
        "relationships": _deduplicate_relationships(relationships),
    }


def _pydoc_extract_graph_from_code(
    code: str, module: str, current_class: str,
    repo: str, source_id: str, file_path: str,
    entities: list[dict], relationships: list[dict],
    known_entities: set[str],
) -> None:
    """Extract graph entities/relationships from Python code blocks."""
    # Class definitions with inheritance
    cm = re.search(r"class\s+(\w+)\s*(?:\(([^)]*)\))?:", code)
    if cm:
        name = cm.group(1)
        bases = cm.group(2)
        qname = qualify_name(name, repo, module)
        if bases:
            for base in bases.split(","):
                base = base.strip()
                if base and not _is_primitive(base):
                    base_qname = qualify_name(base, repo, module)
                    relationships.append(_make_relationship(
                        qname, base_qname,
                        f"{name} inherits from {base}",
                        "inherits, class-inherits-from",
                        source_id,
                    ))
        return

    # Function/method signatures -- extract return type relationships
    fm = re.search(r"def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*(.+?))?:", code)
    if fm:
        func_name = fm.group(1)
        ret_type = fm.group(3)
        if func_name in ("__init__", "self", "cls"):
            return
        if ret_type:
            ret_type = ret_type.strip()
            if not _is_primitive(ret_type):
                if current_class:
                    source_qname = qualify_name(current_class, repo, module)
                else:
                    source_qname = qualify_name(func_name, repo, module)
                target_qname = qualify_name(ret_type, repo, module)
                if target_qname in known_entities:
                    relationships.append(_make_relationship(
                        source_qname, target_qname,
                        f"{func_name} returns {ret_type}",
                        "returns, function-returns-type",
                        source_id,
                    ))


# ---------------------------------------------------------------------------
# protoc-gen-doc JSON transformer
# ---------------------------------------------------------------------------

def transform_protoc_json(path: Path, repo: str, commit: str) -> list[dict]:
    """Transform protoc-gen-doc JSON into entity-rich text.

    protoc-gen-doc JSON has structure:
    {
      "files": [
        {
          "name": "foo.proto",
          "package": "com.example",
          "services": [...],
          "messages": [...],
          "enums": [...]
        }
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines: list[str] = []
    files = data.get("files", [])

    # Handle case where top-level is a list
    if isinstance(data, list):
        files = data

    seen: set[str] = set()

    for proto_file in files:
        fname = proto_file.get("name", "unknown")
        package = proto_file.get("package", "")
        pkg_label = f" in package {package}" if package else ""
        # Use the proto package (e.g. "payment.v1") as the namespace package
        ns_pkg = package or fname.rsplit("/", 1)[0] if "/" in fname else ""

        def _mention(name: str) -> str:
            qname = qualify_name(name, repo, ns_pkg)
            if qname not in seen:
                seen.add(qname)
                return _first_mention(name, repo, ns_pkg)
            return qname

        # Services
        for svc in proto_file.get("services", []):
            svc_name = svc.get("name", "unknown")
            desc = svc.get("description", "")
            desc_str = f" {desc}" if desc else ""
            lines.append(f"Service {_mention(svc_name)}{pkg_label} in repo {repo}:{desc_str}")

            for method in svc.get("methods", []):
                method_name = method.get("name", "unknown")
                req_type = method.get("requestType", "unknown")
                resp_type = method.get("responseType", "unknown")
                req_streaming = method.get("requestStreaming", False)
                resp_streaming = method.get("responseStreaming", False)
                method_desc = method.get("description", "")

                streaming_info = ""
                if req_streaming:
                    streaming_info += " with client streaming"
                if resp_streaming:
                    streaming_info += " with server streaming"

                svc_qname = qualify_name(svc_name, repo, ns_pkg)
                desc_info = f" {method_desc}" if method_desc else ""
                lines.append(
                    f"  RPC {method_name} on {svc_qname} accepts {_mention(req_type)} and returns "
                    f"{_mention(resp_type)}{streaming_info}.{desc_info}"
                )

        # Messages
        for msg in proto_file.get("messages", []):
            msg_name = msg.get("name", "unknown")
            desc = msg.get("description", "")
            desc_str = f" {desc}" if desc else ""
            lines.append(f"Message {_mention(msg_name)}{pkg_label}:{desc_str}")

            fields = msg.get("fields", [])
            if fields:
                field_parts = []
                for field in fields:
                    field_name = field.get("name", "?")
                    field_type = field.get("type", "?")
                    field_label = field.get("label", "")
                    field_desc = field.get("description", "")

                    entry = f"{field_name} ({field_type}"
                    if field_label and field_label != "optional":
                        entry += f", {field_label}"
                    entry += ")"
                    if field_desc:
                        entry += f" - {field_desc}"
                    field_parts.append(entry)
                lines.append(f"  Fields: {', '.join(field_parts)}.")

            # Nested enums within messages
            for enum in msg.get("enums", []):
                _protoc_process_enum(enum, lines, pkg_label, repo, ns_pkg, seen)

        # Top-level enums
        for enum in proto_file.get("enums", []):
            _protoc_process_enum(enum, lines, pkg_label, repo, ns_pkg, seen)

    if not lines:
        return []

    content = "\n".join(lines)
    return [
        {
            "content": content,
            "metadata": {
                "repo": repo,
                "commit": commit,
                "language": "protobuf",
                "source_file": str(path),
            },
        }
    ]


def _protoc_process_enum(
    enum: dict, lines: list[str], pkg_label: str,
    repo: str = "", ns_pkg: str = "", seen: set[str] | None = None,
) -> None:
    """Process a protobuf enum definition."""
    if seen is None:
        seen = set()
    enum_name = enum.get("name", "unknown")
    desc = enum.get("description", "")
    desc_str = f" {desc}" if desc else ""

    qname = qualify_name(enum_name, repo, ns_pkg)
    if qname not in seen:
        seen.add(qname)
        display = _first_mention(enum_name, repo, ns_pkg)
    else:
        display = qname
    lines.append(f"Enum {display}{pkg_label}:{desc_str}")

    values = enum.get("values", [])
    if values:
        val_parts = []
        for v in values:
            vname = v.get("name", "?")
            vnumber = v.get("number", "?")
            val_parts.append(f"{vname} ({vnumber})")
        lines.append(f"  Values: {', '.join(val_parts)}.")


# ---------------------------------------------------------------------------
# protoc-gen-doc graph extractor
# ---------------------------------------------------------------------------

def extract_graph_protoc_json(path: Path, repo: str, commit: str) -> dict:
    """Extract entity/relationship graph from protoc-gen-doc JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    source_id = _make_source_id(repo, path)
    file_path = str(path)
    entities: list[dict] = []
    relationships: list[dict] = []
    known_entities: set[str] = set()

    files = data.get("files", [])
    if isinstance(data, list):
        files = data

    for proto_file in files:
        package = proto_file.get("package", "")
        fname = proto_file.get("name", "unknown")
        ns_pkg = package or (fname.rsplit("/", 1)[0] if "/" in fname else "")

        # Services
        for svc in proto_file.get("services", []):
            svc_name = svc.get("name", "unknown")
            desc = svc.get("description", "")
            svc_qname = qualify_name(svc_name, repo, ns_pkg)
            entities.append(_make_entity(
                svc_qname, desc or f"gRPC service {svc_name}",
                "service", source_id, file_path,
            ))
            known_entities.add(svc_qname)

            for method in svc.get("methods", []):
                method_name = method.get("name", "unknown")
                req_type = method.get("requestType", "unknown")
                resp_type = method.get("responseType", "unknown")
                method_desc = method.get("description", "")

                # service-has-rpc
                relationships.append(_make_relationship(
                    svc_qname, svc_qname,
                    f"{svc_name} has RPC {method_name}",
                    "has-rpc, service-has-rpc",
                    source_id,
                ))

                # rpc-accepts-message (if request type is not a primitive)
                if not _is_primitive(req_type):
                    req_qname = qualify_name(req_type, repo, ns_pkg)
                    relationships.append(_make_relationship(
                        svc_qname, req_qname,
                        f"RPC {method_name} on {svc_name} accepts {req_type}",
                        "accepts, rpc-accepts-message",
                        source_id,
                    ))

                # rpc-returns-message
                if not _is_primitive(resp_type):
                    resp_qname = qualify_name(resp_type, repo, ns_pkg)
                    relationships.append(_make_relationship(
                        svc_qname, resp_qname,
                        f"RPC {method_name} on {svc_name} returns {resp_type}",
                        "returns, rpc-returns-message",
                        source_id,
                    ))

        # Messages
        for msg in proto_file.get("messages", []):
            msg_name = msg.get("name", "unknown")
            desc = msg.get("description", "")
            msg_qname = qualify_name(msg_name, repo, ns_pkg)
            entities.append(_make_entity(
                msg_qname, desc or f"Protobuf message {msg_name}",
                "model", source_id, file_path,
            ))
            known_entities.add(msg_qname)

            # Nested enums
            for enum in msg.get("enums", []):
                _protoc_extract_graph_enum(
                    enum, repo, ns_pkg, source_id, file_path,
                    entities, known_entities,
                )

        # Top-level enums
        for enum in proto_file.get("enums", []):
            _protoc_extract_graph_enum(
                enum, repo, ns_pkg, source_id, file_path,
                entities, known_entities,
            )

    return {
        "entities": _deduplicate_entities(entities),
        "relationships": _deduplicate_relationships(relationships),
    }


def _protoc_extract_graph_enum(
    enum: dict, repo: str, ns_pkg: str,
    source_id: str, file_path: str,
    entities: list[dict], known_entities: set[str],
) -> None:
    """Extract a protobuf enum as a concept entity."""
    enum_name = enum.get("name", "unknown")
    desc = enum.get("description", "")
    qname = qualify_name(enum_name, repo, ns_pkg)
    values = enum.get("values", [])
    val_names = ", ".join(v.get("name", "?") for v in values) if values else ""
    description = desc or f"Protobuf enum {enum_name}"
    if val_names:
        description += f" with values: {val_names}"
    entities.append(_make_entity(
        qname, description, "concept", source_id, file_path,
    ))
    known_entities.add(qname)


# ---------------------------------------------------------------------------
# helm-docs Markdown transformer
# ---------------------------------------------------------------------------

def transform_helm_docs(path: Path, repo: str, commit: str) -> list[dict]:
    """Transform helm-docs markdown into entity-rich text.

    helm-docs outputs markdown with:
    - H1 with chart name
    - A metadata section with version, appVersion, type
    - Values tables in markdown table format with columns:
      Key | Type | Default | Description
    """
    text = path.read_text(encoding="utf-8")
    lines: list[str] = []

    chart_name = ""
    chart_version = ""
    app_version = ""

    # Extract chart name from H1 (strip any trailing badge markdown)
    h1 = re.search(r"^#\s+(\S+)", text, re.MULTILINE)
    if h1:
        chart_name = h1.group(1).strip()

    # Extract version info -- prefer badge format (Version-X.Y.Z) over plain
    ver_badge = re.search(r"Version[:-]\s*([0-9][0-9A-Za-z._-]*)", text)
    if ver_badge:
        chart_version = ver_badge.group(1).rstrip(")")

    appver_badge = re.search(r"AppVersion[:-]\s*([0-9][0-9A-Za-z._-]*)", text)
    if appver_badge:
        app_version = appver_badge.group(1).rstrip(")")

    header = f"Helm chart {chart_name or path.stem}"
    if chart_version:
        header += f" version {chart_version}"
    if app_version:
        header += f" (appVersion {app_version})"
    header += f" in repo {repo}:"
    lines.append(header)

    # Extract description if present
    desc_match = re.search(
        r"(?:^|\n)(?:>|)\s*(.+?)(?:\n\n|\n#|\n\|)", text
    )

    # Parse values tables
    # helm-docs tables: | Key | Type | Default | Description |
    table_rows = re.findall(
        r"^\|\s*`?([^|`]+)`?\s*\|\s*([^|]+)\s*\|\s*`?([^|`]*)`?\s*\|\s*([^|]*)\s*\|",
        text,
        re.MULTILINE,
    )

    for row in table_rows:
        key = row[0].strip().strip("`")
        vtype = row[1].strip()
        default = row[2].strip().strip("`")
        description = row[3].strip()

        # Skip header and separator rows
        if key.lower() == "key" or key.startswith("-"):
            continue
        if all(c in "-: " for c in key):
            continue

        entry = f"  Value {key} ({vtype}"
        if default:
            entry += f", default {default}"
        entry += ")"
        if description:
            entry += f": {description}"
        entry += "."
        lines.append(entry)

    if len(lines) <= 1:
        # Fallback: look for simpler key-value patterns
        kv_pattern = re.findall(
            r"^\|\s*([^|]+)\s*\|\s*([^|]+)\s*\|", text, re.MULTILINE
        )
        for kv in kv_pattern:
            key = kv[0].strip().strip("`")
            value = kv[1].strip()
            if key.lower() == "key" or key.startswith("-") or all(c in "-: " for c in key):
                continue
            lines.append(f"  Value {key}: {value}.")

    if len(lines) <= 1:
        return []

    content = "\n".join(lines)
    return [
        {
            "content": content,
            "metadata": {
                "repo": repo,
                "commit": commit,
                "language": "helm",
                "source_file": str(path),
            },
        }
    ]


# ---------------------------------------------------------------------------
# helm-docs graph extractor
# ---------------------------------------------------------------------------

def extract_graph_helm_docs(path: Path, repo: str, commit: str) -> dict:
    """Extract entity/relationship graph from helm-docs markdown."""
    text = path.read_text(encoding="utf-8")
    source_id = _make_source_id(repo, path)
    file_path = str(path)
    entities: list[dict] = []

    chart_name = ""
    chart_version = ""

    h1 = re.search(r"^#\s+(\S+)", text, re.MULTILINE)
    if h1:
        chart_name = h1.group(1).strip()

    ver_badge = re.search(r"Version[:-]\s*([0-9][0-9A-Za-z._-]*)", text)
    if ver_badge:
        chart_version = ver_badge.group(1).rstrip(")")

    name = chart_name or path.stem
    qname = qualify_name(name, repo, "")
    desc = f"Helm chart {name}"
    if chart_version:
        desc += f" version {chart_version}"
    desc += f" in repo {repo}"

    entities.append(_make_entity(
        qname, desc, "tool", source_id, file_path,
    ))

    return {
        "entities": _deduplicate_entities(entities),
        "relationships": [],
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform doc tool output into RAGAnything-friendly entity-rich text"
    )
    parser.add_argument("docs_dir", help="Directory containing generated docs")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument("--commit", default="unknown", help="Commit SHA")
    parser.add_argument(
        "--output-format",
        choices=["prose", "graph", "dual"],
        default="prose",
        help="Output format: prose (default, JSONL text), graph (entity/relationship JSON), dual (both)",
    )
    args = parser.parse_args()

    docs = Path(args.docs_dir)
    if not docs.is_dir():
        print(f"Error: {docs} is not a directory", file=sys.stderr)
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Graph output mode
    # -----------------------------------------------------------------------
    if args.output_format in ("graph", "dual"):
        graph_results: dict[str, list] = {"entities": [], "relationships": []}

        # TypeDoc JSON
        typedoc_path = docs / "ts" / "typedoc.json"
        if typedoc_path.exists():
            g = extract_graph_typedoc_json(typedoc_path, args.repo, args.commit)
            graph_results["entities"].extend(g["entities"])
            graph_results["relationships"].extend(g["relationships"])

        # gomarkdoc markdown
        go_dir = docs / "go"
        if go_dir.is_dir():
            for md in sorted(go_dir.glob("*.md")):
                g = extract_graph_gomarkdoc(md, args.repo, args.commit)
                graph_results["entities"].extend(g["entities"])
                graph_results["relationships"].extend(g["relationships"])

        # pydoc-markdown
        python_dir = docs / "python"
        if python_dir.is_dir():
            for md in sorted(python_dir.glob("*.md")):
                g = extract_graph_pydoc_markdown(md, args.repo, args.commit)
                graph_results["entities"].extend(g["entities"])
                graph_results["relationships"].extend(g["relationships"])

        # protoc-gen-doc JSON
        proto_path = docs / "proto" / "proto-docs.json"
        if proto_path.exists():
            g = extract_graph_protoc_json(proto_path, args.repo, args.commit)
            graph_results["entities"].extend(g["entities"])
            graph_results["relationships"].extend(g["relationships"])

        # helm-docs markdown
        helm_dir = docs / "helm"
        if helm_dir.is_dir():
            for md in sorted(helm_dir.glob("*.md")):
                g = extract_graph_helm_docs(md, args.repo, args.commit)
                graph_results["entities"].extend(g["entities"])
                graph_results["relationships"].extend(g["relationships"])

        # Final deduplication across all sources
        graph_results["entities"] = _deduplicate_entities(graph_results["entities"])
        graph_results["relationships"] = _deduplicate_relationships(graph_results["relationships"])

        if args.output_format == "graph":
            if not graph_results["entities"]:
                print("Warning: no graph entities produced from input", file=sys.stderr)
                sys.exit(0)
            print(json.dumps(graph_results, ensure_ascii=False))
            return

        # For "dual", print graph JSON on first line, then prose JSONL below
        print(json.dumps(graph_results, ensure_ascii=False))

    # -----------------------------------------------------------------------
    # Prose output mode (also used for dual's second half)
    # -----------------------------------------------------------------------
    results: list[dict] = []

    # TypeDoc JSON
    typedoc_path = docs / "ts" / "typedoc.json"
    if typedoc_path.exists():
        results.extend(transform_typedoc_json(typedoc_path, args.repo, args.commit))

    # gomarkdoc markdown
    go_dir = docs / "go"
    if go_dir.is_dir():
        for md in sorted(go_dir.glob("*.md")):
            results.extend(transform_gomarkdoc(md, args.repo, args.commit))

    # pydoc-markdown
    python_dir = docs / "python"
    if python_dir.is_dir():
        for md in sorted(python_dir.glob("*.md")):
            results.extend(transform_pydoc_markdown(md, args.repo, args.commit))

    # protoc-gen-doc JSON
    proto_path = docs / "proto" / "proto-docs.json"
    if proto_path.exists():
        results.extend(transform_protoc_json(proto_path, args.repo, args.commit))

    # helm-docs markdown
    helm_dir = docs / "helm"
    if helm_dir.is_dir():
        for md in sorted(helm_dir.glob("*.md")):
            results.extend(transform_helm_docs(md, args.repo, args.commit))

    if not results:
        print("Warning: no documents produced from input", file=sys.stderr)
        sys.exit(0)

    # Output JSONL
    for doc in results:
        print(json.dumps(doc, ensure_ascii=False))


if __name__ == "__main__":
    main()
