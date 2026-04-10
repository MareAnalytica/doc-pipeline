#!/usr/bin/env python3
"""Clean up orphan nodes in the RAGAnything Neo4j knowledge graph.

Removes noise nodes (language primitives, ARIA attributes, DOM handlers,
generated types) and merges duplicate entities where a qualified version exists.

Usage:
    python neo4j_orphan_cleanup.py --dry-run --stages all --verbose
    python neo4j_orphan_cleanup.py --execute --stages 1,2
"""

import argparse
import os
import sys

from neo4j import GraphDatabase


class OrphanCleaner:
    def __init__(self, uri: str, password: str, verbose: bool = False):
        self.driver = GraphDatabase.driver(uri, auth=("neo4j", password))
        self.verbose = verbose
        self.total_affected = 0

    def close(self):
        self.driver.close()

    def _run_query(self, query: str, **kwargs):
        with self.driver.session() as session:
            result = session.run(query, **kwargs)
            return [record.data() for record in result]

    def run_stage(self, stage_num: int, execute: bool = False) -> int:
        dispatch = {
            1: self.stage1_noise,
            2: self.stage2_merge_duplicates,
            3: self.stage3_hydra_web_stubs,
            4: self.stage4_config_fields,
            5: self.stage5_api_types,
        }
        if stage_num not in dispatch:
            print(f"  Unknown stage: {stage_num}")
            return 0
        return dispatch[stage_num](execute=execute)

    def stage1_noise(self, execute: bool = False) -> int:
        """Stage 1: Delete noise nodes -- language primitives, ARIA, DOM handlers, generated wrappers, env vars."""
        print("\n== Stage 1: Noise Nodes ==")
        print("   Targets: language primitives, ARIA attributes, DOM/React event handlers,")
        print("            generated API client wrappers, environment variable names\n")

        condition = """
            NOT (n)--()
            AND (
                toLower(n.entity_id) IN [
                    'error', 'string', 'int', 'int64', 'float64', 'bool', 'nil',
                    'void', 'null', 'undefined', 'any', 'number', 'byte', 'rune',
                    'readonly', 'context', 'boolean', 'object'
                ]
                OR n.entity_id STARTS WITH 'Aria'
                OR n.entity_id =~ '(?i).*EventHandler<.*>'
                OR n.entity_id =~ '(?i)^on(Drag|Pointer|Mouse|Click|Change|Focus|Blur|Key|Scroll|Wheel|Touch|Animation|Transition|Composition|Clipboard|Input|Submit|Load|Error|Select|Copy|Cut|Paste|Play|Pause|Ended|Seeking).*'
                OR n.entity_id =~ '.*Options<.*>'
                OR n.entity_id =~ '.*RequestResult<.*>'
                OR n.entity_id =~ '.*ApiResponse<.*>'
                OR n.entity_id STARTS WITH 'Cb_'
                OR n.entity_id STARTS WITH 'CB_'
            )
        """

        # Dry-run query
        dry_query = f"MATCH (n) WHERE {condition} RETURN n.entity_id AS name, labels(n) AS labels ORDER BY name"
        records = self._run_query(dry_query)

        if not records:
            print("   No noise nodes found.")
            return 0

        if self.verbose:
            for r in records:
                print(f"   - {r['name']}  {r['labels']}")

        count = len(records)

        if execute:
            delete_query = f"MATCH (n) WHERE {condition} DELETE n RETURN count(n) AS deleted"
            result = self._run_query(delete_query)
            deleted = result[0]["deleted"] if result else 0
            print(f"\n   Stage 1: {deleted} nodes deleted")
            self.total_affected += deleted
            return deleted
        else:
            print(f"\n   Stage 1: {count} nodes found (dry-run)")
            self.total_affected += count
            return count

    def stage2_merge_duplicates(self, execute: bool = False) -> int:
        """Stage 2: Merge orphan duplicates into qualified connected nodes."""
        print("\n== Stage 2: Merge Duplicates ==")
        print("   Targets: orphan nodes where a qualified version (e.g., pkg.Name) exists with connections\n")

        dry_query = """
            MATCH (orphan) WHERE NOT (orphan)--()
            WITH orphan, orphan.entity_id AS short_name
            MATCH (qualified) WHERE qualified.entity_id ENDS WITH ('.' + short_name) AND (qualified)--()
            RETURN orphan.entity_id AS orphan_name, qualified.entity_id AS qualified_name
            ORDER BY orphan_name
        """
        records = self._run_query(dry_query)

        if not records:
            print("   No duplicate orphans found.")
            return 0

        if self.verbose:
            for r in records:
                print(f"   - {r['orphan_name']}  -->  {r['qualified_name']}")

        count = len(records)

        if execute:
            merge_query = """
                MATCH (orphan) WHERE NOT (orphan)--()
                WITH orphan, orphan.entity_id AS short_name
                MATCH (qualified) WHERE qualified.entity_id ENDS WITH ('.' + short_name) AND (qualified)--()
                SET qualified.description = qualified.description + '<SEP>' + coalesce(orphan.description, '')
                DELETE orphan
                RETURN count(orphan) AS merged
            """
            result = self._run_query(merge_query)
            merged = result[0]["merged"] if result else 0
            print(f"\n   Stage 2: {merged} orphan nodes merged and deleted")
            self.total_affected += merged
            return merged
        else:
            print(f"\n   Stage 2: {count} orphan-qualified pairs found (dry-run)")
            self.total_affected += count
            return count

    def stage3_hydra_web_stubs(self, execute: bool = False) -> int:
        """Stage 3: Delete Hydra-Web generated OpenAPI type stubs."""
        print("\n== Stage 3: Hydra-Web Generated Type Stubs ==")
        print("   Targets: auto-generated OpenAPI client types from hydra-web\n")

        condition = """
            NOT (n)--()
            AND (
                n.entity_id STARTS WITH 'Hydra-Web/'
                OR n.entity_id STARTS WITH 'hydra-web/'
            )
            AND (
                n.entity_id CONTAINS 'ApiV1'
                OR n.entity_id CONTAINS 'Response'
                OR n.entity_id CONTAINS 'ListJobsApi'
                OR n.entity_id CONTAINS 'GetJobApi'
            )
        """

        dry_query = f"MATCH (n) WHERE {condition} RETURN n.entity_id AS name ORDER BY name"
        records = self._run_query(dry_query)

        if not records:
            print("   No Hydra-Web stubs found.")
            return 0

        if self.verbose:
            for r in records:
                print(f"   - {r['name']}")

        count = len(records)

        if execute:
            delete_query = f"MATCH (n) WHERE {condition} DELETE n RETURN count(n) AS deleted"
            result = self._run_query(delete_query)
            deleted = result[0]["deleted"] if result else 0
            print(f"\n   Stage 3: {deleted} nodes deleted")
            self.total_affected += deleted
            return deleted
        else:
            print(f"\n   Stage 3: {count} nodes found (dry-run)")
            self.total_affected += count
            return count

    def stage4_config_fields(self, execute: bool = False) -> int:
        """Stage 4: Delete config struct field expansion nodes."""
        print("\n== Stage 4: Config Struct Field Expansions ==")
        print("   Targets: overly-granular config property entities (e.g., Config.MaxOpenConns)\n")

        condition = """
            NOT (n)--()
            AND n.entity_id =~ '.*Config\\..*'
            AND (
                n.entity_id CONTAINS 'MaxOpenConns'
                OR n.entity_id CONTAINS 'MaxIdleConns'
                OR n.entity_id CONTAINS 'TTL'
                OR n.entity_id CONTAINS 'Timeout'
                OR n.entity_id CONTAINS 'Port'
                OR n.entity_id CONTAINS 'Host'
                OR n.entity_id CONTAINS 'Password'
                OR n.entity_id CONTAINS 'Username'
                OR n.entity_id CONTAINS 'Database'
                OR n.entity_id CONTAINS 'ConnMaxLifetime'
            )
        """

        dry_query = f"MATCH (n) WHERE {condition} RETURN n.entity_id AS name ORDER BY name"
        records = self._run_query(dry_query)

        if not records:
            print("   No config field expansions found.")
            return 0

        if self.verbose:
            for r in records:
                print(f"   - {r['name']}")

        count = len(records)

        if execute:
            delete_query = f"MATCH (n) WHERE {condition} DELETE n RETURN count(n) AS deleted"
            result = self._run_query(delete_query)
            deleted = result[0]["deleted"] if result else 0
            print(f"\n   Stage 4: {deleted} nodes deleted")
            self.total_affected += deleted
            return deleted
        else:
            print(f"\n   Stage 4: {count} nodes found (dry-run)")
            self.total_affected += count
            return count

    def stage5_api_types(self, execute: bool = False) -> int:
        """Stage 5: Delete remaining generated API types."""
        print("\n== Stage 5: Remaining Generated API Types ==")
        print("   Targets: broader pattern for generated REST API types (Get/Post/Put/Delete + Response/Error/Data)\n")

        condition = """
            NOT (n)--()
            AND n.entity_id =~ '.*ApiV1.*(?:Get|Post|Put|Delete|Patch)(?:Response|Error|Data).*'
        """

        dry_query = f"MATCH (n) WHERE {condition} RETURN n.entity_id AS name ORDER BY name"
        records = self._run_query(dry_query)

        if not records:
            print("   No remaining generated API types found.")
            return 0

        if self.verbose:
            for r in records:
                print(f"   - {r['name']}")

        count = len(records)

        if execute:
            delete_query = f"MATCH (n) WHERE {condition} DELETE n RETURN count(n) AS deleted"
            result = self._run_query(delete_query)
            deleted = result[0]["deleted"] if result else 0
            print(f"\n   Stage 5: {deleted} nodes deleted")
            self.total_affected += deleted
            return deleted
        else:
            print(f"\n   Stage 5: {count} nodes found (dry-run)")
            self.total_affected += count
            return count


def main():
    parser = argparse.ArgumentParser(
        description="Clean up orphan nodes in the RAGAnything Neo4j knowledge graph.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --dry-run --stages all --verbose   Show all cleanup targets
  %(prog)s --execute --stages 1,2             Execute stages 1 and 2 only
  %(prog)s --execute --stages all             Execute all cleanup stages
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be deleted without making changes (default)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the deletions",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stage numbers (1-5) or 'all' (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show individual node names being affected",
    )

    args = parser.parse_args()

    # Parse stages
    if args.stages.strip().lower() == "all":
        stages = [1, 2, 3, 4, 5]
    else:
        try:
            stages = [int(s.strip()) for s in args.stages.split(",")]
        except ValueError:
            print(f"Error: Invalid stages value '{args.stages}'. Use comma-separated numbers (1-5) or 'all'.")
            sys.exit(1)

    # Determine mode
    execute = args.execute
    mode = "EXECUTE" if execute else "DRY-RUN"

    # Connection
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not password:
        print("Error: NEO4J_PASSWORD environment variable is required.")
        sys.exit(1)

    print(f"=== Neo4j Orphan Cleanup [{mode}] ===")
    print(f"    URI: {uri}")
    print(f"    Stages: {stages}")
    print(f"    Verbose: {args.verbose}")

    cleaner = OrphanCleaner(uri=uri, password=password, verbose=args.verbose)

    try:
        # Get baseline orphan count
        baseline = cleaner._run_query("MATCH (n) WHERE NOT (n)--() RETURN count(n) AS orphan_count")
        total = cleaner._run_query("MATCH (n) RETURN count(n) AS total_nodes")
        orphan_count = baseline[0]["orphan_count"] if baseline else 0
        total_nodes = total[0]["total_nodes"] if total else 0
        print(f"\n    Baseline: {orphan_count} orphan nodes / {total_nodes} total nodes ({100*orphan_count/total_nodes:.1f}%)")

        # Run stages
        for stage_num in stages:
            cleaner.run_stage(stage_num, execute=execute)

        # Summary
        print(f"\n{'='*50}")
        print(f"  Total nodes affected: {cleaner.total_affected}")
        if execute:
            post = cleaner._run_query("MATCH (n) WHERE NOT (n)--() RETURN count(n) AS orphan_count")
            post_total = cleaner._run_query("MATCH (n) RETURN count(n) AS total_nodes")
            post_orphan = post[0]["orphan_count"] if post else 0
            post_nodes = post_total[0]["total_nodes"] if post_total else 0
            print(f"  Orphans before: {orphan_count} / {total_nodes}")
            print(f"  Orphans after:  {post_orphan} / {post_nodes} ({100*post_orphan/post_nodes:.1f}%)")
            print(f"  Reduction:      {orphan_count - post_orphan} nodes removed")
        print(f"{'='*50}")
    finally:
        cleaner.close()


if __name__ == "__main__":
    main()
