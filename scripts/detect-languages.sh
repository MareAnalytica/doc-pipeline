#!/usr/bin/env bash
#
# detect-languages.sh
#
# Detect which languages and tool ecosystems are present in the current
# repository by checking for well-known marker files. Designed to run
# inside the doc-pipeline GitHub Actions workflow but can also be
# executed standalone for local testing.
#
# Usage:
#   ./scripts/detect-languages.sh          # prints KEY=VALUE lines
#   source <(./scripts/detect-languages.sh) # imports into current shell
#
# When GITHUB_OUTPUT is set (inside Actions), the results are also
# appended there so subsequent workflow steps can reference them.

set -euo pipefail

HAS_GO=false
HAS_TS=false
HAS_PYTHON=false
HAS_PROTO=false
HAS_HELM=false

[ -f "go.mod" ] && HAS_GO=true
[ -f "tsconfig.json" ] && HAS_TS=true
([ -f "pyproject.toml" ] || [ -f "setup.py" ] || [ -f "requirements.txt" ]) && HAS_PYTHON=true
find . -name "*.proto" -maxdepth 3 2>/dev/null | grep -q . && HAS_PROTO=true
[ -f "Chart.yaml" ] && HAS_HELM=true

# Print to stdout for local use / debugging
echo "has_go=$HAS_GO"
echo "has_ts=$HAS_TS"
echo "has_python=$HAS_PYTHON"
echo "has_proto=$HAS_PROTO"
echo "has_helm=$HAS_HELM"

# Append to GITHUB_OUTPUT when running inside Actions
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "has_go=$HAS_GO"     >> "$GITHUB_OUTPUT"
  echo "has_ts=$HAS_TS"     >> "$GITHUB_OUTPUT"
  echo "has_python=$HAS_PYTHON" >> "$GITHUB_OUTPUT"
  echo "has_proto=$HAS_PROTO"   >> "$GITHUB_OUTPUT"
  echo "has_helm=$HAS_HELM"     >> "$GITHUB_OUTPUT"
fi
