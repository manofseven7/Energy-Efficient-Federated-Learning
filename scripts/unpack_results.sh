#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ARCHIVE="results/archive/raw_experiment_runs.tar.gz"
if [[ ! -f "$ARCHIVE" ]]; then
  echo "Missing $ARCHIVE" >&2
  exit 1
fi
echo "Restoring raw experiment logs from $ARCHIVE ..."
tar -xzf "$ARCHIVE"
echo "Done. Raw runs restored under results/."
