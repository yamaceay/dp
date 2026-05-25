#!/bin/bash
# Copy the MDS CSV subset needed by docs/paper/data_export.py into
# docs/paper/mds/, then regenerate all paper data CSVs.
#
# Usage (from repo root):  bash scripts/paper_data.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR/.."
SRC="$ROOT/mds"
DST="$ROOT/docs/paper/mds"
EXPORT="$ROOT/docs/paper/data_export.py"

# Scrape data_export.py for all .csv stems referenced via MDS, just like
# images.sh scrapes .typ files for image paths.
#
# Matches patterns like:
#   _load(MDS / ds / "token_level_rewriters_with_threshold_k.csv")
#   _load(MDS / "tab" / "token_level_rewriters_with_threshold_lambda.csv")
stems=$(grep 'MDS /' "$EXPORT" \
  | grep -oE '"[a-z_]+\.csv"' \
  | tr -d '"' \
  | sort -u)

if [ -z "$stems" ]; then
  echo "No CSV stems found in $EXPORT" >&2
  exit 1
fi

datasets=("db_bio" "tab")

echo "Copying MDS subset → $DST"
for stem in $stems; do
  for ds in "${datasets[@]}"; do
    src="$SRC/$ds/$stem"
    dst="$DST/$ds/$stem"
    if [ ! -f "$src" ]; then
      continue
    fi
    mkdir -p "$(dirname "$dst")"
    cp -- "$src" "$dst"
    echo "  $ds/$stem"
  done
done

echo ""
echo "Running data_export.py ..."
python3 "$EXPORT"
