#!/usr/bin/env bash
# Render all defense animations to 1080p MP4.
set -e
cd "$(dirname "$0")"

SCENES=(
  "scene_01_token_risk.py:TokenRisk"
  "scene_02_budget_allocation.py:BudgetAllocation"
  "scene_03_k_stopping.py:KStopping"
  "scene_04_rho_stopping.py:RhoStopping"
)

for entry in "${SCENES[@]}"; do
  file="${entry%%:*}"
  scene="${entry##*:}"
  echo "=== Rendering $scene from $file ==="
  manim -qh --media_dir ./media "$file" "$scene"
done

echo ""
echo "Done. Videos are in ./media/videos/"
