#!/bin/bash

used_list() {
  rg 'csv\(' docs/thesis -g '*.typ' |
    sed -En 's|.*csv\("([^"]+\.csv)".*|\1|p' |
    grep -E '^(\./)?(\.\./)?data/.*\.csv$' |
    while IFS= read -r p; do
      echo "docs/thesis/${p#../}"
    done |
    sort -u
}

existing_list() {
  find docs/thesis/data -maxdepth 1 -name '*.csv' -print | sort -u
}

echo "USED"
used_list

echo

echo "MISSING"
comm -23 <(used_list) <(existing_list)

echo

echo "UNUSED"
comm -13 <(used_list) <(existing_list)
