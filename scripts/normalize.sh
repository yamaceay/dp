#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 [-n|--dry-run] [-r|--regex <prefix_regex>] <output_dir> [<prefix_regex>]" >&2
  echo "Renames files under <output_dir> (recursively) by removing the leading portion that matches <prefix_regex>" >&2
  echo "and stripping a trailing '_None' just before the extension when present." >&2
  echo "You can pass the regex via -r/--regex or as the final positional argument (backward compatible)." >&2
}

dry_run=false
prefix_regex=""
output_dir=""

# Parse flags and positional args.
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run)
      dry_run=true
      shift
      ;;
    -r|--regex)
      if [[ $# -lt 2 ]]; then
        echo "Error: -r|--regex requires an argument." >&2
        usage
        exit 1
      fi
      prefix_regex=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      if [[ -z "$output_dir" ]]; then
        output_dir=$1
      elif [[ -z "$prefix_regex" ]]; then
        prefix_regex=$1 # backward-compatible positional regex
      else
        echo "Error: unexpected argument '$1'." >&2
        usage
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ -z "$output_dir" || -z "$prefix_regex" ]]; then
  usage
  exit 1
fi

if [[ ! -d "$output_dir" ]]; then
  echo "Error: output_dir '$output_dir' is not a directory." >&2
  exit 1
fi

# Validate regex early to fail fast on invalid patterns.
if ! [[ "" =~ $prefix_regex ]]; then
  : # validation only; result unused
fi

changed=0

while IFS= read -r -d '' file; do
  dir_name=$(dirname "$file")
  base_name=$(basename "$file")

  # Match the prefix regex at the start of the filename.
  if [[ $base_name =~ ^($prefix_regex)(.*)$ ]]; then
    new_name=${BASH_REMATCH[2]}
  else
    continue
  fi

  # Strip a trailing "_None" before the extension (e.g., foo_None.jsonl -> foo.jsonl).
  if [[ $new_name =~ ^(.*)_None(\.[^.]+)$ ]]; then
    new_name="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
  fi

  # If the first two underscore-delimited parts are identical (e.g., foo_foo_bar -> foo_bar),
  # drop the first one. Handles both two-part names (foo_foo.jsonl -> foo.jsonl) and longer ones.
  if [[ $new_name =~ ^([^_]+)_([^_.]+)(.*)$ ]]; then
    first=${BASH_REMATCH[1]}
    second=${BASH_REMATCH[2]}
    rest=${BASH_REMATCH[3]}
    if [[ $first == "$second" ]]; then
      new_name="${second}${rest}"
    fi
  fi

  if [[ -z "$new_name" ]]; then
    echo "Skip: regex would remove entire filename '$base_name'." >&2
    continue
  fi

  target="$dir_name/$new_name"

  if [[ -e "$target" ]]; then
    echo "Skip: '$target' already exists; not overwriting." >&2
    continue
  fi

  if $dry_run; then
    echo "Dry run: $base_name -> $new_name"
  else
    mv "$file" "$target"
    echo "Renamed: $base_name -> $new_name"
  fi
  ((++changed)) # prefix increment keeps exit status 0 under `set -e`
done < <(find "$output_dir" -type f -print0)

if [[ $changed -eq 0 ]]; then
  echo "No files processed."
else
  if $dry_run; then
    echo "Total files that would be renamed: $changed"
  else
    echo "Total files renamed: $changed"
  fi
fi
