#!/usr/bin/env bash
set -euo pipefail

max_rows=""
sort_output=false
filter_threshold=""
declare -a files=()

usage() {
  printf "Usage:\n"
  printf "  %s --max <rows> <file1> [file2 ...]\n" "$0"
  printf "  wc -l <files...> | %s --max <rows>\n" "$0"
  printf "\nOptions:\n"
  printf "  --sort                Sort output by ratio (default: no sorting)\n"
  printf "  --filter \$THRESHOLD   Only show files with ratio below a certain threshold\n"
}

while (($# > 0)); do
  case "$1" in
    --max)
      shift
      [[ $# -gt 0 ]] || { usage; exit 1; }
      max_rows="$1"
      ;;
    --sort)
      sort_output=true
      ;;
    --filter)
      shift
      [[ $# -gt 0 ]] || { usage; exit 1; }
      filter_threshold="$1"
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      files+=("$1")
      ;;
  esac
  shift
done

[[ -n "$max_rows" ]] || { printf "Missing --max\n" >&2; usage; exit 1; }
[[ "$max_rows" =~ ^[0-9]+$ ]] || { printf "--max must be a positive integer\n" >&2; exit 1; }
(( max_rows > 0 )) || { printf "--max must be > 0\n" >&2; exit 1; }

print_ratio() {
  local count=$1 path=$2
  local ratio pad_width num_digits
  
  printf -v ratio "%.2f" "$(awk -v c="$count" -v m="$max_rows" 'BEGIN { print c / m }')"
  pad_width=${PATH_PAD:-100}
  num_digits=${#max_rows}
  printf "%s\t%-${pad_width}s\t%+${num_digits}s/%s\n" "$ratio" "$path" "$count" "$max_rows"
}

run_counting() {
  if ((${#files[@]} > 0)); then
    for f in "${files[@]}"; do
      [[ -f "$f" ]] || { printf "File not found: %s\n" "$f" >&2; exit 1; }
      count="$(wc -l < "$f")"
      if [[ "$count" -ge "$max_rows" ]]; then
        continue
      fi
      print_ratio "$count" "$f"
    done
  else
    while IFS= read -r line; do
      [[ -n "$line" ]] || continue
      count="$(awk '{print $1}' <<< "$line")"
      path="$(awk '{$1=""; sub(/^[[:space:]]+/, ""); print}' <<< "$line")"
      [[ "$count" =~ ^[0-9]+$ ]] || continue
      [[ "$path" == "total" ]] && continue
      print_ratio "$count" "$path"
    done
  fi
}

output=$(run_counting)

if [[ -n "$filter_threshold" ]]; then
  output=$(awk -v t="$filter_threshold" '$1 != 1.00 && $1 < t' <<< "$output")
fi

if $sort_output; then
  sort -k1,1n <<< "$output"
else
  cat <<< "$output"
fi
