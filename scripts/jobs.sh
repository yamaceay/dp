#!/bin/bash

# Usage: scripts/jobs.sh [--datasets d1,d2,...] [--methods m1,m2,...] [output_file]
# Default output: scripts/jobs.table

OUTPUT_FILE=""
FILTER_DATASETS=""
FILTER_METHODS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets=*)
      FILTER_DATASETS="${1#*=}"
      shift
      ;;
    --methods=*)
      FILTER_METHODS="${1#*=}"
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--datasets d1,d2,...] [--methods m1,m2,...] [output_file]"
      echo "  --datasets: Filter by dataset names (comma-separated, e.g., reddit,tab)"
      echo "  --methods: Filter by method names (comma-separated, e.g., dpmlm,risk)"
      echo "  output_file: Output file path (default: scripts/jobs.table)"
      exit 0
      ;;
    *)
      OUTPUT_FILE="$1"
      shift
      ;;
  esac
done

OUTPUT_FILE="${OUTPUT_FILE:-scripts/jobs.table}"

models_dir="configs/model"
runtime_dir="configs/runtime"

cmd_tpl="python model.py \
  --data %s --data_in %s \
  --model %s --model_in %s \
  %s \
  --output jsonl --unique_name %s %s"

function all_methods() {
  find "$models_dir" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' path
  do 
    # Extract basename without extension (bash equivalent of zsh's :t:r)
    base=$(basename "$path")
    base_no_ext="${base%.yaml}"
    
    if [[ -f "$path" ]]; then
      printf '%s,%s\n' "${path}" "$base_no_ext"
    elif [[ -d "$path" ]]; then
      for subpath in "$path"/*; do
        if [[ -f "$subpath" ]]; then
          printf '%s,%s\n' "${subpath}" "$base_no_ext"
        elif [[ -d "$subpath" ]]; then
          for subsubpath in "$subpath"/*; do
            if [[ -f "$subsubpath" ]]; then
              printf '%s,%s\n' "${subsubpath}" "$base_no_ext"
            fi
          done
        fi
      done
    else 
      printf 'Not_a_file_or_directory:_ %s\n' "$path" >&2
    fi
  done
}

function all_datasets() {
  printf 'tab,data/TAB/splitted/test.json\n'
  printf 'reddit,data/reddit/reddit.jsonl\n'
  # printf 'trustpilot,data/trustpilot/sample_300.jsonl\n'
  # printf 'db_bio,data/db_bio/train/data-00000-of-00001.arrow\n'
}

function runtime_args_for_method() {
  case "$1" in
    baroud)
      echo "--runtime_in $runtime_dir/pii_confidence/lambda_*.yaml"
      ;;
    risk)
      echo "--runtime_in $runtime_dir/risk_tolerance/rho_*.yaml"
      ;;
    petre)
      echo "--runtime_in $runtime_dir/k_anon/k_*.yaml"
      ;;
    dpprompt|dpparaphrase|dpbart|dpmlm)
      echo "--runtime_in $runtime_dir/dp/eps_*.yaml"
      ;;
    manual|presidio|spacy)
      echo ""
      ;;
    *)
      echo ""
      ;;
  esac
}

function all_methods_runtimes() {
  dataset_entries=()
  while IFS= read -r line; do
    dataset_entries+=("$line")
  done < <(all_datasets)

  method_entries=()
  while IFS= read -r line; do
    method_entries+=("$line")
  done < <(all_methods)

  ordered_methods=("baroud" "risk" "spacy" "presidio" "manual" "petre" "dpprompt" "dpparaphrase" "dpbart" "dpmlm")

  # Parse filter arrays
  IFS=',' read -ra filter_datasets_arr <<< "$FILTER_DATASETS"
  IFS=',' read -ra filter_methods_arr <<< "$FILTER_METHODS"

  for dataset_entry in "${dataset_entries[@]}"; do
    IFS=, read -r dataset_name dataset_path <<< "$dataset_entry"

    # Filter datasets if specified
    if [[ -n "$FILTER_DATASETS" ]]; then
      match=0
      for fd in "${filter_datasets_arr[@]}"; do
        [[ "$dataset_name" == "$fd" ]] && match=1 && break
      done
      [[ $match -eq 0 ]] && continue
    fi

    for method_key in "${ordered_methods[@]}"; do
      # Filter methods if specified
      if [[ -n "$FILTER_METHODS" ]]; then
        match=0
        for fm in "${filter_methods_arr[@]}"; do
          [[ "$method_key" == "$fm" ]] && match=1 && break
        done
        [[ $match -eq 0 ]] && continue
      fi
      for method_entry in "${method_entries[@]}"; do
        IFS=, read -r method_path method_base <<< "$method_entry"
        [[ "$method_base" != "$method_key" ]] && continue

        if [[ "$method_path" =~ configs/model/[^/]+/([^/]+)/[^/]+\.yaml$ ]]; then
          ref_dataset_name="${BASH_REMATCH[1]}"
          if [[ "$ref_dataset_name" != "$dataset_name" ]]; then
            continue
          fi
        fi

        runtime_args=$(runtime_args_for_method "$method_base")

        flags=" "
        case "$method_base" in
          risk|petre|dpprompt|dpparaphrase|dpbart|dpmlm)
            flags="--stream"
            ;;
        esac

        method_unique_config="${method_path##*/}"
        method_unique_config="${method_unique_config%.yaml}"
        job_name="${dataset_name}_${method_base}_${method_unique_config}"
        cmd=$(printf "$cmd_tpl" \
            "$dataset_name" "$dataset_path" \
            "$method_base" "$method_path" \
            "$runtime_args" \
            "$job_name" \
            $flags)
        
        printf '%s|%s\n' "$job_name" "$cmd"
      done
    done
  done
}

all_methods_runtimes > "$OUTPUT_FILE"
echo "Generated job table: $OUTPUT_FILE" >&2
