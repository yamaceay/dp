#!/bin/bash

# Usage: scripts/jobs.sh [output_file]
# Default output: scripts/jobs.table

OUTPUT_FILE="${1:-scripts/jobs.table}"

models_dir="configs/model"
runtime_dir="configs/runtime"

cmd_tpl="python3 model.py \
  --data %s --data_in %s \
  --model %s --model_in %s \
  %s \
  --output jsonl %s --start_idx 0"

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
    baroud|spacy)
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
    manual|presidio)
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

  for dataset_entry in "${dataset_entries[@]}"; do
    IFS=, read -r dataset_name dataset_path <<< "$dataset_entry"

    for method_key in "${ordered_methods[@]}"; do
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

        job_name="${dataset_name}_${method_base}"
        cmd=$(printf "$cmd_tpl" \
            "$dataset_name" "$dataset_path" \
            "$method_base" "$method_path" \
            "$runtime_args" \
            $flags)
        
        printf '%s|%s\n' "$job_name" "$cmd"
      done
    done
  done
}

all_methods_runtimes > "$OUTPUT_FILE"
echo "Generated job table: $OUTPUT_FILE" >&2
