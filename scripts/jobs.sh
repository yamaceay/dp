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
      echo "__EPS_FIXED__"
      ;;
    manual|presidio|spacy)
      echo ""
      ;;
    *)
      echo ""
      ;;
  esac
}

function runtime_specs_from_model_config() {
  local model_config_path="$1"
  python3 - "$model_config_path" <<'PY'
import sys
import yaml

path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f) or {}

has_params = "params" in cfg
params = cfg.get("params") if isinstance(cfg, dict) else None
if params is None:
    print("HAS\t0")
    sys.exit(0)
if not isinstance(params, dict):
    raise ValueError("params must be a mapping")

def norm(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        out = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("params entries must be strings")
            out.append(item)
        return out
    raise ValueError("params entries must be a string or list of strings")

def zero_pad(tokens):
    if not tokens:
        return []
    max_len = max(len(t) for t in tokens)
    return [t.zfill(max_len) for t in tokens]

def emit(prefix, base, name, pats):
    padded = zero_pad(pats)
    for pat in padded:
        print(f"{prefix}\t{base}/{name}_{pat}.yaml")

print("HAS\t1" if has_params else "HAS\t0")

emit("EPS", "configs/runtime/dp", "eps", norm(params.get("epsilon")))
emit("ARG", "configs/runtime/pii_confidence", "lambda", norm(params.get("lambdas")))
emit("ARG", "configs/runtime/risk_tolerance", "rho", norm(params.get("rhos")))
emit("ARG", "configs/runtime/k_anon", "k", norm(params.get("ks")))
PY
}

function expand_singleton_glob_or_fail() {
  local pattern="$1"
  local matches
  matches=$(compgen -G "$pattern" || true)
  if [[ -z "$matches" ]]; then
    echo "No matches for runtime pattern: $pattern" >&2
    exit 1
  fi
  printf '%s\n' $matches
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

  IFS=',' read -ra filter_datasets_arr <<< "$FILTER_DATASETS"
  IFS=',' read -ra filter_methods_arr <<< "$FILTER_METHODS"

  for dataset_entry in "${dataset_entries[@]}"; do
    IFS=, read -r dataset_name dataset_path <<< "$dataset_entry"

    if [[ -n "$FILTER_DATASETS" ]]; then
      match=0
      for fd in "${filter_datasets_arr[@]}"; do
        [[ "$dataset_name" == "$fd" ]] && match=1 && break
      done
      [[ $match -eq 0 ]] && continue
    fi

    for method_key in "${ordered_methods[@]}"; do
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

        method_unique_config="${method_path##*/}"
        method_unique_config="${method_unique_config%.yaml}"

        has_params=0
        eps_patterns=()
        arg_patterns=()
        while IFS=$'\t' read -r tag val; do
          [[ -z "$tag" ]] && continue
          if [[ "$tag" == "HAS" ]]; then
            [[ "$val" == "1" ]] && has_params=1
            continue
          fi
          if [[ "$tag" == "EPS" ]]; then
            eps_patterns+=("$val")
            continue
          fi
          if [[ "$tag" == "ARG" ]]; then
            arg_patterns+=("$val")
            continue
          fi
        done < <(runtime_specs_from_model_config "$method_path")

        if [[ $has_params -eq 1 ]]; then
          if [[ ${#eps_patterns[@]} -gt 0 ]]; then
            # Collect all eps files first to compute padding
            all_eps_files=()
            for eps_pattern in "${eps_patterns[@]}"; do
              while IFS= read -r eps_path; do
                all_eps_files+=("$eps_path")
              done < <(expand_singleton_glob_or_fail "$eps_pattern")
            done
            
            # Find max length of epsilon values for zero-padding
            max_eps_len=0
            for eps_path in "${all_eps_files[@]}"; do
              eps_name=$(basename "$eps_path")
              eps_name="${eps_name%.yaml}"
              eps_val="${eps_name#eps_}"
              (( ${#eps_val} > max_eps_len )) && max_eps_len=${#eps_val}
            done
            
            for eps_path in "${all_eps_files[@]}"; do
              eps_name=$(basename "$eps_path")
              eps_name="${eps_name%.yaml}"
              eps_val="${eps_name#eps_}"
              eps_val_padded=$(printf "%0${max_eps_len}s" "$eps_val" | tr ' ' '0')
              eps_name_padded="eps_${eps_val_padded}"

              if [[ "$method_unique_config" == "$method_base" ]]; then
                job_name="${dataset_name}_${method_base}_${eps_name_padded}"
              else
                job_name="${dataset_name}_${method_base}_${method_unique_config}_${eps_name_padded}"
              fi

              runtime_args="--runtime_in $eps_path"
              if [[ ${#arg_patterns[@]} -gt 0 ]]; then
                runtime_args="$runtime_args ${arg_patterns[*]}"
              fi

              cmd=$(printf "$cmd_tpl" \
                  "$dataset_name" "$dataset_path" \
                  "$method_base" "$method_path" \
                  "$runtime_args" \
                  "$job_name")

              printf '%s|%s\n' "$job_name" "$cmd"
            done
            continue
          fi

          runtime_args=""
          if [[ ${#arg_patterns[@]} -gt 0 ]]; then
            runtime_args="--runtime_in ${arg_patterns[*]}"
          fi

          if [[ "$method_unique_config" == "$method_base" ]]; then
            job_name="${dataset_name}_${method_base}"
          else
            job_name="${dataset_name}_${method_base}_${method_unique_config}"
          fi

          cmd=$(printf "$cmd_tpl" \
              "$dataset_name" "$dataset_path" \
              "$method_base" "$method_path" \
              "$runtime_args" \
              "$job_name")

          printf '%s|%s\n' "$job_name" "$cmd"
          continue
        fi

        runtime_args=$(runtime_args_for_method "$method_base")
        if [[ "$runtime_args" == "__EPS_FIXED__" ]]; then
          eps_files=()
          for eps_path in "$runtime_dir/dp"/eps_*.yaml; do
            [[ -f "$eps_path" ]] && eps_files+=("$eps_path")
          done
          
          # Find max length of epsilon values for zero-padding
          max_len=0
          for eps_path in "${eps_files[@]}"; do
            eps_name=$(basename "$eps_path")
            eps_name="${eps_name%.yaml}"
            eps_val="${eps_name#eps_}"
            (( ${#eps_val} > max_len )) && max_len=${#eps_val}
          done
          
          for eps_path in "${eps_files[@]}"; do
            eps_name=$(basename "$eps_path")
            eps_name="${eps_name%.yaml}"
            eps_val="${eps_name#eps_}"
            eps_val_padded=$(printf "%0${max_len}s" "$eps_val" | tr ' ' '0')
            eps_name_padded="eps_${eps_val_padded}"

            if [[ "$method_unique_config" == "$method_base" ]]; then
              job_name="${dataset_name}_${method_base}_${eps_name_padded}"
            else
              job_name="${dataset_name}_${method_base}_${method_unique_config}_${eps_name_padded}"
            fi

            cmd=$(printf "$cmd_tpl" \
                "$dataset_name" "$dataset_path" \
                "$method_base" "$method_path" \
                "--runtime_in $eps_path" \
                "$job_name")

            printf '%s|%s\n' "$job_name" "$cmd"
          done
          continue
        fi

        if [[ "$method_unique_config" == "$method_base" ]]; then
          job_name="${dataset_name}_${method_base}"
        else
          job_name="${dataset_name}_${method_base}_${method_unique_config}"
        fi
        cmd=$(printf "$cmd_tpl" \
            "$dataset_name" "$dataset_path" \
            "$method_base" "$method_path" \
            "$runtime_args" \
            "$job_name")
        
        printf '%s|%s\n' "$job_name" "$cmd"
      done
    done
  done
}

all_methods_runtimes > "$OUTPUT_FILE"
echo "Generated job table: $OUTPUT_FILE" >&2
