#!/bin/zsh

models_dir="configs/model"
runtime_dir="configs/runtime"

cmd_tpl="python3 model.py \
  --data %s --data_in %s \
  --model %s --model_in %s \
  --runtime_in %s \
  --output jsonl %s --start_idx 0 2>&1 >> logs/%s_%s.log"

function all_methods() {
  find "$models_dir" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' path
  do if [[ -f "$path" ]]
  then printf '%s,%s\n' "${path}" "${path:t:r}"
  elif [[ -d "$path" ]]
  then for subpath in "$path"/*; do
    if [[ -f "$subpath" ]]; then
      printf '%s,%s\n' "${subpath}" "${path:t:r}"
    elif [[ -d "$subpath" ]]; then
      for subsubpath in "$subpath"/*; do
        if [[ -f "$subsubpath" ]]; then
          printf '%s,%s\n' "${subsubpath}" "${path:t:r}"
        fi
      done
    fi
  done
  else printf 'Not_a_file_or_directory:_ %s\n' "$path" >&2
  fi
  done
}

function all_datasets() {
  printf 'trustpilot,data/trustpilot/sample_300.jsonl\n'
  printf 'tab,data/tab/splitted/test.json\n'
  printf 'reddit,data/reddit/train.jsonl\n'
  printf 'db_bio,data/db_bio/train/data-00000-of-00001.arrow\n'
}

function all_runtimes() {
  find "$runtime_dir" -mindepth 1 -maxdepth 1 -name '*.yaml' -print0 | while IFS= read -r -d '' path; do
  if [[ -f "$path" ]]; then
    if [[ "${path:t:r}" =~ ^k_([0-9]+)$ ]]; then
      method="k_anon,${match[1]}"
    elif [[ "${path:t:r}" =~ ^eps_([0-9]+)$ ]]; then
      method="dp,${match[1]}"
    else
      method="simple,${path:t:r}"
    fi
    printf '%s,%s\n' "$method" "$path"
  else printf 'Not_a_file:_ %s\n' "$path" >&2
  fi
  done
}

function all_methods_runtimes() {
  declare -A method_runtime_map
  method_runtime_map=(
    ["petre"]="k_anon"
    ["dpprompt"]="dp"
    ["dpparaphrase"]="dp"
    ["dpbart"]="dp"
    ["dpmlm"]="dp"
    ["baroud"]="simple"
    ["spacy"]="simple"
    ["presidio"]="simple"
    ["manual"]="simple"
  )

  while IFS=, read -r dataset_name dataset_path; do
    while IFS=, read -r method_path method_base; do
      runtime_type="${method_runtime_map[$method_base]}"
      
      if [[ -n "$runtime_type" ]]; then
        if [[ "$method_path" =~ configs/model/[^/]+/([^/]+)/[^/]+\.yaml$ ]]; then
          ref_dataset_name="${match[1]}"
          if [[ "$ref_dataset_name" != "$dataset_name" ]]; then
            continue
          fi
        fi

        flags=" "
        if [[ "$runtime_type" != "simple" ]]; then
          flags="--stream"
        fi
        all_runtimes | while IFS=, read -r runtime_method runtime_path; do
          runtime_path_ext="${runtime_dir}/${runtime_path##*/}"
          if [[ "$runtime_method" == "$runtime_type"* ]]; then
            printf "$cmd_tpl\n" \
              "$dataset_name" "$dataset_path" \
              "$method_base" "$method_path" \
              "$runtime_path_ext" \
              $flags \
              "$dataset_name" "$method_base"
          fi
        done
      fi
    done < <(all_methods)
  done < <(all_datasets)
}

all_methods_runtimes