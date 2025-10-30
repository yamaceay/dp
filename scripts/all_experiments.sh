#!/bin/zsh

models_dir="configs/model"
runtime_dir="configs/runtime"

cmd_tpl="python3 model.py \
  --data %s --data_in %s \
  --model %s --model_in %s \
  --runtime_in %s \
  --output jsonl %s --start_idx 0 2>&1"

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

        flags=""
        if [[ "$runtime_type" != "simple" ]]; then
          flags="--stream"
        fi
        all_runtimes | while IFS=, read -r runtime_method runtime_path; do
          if [[ "$runtime_method" == "$runtime_type"* ]]; then
            printf "$cmd_tpl\n" \
              "$dataset_name" "$dataset_path" \
              "$method_base" "$method_path" \
              "$runtime_path" \
              $flags
          fi
        done
      fi
    done < <(all_methods)
  done < <(all_datasets)
}

all_methods_runtimes

# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model manual --model_in configs/model/manual.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model petre --model_in configs/model/petre/trustpilot/greedy.yaml   --runtime_in 5,configs/runtime/k_5.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model petre --model_in configs/model/petre/trustpilot/greedy.yaml   --runtime_in 3,configs/runtime/k_3.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model petre --model_in configs/model/petre/trustpilot/greedy.yaml   --runtime_in 2,configs/runtime/k_2.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model petre --model_in configs/model/petre/trustpilot/shap.yaml   --runtime_in 5,configs/runtime/k_5.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model petre --model_in configs/model/petre/trustpilot/shap.yaml   --runtime_in 3,configs/runtime/k_3.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model petre --model_in configs/model/petre/trustpilot/shap.yaml   --runtime_in 2,configs/runtime/k_2.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model presidio --model_in configs/model/presidio.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model spacy --model_in configs/model/spacy.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model baroud --model_in configs/model/baroud.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/trustpilot/greedy.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/trustpilot/greedy.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/trustpilot/greedy.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/trustpilot/greedy.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/trustpilot/shap.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/trustpilot/shap.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/trustpilot/shap.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/trustpilot/shap.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data trustpilot --data_in data/trustpilot/sample_300.jsonl   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model manual --model_in configs/model/manual.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model petre --model_in configs/model/petre/tab/greedy.yaml   --runtime_in 5,configs/runtime/k_5.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model petre --model_in configs/model/petre/tab/greedy.yaml   --runtime_in 3,configs/runtime/k_3.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model petre --model_in configs/model/petre/tab/greedy.yaml   --runtime_in 2,configs/runtime/k_2.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model petre --model_in configs/model/petre/tab/shap.yaml   --runtime_in 5,configs/runtime/k_5.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model petre --model_in configs/model/petre/tab/shap.yaml   --runtime_in 3,configs/runtime/k_3.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model petre --model_in configs/model/petre/tab/shap.yaml   --runtime_in 2,configs/runtime/k_2.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model presidio --model_in configs/model/presidio.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model spacy --model_in configs/model/spacy.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model baroud --model_in configs/model/baroud.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy_pii.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy_pii.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy_pii.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy_pii.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy_plus.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy_plus.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy_plus.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/greedy_plus.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/shap.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/shap.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/shap.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/tab/shap.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data tab --data_in data/tab/splitted/test.json   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model manual --model_in configs/model/manual.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model petre --model_in configs/model/petre/reddit/greedy.yaml   --runtime_in 5,configs/runtime/k_5.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model petre --model_in configs/model/petre/reddit/greedy.yaml   --runtime_in 3,configs/runtime/k_3.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model petre --model_in configs/model/petre/reddit/greedy.yaml   --runtime_in 2,configs/runtime/k_2.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model petre --model_in configs/model/petre/reddit/shap.yaml   --runtime_in 5,configs/runtime/k_5.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model petre --model_in configs/model/petre/reddit/shap.yaml   --runtime_in 3,configs/runtime/k_3.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model petre --model_in configs/model/petre/reddit/shap.yaml   --runtime_in 2,configs/runtime/k_2.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model presidio --model_in configs/model/presidio.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model spacy --model_in configs/model/spacy.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model baroud --model_in configs/model/baroud.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/reddit/greedy.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/reddit/greedy.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/reddit/greedy.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/reddit/greedy.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/reddit/shap.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/reddit/shap.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/reddit/shap.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/reddit/shap.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data reddit --data_in data/reddit/train.jsonl   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model manual --model_in configs/model/manual.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model petre --model_in configs/model/petre/db_bio/greedy.yaml   --runtime_in 5,configs/runtime/k_5.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model petre --model_in configs/model/petre/db_bio/greedy.yaml   --runtime_in 3,configs/runtime/k_3.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model petre --model_in configs/model/petre/db_bio/greedy.yaml   --runtime_in 2,configs/runtime/k_2.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model petre --model_in configs/model/petre/db_bio/shap.yaml   --runtime_in 5,configs/runtime/k_5.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model petre --model_in configs/model/petre/db_bio/shap.yaml   --runtime_in 3,configs/runtime/k_3.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model petre --model_in configs/model/petre/db_bio/shap.yaml   --runtime_in 2,configs/runtime/k_2.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpprompt --model_in configs/model/dpprompt.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model presidio --model_in configs/model/presidio.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpbart --model_in configs/model/dpbart.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model spacy --model_in configs/model/spacy.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model baroud --model_in configs/model/baroud.yaml   --runtime_in simple,configs/runtime/simple.yaml   --output jsonl  --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/db_bio/greedy.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/db_bio/greedy.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/db_bio/greedy.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/db_bio/greedy.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/db_bio/shap.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/db_bio/shap.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/db_bio/shap.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/db_bio/shap.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/uniform.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpmlm --model_in configs/model/dpmlm/uniform_plus.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 250,configs/runtime/eps_250.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 25,configs/runtime/eps_25.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 100,configs/runtime/eps_100.yaml   --output jsonl --stream --start_idx 0 2>&1
# python3 model.py   --data db_bio --data_in data/db_bio/train/data-00000-of-00001.arrow   --model dpparaphrase --model_in configs/model/dpparaphrase.yaml   --runtime_in 10,configs/runtime/eps_10.yaml   --output jsonl --stream --start_idx 0 2>&1