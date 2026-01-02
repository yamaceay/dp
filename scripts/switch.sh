#!/bin/bash


MODELS_FILE=""
if [[ "$1" == "local" ]]; then
    MODELS_FILE="configs/environment/local.yaml"
elif [[ "$1" == "hpc" ]]; then
    MODELS_FILE="configs/environment/hpc.yaml"
fi

if [[ -z "$MODELS_FILE" ]]; then
    echo "Usage: $0 {local|hpc}"
    exit 1
fi

if [[ ! -f "$MODELS_FILE" ]]; then
    echo "Error: $MODELS_FILE not found"
    exit 1
fi

echo "Using models file: $MODELS_FILE"

CONFIG_DIRS=("configs/model" "configs/lrec" "configs/experiments" "configs/tri_training")

yaml_top_value() {
    local key="$1"
    awk -F': *' -v key="$key" '$1==key { print $2; exit }' "$MODELS_FILE"
}

yaml_section_value() {
    local section="$1"
    local key="$2"
    awk -F': *' -v section="$section" -v want_key="$key" '
        function trim(s) { gsub(/^[ \t]+|[ \t]+$/, "", s); return s }
        /^[^ \t]/ {
            top = $1
            top = trim(top)
            in_section = (top == section)
            next
        }
        {
            if (!in_section) next
            k = trim($1)
            if (k == want_key) {
                print $2
                exit
            }
        }
    ' "$MODELS_FILE"
}

GPT2_PARAPHRASER=$(yaml_top_value "gpt2-paraphraser")
REDDIT_TRI_PIPELINE=$(yaml_section_value "reddit" "tri_pipeline")
REDDIT_UNIVERSAL_TRI_PIPELINE=$(yaml_section_value "reddit" "universal_tri_pipeline")
REDDIT_RESULT_IN=$(yaml_section_value "reddit" "result_in")
REDDIT_RISK_IN=$(yaml_section_value "reddit" "risk_in")
TAB_TRI_PIPELINE=$(yaml_section_value "tab" "tri_pipeline")
TAB_PII_ANNOTATOR=$(yaml_section_value "tab" "pii_annotator")
TAB_UNIVERSAL_TRI_PIPELINE=$(yaml_section_value "tab" "universal_tri_pipeline")
TAB_RESULT_IN=$(yaml_section_value "tab" "result_in")
TAB_RISK_IN=$(yaml_section_value "tab" "risk_in")

if [[ -z "$GPT2_PARAPHRASER" ]]; then
    echo "Error: failed to extract gpt2-paraphraser from $MODELS_FILE" >&2
    exit 1
fi
if [[ -z "$REDDIT_TRI_PIPELINE" ]]; then
    echo "Error: failed to extract reddit.tri_pipeline from $MODELS_FILE" >&2
    exit 1
fi
if [[ -z "$TAB_TRI_PIPELINE" ]]; then
    echo "Error: failed to extract tab.tri_pipeline from $MODELS_FILE" >&2
    exit 1
fi

echo "Extracted values:"
echo "  gpt2-paraphraser: $GPT2_PARAPHRASER"
echo "  reddit tri_pipeline: $REDDIT_TRI_PIPELINE"
echo "  reddit universal_tri_pipeline: $REDDIT_UNIVERSAL_TRI_PIPELINE"
echo "  reddit result_in: $REDDIT_RESULT_IN"
echo "  reddit risk_in: $REDDIT_RISK_IN"
echo "  tab tri_pipeline: $TAB_TRI_PIPELINE"
echo "  tab universal_tri_pipeline: $TAB_UNIVERSAL_TRI_PIPELINE"
echo "  tab pii_annotator: $TAB_PII_ANNOTATOR"
echo "  tab result_in: $TAB_RESULT_IN"
echo "  tab risk_in: $TAB_RISK_IN"
echo ""

for CONFIG_DIR in "${CONFIG_DIRS[@]}"; do
    if [[ ! -d "$CONFIG_DIR" ]]; then
        echo "Skipping missing config dir: $CONFIG_DIR"
        continue
    fi

    echo "Updating model_checkpoint references in $CONFIG_DIR..."
    find "$CONFIG_DIR" -name "*.yaml" -type f | while read config; do
        if grep -q "model_checkpoint:" "$config"; then
            if [[ "$config" == *"dpparaphrase"* ]]; then
                sed -i.bak "s|model_checkpoint:.*|model_checkpoint: $GPT2_PARAPHRASER|g" "$config"
                echo "  Updated: $config"
            fi
        fi
    done


    echo "Updating tri_pipeline in reddit configs under $CONFIG_DIR..."
    find "$CONFIG_DIR" -path "*reddit*" -name "*.yaml" -type f | while read config; do
        if grep -q "tri_pipeline:" "$config"; then
            sed -i.bak "s|tri_pipeline:.*|tri_pipeline: $REDDIT_TRI_PIPELINE|g" "$config"
            echo "  Updated: $config"
        fi
    done

    echo "Updating tri_pipeline in tab configs under $CONFIG_DIR..."
    find "$CONFIG_DIR" -path "*tab*" -name "*.yaml" -type f | while read config; do
        if grep -q "tri_pipeline:" "$config"; then
            sed -i.bak "s|tri_pipeline:.*|tri_pipeline: $TAB_TRI_PIPELINE|g" "$config"
            echo "  Updated: $config"
        fi
    done

    echo "Updating universal_tri_pipeline in reddit configs under $CONFIG_DIR..."
    find "$CONFIG_DIR" -path "*reddit*" -name "*.yaml" -type f | while read config; do
        if grep -q "universal_tri_pipeline:" "$config"; then
            sed -i.bak "s|universal_tri_pipeline:.*|universal_tri_pipeline: $REDDIT_UNIVERSAL_TRI_PIPELINE|g" "$config"
            echo "  Updated: $config"
        fi
    done

    echo "Updating universal_tri_pipeline in tab configs under $CONFIG_DIR..."
    find "$CONFIG_DIR" -path "*tab*" -name "*.yaml" -type f | while read config; do
        if grep -q "universal_tri_pipeline:" "$config"; then
            sed -i.bak "s|universal_tri_pipeline:.*|universal_tri_pipeline: $TAB_UNIVERSAL_TRI_PIPELINE|g" "$config"
            echo "  Updated: $config"
        fi
    done
    
    echo "Updating pii_annotator in tab configs under $CONFIG_DIR..."
    find "$CONFIG_DIR" -path "*tab*" -name "*.yaml" -type f | while read config; do
        if grep -q "pii_annotator:" "$config"; then
            sed -i.bak "s|pii_annotator:.*|pii_annotator: $TAB_PII_ANNOTATOR|g" "$config"
            echo "  Updated: $config"
        fi
    done

    echo "Updating result_in in reddit configs under $CONFIG_DIR..."
    if [[ -n "$REDDIT_RESULT_IN" ]]; then
        find "$CONFIG_DIR" -path "*reddit*" -name "*.yaml" -type f | while read config; do
            if grep -q "result_in:" "$config"; then
                sed -i.bak "s|result_in:.*|result_in: $REDDIT_RESULT_IN|g" "$config"
                echo "  Updated: $config"
            fi
        done
    fi

    echo "Updating result_in in tab configs under $CONFIG_DIR..."
    if [[ -n "$TAB_RESULT_IN" ]]; then
        find "$CONFIG_DIR" -path "*tab*" -name "*.yaml" -type f | while read config; do
            if grep -q "result_in:" "$config"; then
                sed -i.bak "s|result_in:.*|result_in: $TAB_RESULT_IN|g" "$config"
                echo "  Updated: $config"
            fi
        done
    fi

    echo "Updating risk_in in reddit configs under $CONFIG_DIR..."
    if [[ -n "$REDDIT_RISK_IN" ]]; then
        find "$CONFIG_DIR" -path "*reddit*" -name "*.yaml" -type f | while read config; do
            if grep -q "risk_scores:" "$config"; then
                sed -i.bak "s|risk_scores:.*|risk_scores: $REDDIT_RISK_IN|g" "$config"
                echo "  Updated: $config"
            fi
        done
    fi

    echo "Updating risk_in in tab configs under $CONFIG_DIR..."
    if [[ -n "$TAB_RISK_IN" ]]; then
        find "$CONFIG_DIR" -path "*tab*" -name "*.yaml" -type f | while read config; do
            if grep -q "risk_scores:" "$config"; then
                sed -i.bak "s|risk_scores:.*|risk_scores: $TAB_RISK_IN|g" "$config"
                echo "  Updated: $config"
            fi
        done
    fi

    echo "Cleaning up backup files in $CONFIG_DIR..."
    find "$CONFIG_DIR" -name "*.bak" -delete
done

echo "Done!"
