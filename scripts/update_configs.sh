#!/bin/bash

MODELS_FILE="${1:-configs/environment/hpc.yaml}"

if [[ ! -f "$MODELS_FILE" ]]; then
    echo "Error: $MODELS_FILE not found"
    exit 1
fi

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
TAB_TRI_PIPELINE=$(yaml_section_value "tab" "tri_pipeline")
TAB_PII_ANNOTATOR=$(yaml_section_value "tab" "pii_annotator")
TAB_UNIVERSAL_TRI_PIPELINE=$(yaml_section_value "tab" "universal_tri_pipeline")
TAB_RESULT_IN=$(yaml_section_value "tab" "result_in")

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
echo "  reddit result_in: $REDDIT_RESULT_IN"
echo "  tab tri_pipeline: $TAB_TRI_PIPELINE"
echo "  tab pii_annotator: $TAB_PII_ANNOTATOR"
echo "  tab result_in: $TAB_RESULT_IN"
echo ""

# Extract tri_location for output_root
# Select tri_location from tab or reddit section based on config file name
if [[ "$MODELS_FILE" == *tab* ]]; then
    TRI_LOCATION=$(awk -F': *' '
        /^tab:/ {in_tab=1; next}
        /^[^ \t]/ && $1!="tab:" {in_tab=0}
        in_tab && /^[ \t]+tri_location:/ {print $2; exit}
    ' "$MODELS_FILE")
    SECTION=tab
else
    TRI_LOCATION=$(awk -F': *' '
        /^reddit:/ {in_reddit=1; next}
        /^[^ \t]/ && $1!="reddit:" {in_reddit=0}
        in_reddit && /^[ \t]+tri_location:/ {print $2; exit}
    ' "$MODELS_FILE")
    SECTION=reddit
fi
if [[ -z "$TRI_LOCATION" ]]; then
    echo "Error: failed to extract $SECTION.tri_location from $MODELS_FILE" >&2
    exit 1
fi
echo "  $SECTION.tri_location (output_root): $TRI_LOCATION"
echo ""

for CONFIG_DIR in "${CONFIG_DIRS[@]}"; do
    if [[ ! -d "$CONFIG_DIR" ]]; then
        echo "Skipping missing config dir: $CONFIG_DIR"
        continue
    fi

    echo "Setting output_root in configs under $CONFIG_DIR..."
    find "$CONFIG_DIR" -name "*.yaml" -type f | while read config; do
        if grep -q "output_root:" "$config"; then
            sed -i.bak "s|output_root:.*|output_root: $TRI_LOCATION|g" "$config"
            echo "  Updated: $config"
        fi
    done

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

    echo "Cleaning up backup files in $CONFIG_DIR..."
    find "$CONFIG_DIR" -name "*.bak" -delete
done

echo "Done!"
