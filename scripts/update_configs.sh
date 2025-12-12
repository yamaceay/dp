#!/bin/bash

MODELS_FILE="${1:-configs/environment/hpc.yaml}"
CONFIG_DIR="configs/model"

if [[ ! -f "$MODELS_FILE" ]]; then
    echo "Error: $MODELS_FILE not found"
    exit 1
fi

GPT2_PARAPHRASER=$(grep "^gpt2-paraphraser:" "$MODELS_FILE" | awk '{print $2}')
REDDIT_TRI_PIPELINE=$(grep -A 1 "^reddit:" "$MODELS_FILE" | grep "tri_pipeline:" | awk '{print $2}')
TAB_TRI_PIPELINE=$(grep -A 10 "^tab:" "$MODELS_FILE" | grep "tri_pipeline:" | head -1 | awk '{print $2}')
TAB_PII_ANNOTATOR=$(grep -A 10 "^tab:" "$MODELS_FILE" | grep "pii_annotator:" | awk '{print $2}')

echo "Extracted values:"
echo "  gpt2-paraphraser: $GPT2_PARAPHRASER"
echo "  reddit tri_pipeline: $REDDIT_TRI_PIPELINE"
echo "  tab tri_pipeline: $TAB_TRI_PIPELINE"
echo "  tab pii_annotator: $TAB_PII_ANNOTATOR"
echo ""

echo "Updating model_checkpoint references..."
find "$CONFIG_DIR" -name "*.yaml" -type f | while read config; do
    if grep -q "model_checkpoint:" "$config"; then
        
        if [[ "$config" == *"dpparaphrase"* ]] || [[ "$config" == *"dpprompt"* ]]; then
            sed -i.bak "s|model_checkpoint:.*|model_checkpoint: $GPT2_PARAPHRASER|g" "$config"
            echo "  Updated: $config"
        fi
    fi
done

echo "Updating tri_pipeline in reddit configs..."
find "$CONFIG_DIR" -path "*reddit*" -name "*.yaml" -type f | while read config; do
    if grep -q "tri_pipeline:" "$config"; then
        sed -i.bak "s|tri_pipeline:.*|tri_pipeline: $REDDIT_TRI_PIPELINE|g" "$config"
        echo "  Updated: $config"
    fi
done

echo "Updating tri_pipeline in tab configs..."
find "$CONFIG_DIR" -path "*tab*" -name "*.yaml" -type f | while read config; do
    if grep -q "tri_pipeline:" "$config"; then
        sed -i.bak "s|tri_pipeline:.*|tri_pipeline: $TAB_TRI_PIPELINE|g" "$config"
        echo "  Updated: $config"
    fi
done

echo "Updating pii_annotator in tab configs..."
find "$CONFIG_DIR" -path "*tab*" -name "*.yaml" -type f | while read config; do
    if grep -q "pii_annotator:" "$config"; then
        sed -i.bak "s|pii_annotator:.*|pii_annotator: $TAB_PII_ANNOTATOR|g" "$config"
        echo "  Updated: $config"
    fi
done

echo ""
echo "Cleaning up backup files..."
find "$CONFIG_DIR" -name "*.bak" -delete
echo "Done!"
