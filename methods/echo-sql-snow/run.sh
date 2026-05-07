#!/bin/bash
# ECHO-SQL agent for Spider 2.0-Snow.

INPUT_FILE="../../spider2-snow/spider2-snow.jsonl"
DATABASES_PATH="../../spider2-snow/resource/databases"
DOCUMENTS_PATH="../../spider2-snow/resource/documents"
CREDENTIAL_PATH="./snowflake_credential.json"

# Single-instance smoke test by default. Comment out to run all 547.
IDS_FILE="./test_one_id.txt"

MODEL="gpt-5.4-nano"
TEMPERATURE=0.6
MAX_STEPS=12
MAX_TOKENS=200000
LLM_TIMEOUT_S=240
# gpt-5-nano (reasoning) needs headroom for hidden CoT + visible output;
# 8K causes silent truncation when combined with tool calling.
LLM_DEFAULT_MAX_TOKENS=24000
PROFILE_HARD_CAP=30
NUM_THREADS=1

EXPERIMENT_SUFFIX="echo_v0"
OUTPUT_FOLDER="./results/${MODEL}_${EXPERIMENT_SUFFIX}"

mkdir -p ./results
echo "Model: $MODEL"
echo "Output: $OUTPUT_FOLDER"
echo "IDS file: ${IDS_FILE:-all}"

uv run python -m echo_sql.main \
    --input_file "$INPUT_FILE" \
    --databases_path "$DATABASES_PATH" \
    --documents_path "$DOCUMENTS_PATH" \
    --credential_path "$CREDENTIAL_PATH" \
    --output_folder "$OUTPUT_FOLDER" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --llm_timeout_s "$LLM_TIMEOUT_S" \
    --llm_default_max_tokens "$LLM_DEFAULT_MAX_TOKENS" \
    --max_steps "$MAX_STEPS" \
    --max_tokens "$MAX_TOKENS" \
    --profile_hard_cap "$PROFILE_HARD_CAP" \
    --num_threads "$NUM_THREADS" \
    ${IDS_FILE:+--ids_file "$IDS_FILE"}
