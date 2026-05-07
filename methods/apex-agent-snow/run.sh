#!/bin/bash
# APEX-SQL agent for Spider 2.0-Snow.
# See ../../APEX-SQL PLAN.md.

INPUT_FILE="../../spider2-snow/spider2-snow.jsonl"
DATABASES_PATH="../../spider2-snow/resource/databases"
DOCUMENTS_PATH="../../spider2-snow/resource/documents"
CREDENTIAL_PATH="./snowflake_credential.json"

# Subset filter. Leave empty to run all 547 instances.
IDS_FILE="../spider-agent-snow/completed_ids.txt"

MODEL="gpt-5-nano"
TEMPERATURE=0.6
N_SAMPLES=1
MAX_ACTIONS=40
MAX_TOKENS=56000             # action-loop total budget
LLM_TIMEOUT_S=120
LLM_DEFAULT_MAX_TOKENS=8000  # per-call (max_completion_tokens for reasoning models)
NUM_THREADS=4

EXPERIMENT_SUFFIX="apex_v0"
OUTPUT_FOLDER="./results/${MODEL}_${EXPERIMENT_SUFFIX}"

mkdir -p ./results

echo "Model: $MODEL"
echo "Output: $OUTPUT_FOLDER"
echo "IDS file: ${IDS_FILE:-all}"

uv run python -m apex_agent.main \
    --input_file "$INPUT_FILE" \
    --databases_path "$DATABASES_PATH" \
    --documents_path "$DOCUMENTS_PATH" \
    --credential_path "$CREDENTIAL_PATH" \
    --output_folder "$OUTPUT_FOLDER" \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --llm_timeout_s "$LLM_TIMEOUT_S" \
    --llm_default_max_tokens "$LLM_DEFAULT_MAX_TOKENS" \
    --n_samples "$N_SAMPLES" \
    --max_actions "$MAX_ACTIONS" \
    --max_tokens "$MAX_TOKENS" \
    --num_threads "$NUM_THREADS" \
    ${IDS_FILE:+--ids_file "$IDS_FILE"}
