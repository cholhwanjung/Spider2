INPUT_FILE="../../spider2-snow/spider2-snow.jsonl"
SYSTEM_PROMPT="./prompts/spider_agent.txt"
DATABASES_PATH="/Users/cholhwan/Documents/ai/Spider2/spider2-snow/resource/databases"
DOCUMENTS_PATH="../../spider2-snow/resource/documents"

# Instance filtering (leave empty to run all)
IDS_FILE="../spider-agent-snow/completed_ids.txt"

MODEL="gpt-5-nano"
# Temperature/top_p/max_new_tokens: leave unset to use model-specific defaults from main.py
# TEMPERATURE=0.7
# TOP_P=0.9
# MAX_NEW_TOKENS=4096

MAX_ROUNDS=20
NUM_THREADS=4
ROLLOUT_NUMBER=1
EXPERIMENT_SUFFIX="test1"

OUTPUT_FOLDER="./results/${MODEL}_${EXPERIMENT_SUFFIX}"

mkdir -p "./results"

echo "Model: $MODEL"
echo "Output folder: $OUTPUT_FOLDER"
echo "IDS file: ${IDS_FILE:-all}"

host="localhost"
port=$(shuf -i 30000-31000 -n 1)
uv run python -m servers.serve --workers_per_tool 8 --host $host --port $port &
server_pid=$!

echo "Server (pid=$server_pid) started at http://$host:$port"

sleep 3

uv run python agent/main.py \
    --input_file "$INPUT_FILE" \
    --output_folder "$OUTPUT_FOLDER" \
    --system_prompt_path "$SYSTEM_PROMPT" \
    --databases_path "$DATABASES_PATH" \
    --documents_path "$DOCUMENTS_PATH" \
    --model "$MODEL" \
    --api_host "$host" \
    --api_port "$port" \
    --max_rounds "$MAX_ROUNDS" \
    --num_threads "$NUM_THREADS" \
    --rollout_number "$ROLLOUT_NUMBER" \
    ${IDS_FILE:+--ids_file "$IDS_FILE"}

kill $server_pid
