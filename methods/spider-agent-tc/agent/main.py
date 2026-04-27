import argparse

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from llm_agent import LLMAgent

# Model-specific default parameters
MODEL_DEFAULTS = {
    "gpt-4o": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 4096,
    },
    "gpt-4o-mini": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 4096,
    },
    "gpt-5-nano": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 4096,
    },
    "o1": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 10000,
    },
    "o3-mini": {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": 10000,
    },
}

def get_model_defaults(model: str) -> dict:
    for prefix, defaults in MODEL_DEFAULTS.items():
        if model.startswith(prefix):
            return defaults
    return {"temperature": 0.7, "top_p": 0.9, "max_new_tokens": 4096}


def main():
    parser = argparse.ArgumentParser()

    # Data paths
    parser.add_argument("--input_file", required=True, help="Input jsonl file path")
    parser.add_argument("--output_folder", required=True, help="Output folder path")
    parser.add_argument("--system_prompt_path", required=True, help="System prompt file path")
    parser.add_argument("--databases_path", required=True, help="Databases directory path")
    parser.add_argument("--documents_path", required=True, help="Documents directory path")

    # LLM settings
    parser.add_argument("--model", default="gpt-4o", help="Model name")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature (default: model-specific)")
    parser.add_argument("--top_p", type=float, default=None, help="Top-p (default: model-specific)")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="Max new tokens (default: model-specific)")

    # Execution settings
    parser.add_argument("--api_host", default="localhost", help="API host")
    parser.add_argument("--api_port", default="5000", help="API port")
    parser.add_argument("--max_rounds", type=int, default=20, help="Max conversation rounds")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads")
    parser.add_argument("--rollout_number", type=int, default=1, help="Number of rollouts per example")

    parser.add_argument("--prompt_strategy", default="spider-agent",
                       choices=["spider-agent"],
                       help="Prompt building strategy")

    # Instance filtering
    parser.add_argument("--ids_file", type=str, default="",
                       help="Path to a text file with one instance_id per line to run")

    args = parser.parse_args()

    # Apply model-specific defaults for unset parameters
    defaults = get_model_defaults(args.model)
    if args.temperature is None:
        args.temperature = defaults["temperature"]
    if args.top_p is None:
        args.top_p = defaults["top_p"]
    if args.max_new_tokens is None:
        args.max_new_tokens = defaults["max_new_tokens"]

    print(f"Model: {args.model}")
    print(f"Temperature: {args.temperature}, Top-p: {args.top_p}, Max tokens: {args.max_new_tokens}")

    agent = LLMAgent(args)
    agent.run()

if __name__ == "__main__":
    main()
