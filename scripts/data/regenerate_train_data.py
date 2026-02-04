import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from openai import OpenAI
import random

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument("--output-file-path", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--server-address", type=str, nargs="+", required=True)
    parser.add_argument("--is-gpt-oss", action="store_true")
    parser.add_argument("--thinking", action="store_true", default=False)
    return parser.parse_args()

def get_random_reasoning_effort() -> str:
    reasoning_efforts = [
        "low",
        "medium",
        "high",
    ]
    weights = [4, 4, 2]
    return random.choices(reasoning_efforts, weights=weights, k=1)[0]

def build_query_kwargs(args, messages):
    query_kwargs = dict(
        model=args.model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        stream=False,
    )
    if args.top_p is not None:
        query_kwargs["top_p"] = args.top_p
    if args.repetition_penalty is not None:
        query_kwargs["presence_penalty"] = args.repetition_penalty
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.is_gpt_oss:
        query_kwargs["reasoning_effort"] = get_random_reasoning_effort()
    else:
        if args.thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            extra_body["chat_template_kwargs"] = {"enable_thinking": False}
    if extra_body:
        query_kwargs["extra_body"] = extra_body
        
    return query_kwargs

def call_sglang_generate(args, server_address, data):
    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")
    original_msgs = data["conversations"]
    regen_conversation = []
    regen_conversation.append(original_msgs[0])
    query_kwargs = build_query_kwargs(args, regen_conversation)
    try:
        response = client.chat.completions.create(**query_kwargs)
    except Exception as e:
        data["status"] = "error"
        data["error"] = str(e)
        return data
        
    response_text = response.choices[0].message.content
    resp_msg = {
        "role": "assistant",
        "content": response_text,
    }
    regen_conversation.append(resp_msg)

    if args.is_gpt_oss:
        data["reasoning_effort"] = query_kwargs.get("reasoning_effort", "")
    else:
        if args.thinking:
            data["thinking"] = "on"
        else:
            data["thinking"] = "off"
    data["conversations"] = regen_conversation
    data["status"] = "success"
    return data

def main():
    args = parse_arguments()
    
    with open(args.input_file_path, "r") as f_in, \
         open(args.output_file_path, "w") as f_out, \
         open(args.output_file_path.replace(".jsonl", "_error.jsonl"), "w") as f_err:
        
        lines = f_in.readlines()
        if args.num_samples:
            lines = lines[:args.num_samples]

        with ThreadPoolExecutor(max_workers=args.concurrency * len(args.server_address)) as executor:
            futures = []
            for i, line in enumerate(lines):
                server = args.server_address[i % len(args.server_address)]
                futures.append(executor.submit(call_sglang_generate, args, server, json.loads(line)))

            for future in tqdm(futures):
                result = future.result()
                if result.get("status") == "success":
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                elif result.get("status") == "error":
                    f_err.write(json.dumps(result, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()