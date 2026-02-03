import argparse
import json
from pathlib import Path
from typing import Dict

from datasets import load_dataset
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "nemotron",
            "evol_codealpaca"
        ],
    )
    parser.add_argument("--output-path", type=str, default="./cache/datasets")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--nemotron-split", type=str, default="code")
    return parser.parse_args()

def process_nemotron_row(row: Dict) -> Dict:
    conversations = row["messages"]
    formatted_conversations = []
    
    for message in conversations:
        role = message["role"]
        content = message["content"]
        if role in ["user", "assistant"]:
            formatted_conversations.append({"role": role, "content": content})

    return {"conversations": formatted_conversations}

def process_evol_codealpaca_row(row: Dict) -> Dict:
    processed_row = {
        "conversations": [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["output"]},
        ],
    }
    return processed_row

def process_and_save_ds(train_ds, output_path, proc_fn, dataset_name):
    train_output_jsonl_path = output_path.joinpath(f"{dataset_name}_train.jsonl")
    
    with open(train_output_jsonl_path, "w") as f:
        for item in tqdm(train_ds, desc=f"Processing {dataset_name} dataset"):
            if proc_fn is not None:
                row = proc_fn(item)
                if row is None:
                    continue
            else:
                row = item
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def main():
    args = parse_args()

    if args.dataset == "nemotron":
        ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split=args.nemotron_split)
        proc_fn = process_nemotron_row

    elif args.dataset == "evol_codealpaca":
        ds = load_dataset("theblackcat102/evol-codealpaca-v1")["train"]
        proc_fn = process_evol_codealpaca_row

    else:
        raise ValueError(f"Dataset {args.dataset} not supported.")

    if args.sample_size is not None and args.sample_size < len(ds):
        ds = ds.shuffle(seed=0).select(range(args.sample_size))

    output_path = Path(args.output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    process_and_save_ds(ds, output_path, proc_fn, f"{args.dataset}_{args.nemotron_split}" if args.dataset=="nemotron" else args.dataset)

if __name__ == "__main__":
    main()