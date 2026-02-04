import torch
from datasets import load_dataset
from transformers import AutoTokenizer

def process_batch(examples, tokenizer, max_length):
    batch_input_ids = []
    batch_loss_mask = []
    batch_attention_mask = []
    batch_size = len(examples["conversations"])

    for i in range(batch_size):
        messages = examples["conversations"][i]
        reasoning_effort = examples["reasoning_effort"][i] if "reasoning_effort" in examples else None
        if reasoning_effort:
            prompt_ids = tokenizer.apply_chat_template(
                messages[:1], 
                tokenize=True, 
                add_generation_prompt=True,
                reasoning_effort=reasoning_effort,
                return_tensors="pt"
            )[0]
            response_ids = tokenizer.encode(messages[1]["content"]+"<|end|>", return_tensors="pt")[0]
            input_ids = torch.cat([prompt_ids, response_ids], dim=0)
        else:
            enable_thinking = (examples["thinking"][i] == "on")
            prompt_ids = tokenizer.apply_chat_template(
                messages[:1], 
                tokenize=True, 
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_tensors="pt"
            )[0]
            response_ids = tokenizer.encode(messages[1]["content"]+"<|im_end|>", return_tensors="pt")[0]
            input_ids = torch.cat([prompt_ids, response_ids], dim=0)

        prompt_len = prompt_ids.shape[0]
        loss_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        loss_mask[prompt_len:] = True

        if input_ids.shape[0] > max_length:
            input_ids = input_ids[:max_length]
            loss_mask = loss_mask[:max_length]

        batch_input_ids.append(input_ids[None, :])
        batch_loss_mask.append(loss_mask[None, :])
        batch_attention_mask.append(torch.ones_like(loss_mask, dtype=torch.bool)[None, :])

    return {
        "input_ids": batch_input_ids,
        "loss_mask": batch_loss_mask,
        "attention_mask": batch_attention_mask,
    }

def preprocess_and_save(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    raw_dataset = load_dataset("json", data_files=args.data_path, split="train")
    processed_dataset = raw_dataset.map(
        process_batch,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": args.max_length,
        },
        remove_columns=raw_dataset.column_names,
        num_proc=args.num_proc, 
    )
    processed_dataset.save_to_disk(args.save_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess dataset for training.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--num-proc", type=int, default=8)
    
    args = parser.parse_args()
    preprocess_and_save(args)