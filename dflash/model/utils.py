import os
import json
import glob
import copy
import torch
from typing import Optional
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Config,
)
from transformers import AutoConfig
from huggingface_hub import snapshot_download
from safetensors import safe_open
from datasets import load_dataset, Features, Sequence, Value


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [(num_target_layers // 2)]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def make_draft_config(target_path, mask_token_id: int, num_target_hidden: int, num_draft_layers: int, block_size: int = 16, attn_implementation: str = "sdpa") -> Qwen3Config:
    target_config = AutoConfig.from_pretrained(target_path, trust_remote_code=True)

    if getattr(target_config, "model_type", "") == "qwen3_moe":
        # For Qwen3 MoE Target Models
        config_dict = target_config.to_dict()
        moe_keys = [
            "num_experts",
            "num_experts_per_tok",
            "moe_intermediate_size", 
            "shared_expert_intermediate_size",
            "decoder_sparse_step",
            "norm_topk_prob",
            "router_aux_loss_coef",
            "mlp_only_layers",
            "output_router_logits"
        ]   
        for key in moe_keys:
            config_dict.pop(key, None)
        config_dict["architectures"] = ["Qwen3ForCausalLM"]
        config_dict["model_type"] = "qwen3"
        draft_config = Qwen3Config(**config_dict)

    elif getattr(target_config, "model_type", "") == "llama":
        # For LLaMA-3.1-8B-Instruct Target Model
        qwen3_8b_config = AutoConfig.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        draft_config = copy.deepcopy(qwen3_8b_config)
        draft_config.vocab_size = target_config.vocab_size
        draft_config.pad_token_id = target_config.pad_token_id

    elif getattr(target_config, "model_type", "") == "gpt_oss":
        # For GPT-OSS-20B Target Model
        qwen3_4b_config = AutoConfig.from_pretrained("Qwen/Qwen3-4B", trust_remote_code=True)
        draft_config = copy.deepcopy(qwen3_4b_config)
        draft_config.hidden_size = target_config.hidden_size
        draft_config.vocab_size = target_config.vocab_size
        draft_config.pad_token_id = target_config.pad_token_id
        draft_config.eos_token_id = target_config.eos_token_id

    else:
        # For Qwen3 Dense (4B/8B) Target Models
        draft_config = copy.deepcopy(target_config)

    draft_config.num_target_layers = target_config.num_hidden_layers
    draft_config.num_hidden_layers = num_draft_layers
    draft_config.max_window_layers = num_draft_layers
    draft_config._attn_implementation = attn_implementation
    draft_config.layer_types = ["full_attention"] * num_draft_layers
    draft_config.block_size = block_size
    draft_config.dflash_config = {
        "target_layer_ids": build_target_layer_ids(draft_config.num_target_layers, num_target_hidden),
        "mask_token_id": mask_token_id,
    }
    return draft_config

def load_embed_lm_head(draft_model, model_path: str):
    emb_key = "model.embed_tokens.weight"
    head_key = "lm_head.weight"
    tied = getattr(draft_model.config, "tie_word_embeddings", False)
    if not os.path.exists(model_path):
        model_path = snapshot_download(repo_id=model_path)
    index_files = glob.glob(os.path.join(model_path, "*.index.json"))
    sharded = len(index_files) == 1
    def load_tensor(k: str):
        if sharded:
            idx = json.load(open(index_files[0], "r"))
            ckpt = os.path.join(model_path, idx["weight_map"][k])
            if ckpt.endswith(".safetensors"):
                with safe_open(ckpt, framework="pt") as f:
                    return f.get_tensor(k)
            return torch.load(ckpt, map_location="cpu")[k]
        st = os.path.join(model_path, "model.safetensors")
        if os.path.exists(st):
            with safe_open(st, framework="pt") as f:
                return f.get_tensor(k)
        return torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")[k]
    emb = load_tensor(emb_key)
    draft_model.embed_tokens.weight.copy_(emb)
    if tied:
        draft_model.lm_head.weight = draft_model.embed_tokens.weight
    else:
        try:
            head = load_tensor(head_key)
        except:
            head = emb
        draft_model.lm_head.weight.copy_(head)


def freeze_embedding_lm_head(draft_model) -> None:
    draft_model.embed_tokens.weight.requires_grad = False
    draft_model.lm_head.weight.requires_grad = False


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def process_batch_with_padding(examples, tokenizer, max_length, block_size=16):
    batch_input_ids = []
    batch_valid_indices = []
    
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 128004
    padding_tensor = torch.full((block_size - 2,), pad_id, dtype=torch.long)
    padding_mask = torch.zeros(block_size - 2, dtype=torch.bool)

    for messages in examples["conversations"]:
        full_ids = []
        full_masks = []
        current_len = 0

        for i in range(0, len(messages), 2):
            prompt_ids = tokenizer.apply_chat_template(
                messages[:i+1], 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            )[0]
            
            turn_ids = tokenizer.apply_chat_template(
                messages[:i+2], 
                tokenize=True, 
                return_tensors="pt"
            )[0]

            if i == 0:
                new_ids = turn_ids
                assistant_start = len(prompt_ids)
            else:
                prev_turn_len = len(tokenizer.apply_chat_template(messages[:i], tokenize=True))
                new_ids = turn_ids[prev_turn_len:]
                assistant_start = len(prompt_ids) - prev_turn_len

            # Define assistant mask for this segment
            new_mask = torch.zeros(len(new_ids), dtype=torch.bool)
            new_mask[assistant_start:-1] = True

            # 3. Append turn data + padding block
            seg_ids = torch.cat([new_ids, padding_tensor])
            seg_mask = torch.cat([new_mask, padding_mask])

            # 4. Truncation check
            if current_len + len(new_ids) > max_length:
                remaining = max_length - current_len
                if remaining > 0:
                    new_ids = new_ids[:remaining]
                    new_mask = new_mask[:remaining]
                    new_mask[-1] = False 
                    full_ids.append(torch.cat([new_ids, padding_tensor]))
                    full_masks.append(torch.cat([new_mask, padding_mask]))
                break
            
            full_ids.append(seg_ids)
            full_masks.append(seg_mask)
            current_len += len(seg_ids)

        # Finalize sequence
        final_input_ids = torch.cat(full_ids)
        final_mask = torch.cat(full_masks)
        
        batch_input_ids.append(final_input_ids)
        batch_valid_indices.append(torch.nonzero(final_mask, as_tuple=True)[0])

    return {
        "input_ids": batch_input_ids,
        "valid_indices": batch_valid_indices,
    }


def sample(logits, temperature=0.0, top_k=0, top_p=1.0):
    bsz, seq_len, vocab_size = logits.shape
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    
    flat_logits = logits.view(-1, vocab_size)
    flat_logits = flat_logits / temperature

    if 0 < top_k < vocab_size:
        top_k_values, _ = torch.topk(flat_logits, top_k)
        kth_largest = top_k_values[..., -1].unsqueeze(-1)
        flat_logits = torch.where(flat_logits < kth_largest, torch.tensor(-float('inf'), device=logits.device), flat_logits)

    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(flat_logits, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        flat_logits = flat_logits.masked_fill(indices_to_remove, -float('inf'))
    probs = torch.softmax(flat_logits, dim=-1)
    
    sampled_tokens = torch.multinomial(probs, num_samples=1)
    return sampled_tokens.view(bsz, seq_len)


def load_and_process_dataset(data_name: str):
    # Math datasets
    if data_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    elif data_name == "aime25":
        dataset = load_dataset("MathArena/aime_2025", split="train")
        prompt_fmt = "{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})

    # Chat datasets 
    elif data_name == "alpaca":
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        dataset = dataset.map(lambda x: {"formatted_input": (f"{x['instruction']}\n\nInput:\n{x['input']}" if x['input'] else x['instruction'])})
        dataset = dataset.map(lambda x: {"turns": [x["formatted_input"]]})

    elif data_name == "mt-bench":
        dataset = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
        dataset = dataset.map(lambda x: {"turns": x["prompt"]})

    # Coding datasets
    elif data_name == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{prompt}\n```"
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        dataset = dataset.map(lambda x: {"turns": [x["prompt"]]})
    
    elif data_name == "lbpp":
        LBPP_PY_TEST_URL = "https://huggingface.co/datasets/CohereLabs/lbpp/resolve/main/python/test.parquet"
        dataset = load_dataset("parquet", data_files={"test": LBPP_PY_TEST_URL})["test"]
        dataset = dataset.map(lambda x: {"turns": [x["instruction"]]})

    elif data_name == "swe-bench":
        dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
        prompt_fmt = "Problem Statement:\n{problem_statement}\nPlease fix the issue described above."
        dataset = dataset.map(lambda x: {"turns": [prompt_fmt.format(**x)]})
    
    elif data_name == "livecodebench":
        base = "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/main/"
        allowed_files = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
        urls = [base + fn for fn in allowed_files]
        dataset = load_dataset("json", data_files={"test": urls})["test"]
        def format_lcb(doc):
            system_prompt = (
                "You are an expert Python programmer. You will be given a question (problem specification) "
                "and will generate a correct Python program that matches the specification and passes all tests. "
                "You will NOT return anything except for the program"
            )
            question_block = f"### Question:\n{doc['question_content']}"
            if doc.get("starter_code"):
                format_message = "### Format: Use the following code structure:"
                code_block = f"```python\n{doc['starter_code']}\n```"
            else:
                format_message = "### Format: Write your code in the following format:"
                code_block = "```python\n# YOUR CODE HERE\n```"
            answer_footer = "### Answer: (use the provided format with backticks)"
            return f"{system_prompt}\n\n{question_block}\n\n{format_message}\n{code_block}\n\n{answer_footer}"
        target_features = Features({"turns": Sequence(Value("large_string"))})
        dataset = dataset.map(
            lambda x: {"turns": [format_lcb(x)]},
            remove_columns=dataset.column_names,
            features=target_features
        )
    
    return dataset
