import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import transformers
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, Trainer
from model import DFlashDraftModel, make_draft_config, freeze_embedding_lm_head, load_embed_lm_head
from model import JointModel
from deepspeed.runtime.zero.stage3 import GatheredParameters
from datasets import load_from_disk

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default=None)
    data_paths: Optional[str] = field(default=None)
    block_size: Optional[int] = field(default=16)
    num_anchors: Optional[int] = field(default=512)
    num_draft_layers: Optional[int] = field(default=5)
    remove_unused_columns: Optional[bool] = field(default=False)
    save_safetensors: bool = field(default=False)
    dataloader_num_workers: int = field(default=8)
    dataloader_prefetch_factor: int = field(default=2)

class DiffusionTrainer(Trainer):
    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        draft_folder = f"draft-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, draft_folder)
        os.makedirs(output_dir, exist_ok=True)
        unwrapped_model = self.accelerator.unwrap_model(model)
        draft_model = unwrapped_model.draft_model
        with GatheredParameters(draft_model.parameters(), modifier_rank=0):
            state_dict = draft_model.state_dict()
            if self.args.should_save:
                draft_model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=True)
                
    def num_tokens(self, train_dl, max_steps=None):
        # On a network volume, this is suicide. We return a dummy value.
        logger.info("Skipping expensive num_tokens pre-scan.")
        return 1000000000

@dataclass
class AnchorDataCollator:
    tokenizer: AutoTokenizer
    block_size: int
    num_anchors: int
    mask_token_id: int
    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        prefix_input_ids_list = []
        prefix_batch_ids_list = []
        prefix_pos_ids_list = []        
        tail_input_ids_list = []
        tail_labels_list = []
        tail_pos_ids_list = []
        tail_batch_ids_list = []

        current_prefix_offset = 0

        valid_batch_idx = 0
        for instance in instances:
            seq = instance["input_ids"].squeeze(0)
            seq_len = seq.shape[0]
            valid_indices = torch.nonzero(instance["loss_mask"].flatten()).flatten()
            valid_indices = valid_indices[valid_indices < seq_len - 1]
            if valid_indices.numel() > 0:
                prefix_input_ids_list.append(seq)
                prefix_batch_ids_list.append(torch.full((seq_len,), valid_batch_idx, dtype=torch.long))
                prefix_pos_ids_list.append(torch.arange(seq_len, dtype=torch.long))

                k = min(self.num_anchors, valid_indices.numel())
                sampled_anchors = valid_indices[torch.randperm(valid_indices.numel())[:k]]

                num_blocks = sampled_anchors.numel()                
                block_input = torch.full((num_blocks, self.block_size), self.mask_token_id, dtype=torch.long)
                block_input[:, 0] = seq[sampled_anchors]

                block_label = torch.full((num_blocks, self.block_size), -100, dtype=torch.long)                
                target_indices = sampled_anchors.unsqueeze(1) + torch.arange(1, self.block_size).unsqueeze(0)
                block_label[:, 1:] = self._pad_block(seq, -100)[target_indices]
                
                block_pos = sampled_anchors.unsqueeze(1) + torch.arange(self.block_size).unsqueeze(0)
                block_batch = torch.full((num_blocks, self.block_size), valid_batch_idx, dtype=torch.long)

                tail_input_ids_list.append(block_input.flatten())
                tail_labels_list.append(block_label.flatten())
                tail_pos_ids_list.append(block_pos.flatten())
                tail_batch_ids_list.append(block_batch.flatten())
                valid_batch_idx += 1
                current_prefix_offset += seq_len

        batch = {}       
        if valid_batch_idx == 0:
            batch["prefix_input_ids"] = torch.empty(0, dtype=torch.long)
            batch["prefix_batch_ids"] = torch.empty(0, dtype=torch.long)
            batch["prefix_position_ids"] = torch.empty(0, dtype=torch.long)
            batch["input_ids"] = torch.empty(0, dtype=torch.long)
            batch["labels"] = torch.empty(0, dtype=torch.long)
            batch["position_ids"] = torch.empty(0, dtype=torch.long)
            batch["batch_ids"] = torch.empty(0, dtype=torch.long)
        else:
            batch["prefix_input_ids"] = torch.cat(prefix_input_ids_list)
            batch["prefix_batch_ids"] = torch.cat(prefix_batch_ids_list)
            batch["prefix_position_ids"] = torch.cat(prefix_pos_ids_list)
            batch["input_ids"] = torch.cat(tail_input_ids_list)
            batch["labels"] = torch.cat(tail_labels_list)
            batch["position_ids"] = torch.cat(tail_pos_ids_list)
            batch["batch_ids"] = torch.cat(tail_batch_ids_list)
        return batch
    
    def _pad_block(self, tensor: torch.Tensor, value: int) -> torch.Tensor:
        return torch.cat([tensor, torch.full((self.block_size-2,), value, device=tensor.device)])


def main() -> None:
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

    target_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flex_attention",
        dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.mask_token_id is None:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.convert_tokens_to_ids("<|MASK|>")
    else:
        mask_token_id = args.mask_token_id
    
    draft_config = make_draft_config(
            args.model_name_or_path,
            mask_token_id=mask_token_id,
            num_target_hidden=args.num_target_hidden,
            num_draft_layers=args.num_draft_layers,
            block_size=args.block_size,
            attn_implementation="flex_attention")
    
    draft_model = DFlashDraftModel(draft_config)
    freeze_embedding_lm_head(draft_model)
    load_embed_lm_head(draft_model, args.model_name_or_path)
    logger.info(f"Selected target hidden state layer ids: {draft_model.aux}")

    model = JointModel(
        draft_model=draft_model,
        target_model=target_model,
        block_size=args.block_size
    )

    logger.info(f"Loading preprocessed dataset from {args.data_paths}...")
    train_dataset = load_from_disk(args.data_paths)
    train_dataset.set_format("torch")

    data_collator = AnchorDataCollator(
        tokenizer=tokenizer,
        block_size=args.block_size,
        num_anchors=args.num_anchors
    )

    trainer = DiffusionTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    checkpoint_dirs = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
    if checkpoint_dirs:
        resume_from_checkpoint = max(checkpoint_dirs, key=lambda s: int(s.split("-")[-1]))
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
    else:
        resume_from_checkpoint = None
        logger.info("No checkpoint found. Starting training from scratch.")

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

if __name__ == "__main__":
    main()