import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import transformers
from loguru import logger
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers import AutoTokenizer, HfArgumentParser, Trainer
from model import DFlashDraftModel, make_draft_config, freeze_embedding_lm_head, load_embed_lm_head
from torch.utils.data import Dataset

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    model_name_or_path: Optional[str] = field(default=None)
    data_paths: Optional[str] = field(default=None)
    block_size: Optional[int] = field(default=16)
    num_draft_layers: Optional[int] = field(default=5)
    num_target_hidden: Optional[int] = field(default=5)
    mask_token_id: Optional[int] = field(default=None)
    num_anchors: Optional[int] = field(default=512)
    gamma: Optional[float] = field(default=7.0)

    remove_unused_columns: Optional[bool] = field(default=False)
    dataloader_num_workers: int = field(default=8)
    dataloader_prefetch_factor: int = field(default=4)

class CachedHiddenStateDataset(Dataset):
    def __init__(self, data_path):
        self.files = sorted(glob.glob(os.path.join(data_path, "**/*.ckpt"), recursive=True))
        logger.info(f"Found {len(self.files)} cached samples in {data_path}")

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], map_location="cpu")
        except Exception as e:
            logger.warning(f"CORRUPTED FILE DETECTED: {self.files[idx]} - Error: {e}. replacing with random sample.")
            return self.__getitem__(torch.randint(0, len(self), (1,)).item())
        seq_len = data["input_ids"].shape[0]
        data["valid_indices"] = torch.nonzero(data["loss_mask"].flatten()).flatten()
        data["valid_indices"] = data["valid_indices"][data["valid_indices"] < seq_len - 1]
        return data

@dataclass
class AnchorDataCollator:
    block_size: int
    num_anchors: int
    mask_token_id: int

    def __call__(self, instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        prefix_batch_ids_list = []
        prefix_pos_ids_list = []        
        tail_input_ids_list = []
        tail_labels_list = []
        tail_pos_ids_list = []
        tail_batch_ids_list = []

        aux_hidden_list = []

        current_prefix_offset = 0

        valid_batch_idx = 0
        for instance in instances:
            seq = instance["input_ids"]
            valid_indices = instance["valid_indices"]
            aux_hidden = instance["aux_hidden_state"]
            seq_len = seq.shape[0]

            if valid_indices.numel() > 0:
                aux_hidden_list.append(aux_hidden)
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
            batch["prefix_batch_ids"] = torch.empty(0, dtype=torch.long)
            batch["prefix_position_ids"] = torch.empty(0, dtype=torch.long)
            batch["input_ids"] = torch.empty(0, dtype=torch.long)
            batch["labels"] = torch.empty(0, dtype=torch.long)
            batch["position_ids"] = torch.empty(0, dtype=torch.long)
            batch["batch_ids"] = torch.empty(0, dtype=torch.long)
            batch["aux_hidden_state"] = torch.empty(0, dtype=torch.float)
        else:
            batch["prefix_batch_ids"] = torch.cat(prefix_batch_ids_list)
            batch["prefix_position_ids"] = torch.cat(prefix_pos_ids_list)
            batch["input_ids"] = torch.cat(tail_input_ids_list)
            batch["labels"] = torch.cat(tail_labels_list)
            batch["position_ids"] = torch.cat(tail_pos_ids_list)
            batch["batch_ids"] = torch.cat(tail_batch_ids_list)
            batch["aux_hidden_state"] = torch.cat(aux_hidden_list)
        return batch
    
    def _pad_block(self, tensor: torch.Tensor, value: int) -> torch.Tensor:
        return torch.cat([tensor, torch.full((self.block_size-2,), value, device=tensor.device)])


class DiffusionTrainer(Trainer):
    def __init__(self, block_size: int, gamma: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.block_size = block_size
        self.gamma = gamma
    
    def num_tokens(self, train_dl, max_steps=None):
        logger.info("Skipping expensive num_tokens pre-scan.")
        return 1000000000

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs["prefix_batch_ids"].numel() == 0 or num_items_in_batch == 0:
            return torch.tensor(0.0, device=inputs["prefix_batch_ids"].device, requires_grad=True)
        
        logits = self._forward_draft(model, inputs, inputs["aux_hidden_state"])
        labels = inputs["labels"]

        if self.state.global_step % self.args.logging_steps == 0:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                correct = (preds == labels) & (labels != -100)
                correct_per_pos = correct.view(-1, self.block_size)
                labels_per_pos = labels.view(-1, self.block_size)
                logs = {}
                for i in range(1, self.block_size):
                    mask_i = labels_per_pos[:, i] != -100
                    if mask_i.any():
                        acc_i = correct_per_pos[:, i][mask_i].float().mean()
                        avg_acc_i = self.accelerator.reduce(acc_i, reduction="mean")
                        logs[f"train/pos_{i}_acc"] = avg_acc_i.item()
                local_accept_len = correct_per_pos[:, 1:].cumprod(dim=1).sum(dim=1).float().mean()
                global_accept_len = self.accelerator.reduce(local_accept_len, reduction="mean")
                logs["train/accept_length"] = global_accept_len.item()
                self.log(logs)

        ce_loss_all = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            reduction="none",
            ignore_index=-100
        )

        num_tokens = ce_loss_all.shape[0]
        device = ce_loss_all.device
        steps_within_block = torch.arange(num_tokens, device=device) % self.block_size
        prediction_step = (steps_within_block - 1).float()
        weights = torch.exp(-prediction_step / self.gamma)
        weighted_loss = (ce_loss_all * weights).sum()
        loss = weighted_loss / num_items_in_batch
        
        if (
            self.args.average_tokens_across_devices
            and self.model_accepts_loss_kwargs
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes
        
        if return_outputs:
            raise NotImplementedError("`return_outputs` is not supported.")
        return loss

    def _forward_draft(
        self,
        model: nn.Module,
        inputs: Dict[str, Any],
        hidden: Dict[str, Any],
    ) -> torch.Tensor:
        tail_input_ids = inputs["input_ids"]
        tail_batch_ids = inputs["batch_ids"]
        tail_pos_ids = inputs["position_ids"]
        
        prefix_batch_ids = inputs["prefix_batch_ids"]
        prefix_pos_ids = inputs["prefix_position_ids"]
        prefix_length = prefix_batch_ids.shape[0]

        full_batch_ids = torch.cat([prefix_batch_ids, tail_batch_ids])
        full_pos_ids = torch.cat([prefix_pos_ids, tail_pos_ids])
        block_size = self.block_size

        def flex_mask_mod(b, h, q_idx, kv_idx):
            q_block_num = q_idx // block_size
            q_b_id = tail_batch_ids[q_idx]
            q_anchor_id = q_block_num * block_size
            q_anchor_p_id = tail_pos_ids[q_anchor_id]
            kv_b_id = full_batch_ids[kv_idx]
            kv_p_id = full_pos_ids[kv_idx]
            batch_match = q_b_id == kv_b_id
            is_kv_prefix = kv_idx < prefix_length
            prefix_visible = is_kv_prefix & (q_anchor_p_id > kv_p_id) 
            kv_in_tail_idx = kv_idx - prefix_length
            kv_block_num = kv_in_tail_idx // block_size
            tail_block_match = (q_block_num == kv_block_num)
            return batch_match & (prefix_visible | tail_block_match)

        attention_mask = create_block_mask(
            flex_mask_mod,
            1,
            1,
            tail_input_ids.shape[0],
            full_batch_ids.shape[0],
            device=tail_input_ids.device,
        )

        logits = model(
            position_ids=full_pos_ids.unsqueeze(0),
            attention_mask=attention_mask,
            noise_ids=tail_input_ids.unsqueeze(0),
            hidden_states=hidden.unsqueeze(0),
            use_cache=False,
        ).logits[0]
            
        return logits


def main() -> None:
    parser = HfArgumentParser(TrainingArguments)
    (args,) = parser.parse_args_into_dataclasses()

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
    
    model = DFlashDraftModel(draft_config)
    
    logger.info(f"Selected target hidden state layer ids: {model.aux}")

    freeze_embedding_lm_head(model)
    load_embed_lm_head(model, args.model_name_or_path)

    model = model.to(torch.bfloat16)

    logger.info(f"Loading preprocessed dataset from {args.data_paths}...")
    train_dataset = CachedHiddenStateDataset(args.data_paths)

    data_collator = AnchorDataCollator(
        block_size=args.block_size,
        num_anchors=args.num_anchors,
        mask_token_id=mask_token_id,
    )

    trainer = DiffusionTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        block_size=args.block_size,
        gamma=args.gamma,
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