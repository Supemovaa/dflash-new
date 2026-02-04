import torch
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from .utils import extract_context_feature

class JointModel(nn.Module):
    def __init__(self, draft_model: nn.Module, target_model: nn.Module, block_size: int):
        super().__init__()
        self.draft_model = draft_model
        self.target_model = target_model
        self.block_size = block_size
        for p in self.target_model.parameters():
            p.requires_grad = False
        self.target_model.eval()
        self.config = target_model.config
    
    def train(self, mode=True):
        super().train(mode)
        self.target_model.eval()
        return self
            
    def forward(
        self,
        prefix_input_ids,
        prefix_batch_ids,
        prefix_position_ids,
        input_ids,
        batch_ids,
        position_ids,
        tail_gather_indices,
        labels=None,
        **kwargs
    ):
        if prefix_input_ids.numel() == 0 or kwargs["num_items_in_batch"] == 0:
            return {"loss": torch.tensor(0.0, device=prefix_input_ids.device, requires_grad=True)}
        hidden_states, target_full_logits = self._casual_prefill(
            prefix_input_ids, prefix_batch_ids, prefix_position_ids
        )
        draft_logits = self._forward_draft(
            input_ids, batch_ids, position_ids, 
            prefix_batch_ids, prefix_position_ids,
            hidden_states
        )

        teacher_logits = target_full_logits[tail_gather_indices]

        # loss_ce = nn.functional.cross_entropy(draft_logits, labels, ignore_index=-100, reduction="none")
        # loss_ce = loss_ce.sum() / kwargs["num_items_in_batch"]
        loss_kd = self._compute_kd_loss(draft_logits, teacher_logits, labels)
        loss_kd = loss_kd.sum() / kwargs["num_items_in_batch"]
        # loss = loss_ce + 0.5 * loss_kd
        loss = loss_kd
        return {"loss": loss, "logits": draft_logits}

    def _casual_prefill(self, input_ids, batch_ids, position_ids):
        def mask_mod(b, h, q_idx, kv_idx):
            return (batch_ids[q_idx] == batch_ids[kv_idx]) & (position_ids[q_idx] >= position_ids[kv_idx])
        attention_mask = create_block_mask(
            mask_mod, 1, 1, input_ids.shape[0], input_ids.shape[0], device=input_ids.device
        )
        self.target_model.eval()
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=input_ids.unsqueeze(0),
                position_ids=position_ids.unsqueeze(0),
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        
        return extract_context_feature(outputs.hidden_states, self.draft_model.aux), outputs.logits[0]

    def _forward_draft(
        self, 
        tail_input_ids, tail_batch_ids, tail_pos_ids,
        prefix_batch_ids, prefix_pos_ids,
        hidden
    ):
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
            flex_mask_mod, 1, 1, tail_input_ids.shape[0], full_batch_ids.shape[0], device=tail_input_ids.device
        )

        return self.draft_model(
            position_ids=full_pos_ids.unsqueeze(0),
            attention_mask=attention_mask,
            noise_ids=tail_input_ids.unsqueeze(0),
            hidden_states=hidden,
            use_cache=False,
        ).logits[0]