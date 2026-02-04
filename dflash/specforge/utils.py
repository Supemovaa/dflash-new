from typing import Any, Dict, List, Optional
import json
import logging
from contextlib import contextmanager
import os
from random import random
import random
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, PretrainedConfig
from datasets import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from dflash.specforge.distributed import get_draft_sp_group

logger = logging.getLogger(__name__)


@contextmanager
def rank_0_priority():
    rank = dist.get_rank()

    if rank == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


@contextmanager
def default_torch_dtype(dtype: torch.dtype):
    current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(current_dtype)


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def load_config_from_file(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    return PretrainedConfig.from_dict(config)


def print_with_rank(message):
    if dist.is_available() and dist.is_initialized():
        logger.info(f"rank {dist.get_rank()}: {message}")
    else:
        logger.info(f"non-distributed: {message}")


def print_args_with_dots(args):
    if dist.get_rank() == 0:
        args_dict = vars(args)
        max_key_length = max(len(key) for key in args_dict.keys())
        total_width = 50

        print("\n -----------【args】-----------")
        for key, value in args_dict.items():
            key_str = f"{key:<{max_key_length}}"
            value_str = str(value)
            dot_count = total_width - len(key_str) - len(value_str)
            dot_fill = "·" * dot_count
            print(f"{key_str} {dot_fill} {value_str}")


def print_on_rank0(message):
    if dist.get_rank() == 0:
        logger.info(message)


class DataCollatorWithPadding:
    """
    Datacollator that will dynamically pad the inputs for batching.
    """

    def __init__(self):
        self.sp_degree = torch.distributed.get_world_size(get_draft_sp_group())

    def paddingtensor(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad to the longest sequence in the batch.

        Args:
            intensors: (B, n, S)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N, S)
        """
        B, n, S = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, S, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors: torch.Tensor, N: int) -> torch.Tensor:
        """
        Pad 2D tensor to the longest sequence in the batch.

        Args:
            intensors: (B, n)
            N: the length to pad to, N >= n

        Returns:
            outtensors: (B, N)
        """
        B, n = intensors.shape
        padding_tensor = torch.zeros(
            B, N - n, dtype=intensors.dtype, device=intensors.device
        )
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        # pad for sequence parrel
        max_length = (
            (max_length + self.sp_degree - 1) // self.sp_degree
        ) * self.sp_degree
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )
        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
            "hidden_state": None,
            "target": None,
        }
        if all("hidden_state" in item for item in features):
            assert all(
                "target" in item for item in features
            ), "target is required when hidden_state is provided"
            batch["hidden_state"] = torch.cat(
                [
                    self.paddingtensor(item["hidden_state"], max_length)
                    for item in features
                ]
            )
            batch["target"] = torch.cat(
                [self.paddingtensor(item["target"], max_length) for item in features]
            )
        return batch


def prepare_dp_dataloaders(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    process_group: Optional[dist.ProcessGroup] = None,
    pin_memory: Optional[bool] = False,
    shuffle: Optional[bool] = False,
    prefetch_factor: Optional[int] = 2,
    **dataloader_kwargs
) -> DataLoader:

    world_size = dist.get_world_size(process_group)
    rank = dist.get_rank(process_group)
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
    )

    datacollator_cls = DataCollatorWithPadding

    if num_workers == 0:
        prefetch_factor = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        collate_fn=datacollator_cls(),
        drop_last=True,
        **dataloader_kwargs
    )
    return dataloader