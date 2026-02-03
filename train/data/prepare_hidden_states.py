import argparse
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import List, Optional

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoConfig
from datasets import load_from_disk

from specforge.args import SGLangBackendArgs
from specforge.data import prepare_dp_dataloaders
from specforge.distributed import (
    destroy_distributed,
    get_dp_group,
    get_tp_group,
    init_distributed,
    is_tp_rank_0,
)
from specforge.modeling.target import get_eagle3_target_model
from specforge.utils import print_with_rank, rank_0_priority
from train.model.utils import build_target_layer_ids


@dataclass
class DataPoint:
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    aux_hidden_state: Optional[torch.Tensor] = None


def parse_args():
    parser = argparse.ArgumentParser()
    
    model_group = parser.add_argument_group("model")
    model_group.add_argument("--target-model-path", type=str, required=True)
    model_group.add_argument("--enable-aux-hidden-states", action="store_true")
    model_group.add_argument("--num-draft-layers", type=int, default=5)
    model_group.add_argument("--aux-hidden-states-layers", type=str, default=None)

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--data-path", type=str, required=True)
    data_group.add_argument("--max-length", type=int, default=2048)
    data_group.add_argument("--num-samples", type=int, default=None)

    inference_group = parser.add_argument_group("inference")
    inference_group.add_argument("--tp-size", type=int, default=1)
    inference_group.add_argument("--batch-size", type=int, default=32)

    others_group = parser.add_argument_group("others")
    others_group.add_argument("--output-path", type=str, default=None)
    others_group.add_argument("--dist-timeout", type=int, default=2000)
    others_group.add_argument("--num-io-threads", type=int, default=4)
    others_group.add_argument("--num-workers", type=int, default=4)
    others_group.add_argument("--io-queue-size", type=int, default=50)
    others_group.add_argument("--file-group-size", type=int, default=2000)

    sglang_group = parser.add_argument_group("sglang")
    SGLangBackendArgs.add_args(sglang_group)
    return parser.parse_args()


def build_target_model(args: argparse.Namespace, model_config: AutoConfig):
    target_model_kwargs = SGLangBackendArgs.from_args(args).to_kwargs()
    target_model = get_eagle3_target_model(
        pretrained_model_name_or_path=args.target_model_path,
        backend="sglang",
        torch_dtype=(
            model_config.dtype
            if hasattr(model_config, "dtype")
            else model_config.torch_dtype
        ),
        device="cuda",
        **target_model_kwargs,
    )
    target_model.set_aux_hidden_states_layers(args.aux_hidden_states_layers)
    return target_model


class HiddenStatesGenerator:
    def __init__(
        self,
        target_model,
        enable_aux_hidden_states: bool = True,
        num_io_threads: int = 4,
        io_queue_size: int = 50,
        file_group_size: int = 2000,
    ):
        self.model = target_model
        self.enable_aux_hidden_states = enable_aux_hidden_states
        self.num_io_threads = num_io_threads
        self.io_queue_size = io_queue_size
        self.file_group_size = file_group_size
        self.show_progress = dist.get_rank(get_tp_group()) == 0
        self.io_executor = None
        self.pending_futures = []

    def __enter__(self):
        if is_tp_rank_0():
            self.io_executor = ThreadPoolExecutor(max_workers=self.num_io_threads)
        self.pending_futures = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if is_tp_rank_0() and self.io_executor is not None:
            self._wait_all_saves()
            self.io_executor.shutdown(wait=True)
            self.io_executor = None
        dist.barrier()

    def _save_tensor_sync(self, data_point: DataPoint, output_file: str) -> None:
        if data_point.hidden_state is not None and torch.any(torch.isnan(data_point.hidden_state)):
            return

        if data_point.aux_hidden_state is not None and torch.any(torch.isnan(data_point.aux_hidden_state)):
            return

        torch.save(asdict(data_point), output_file)

    def _save_tensor_async(self, data_point: DataPoint, output_file: str) -> None:
        assert is_tp_rank_0()
        if len(self.pending_futures) >= self.io_queue_size:
            self.pending_futures = [f for f in self.pending_futures if not f.done()]
            if len(self.pending_futures) >= self.io_queue_size:
                self.pending_futures.pop(0).result()

        future = self.io_executor.submit(self._save_tensor_sync, data_point, output_file)
        self.pending_futures.append(future)

    def _wait_all_saves(self):
        if is_tp_rank_0() and self.pending_futures:
            for future in self.pending_futures:
                future.result()
            self.pending_futures.clear()

    def _prepare_output_dirs(self, output_path: str, start_idx: int, total_samples: int) -> None:
        if not is_tp_rank_0() or total_samples == 0:
            return
        start_group = (start_idx // self.file_group_size) * self.file_group_size
        end_sample_idx = start_idx + total_samples - 1
        end_group = (end_sample_idx // self.file_group_size) * self.file_group_size
        for group_start_idx in range(start_group, end_group + 1, self.file_group_size):
            grouped_subdir = f"rows_{group_start_idx}-{group_start_idx + self.file_group_size}"
            output_dir = os.path.join(output_path, grouped_subdir)
            os.makedirs(output_dir, exist_ok=True)

    def _check_existing_files_batch(self, output_path: str, global_indices: List[int]) -> List[bool]:
        if not is_tp_rank_0():
            return [False] * len(global_indices)

        def check_single_file(idx):
            return os.path.exists(self._get_file_path(output_path, idx))

        with ThreadPoolExecutor(max_workers=self.num_io_threads) as executor:
            exists = list(executor.map(check_single_file, global_indices))
        return exists

    def _get_file_path(self, output_path: str, idx: int) -> str:
        group_idx = (idx // self.file_group_size) * self.file_group_size
        grouped_subdir = f"rows_{group_idx}-{group_idx + self.file_group_size}"
        return os.path.join(output_path, grouped_subdir, f"data_{idx}.ckpt")

    @torch.no_grad()
    def generate(
        self,
        data_loader: torch.utils.data.DataLoader,
        output_path: str,
        start_idx: int = 0,
        samples_per_dp: int = 0,
    ):
        self._prepare_output_dirs(output_path, start_idx, samples_per_dp)

        tp_group = get_tp_group()
        tp_rank_0_global = dist.get_process_group_ranks(tp_group)[0]
        global_idx = start_idx

        progress_bar = tqdm(
            data_loader,
            disable=(not self.show_progress),
            desc="Generating Hidden States",
            position=dist.get_rank(get_dp_group()),
            leave=True,
        )

        total_skipped, total_processed = 0, 0

        for batch_idx, batch in enumerate(progress_bar):
            batch_size = batch["input_ids"].size(0)
            current_batch_indices = list(range(global_idx, global_idx + batch_size))

            if is_tp_rank_0():
                exists_list = self._check_existing_files_batch(output_path, current_batch_indices)
                exists_tensor = torch.tensor(exists_list, dtype=torch.bool, device="cuda")
            else:
                exists_tensor = torch.tensor([False] * batch_size, dtype=torch.bool, device="cuda")
            
            dist.broadcast(exists_tensor, src=tp_rank_0_global, group=tp_group)

            valid_indices_in_batch = [i for i, exists in enumerate(exists_tensor) if not exists]
            sample_global_indices = [current_batch_indices[i] for i in valid_indices_in_batch]
            num_valid = len(valid_indices_in_batch)
            total_skipped += batch_size - num_valid

            global_idx += batch_size
            filtered_batch = {
                "input_ids": batch["input_ids"][valid_indices_in_batch],
                "attention_mask": batch["attention_mask"][valid_indices_in_batch],
                "loss_mask": batch["loss_mask"][valid_indices_in_batch],
            }
            del batch

            if num_valid == 0:
                if self.show_progress:
                    progress_bar.set_postfix(
                        {
                            "processed": total_processed,
                            "skipped": total_skipped,
                            "pending_io": (len(self.pending_futures) if is_tp_rank_0() else 0),
                        }
                    )
                continue

            filtered_batch_gpu = {k: v.cuda(non_blocking=True) for k, v in filtered_batch.items()}

            _, _, aux_hidden_states_list, _ = self.model.extend(
                **filtered_batch_gpu,
                return_last_hidden_states=False,
                return_logits=False,
            )

            del filtered_batch_gpu

            if is_tp_rank_0():
                for i, (current_global_idx, aux_hidden_states) in enumerate(
                    zip(sample_global_indices, aux_hidden_states_list)
                ):
                    valid_len = filtered_batch["attention_mask"][i].sum().item()
                    
                    aux_hidden_states = (
                        aux_hidden_states[:valid_len, :].cpu().clone().to(torch.bfloat16)
                        if aux_hidden_states is not None
                        else None
                    )
                    
                    data_point = DataPoint(
                        input_ids=filtered_batch["input_ids"][i][:valid_len].clone(),
                        loss_mask=filtered_batch["loss_mask"][i][:valid_len].clone(),
                        aux_hidden_state=aux_hidden_states,
                    )

                    output_file = self._get_file_path(output_path, current_global_idx)
                    self._save_tensor_async(data_point, output_file)

                    del aux_hidden_states

                total_processed += len(sample_global_indices)

            del aux_hidden_states_list, filtered_batch

            if batch_idx % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            if self.show_progress:
                progress_bar.set_postfix(
                    {
                        "processed": total_processed,
                        "skipped": total_skipped,
                        "pending_io": (len(self.pending_futures) if is_tp_rank_0() else 0),
                    }
                )

        dist.barrier()


def main():
    args = parse_args()
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)

    target_model_config = AutoConfig.from_pretrained(args.target_model_path)
    args.enable_aux_hidden_states = True
    args.aux_hidden_states_layers = build_target_layer_ids(
        num_target_layers=target_model_config.num_hidden_layers,
        num_draft_layers=args.num_draft_layers,
    )

    target_model = build_target_model(args, target_model_config)
    print_with_rank(f"DP Size {dist.get_world_size(get_dp_group())}, TP Size {dist.get_world_size(get_tp_group())}")

    with rank_0_priority():
        dataset = load_from_disk(args.data_path)
        if args.num_samples is not None:
            dataset = dataset.shuffle(seed=0).select(range(args.num_samples))

    print_with_rank(f"Dataset prepared with {len(dataset)} samples.")

    data_loader = prepare_dp_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        process_group=get_dp_group(),
    )

    total = len(dataset)
    dp_rank = dist.get_rank(get_dp_group())
    dp_size = dist.get_world_size(get_dp_group())
    samples_per_dp = total // dp_size
    remainder = total % dp_size

    if dp_rank < remainder:
        samples_per_dp += 1
        start_idx = dp_rank * samples_per_dp
    else:
        start_idx = dp_rank * samples_per_dp + remainder

    try:
        with HiddenStatesGenerator(
            target_model,
            args.enable_aux_hidden_states,
            num_io_threads=args.num_io_threads,
            io_queue_size=args.io_queue_size,
            file_group_size=args.file_group_size,
        ) as hidden_states_generator:
            hidden_states_generator.generate(
                data_loader,
                output_path=args.output_path,
                start_idx=start_idx,
                samples_per_dp=samples_per_dp,
            )
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()