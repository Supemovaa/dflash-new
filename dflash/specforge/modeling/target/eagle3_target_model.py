from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather

from dflash.specforge.distributed import get_tp_group

from .sglang_backend import SGLangRunner, wrap_eagle3_logits_processors_in_module
from .sglang_backend.utils import LogitsProcessorForEAGLE3


@dataclass
class Eagle3TargetOutput:
    hidden_states: torch.Tensor
    target: torch.Tensor
    loss_mask: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    last_hidden_states: Optional[torch.Tensor] = None


class Eagle3TargetModel(ABC):
    """
    This  offers a layer of abstraction for the target model backend. The user can choose different backends to suit their needs:
    1. SGLang backend: for the mainstream model support with the fastest inference speed
    2. HuggingFace backend: for models that are not supported by SGLang but can be loaded by HuggingFace.
    3. Custom backend: for models with customized architecture and inference plan.
    """

    def __init__(self):
        self.aux_hidden_states_layers = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "Eagle3TargetModel":
        """
        Initialize the target model backend from a pretrained model path.
        """

    @abstractmethod
    def generate_eagle3_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Eagle3TargetOutput:
        """
        Generate the eagle3 data from the target model.
        """

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        """
        Set the layers to capture the aux hidden states from the target model outputs.
        """
        if aux_hidden_states_layers is None:
            if hasattr(self.model.config, "num_hidden_layers"):
                num_layers = self.model.config.num_hidden_layers
            else:
                raise ValueError(
                    f"Failed to set aux hidden states layers as model config {self.model.config} does not have num_hidden_layers"
                )
            aux_hidden_states_layers = [
                1,
                num_layers // 2 - 1,
                num_layers - 4,
            ]
        self.aux_hidden_states_layers = aux_hidden_states_layers
        assert (
            len(self.aux_hidden_states_layers) == 3
        ), "aux_hidden_states_layers is expected to be 3 layers for EAGLE3"


class SGLangEagle3TargetModel(Eagle3TargetModel):

    def __init__(self, model_runner: SGLangRunner):
        super().__init__()
        self.model_runner = model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangEagle3TargetModel":
        tp_size = dist.get_world_size(get_tp_group())
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            enable_return_hidden_states=True,
            disable_cuda_graph=True,  # we use piecewise cuda graph for prefill instead
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)
        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=0.7,
            gpu_id=torch.cuda.current_device(),
            tp_rank=dist.get_rank(get_tp_group()),
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
        )
        wrap_eagle3_logits_processors_in_module(
            model_runner.model, return_full_logits=False
        )
        return cls(model_runner)

    def set_aux_hidden_states_layers(
        self, aux_hidden_states_layers: Optional[List[int]] = None
    ) -> None:
        self.model_runner.model.set_eagle3_layers_to_capture(aux_hidden_states_layers)

    @torch.no_grad
    def _extend(
        self,
        reqs,
        capture_aux_hidden_states: bool = True,
        return_last_hidden_states: bool = False,
        return_logits: bool = False,
    ):
        # set the logits processor for the model runner
        for name, module in self.model_runner.model.named_modules():
            if isinstance(module, LogitsProcessorForEAGLE3):
                module.return_last_hidden_states = return_last_hidden_states
                module.return_logits = return_logits

        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        tree_cache = RadixCache(cache_params)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        self._maybe_prepare_mlp_sync_batch(batch)
        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
        eagle3_output, _ = self.model_runner.forward(forward_batch)

        aux_hidden_states_list = None
        input_lens = [len(req.origin_input_ids) for req in reqs]

        if return_logits:
            logits = torch.split(eagle3_output.logits, input_lens, dim=0)
        else:
            logits = [None] * len(reqs)

        if capture_aux_hidden_states:
            aux_hidden_states_list = torch.split(
                eagle3_output.aux_hidden_states, input_lens, dim=0
            )
        else:
            aux_hidden_states_list = [None] * len(reqs)

        if return_last_hidden_states:
            last_hidden_states = torch.split(
                eagle3_output.last_hidden_states, input_lens, dim=0
            )
        else:
            last_hidden_states = [None] * len(reqs)

        # TODO: can we not clear?
        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()
        return logits, aux_hidden_states_list, last_hidden_states

    def _maybe_prepare_mlp_sync_batch(self, batch: ScheduleBatch):
        if require_mlp_sync(self.model_runner.server_args):
            Scheduler.prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                speculative_num_draft_tokens=None,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

    def extend(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
        return_last_hidden_states: bool = False,
        return_logits: bool = True,
    ):
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1, top_k=1)
        reqs, data_cache = [], []

        if isinstance(input_ids, torch.Tensor):
            input_ids = torch.split(input_ids, 1, dim=0)
            attention_mask = torch.split(attention_mask, 1, dim=0)
            loss_mask = torch.split(loss_mask, 1, dim=0)

        for idx, (input_id_, attention_mask_, loss_mask_) in enumerate(
            zip(
                input_ids,
                attention_mask,
                loss_mask,
            )
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=input_id_.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            req.logprob_start_len = len(req.origin_input_ids) - 1
            data_cache.append([input_id_, attention_mask_, loss_mask_])
            reqs.append(req)

        logits_list, aux_hidden_states_list, last_hidden_states_list = self._extend(
            reqs,
            capture_aux_hidden_states=True,
            return_last_hidden_states=return_last_hidden_states,
            return_logits=return_logits,
        )

        return data_cache, logits_list, aux_hidden_states_list, last_hidden_states_list
    

def get_eagle3_target_model(
    pretrained_model_name_or_path: str,
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> Eagle3TargetModel:
    return SGLangEagle3TargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )