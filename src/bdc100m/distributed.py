from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(slots=True)
class DistEnv:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistEnv:
    enabled = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if enabled:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if not torch.cuda.is_available():
            raise RuntimeError("DDP mode currently requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return DistEnv(True, rank, local_rank, world_size, torch.device("cuda", local_rank))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return DistEnv(False, 0, 0, 1, device)


def cleanup_distributed(env: DistEnv) -> None:
    if env.enabled and dist.is_initialized():
        dist.destroy_process_group()
