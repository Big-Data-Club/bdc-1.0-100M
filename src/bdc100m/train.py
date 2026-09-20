from __future__ import annotations

import argparse
import contextlib
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import load_checkpoint, save_checkpoint
from .config import load_config
from .data import TokenBin
from .distributed import cleanup_distributed, init_distributed
from .model import BDCModel
from .utils import append_jsonl, cosine_lr, human_count, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pretrain BDC-1.0-100M from scratch")
    p.add_argument("--config", default="configs/bdc_1_0_100m.yaml")
    return p.parse_args()


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    if name == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            print("[warning] bf16 unsupported on this GPU; falling back to float16")
            return torch.float16
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, betas: tuple[float, float]):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Matrix weights decay; norm vectors and other 1D params do not.
        (decay if p.dim() >= 2 else no_decay).append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused_ok = torch.cuda.is_available() and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    kwargs = {"lr": lr, "betas": betas}
    if fused_ok:
        kwargs["fused"] = True
    return torch.optim.AdamW(groups, **kwargs)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_data: TokenBin,
    batches: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> float:
    model.eval()
    losses = []
    amp = torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda" and dtype != torch.float32)
    with amp:
        for _ in range(batches):
            x, y = val_data.get_batch(batch_size, seq_len, device, generator)
            _, loss = model(x, y)
            assert loss is not None
            losses.append(loss.detach().float())
    model.train()
    return torch.stack(losses).mean().item()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    env = init_distributed()
    tcfg, mcfg = cfg.training, cfg.model

    seed_everything(tcfg.seed + env.rank)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    train_data = TokenBin(tcfg.train_data)
    val_data = TokenBin(tcfg.val_data)

    raw_model = BDCModel(mcfg).to(env.device)
    param_count = raw_model.num_parameters()
    if env.is_main:
        print(f"device={env.device} world_size={env.world_size}")
        print(f"parameters={param_count:,} ({human_count(param_count)})")
        print(f"train_tokens={len(train_data):,} val_tokens={len(val_data):,}")

    if tcfg.compile:
        raw_model = torch.compile(raw_model)

    model: torch.nn.Module = raw_model
    if env.enabled:
        model = DDP(raw_model, device_ids=[env.local_rank], broadcast_buffers=False)

    base_model = model.module if isinstance(model, DDP) else model
    optimizer = build_optimizer(
        base_model,
        tcfg.learning_rate,
        tcfg.weight_decay,
        (tcfg.beta1, tcfg.beta2),
    )

    start_step = 0
    if tcfg.resume_from:
        start_step = load_checkpoint(tcfg.resume_from, base_model, optimizer, map_location=env.device)
        if env.is_main:
            print(f"resumed_from={tcfg.resume_from} step={start_step}")

    dtype = resolve_dtype(tcfg.dtype, env.device)
    use_amp = env.device.type == "cuda" and dtype != torch.float32
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)

    train_gen = torch.Generator(device="cpu").manual_seed(tcfg.seed + 10_000 * env.rank)
    val_gen = torch.Generator(device="cpu").manual_seed(tcfg.seed + 999_999)

    tokens_per_step = (
        tcfg.micro_batch_size
        * mcfg.max_seq_len
        * tcfg.gradient_accumulation_steps
        * env.world_size
    )
    out_dir = Path(tcfg.output_dir)
    metrics_path = out_dir / "metrics.jsonl"
    if env.is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"effective_tokens_per_step={tokens_per_step:,}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    last_time = time.perf_counter()

    try:
        for step in range(start_step, tcfg.max_steps):
            lr = cosine_lr(
                step,
                tcfg.max_steps,
                tcfg.warmup_steps,
                tcfg.learning_rate,
                tcfg.min_learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            step_loss = 0.0
            for micro in range(tcfg.gradient_accumulation_steps):
                sync = micro == tcfg.gradient_accumulation_steps - 1
                ddp_ctx = contextlib.nullcontext()
                if isinstance(model, DDP) and not sync:
                    ddp_ctx = model.no_sync()

                x, y = train_data.get_batch(
                    tcfg.micro_batch_size, mcfg.max_seq_len, env.device, train_gen
                )
                amp_ctx = torch.autocast(
                    device_type=env.device.type,
                    dtype=dtype,
                    enabled=use_amp,
                )
                with ddp_ctx, amp_ctx:
                    _, loss = model(x, y)
                    assert loss is not None
                    loss = loss / tcfg.gradient_accumulation_steps
                scaler.scale(loss).backward()
                step_loss += loss.detach().float().item()

            if tcfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(base_model.parameters(), tcfg.grad_clip)
            else:
                grad_norm = torch.tensor(float("nan"), device=env.device)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            completed_step = step + 1
            if env.is_main and completed_step % tcfg.log_interval == 0:
                now = time.perf_counter()
                elapsed = now - last_time
                last_time = now
                toks_s = tokens_per_step * tcfg.log_interval / max(elapsed, 1e-9)
                record = {
                    "step": completed_step,
                    "train_loss": step_loss,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_sec": toks_s,
                    "tokens_seen": completed_step * tokens_per_step,
                }
                append_jsonl(metrics_path, record)
                print(
                    f"step={completed_step:7d} loss={step_loss:.4f} lr={lr:.3e} "
                    f"grad={float(grad_norm):.3f} tok/s={toks_s:,.0f}"
                )

            if completed_step % tcfg.eval_interval == 0:
                # Only rank 0 evaluates; synchronize ranks around eval to keep DDP in lock-step.
                if env.enabled:
                    torch.distributed.barrier()
                if env.is_main:
                    val_loss = evaluate(
                        base_model,
                        val_data,
                        tcfg.eval_batches,
                        tcfg.micro_batch_size,
                        mcfg.max_seq_len,
                        env.device,
                        dtype,
                        val_gen,
                    )
                    ppl = math.exp(min(val_loss, 20.0))
                    append_jsonl(
                        metrics_path,
                        {"step": completed_step, "val_loss": val_loss, "perplexity": ppl},
                    )
                    print(f"eval step={completed_step} val_loss={val_loss:.4f} ppl={ppl:.2f}")
                if env.enabled:
                    torch.distributed.barrier()

            if completed_step % tcfg.checkpoint_interval == 0:
                if env.is_main:
                    ckpt = save_checkpoint(
                        out_dir,
                        completed_step,
                        base_model,
                        optimizer,
                        {"model": asdict(mcfg), "training": asdict(tcfg)},
                        keep_last=tcfg.keep_last_checkpoints,
                    )
                    print(f"checkpoint={ckpt}")
                if env.enabled:
                    torch.distributed.barrier()

        if env.enabled:
            torch.distributed.barrier()
        if env.is_main:
            save_checkpoint(
                out_dir,
                tcfg.max_steps,
                base_model,
                optimizer,
                {"model": asdict(mcfg), "training": asdict(tcfg)},
                keep_last=tcfg.keep_last_checkpoints,
            )
        if env.enabled:
            torch.distributed.barrier()
    finally:
        cleanup_distributed(env)


if __name__ == "__main__":
    main()
