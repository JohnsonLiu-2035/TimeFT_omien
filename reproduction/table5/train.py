"""Run one pre-specified Table 5 experiment from the TimeFT project root."""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
from itertools import islice
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Models.FreqTokenizer import Model as PublicTimeFT
from data_provider.data_factory import data_dict
from reproduction.table5.model import PairedPatchTST
from utils.tools import EarlyStopping, adjust_learning_rate


DATASETS = {
    "ETTh1": ("ETTh1", "ETT-small/ETTh1.csv", "h", 7, 11520),
    "ETTm1": ("ETTm1", "ETT-small/ETTm1.csv", "15min", 7, 46080),
    "Weather": ("custom", "weather/weather.csv", "10min", 21, 42157),
}


def json_file(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def persist_shared(path, values):
    if not path.exists():
        np.save(path, values, allow_pickle=False)
    elif not np.array_equal(np.load(path, mmap_mode="r"), values):
        raise ValueError(f"Paired runs disagree on {path.name}")


def embedding_latency(model, history, device):
    model.eval()
    x = torch.as_tensor(history[:32], device=device).transpose(1, 2)
    x = (x - x.mean(-1, keepdim=True)) / (x.var(-1, keepdim=True, unbiased=False) + 1e-5).sqrt()
    samples = []
    with torch.inference_mode():
        for _ in range(10):
            model.patch_embedding(x)
        for _ in range(5):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(32):
                model.patch_embedding(x)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 32)
    return {"batch_size": len(x), "median_ms": float(np.median(samples)), "sample_ms": samples}


def evaluate(model, loader, device, limit=None, retain=False):
    model.eval()
    losses, predictions, truths, history = [], [], [], []
    with torch.inference_mode():
        for x, y, _, _ in islice(loader, limit):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            target = y[:, -96:].to(device, dtype=torch.float32, non_blocking=True)
            pred = model(x, None, None, None)
            losses.append(torch.nn.functional.mse_loss(pred, target).item())
            if retain:
                predictions.append(pred.cpu().numpy())
                truths.append(target.cpu().numpy())
                history.append(x.cpu().numpy())
    if not losses or not np.isfinite(losses).all():
        raise RuntimeError("Evaluation has no batches or contains non-finite losses")
    if retain:
        return tuple(np.concatenate(values) for values in (predictions, truths, history))
    return float(np.mean(losses))


def audit_public_model(args, model, sample, device):
    with torch.random.fork_rng(devices=[device.index]):
        original = PublicTimeFT(args).to(device).eval()
        optimizer = torch.optim.Adam(original.parameters(), lr=args.learning_rate)
        seen = []
        hook = original.patch_embedding.freq_embedding.register_forward_pre_hook(
            lambda _, inputs: seen.append(inputs[0].detach())
        )
        x = torch.as_tensor(sample, device=device, dtype=torch.float32).unsqueeze(0)
        with patch("layers.FreqEmbed.np.save"):
            prediction = original(x, None, None, None)
        hook.remove()
        head = original.head.linear.weight
        prediction.square().mean().backward()
        before = head.detach().clone()
        optimizer.step()
        after = head.detach()
        if head.grad.norm().item() <= 0 or not torch.equal(before, after):
            raise RuntimeError("Unexpected behavior in the public prediction head audit")
        centered = x - x.mean(1, keepdim=True)
        normalized = centered / (centered.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt()
        paired_tokens = model.patch_embedding.tokenize(normalized.transpose(1, 2)).flatten(-2).flatten(0, 1)
        torch.testing.assert_close(seen[0], paired_tokens, rtol=2e-6, atol=2e-6)
        return {"public_head_gradient_l2": head.grad.norm().item(),
                "public_head_update_l2": (after - before).norm().item(),
                "public_vs_paired_raw_token_max_abs_diff": (seen[0] - paired_tokens).abs().max().item()}


def run(args):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available")
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")

    key, relative_path, frequency, channels, test_start = DATASETS[args.dataset]
    args.root_path = str(args.data_dir / Path(relative_path).parent)
    args.data_path = Path(relative_path).name
    args.enc_in = channels
    args.freq = frequency
    args.task_name = "long_term_forecast"
    args.seq_len, args.label_len, args.pred_len = 320, 48, 96
    args.patch_len, args.stride, args.hidden_dim = 20, 20, 16
    args.d_model, args.d_ff, args.e_layers, args.n_heads = 512, 2048, 2, 4
    args.factor, args.dropout, args.activation = 3, 0.1, "gelu"
    args.augmentation_ratio = 0
    args.learning_rate, args.patience = 5e-5, 5
    args.lradj = getattr(args, "lradj", "type1")
    args.train_epochs, args.batch_size = (1, 4) if args.smoke else (70, 32)

    source = args.data_dir / relative_path
    with source.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    output = args.output / ("smoke" if args.smoke else "formal") / args.dataset
    trial = output / args.variant
    trial.mkdir(parents=True, exist_ok=False)
    data = {
        flag: data_dict[key](args=args, root_path=args.root_path, data_path=args.data_path,
                            flag=flag, size=[320, 48, 96], features="M", target="OT",
                            timeenc=1, freq=frequency)
        for flag in ("train", "val", "test")
    }
    loaders = {
        flag: DataLoader(split, batch_size=args.batch_size, shuffle=flag == "train",
                         num_workers=0, drop_last=False, pin_memory=True,
                         generator=torch.Generator().manual_seed(args.seed + index))
        for index, (flag, split) in enumerate(data.items())
    }

    model = PairedPatchTST(args, args.variant).to(device)
    if args.smoke and args.variant == "timeft":
        audit = audit_public_model(args, model, data["train"][0][0], device)
        json_file(trial / "public_code_audit.json", audit)
        print(f"public code audit {audit}", flush=True)
    adam = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    head = model.head.linear.weight
    if not any(p is head for group in adam.param_groups for p in group["params"]):
        raise RuntimeError("Prediction head is missing from Adam")
    encoder_init = hashlib.sha256()
    for parameter in model.encoder.state_dict().values():
        encoder_init.update(parameter.detach().cpu().numpy().tobytes())
    config = {**vars(args), "data_dir": str(args.data_dir), "output": str(args.output),
              "input_sha256": checksum, "windows": {k: len(v) for k, v in data.items()},
              "parameters": sum(p.numel() for p in model.parameters()), "tokens": model.token_count,
              "head_in_features": head.shape[1], "encoder_initial_sha256": encoder_init.hexdigest(),
              "python": platform.python_version(),
              "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(device), "matmul_precision": torch.get_float32_matmul_precision()}
    json_file(trial / "config.json", config)
    print(json.dumps(config), flush=True)

    stopper = EarlyStopping(patience=args.patience, verbose=True)
    torch.cuda.reset_peak_memory_stats(device)
    records, best_epoch, best_val, head_check = [], 0, math.inf, None
    start = perf_counter()
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        step_losses = []
        epoch_start = perf_counter()
        for x, y, _, _ in islice(loaders["train"], 12 if args.smoke else None):
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            target = y[:, -96:].to(device, dtype=torch.float32, non_blocking=True)
            adam.zero_grad(set_to_none=True)
            pred = model(x, None, None, None)
            if pred.shape != target.shape:
                raise RuntimeError(f"Unexpected forecast shape {pred.shape} vs {target.shape}")
            loss = torch.nn.functional.mse_loss(pred, target)
            loss.backward()
            if head_check is None:
                gradient = head.grad.norm().item() if head.grad is not None else 0
                before = head.detach().clone()
                adam.step()
                update = (head.detach() - before).norm().item()
                if not (math.isfinite(gradient) and math.isfinite(update) and gradient > 0 and update > 0):
                    raise RuntimeError("Prediction head did not receive a finite nonzero update")
                head_check = {"gradient_l2": gradient, "first_update_l2": update}
                print(f"head check {head_check}", flush=True)
            else:
                adam.step()
            step_losses.append(loss.detach())
        train_loss = torch.stack(step_losses).mean().item()
        train_seconds = perf_counter() - epoch_start
        if not math.isfinite(train_loss):
            raise RuntimeError(f"Non-finite training loss in epoch {epoch}")
        val_start = perf_counter()
        val_loss = evaluate(model, loaders["val"], device, 4 if args.smoke else None)
        row = {"epoch": epoch, "train_mse": train_loss, "val_batch_mean_mse": val_loss,
               "train_seconds": train_seconds, "val_seconds": perf_counter() - val_start,
               "lr": adam.param_groups[0]["lr"], "steps": len(step_losses)}
        records.append(row)
        with (trial / "epochs.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=row.keys())
            writer.writeheader()
            writer.writerows(records)
        print(json.dumps(row), flush=True)
        if val_loss <= best_val:
            best_val, best_epoch = val_loss, epoch
        stopper(val_loss, model, str(trial))
        if stopper.early_stop:
            break
        adjust_learning_rate(adam, epoch, args)

    train_wall = perf_counter() - start
    train_peak = torch.cuda.max_memory_allocated(device)
    restored = PairedPatchTST(args, args.variant).to(device)
    restored.load_state_dict(torch.load(trial / "checkpoint.pth", map_location=device, weights_only=True), strict=True)
    infer_start = perf_counter()
    pred, truth, past = evaluate(restored, loaders["test"], device, 4 if args.smoke else None, retain=True)
    infer_wall = perf_counter() - infer_start
    if not np.isfinite(pred).all() or not np.isfinite(truth).all() or not np.isfinite(past).all():
        raise RuntimeError("Non-finite predictions or inputs")
    np.save(trial / "pred.npy", pred, allow_pickle=False)
    persist_shared(output / "true.npy", truth)
    persist_shared(output / "history.npy", past)
    persist_shared(output / "target_row.npy", test_start + np.arange(len(pred)))
    frame = pd.read_csv(source, usecols=["date"])
    persist_shared(output / "target_time.npy", pd.to_datetime(frame["date"]).to_numpy()[test_start:test_start + len(pred)])
    if not args.smoke and len(pred) != len(data["test"]):
        raise RuntimeError("Unexpected number of retained test windows")
    scaler = data["train"].scaler
    header = list(pd.read_csv(source, nrows=0).columns)
    feature_names = [column for column in header if column not in ("date", "OT")] + ["OT"]
    json_file(output / "data.json", {"channels": feature_names, "raw_test_target_start": test_start,
               "row_indexing": "zero-based CSV data row", "scale_mean": scaler.mean_.tolist(),
               "scale_std": scaler.scale_.tolist(), "frequency": frequency, "sha256": checksum})
    error = pred.astype(np.float64) - truth
    summary = {"mse": float(np.mean(error ** 2)), "mae": float(np.mean(np.abs(error))),
               "best_epoch": best_epoch, "best_validation_mse": best_val, "epochs": len(records),
               "train_wall_seconds": train_wall, "test_wall_seconds": infer_wall,
               "peak_training_allocated_mib": train_peak / 2 ** 20, "shape": list(pred.shape),
               "head_check": head_check, "embedding_latency": embedding_latency(restored, past, device)}
    json_file(trial / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--variant", choices=("patch", "timeft"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--lradj", choices=("type1", "cosine"), default="type1")
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())
