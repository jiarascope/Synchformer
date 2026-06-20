from __future__ import annotations

import csv
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from scripts.train_utils import prepare_inputs

from .datasets import build_eval_dataset
from .fields import TRIAL_FIELDNAMES
from .metrics import (
    build_summary_row,
    find_zero_target_idx,
    new_metric_state,
    update_metric_state,
)
from .modeling import load_synchformer
from .utils import (
    coerce_paths,
    get_target_class_and_offset,
    make_amp_context,
    tensor_to_cpu_1d,
    unpack_offset_logits,
)


def _make_loader(dataset, args, device: torch.device) -> DataLoader:
    loader_kwargs: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": (device.type == "cuda"),
        "drop_last": False,
        "persistent_workers": (args.num_workers > 0 and args.persistent_workers),
    }

    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    return DataLoader(dataset, **loader_kwargs)


def _empty_error_row(
    *,
    model_name: str,
    exp_name: str,
    iter_idx: int | str,
    batch_idx: int,
    sample_idx_in_batch: int,
    sample_id: str,
    error: Exception,
):
    row = {
        "model_name": model_name,
        "exp_name": exp_name,
        "iter_idx": iter_idx,
        "batch_idx": batch_idx,
        "sample_idx_in_batch": sample_idx_in_batch,
        "sample_id": sample_id,
        "status": "error",
        "error": repr(error),
    }
    for field in TRIAL_FIELDNAMES:
        row.setdefault(field, "")
    return row


def evaluate_one_model(
    *,
    model_name: str,
    exp_name: str,
    args,
    trial_writer: csv.DictWriter,
    summary_writer: csv.DictWriter,
    device: torch.device,
):
    print("")
    print(f"Loading model {model_name}: {exp_name}")

    cfg, model, transform, grid = load_synchformer(
        exp_name,
        device,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
    )
    grid_tensor = torch.as_tensor(grid).float().cpu()
    zero_target_idx = find_zero_target_idx(grid_tensor)

    print(f"Loaded {model_name}. Offset grid: {[round(x, 4) for x in grid_tensor.tolist()]}")
    print("Building cached tar dataset and flattened stochastic-trial dataset")

    dataset, num_clips = build_eval_dataset(args, transform)
    planned_trials = num_clips * args.data_iter

    print(f"Dataset clips: {num_clips}")
    print(f"data_iter: {args.data_iter}")
    print(f"Total planned trials for this model: {planned_trials}")
    print(
        "Cache: "
        f"decoded={args.cache_decoded}, decoded_cache_size={args.decoded_cache_size}, "
        f"tar_handles={args.cache_tar_handles}, tar_handle_cache_size={args.tar_handle_cache_size}"
    )

    loader = _make_loader(dataset, args, device)

    metric_state = new_metric_state(zero_target_idx=zero_target_idx)
    num_errors = 0

    for batch_idx, batch in enumerate(loader):
        sample_ids = coerce_paths(batch["path"])
        if "iter_idx" in batch:
            iter_idx_cpu = tensor_to_cpu_1d(batch["iter_idx"], dtype=torch.long)
        else:
            iter_idx_cpu = torch.zeros(len(sample_ids), dtype=torch.long)

        try:
            aud, vid, targets = prepare_inputs(batch, device)
            target_idx_cpu, target_offset_cpu = get_target_class_and_offset(targets, grid_tensor)

            v_start_cpu = tensor_to_cpu_1d(targets["v_start_i_sec"], dtype=torch.float32)

            if "a_start_i_sec" in targets:
                a_start_cpu = tensor_to_cpu_1d(targets["a_start_i_sec"], dtype=torch.float32)
            else:
                a_start_cpu = v_start_cpu + target_offset_cpu

            if "crop_duration_sec" in targets:
                crop_duration_cpu = tensor_to_cpu_1d(targets["crop_duration_sec"], dtype=torch.float32)
            else:
                crop_duration_cpu = torch.full_like(v_start_cpu, 5.0)

            v_end_cpu = v_start_cpu + crop_duration_cpu
            a_end_cpu = a_start_cpu + crop_duration_cpu

            with torch.inference_mode():
                with make_amp_context(cfg, device):
                    _, logits = model(vid, aud)

            off_logits = unpack_offset_logits(logits).float()

            # Always compute at least top-5 because the CSV and summary contain top5 fields.
            k = min(max(int(args.topk), 5), off_logits.shape[-1])
            off_probs = torch.softmax(off_logits, dim=-1)
            top_probs, top_indices = torch.topk(off_probs, k=k, dim=-1)
            top_logits = torch.gather(off_logits.detach(), 1, top_indices)

            # Copy only compact per-row outputs to CPU.
            top_probs_cpu = top_probs.detach().cpu()
            top_indices_cpu = top_indices.detach().cpu()
            top_logits_cpu = top_logits.detach().cpu()

            pred_idx_cpu = top_indices_cpu[:, 0].long()
            pred_prob_cpu = top_probs_cpu[:, 0].float()
            pred_offset_cpu = grid_tensor[pred_idx_cpu]

            batch_size_actual = top_probs_cpu.shape[0]
            if batch_size_actual != len(sample_ids):
                raise RuntimeError(
                    f"Batch size mismatch: logits has {batch_size_actual}, "
                    f"sample_ids has {len(sample_ids)}"
                )

            rows = []
            for b in range(batch_size_actual):
                iter_idx = int(iter_idx_cpu[b].item())
                target_idx = int(target_idx_cpu[b].item())
                target_offset = float(target_offset_cpu[b].item())
                pred_idx = int(pred_idx_cpu[b].item())
                pred_offset = float(pred_offset_cpu[b].item())
                pred_prob = float(pred_prob_cpu[b].item())
                top_idx_list = [int(x.item()) for x in top_indices_cpu[b, : min(5, k)]]

                top1_correct = int(pred_idx == target_idx)
                pm1_correct = int(abs(pred_idx - target_idx) <= 1)
                top5_correct = int(target_idx in set(top_idx_list))
                abs_class_error = abs(pred_idx - target_idx)
                abs_offset_error = abs(pred_offset - target_offset)

                v_start = float(v_start_cpu[b].item())
                v_end = float(v_end_cpu[b].item())
                a_start = float(a_start_cpu[b].item())
                a_end = float(a_end_cpu[b].item())
                crop_duration = float(crop_duration_cpu[b].item())

                pred_a_start = v_start + pred_offset
                pred_a_end = pred_a_start + crop_duration

                update_metric_state(
                    metric_state,
                    sample_id=sample_ids[b],
                    target_idx=target_idx,
                    pred_idx=pred_idx,
                    top_indices_for_top5=top_idx_list,
                    target_offset=target_offset,
                    pred_offset=pred_offset,
                    zero_target_idx=zero_target_idx,
                )

                row = {
                    "model_name": model_name,
                    "exp_name": exp_name,
                    "iter_idx": iter_idx,
                    "batch_idx": batch_idx,
                    "sample_idx_in_batch": b,
                    "sample_id": sample_ids[b],
                    "status": "ok",
                    "error": "",
                    "target_offset_sec": target_offset,
                    "target_class_index": target_idx,
                    "pred_offset_sec": pred_offset,
                    "pred_class_index": pred_idx,
                    "pred_probability": pred_prob,
                    "top1_correct": top1_correct,
                    "top1_pm1_class_correct": pm1_correct,
                    "top5_correct": top5_correct,
                    "abs_class_error": abs_class_error,
                    "abs_offset_error_sec": abs_offset_error,
                    "v_start_i_sec": v_start,
                    "v_end_i_sec": v_end,
                    "a_start_i_sec": a_start,
                    "a_end_i_sec": a_end,
                    "crop_duration_sec": crop_duration,
                    "pred_a_start_i_sec": pred_a_start,
                    "pred_a_end_i_sec": pred_a_end,
                }

                for rank in range(1, 6):
                    if rank <= k:
                        cls_idx = int(top_indices_cpu[b, rank - 1].item())
                        row[f"top{rank}_offset_sec"] = float(grid_tensor[cls_idx].item())
                        row[f"top{rank}_probability"] = float(top_probs_cpu[b, rank - 1].item())
                        row[f"top{rank}_class_index"] = cls_idx
                        row[f"top{rank}_logit"] = float(top_logits_cpu[b, rank - 1].item())
                    else:
                        row[f"top{rank}_offset_sec"] = ""
                        row[f"top{rank}_probability"] = ""
                        row[f"top{rank}_class_index"] = ""
                        row[f"top{rank}_logit"] = ""

                rows.append(row)

            trial_writer.writerows(rows)

        except Exception as exc:
            num_errors += len(sample_ids)
            rows = []
            for b, sample_id in enumerate(sample_ids):
                iter_idx_value: int | str = ""
                if b < len(iter_idx_cpu):
                    iter_idx_value = int(iter_idx_cpu[b].item())
                rows.append(
                    _empty_error_row(
                        model_name=model_name,
                        exp_name=exp_name,
                        iter_idx=iter_idx_value,
                        batch_idx=batch_idx,
                        sample_idx_in_batch=b,
                        sample_id=sample_id,
                        error=exc,
                    )
                )
            trial_writer.writerows(rows)

            print(f"ERROR [{model_name}] batch={batch_idx}: {exc}")
            if not args.continue_on_error:
                raise

        if args.log_every > 0 and (batch_idx + 1) % args.log_every == 0:
            done_trials = metric_state["num_trials"]
            if done_trials > 0:
                acc1 = metric_state["top1_correct"] / done_trials
                acc_pm1 = metric_state["pm1_correct"] / done_trials
                print(
                    f"  {model_name}: {done_trials}/{planned_trials} ok trials, "
                    f"Acc@1={acc1:.4f}, Acc@1±1cls={acc_pm1:.4f}"
                )

    summary_row = build_summary_row(
        model_name=model_name,
        exp_name=exp_name,
        num_clips=num_clips,
        data_iter=args.data_iter,
        num_errors=num_errors,
        grid_tensor=grid_tensor,
        state=metric_state,
    )
    summary_writer.writerow(summary_row)

    print("")
    print(f"Summary for {model_name} ({exp_name})")
    print(f"  trials ok:      {summary_row['num_trials']}")
    print(f"  errors:         {summary_row['num_errors']}")
    print(f"  Acc@1:          {summary_row['accuracy_1']:.6f}")
    print(f"  Acc@1 ±1 class: {summary_row['accuracy_1_pm1_class']:.6f}")
    print(f"  Acc@5:          {summary_row['accuracy_5']:.6f}")
    print(f"  Mean abs err:   {summary_row['mean_abs_offset_error_sec']:.6f} sec")
    print(f"  Zero-offset trials: {summary_row['zero_offset_num_trials']}")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
