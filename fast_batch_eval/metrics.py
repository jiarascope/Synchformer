from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Dict, Sequence

import numpy as np
import torch

from .utils import extract_tar_name, offset_key


def nested_defaultdict_int() -> defaultdict:
    """Nested defaultdict for per-tar class/offset counts."""
    return defaultdict(int)


def sorted_nested_counts(value: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    """Convert nested defaultdict counts into sorted plain dicts for JSON output."""
    return {
        str(outer_key): {
            str(inner_key): int(inner_value)
            for inner_key, inner_value in sorted(inner_counts.items())
        }
        for outer_key, inner_counts in sorted(value.items())
    }


def new_metric_state(*, zero_target_idx: int | None) -> Dict[str, Any]:
    return {
        "num_trials": 0,
        "top1_correct": 0,
        "pm1_correct": 0,
        "top5_correct": 0,
        "abs_class_errors": [],
        "abs_offset_errors": [],
        "target_counts": defaultdict(int),
        "per_class_total": defaultdict(int),
        "per_class_top1": defaultdict(int),
        "correct_target_class_count_per_tar": defaultdict(nested_defaultdict_int),
        "correct_target_offset_sec_count_per_tar": defaultdict(nested_defaultdict_int),
        "zero_target_idx": zero_target_idx,
        "zero_offset_num_trials": 0,
        "zero_offset_pred_class_counts": defaultdict(int),
    }


def find_zero_target_idx(grid_tensor: torch.Tensor) -> int | None:
    zero_target_idx = None
    if grid_tensor.numel() > 0:
        closest_zero_idx = int(torch.argmin(torch.abs(grid_tensor)).item())
        if math.isclose(float(grid_tensor[closest_zero_idx].item()), 0.0, abs_tol=1e-6):
            zero_target_idx = closest_zero_idx
    return zero_target_idx


def update_metric_state(
    state: Dict[str, Any],
    *,
    sample_id: str,
    target_idx: int,
    pred_idx: int,
    top_indices_for_top5: Sequence[int],
    target_offset: float,
    pred_offset: float,
    zero_target_idx: int | None,
):
    top1 = int(pred_idx == target_idx)
    pm1 = int(abs(pred_idx - target_idx) <= 1)
    top5 = int(target_idx in set(int(i) for i in top_indices_for_top5))
    abs_cls_err = abs(pred_idx - target_idx)
    abs_off_err = abs(pred_offset - target_offset)

    state["num_trials"] += 1
    state["top1_correct"] += top1
    state["pm1_correct"] += pm1
    state["top5_correct"] += top5
    state["abs_class_errors"].append(abs_cls_err)
    state["abs_offset_errors"].append(abs_off_err)
    state["target_counts"][target_idx] += 1
    state["per_class_total"][target_idx] += 1
    state["per_class_top1"][target_idx] += top1

    tar_name = extract_tar_name(sample_id)

    if top1:
        state["correct_target_class_count_per_tar"][tar_name][str(target_idx)] += 1
        state["correct_target_offset_sec_count_per_tar"][tar_name][offset_key(target_offset)] += 1

    if zero_target_idx is not None and target_idx == zero_target_idx:
        state["zero_offset_num_trials"] += 1
        state["zero_offset_pred_class_counts"][pred_idx] += 1


def build_summary_row(
    *,
    model_name: str,
    exp_name: str,
    num_clips: int,
    data_iter: int,
    num_errors: int,
    grid_tensor: torch.Tensor,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    num_trials = int(state["num_trials"])
    num_classes = int(grid_tensor.numel())

    if num_trials == 0:
        acc1 = acc_pm1 = acc5 = float("nan")
        mean_cls = med_cls = mean_off = med_off = float("nan")
    else:
        acc1 = state["top1_correct"] / num_trials
        acc_pm1 = state["pm1_correct"] / num_trials
        acc5 = state["top5_correct"] / num_trials
        mean_cls = float(np.mean(state["abs_class_errors"]))
        med_cls = float(np.median(state["abs_class_errors"]))
        mean_off = float(np.mean(state["abs_offset_errors"]))
        med_off = float(np.median(state["abs_offset_errors"]))

    per_class_acc_all = []
    per_class_acc_present = []
    for c in range(num_classes):
        total_c = state["per_class_total"].get(c, 0)
        if total_c > 0:
            acc_c = state["per_class_top1"].get(c, 0) / total_c
            per_class_acc_present.append(acc_c)
            per_class_acc_all.append(acc_c)
        else:
            per_class_acc_all.append(0.0)

    median_all = float(np.median(per_class_acc_all)) if per_class_acc_all else float("nan")
    median_present = (
        float(np.median(per_class_acc_present)) if per_class_acc_present else float("nan")
    )

    if num_classes >= 2:
        grid_step = float((grid_tensor[1] - grid_tensor[0]).item())
    else:
        grid_step = float("nan")

    target_counts = {str(c): int(state["target_counts"].get(c, 0)) for c in range(num_classes)}

    zero_offset_num_trials = int(state.get("zero_offset_num_trials", 0))
    zero_pred_counts = {
        str(c): int(state["zero_offset_pred_class_counts"].get(c, 0))
        for c in range(num_classes)
    }
    zero_pred_freq = {}
    for c in range(num_classes):
        count_c = int(state["zero_offset_pred_class_counts"].get(c, 0))
        zero_pred_freq[str(c)] = {
            "count": count_c,
            "frequency": (count_c / zero_offset_num_trials) if zero_offset_num_trials > 0 else 0.0,
            "pred_offset_sec": float(grid_tensor[c].item()),
        }

    zero_target_idx = state.get("zero_target_idx")

    return {
        "model_name": model_name,
        "exp_name": exp_name,
        "num_clips": num_clips,
        "data_iter": data_iter,
        "num_trials": num_trials,
        "num_errors": num_errors,
        "num_classes": num_classes,
        "grid_min_sec": float(grid_tensor.min().item()),
        "grid_max_sec": float(grid_tensor.max().item()),
        "grid_step_sec": grid_step,
        "accuracy_1": acc1,
        "accuracy_1_pm1_class": acc_pm1,
        "accuracy_5": acc5,
        "mean_abs_class_error": mean_cls,
        "median_abs_class_error": med_cls,
        "mean_abs_offset_error_sec": mean_off,
        "median_abs_offset_error_sec": med_off,
        "accuracy_1_median_over_all_classes_missing_as_zero": median_all,
        "accuracy_1_median_over_present_classes": median_present,
        "target_count_per_class_json": json.dumps(target_counts, sort_keys=True),
        "correct_target_class_count_per_tar_json": json.dumps(
            sorted_nested_counts(state["correct_target_class_count_per_tar"]),
            sort_keys=True,
        ),
        "correct_target_offset_sec_count_per_tar_json": json.dumps(
            sorted_nested_counts(state["correct_target_offset_sec_count_per_tar"]),
            sort_keys=True,
        ),
        "zero_offset_target_class_index": "" if zero_target_idx is None else int(zero_target_idx),
        "zero_offset_num_trials": zero_offset_num_trials,
        "zero_offset_pred_class_count_json": json.dumps(zero_pred_counts, sort_keys=True),
        "zero_offset_pred_class_frequency_json": json.dumps(zero_pred_freq, sort_keys=True),
    }
