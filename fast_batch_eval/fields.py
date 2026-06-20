from __future__ import annotations

TRIAL_FIELDNAMES = [
    "model_name",
    "exp_name",
    "iter_idx",
    "batch_idx",
    "sample_idx_in_batch",
    "sample_id",
    "status",
    "error",
    "target_offset_sec",
    "target_class_index",
    "pred_offset_sec",
    "pred_class_index",
    "pred_probability",
    "top1_correct",
    "top1_pm1_class_correct",
    "top5_correct",
    "abs_class_error",
    "abs_offset_error_sec",

    # Exact crop metadata for visualizer.
    "v_start_i_sec",
    "v_end_i_sec",
    "a_start_i_sec",
    "a_end_i_sec",
    "crop_duration_sec",

    # Predicted-offset reconstruction for visualizer.
    "pred_a_start_i_sec",
    "pred_a_end_i_sec",
]

for _rank in range(1, 6):
    TRIAL_FIELDNAMES.extend(
        [
            f"top{_rank}_offset_sec",
            f"top{_rank}_probability",
            f"top{_rank}_class_index",
            f"top{_rank}_logit",
        ]
    )

SUMMARY_FIELDNAMES = [
    "model_name",
    "exp_name",
    "num_clips",
    "data_iter",
    "num_trials",
    "num_errors",
    "num_classes",
    "grid_min_sec",
    "grid_max_sec",
    "grid_step_sec",
    "accuracy_1",
    "accuracy_1_pm1_class",
    "accuracy_5",
    "mean_abs_class_error",
    "median_abs_class_error",
    "mean_abs_offset_error_sec",
    "median_abs_offset_error_sec",
    "accuracy_1_median_over_all_classes_missing_as_zero",
    "accuracy_1_median_over_present_classes",
    "target_count_per_class_json",
    "correct_target_class_count_per_tar_json",
    "correct_target_offset_sec_count_per_tar_json",
    "zero_offset_target_class_index",
    "zero_offset_num_trials",
    "zero_offset_pred_class_count_json",
    "zero_offset_pred_class_frequency_json",
]
