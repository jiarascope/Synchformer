from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --- Input files ---
# Baseline / original run
before_path = "/home/jiaray/mrBean/plots/baseline/graphs/valid+youtubeClips/alignment_scores.csv"

# New / after run
after_path = "/home/jiaray/mrBean/plots/baseline/graphs/youtube+Valid_after/alignment_scoresAFTER.csv"

# Directory containing videos to highlight
highlight_video_dir = "/home/jiaray/mrBean/data/baseline_data/conductingValid_clips"

# --- Output directory ---
out_dir = Path("/home/jiaray/mrBean/plots/baseline/graphs/compare_alignment")
out_dir.mkdir(parents=True, exist_ok=True)

# --- Choose score ---
score_col = "diag_margin"


def collect_exact_stems_from_dir(video_dir):
    """
    Collect exact filename stems from the input video directory.

    Example:
      /path/to/abc123.mp4 -> abc123

    These stems are matched exactly against Path(csv_video_path).stem.
    """
    video_dir = Path(video_dir)

    if not video_dir.exists():
        raise FileNotFoundError(f"Highlight video directory does not exist: {video_dir}")

    video_extensions = {
        ".mp4", ".mkv", ".webm", ".mov", ".avi",
        ".flv", ".m4v", ".mpg", ".mpeg"
    }

    stems = set()

    for path in video_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in video_extensions:
            stems.add(path.stem)

    return stems


# --- Load data ---
before = pd.read_csv(before_path)
after = pd.read_csv(after_path)

# --- Validate ---
for name, df in [("before", before), ("after", after)]:
    if "video" not in df.columns:
        raise ValueError(f"'video' column not found in {name} CSV. Columns: {list(df.columns)}")
    if score_col not in df.columns:
        raise ValueError(f"'{score_col}' not found in {name} CSV. Columns: {list(df.columns)}")


# --- Build exact names ---
before["video_id"] = before["video"].apply(lambda x: Path(str(x)).stem)
after["video_id"] = after["video"].apply(lambda x: Path(str(x)).stem)

highlight_names = collect_exact_stems_from_dir(highlight_video_dir)

print(f"\nFound {len(highlight_names)} video names in highlight directory.")
print("Example names:", sorted(list(highlight_names))[:10])


# --- Match before/after rows by exact video_id ---
# This is better than matching by full path if the two runs live in different folders.
merged = before[["video_id", "video", score_col]].merge(
    after[["video_id", "video", score_col]],
    on="video_id",
    suffixes=("_before_path", "_after_path")
)

# Rename score columns after merge
merged = merged.rename(
    columns={
        f"{score_col}_before_path": f"{score_col}_before",
        f"{score_col}_after_path": f"{score_col}_after",
    }
)

# The rename above may not trigger depending on pandas suffix behavior,
# so handle the expected column names explicitly.
if f"{score_col}_before_path" in merged.columns:
    merged = merged.rename(columns={f"{score_col}_before_path": f"{score_col}_before"})
if f"{score_col}_after_path" in merged.columns:
    merged = merged.rename(columns={f"{score_col}_after_path": f"{score_col}_after"})

# If pandas made score_col_before_path / score_col_after_path differently,
# print columns for debugging.
if f"{score_col}_before" not in merged.columns or f"{score_col}_after" not in merged.columns:
    raise ValueError(f"Could not find merged score columns. Columns are: {list(merged.columns)}")

if merged.empty:
    raise ValueError(
        "No rows matched between the two CSVs using exact filename stem. "
        "Check whether Path(video).stem is the same in both CSVs."
    )


# --- Highlight exact matches ---
merged["is_highlight"] = merged["video_id"].isin(highlight_names)

matched_highlights = int(merged["is_highlight"].sum())
print(f"Matched {matched_highlights} highlighted videos in the comparison.")

if matched_highlights == 0:
    print("\nNo highlighted videos matched.")
    print("First CSV video_ids:", sorted(merged["video_id"].head(10).tolist()))
    print("First highlight names:", sorted(list(highlight_names))[:10])


# --- Compute deltas ---
# Positive means AFTER is higher than BEFORE.
merged["delta"] = merged[f"{score_col}_after"] - merged[f"{score_col}_before"]
merged["abs_delta"] = merged["delta"].abs()

merged = merged.sort_values("delta").reset_index(drop=True)

# Save table
comparison_csv = out_dir / f"{score_col}_comparison_exact_name_highlights.csv"
merged.to_csv(comparison_csv, index=False)


# --- Styling ---
normal_color = "lightgray"
highlight_color = "tab:orange"

bar_colors = [
    highlight_color if is_h else normal_color
    for is_h in merged["is_highlight"]
]


# --- Plot 1: before vs after paired dot plot ---
plt.figure(figsize=(11, max(6, len(merged) * 0.35)))

y = range(len(merged))

for i, row in enumerate(merged.itertuples()):
    line_color = highlight_color if row.is_highlight else normal_color
    line_width = 2.5 if row.is_highlight else 1.0
    alpha = 0.95 if row.is_highlight else 0.45

    before_val = getattr(row, f"{score_col}_before")
    after_val = getattr(row, f"{score_col}_after")

    plt.plot(
        [before_val, after_val],
        [i, i],
        color=line_color,
        linewidth=line_width,
        alpha=alpha,
        zorder=1
    )

plt.scatter(
    merged[f"{score_col}_before"],
    y,
    label="Before",
    color="tab:blue",
    s=[70 if h else 30 for h in merged["is_highlight"]],
    zorder=2
)

plt.scatter(
    merged[f"{score_col}_after"],
    y,
    label="After",
    color="tab:red",
    s=[70 if h else 30 for h in merged["is_highlight"]],
    zorder=3
)

plt.yticks(y, merged["video_id"])

ax = plt.gca()
for tick_label, is_h in zip(ax.get_yticklabels(), merged["is_highlight"]):
    if is_h:
        tick_label.set_fontweight("bold")
        tick_label.set_color(highlight_color)

plt.axvline(0, linestyle="--", linewidth=1, color="black")
plt.xlabel(score_col)
plt.ylabel("Video")
plt.title(f"{score_col}: before vs after\nHighlighted = exact filename matches from video directory")
plt.legend()
plt.tight_layout()

before_after_png = out_dir / f"{score_col}_before_vs_after_exact_name_highlighted.png"
plt.savefig(before_after_png, dpi=200)
plt.close()


# --- Plot 2: delta bar chart ---
plt.figure(figsize=(11, max(6, len(merged) * 0.35)))

plt.barh(
    merged["video_id"],
    merged["delta"],
    color=bar_colors
)

plt.axvline(0, linestyle="--", linewidth=1, color="black")
plt.xlabel(f"Change in {score_col}  (after - before)")
plt.ylabel("Video")
plt.title(f"Difference in {score_col} between runs\nHighlighted = exact filename matches from video directory")

ax = plt.gca()
for tick_label, is_h in zip(ax.get_yticklabels(), merged["is_highlight"]):
    if is_h:
        tick_label.set_fontweight("bold")
        tick_label.set_color(highlight_color)

plt.tight_layout()

delta_png = out_dir / f"{score_col}_delta_exact_name_highlighted.png"
plt.savefig(delta_png, dpi=200)
plt.close()


# --- Plot 3: scatter comparison ---
plt.figure(figsize=(7, 7))

normal_rows = merged[~merged["is_highlight"]]
highlight_rows = merged[merged["is_highlight"]]

plt.scatter(
    normal_rows[f"{score_col}_before"],
    normal_rows[f"{score_col}_after"],
    label="Other videos",
    color=normal_color,
    alpha=0.8
)

plt.scatter(
    highlight_rows[f"{score_col}_before"],
    highlight_rows[f"{score_col}_after"],
    label="Highlighted videos",
    color=highlight_color,
    s=80,
    edgecolors="black"
)

min_val = min(
    merged[f"{score_col}_before"].min(),
    merged[f"{score_col}_after"].min()
)
max_val = max(
    merged[f"{score_col}_before"].max(),
    merged[f"{score_col}_after"].max()
)

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    linewidth=1,
    color="black"
)

plt.xlabel(f"Before {score_col}")
plt.ylabel(f"After {score_col}")
plt.title(f"{score_col}: before vs after scatter")
plt.legend()
plt.tight_layout()

scatter_png = out_dir / f"{score_col}_scatter_exact_name_highlighted.png"
plt.savefig(scatter_png, dpi=200)
plt.close()


# --- Print summary ---
print("\nSummary:")
print(f"Compared {len(merged)} matched videos")
print(f"Highlighted videos matched: {matched_highlights}")
print(f"Mean before: {merged[f'{score_col}_before'].mean():.6f}")
print(f"Mean after:  {merged[f'{score_col}_after'].mean():.6f}")
print(f"Mean delta:  {merged['delta'].mean():.6f}")

if matched_highlights > 0:
    highlighted = merged[merged["is_highlight"]]

    print("\nHighlighted subset summary:")
    print(f"Highlighted count: {len(highlighted)}")
    print(f"Highlighted mean before: {highlighted[f'{score_col}_before'].mean():.6f}")
    print(f"Highlighted mean after:  {highlighted[f'{score_col}_after'].mean():.6f}")
    print(f"Highlighted mean delta:  {highlighted['delta'].mean():.6f}")

    print("\nHighlighted videos:")
    print(
        highlighted
        [["video_id", f"{score_col}_before", f"{score_col}_after", "delta"]]
        .sort_values("delta")
        .to_string(index=False)
    )

print("\nLargest changes:")
print(
    merged
    .sort_values("abs_delta", ascending=False)
    [["video_id", "is_highlight", f"{score_col}_before", f"{score_col}_after", "delta", "abs_delta"]]
    .head(10)
    .to_string(index=False)
)

print("\nSaved files:")
print(f"  Comparison CSV:      {comparison_csv}")
print(f"  Before/after plot:   {before_after_png}")
print(f"  Delta plot:          {delta_png}")
print(f"  Scatter plot:        {scatter_png}")