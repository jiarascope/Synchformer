from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make local imports work when running this file from a repo root.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ncut_modules.model import load_visual_encoder_from_synchformer
from ncut_modules.process_joint import process_video_directory_joint_ncut


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--repo_root", type=str, default=".", help="Path to Synchformer repo root.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to Synchformer/Motionformer checkpoint.")
    parser.add_argument(
        "--video_dir",
        type=str,
        nargs="+",
        required=True,
        help="One or more directories containing videos to include in a single joint NCut.",
    )
    parser.add_argument("--out_dir", type=str, default="outputs/ncut_motionformer")

    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--sampling", type=str, default="uniform", choices=["uniform", "first", "center"])
    parser.add_argument("--crop_mode", type=str, default="resize_short_side",
                        choices=["resize_short_side", "square_center_crop"])
    parser.add_argument("--start_sec", type=float, default=None)
    parser.add_argument("--duration_sec", type=float, default=None)

    parser.add_argument("--feature_key", type=str, default=None)
    parser.add_argument("--num_eig", type=int, default=16)
    parser.add_argument("--num_clusters", type=int, default=6)
    parser.add_argument("--kmeans_dims", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--out_fps", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--embedding_map",
        "--mapping",
        type=str,
        default="umap",
        choices=["umap", "tsne"],
        help="Dimensionality reduction used for the continuous RGB overlay.",
    )

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--segment_sec", type=float, default=0.64)
    parser.add_argument("--stride_sec", type=float, default=0.32)
    parser.add_argument("--max_duration_sec", type=float, default=None)
    parser.add_argument(
        "--ncut_mode",
        type=str,
        default="global",
        choices=["global", "per_segment"],
    )
    parser.add_argument("--eig_rgb_dims", type=int, default=20)
    parser.add_argument(
        "--joint_video_dir_ncut",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()



def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    repo_root = Path(args.repo_root).resolve()
    out_root = Path(args.out_dir)

    visual_encoder = load_visual_encoder_from_synchformer(
        repo_root=repo_root,
        checkpoint=args.checkpoint,
        device=device,
    )
    visual_encoder.eval()

    process_video_directory_joint_ncut(
        video_dirs=[Path(video_dir) for video_dir in args.video_dir],
        out_root=out_root,
        visual_encoder=visual_encoder,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()
