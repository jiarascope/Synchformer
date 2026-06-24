from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch


def load_visual_encoder_from_synchformer(
    repo_root: Path,
    checkpoint: Optional[str],
    device: torch.device,
):
    """
    Load Synchformer's MotionFormer visual feature extractor directly.

    For NCut patch-token visualization, we want:
      extract_features=True
      factorize_space_time=False

    That should preserve spatiotemporal patch tokens instead of pooling them
    into one vector per segment/frame.
    """
    import sys
    sys.path.insert(0, str(repo_root))

    from model.modules.feat_extractors.visual.motionformer import MotionFormer

    visual_encoder = MotionFormer(
        extract_features=True,
        ckpt_path=checkpoint,
        factorize_space_time=False,
        # These are only used when factorize_space_time=True, so keep them None.
        agg_space_module=None,
        agg_time_module=None,
        add_global_repr=False,
        agg_segments_module=None,
        max_segments=None,
    )

    visual_encoder = visual_encoder.to(device).eval()
    return visual_encoder

def _try_forward_features(model: torch.nn.Module, clip: torch.Tensor) -> Any:
    """
    Try common APIs without assuming one exact Synchformer class.
    """
    if hasattr(model, "forward_features"):
        return model.forward_features(clip)

    if hasattr(model, "encode_video"):
        return model.encode_video(clip)

    if hasattr(model, "visual"):
        visual = model.visual
        if hasattr(visual, "forward_features"):
            return visual.forward_features(clip)
        return visual(clip)

    if hasattr(model, "vfeat_extractor"):
        v = model.vfeat_extractor
        if hasattr(v, "forward_features"):
            return v.forward_features(clip)
        return v(clip)

    return model(clip)

def extract_spatiotemporal_tokens(
    visual_encoder: torch.nn.Module,
    clip: torch.Tensor,
    num_frames: int,
    image_size: int,
    patch_size: int = 16,
    feature_key: Optional[str] = None,
) -> torch.Tensor:
    """
    Returns:
        tokens_grid: [T_tok, H_tok, W_tok, D]

    Synchformer MotionFormer with:
      extract_features=True
      factorize_space_time=False

    should return either:
      [B, S, N, D]
    or sometimes:
      [B, S, N+1, D] / [B, N, D]
    depending on exact code path.
    """
    with torch.no_grad():
        out = visual_encoder(clip)

    if isinstance(out, dict):
        if feature_key is not None:
            out = out[feature_key]
        else:
            for k in ["tokens", "x", "video", "visual", "features", "last_hidden_state"]:
                if k in out:
                    out = out[k]
                    break
            else:
                raise ValueError(
                    f"Model returned dict keys {list(out.keys())}; pass --feature_key."
                )

    if isinstance(out, (tuple, list)):
        # If add_global_repr=True, MotionFormer may return (local_x, global_x).
        out = out[0]

    if not torch.is_tensor(out):
        raise TypeError(f"Expected tensor features, got {type(out)}")

    out = out.detach()

    print(f"Raw visual encoder output shape: {tuple(out.shape)}")

    # Common Synchformer MotionFormer case: [B, S, N, D]
    if out.ndim == 4:
        B, S, N, D = out.shape
        if B != 1:
            raise ValueError(f"This debug script expects B=1, got {B}")

        # For now, handle one segment at a time.
        if S != 1:
            print(f"Warning: got S={S}; using first segment only for now.")
        out = out[:, 0]  # [B, N, D]

    # Already [B, N, D]
    if out.ndim == 3:
        B, N, D = out.shape
        if B != 1:
            raise ValueError(f"This debug script expects B=1, got {B}")

        h_tok = image_size // patch_size
        w_tok = image_size // patch_size

        # Drop CLS only if still present. Synchformer wrapper may already drop it.
        if (N - 1) % (h_tok * w_tok) == 0:
            print("Detected possible CLS token; dropping first token.")
            out = out[:, 1:, :]
            N = N - 1

        if N % (h_tok * w_tok) != 0:
            raise ValueError(
                f"Cannot infer [T,H,W] from feature shape {tuple(out.shape)} with "
                f"image_size={image_size}, patch_size={patch_size}. "
                f"N={N}, Htok={h_tok}, Wtok={w_tok}."
            )

        t_tok = N // (h_tok * w_tok)
        tokens_grid = out.reshape(1, t_tok, h_tok, w_tok, D)[0].contiguous()

        print(f"tokens_grid shape: {tuple(tokens_grid.shape)}")
        return tokens_grid

    # If you accidentally used factorize_space_time=True, you may get pooled features.
    if out.ndim == 5:
        # Could be [B, S, T, H, W, D] in some custom path, but unlikely.
        raise ValueError(
            f"Got 5D output {tuple(out.shape)}. Inspect whether factorize_space_time=True "
            "or whether this is channel-first spatiotemporal output."
        )

    raise ValueError(f"Unsupported feature shape: {tuple(out.shape)}")


# -----------------------------
# NCut compatibility
# -----------------------------
