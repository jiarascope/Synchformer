#!/usr/bin/env python
import os
import sys
import time
import signal
import traceback
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

REPO = "/home/jiaray/mrBean/Synchformer"
sys.path.insert(0, REPO)

from scripts.train_utils import get_transforms, get_datasets


def env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.lower() in {"1", "true", "yes", "y", "on"}


def proc_stats(pid):
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        stat = Path(f"/proc/{pid}/stat").read_text().split()

        def grab(key):
            for line in status.splitlines():
                if line.startswith(key + ":"):
                    return line.split(":", 1)[1].strip()
            return "?"

        return {
            "pid": pid,
            "state": grab("State"),
            "rss": grab("VmRSS"),
            "threads": grab("Threads"),
            "fd_count": len(os.listdir(f"/proc/{pid}/fd")),
            "utime": stat[13],
            "stime": stat[14],
        }
    except Exception as e:
        return {"pid": pid, "dead_or_unreadable": repr(e)}


def print_worker_stats(workers, label):
    print(f"\n--- worker stats: {label} ---", flush=True)
    if not workers:
        print("no workers found", flush=True)
        return

    for w in workers:
        pid = getattr(w, "pid", None)
        alive = w.is_alive() if hasattr(w, "is_alive") else "?"
        exitcode = getattr(w, "exitcode", None)
        print(
            {
                "worker_pid": pid,
                "alive": alive,
                "exitcode": exitcode,
                "stats": proc_stats(pid) if pid else None,
            },
            flush=True,
        )


def dump_debug_files():
    print("\n--- heartbeat files ---", flush=True)
    os.system("ls -lh /tmp/wds_heartbeat_* 2>/dev/null || true")
    os.system("tail -n 20 /tmp/wds_heartbeat_* 2>/dev/null || true")

    print("\n--- faulthandler stack files ---", flush=True)
    os.system("ls -lh /tmp/synchformer_stack_* 2>/dev/null || true")
    os.system("tail -n 120 /tmp/synchformer_stack_* 2>/dev/null || true")


def send_stack_dump(workers):
    for w in workers:
        pid = getattr(w, "pid", None)
        if not pid:
            continue
        try:
            if w.is_alive():
                print(f"sending SIGUSR1 to worker pid={pid}", flush=True)
                os.kill(pid, signal.SIGUSR1)
        except Exception as e:
            print(f"SIGUSR1 failed pid={pid}: {e!r}", flush=True)


def batch_size_from_batch(batch):
    if isinstance(batch, dict) and "path" in batch:
        try:
            return len(batch["path"])
        except Exception:
            return 1
    return 1


def summarize_times(times):
    if not times:
        return {
            "count": 0,
            "avg": float("nan"),
            "steady_avg_skip5": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    steady = times[5:]
    return {
        "count": len(times),
        "avg": sum(times) / len(times),
        "steady_avg_skip5": sum(steady) / len(steady) if steady else float("nan"),
        "min": min(times),
        "max": max(times),
    }


def run_loader(name, loader, round_id):
    print(f"\n=== START {name} round={round_id} ===", flush=True)

    t0 = time.time()
    n = 0
    i = -1
    next_times = []

    it = iter(loader)
    workers = getattr(it, "_workers", [])

    print("worker_pids:", [w.pid for w in workers], flush=True)
    print_worker_stats(workers, f"{name} round={round_id} start")

    try:
        while True:
            i += 1

            t_next0 = time.time()
            batch = next(it)
            t_next1 = time.time()

            next_time = t_next1 - t_next0
            next_times.append(next_time)

            bs = batch_size_from_batch(batch)
            n += bs

            if i % 25 == 0:
                recent = next_times[-25:]
                recent_avg = sum(recent) / len(recent)

                stats = summarize_times(next_times)

                print(
                    f"{name} round={round_id} batch={i} samples={n} "
                    f"elapsed={time.time() - t0:.1f}s "
                    f"next_last={next_time:.3f}s "
                    f"next_avg_25={recent_avg:.3f}s "
                    f"next_global_avg={stats['avg']:.3f}s "
                    f"next_steady_avg_skip5={stats['steady_avg_skip5']:.3f}s "
                    f"next_min={stats['min']:.3f}s "
                    f"next_max={stats['max']:.3f}s",
                    flush=True,
                )
                print_worker_stats(workers, f"{name} round={round_id} batch={i}")

    except StopIteration:
        elapsed = time.time() - t0
        stats = summarize_times(next_times)

        print(
            f"=== DONE {name} round={round_id} batches={i + 1} "
            f"samples={n} elapsed={elapsed:.1f}s "
            f"next_global_avg={stats['avg']:.3f}s "
            f"next_steady_avg_skip5={stats['steady_avg_skip5']:.3f}s "
            f"next_min={stats['min']:.3f}s "
            f"next_max={stats['max']:.3f}s ===",
            flush=True,
        )
        print_worker_stats(workers, f"{name} round={round_id} done")
        return

    except BaseException as e:
        print(
            f"\n!!! CRASH/EXCEPTION name={name} round={round_id} "
            f"batch={i} samples={n} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
        print("exception:", repr(e), flush=True)
        traceback.print_exc()

        print_worker_stats(workers, f"{name} round={round_id} exception")
        send_stack_dump(workers)
        time.sleep(1.0)
        dump_debug_files()
        raise


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OPENCV_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    torch.set_num_threads(1)

    cfg = OmegaConf.load(f"{REPO}/configs/sync.yaml")

    cfg.data.dataset.target = "dataset.webdataset_tar_inmemory_cached_sync.WebDatasetTarInMemoryCachedSync"
    cfg.data.vids_path = "/home/jiaray/mrBean/data/webdataset_clips/smoke_train"

    cfg.data.dataset.params.train_vids_dir = "/home/jiaray/mrBean/data/webdataset_clips/smoke_train"
    cfg.data.dataset.params.valid_vids_dir = "/home/jiaray/mrBean/data/webdataset_clips/smoke_train"
    cfg.data.dataset.params.test_vids_dir = "/home/jiaray/mrBean/data/webdataset_clips/valid_set"

    cfg.data.dataset.params.cache_decoded = env_bool("CACHE_DECODED", False)
    cfg.data.dataset.params.decoded_cache_size = int(os.environ.get("DECODED_CACHE_SIZE", "0"))

    cfg.data.dataset.params.cache_tar_handles = env_bool("CACHE_TAR_HANDLES", False)
    cfg.data.dataset.params.tar_handle_cache_size = int(os.environ.get("TAR_HANDLE_CACHE_SIZE", "8"))

    cfg.data.dataset.params.debug_io = env_bool("DEBUG_IO", False)
    cfg.data.dataset.params.debug_signal = env_bool("DEBUG_SIGNAL", True)

    # These require the dataset-file instrumentation patch.
    # Safe to set even if the constructor accepts **unused_kwargs.
    cfg.data.dataset.params.profile_io = env_bool("PROFILE_IO", True)
    cfg.data.dataset.params.profile_every = int(os.environ.get("PROFILE_EVERY", "50"))

    cfg.data.dataset.params.worker_threads = int(os.environ.get("WORKER_THREADS", "1"))
    cfg.data.dataset.params.decode_threads = int(os.environ.get("DECODE_THREADS", "1"))

    cfg.data.dataset.params.strict_video_fps = 25
    cfg.data.dataset.params.strict_audio_fps = 16000
    cfg.data.dataset.params.max_clip_len_sec = None

    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    num_workers = int(os.environ.get("NUM_WORKERS", "2"))
    prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "2"))
    persistent_workers = env_bool("PERSISTENT_WORKERS", True)
    pin_memory = env_bool("PIN_MEMORY", False)
    rounds = int(os.environ.get("ROUNDS", "10"))
    which = os.environ.get("WHICH", "all").strip().lower()

    print("\n=== stress config ===", flush=True)
    print("batch_size:", batch_size, flush=True)
    print("num_workers:", num_workers, flush=True)
    print("prefetch_factor:", prefetch_factor if num_workers > 0 else None, flush=True)
    print("persistent_workers:", persistent_workers if num_workers > 0 else False, flush=True)
    print("pin_memory:", pin_memory, flush=True)
    print("rounds:", rounds, flush=True)
    print("which:", which, flush=True)
    print("debug_io:", cfg.data.dataset.params.debug_io, flush=True)
    print("debug_signal:", cfg.data.dataset.params.debug_signal, flush=True)
    print("profile_io:", cfg.data.dataset.params.profile_io, flush=True)
    print("profile_every:", cfg.data.dataset.params.profile_every, flush=True)
    print("cache_decoded:", cfg.data.dataset.params.cache_decoded, flush=True)
    print("decoded_cache_size:", cfg.data.dataset.params.decoded_cache_size, flush=True)
    print("cache_tar_handles:", cfg.data.dataset.params.cache_tar_handles, flush=True)
    print("tar_handle_cache_size:", cfg.data.dataset.params.tar_handle_cache_size, flush=True)
    print("worker_threads:", cfg.data.dataset.params.worker_threads, flush=True)
    print("decode_threads:", cfg.data.dataset.params.decode_threads, flush=True)

    transforms = get_transforms(cfg)
    datasets = get_datasets(cfg, transforms, which_datasets=["train", "valid", "test"])

    print("train len:", len(datasets["train"]), flush=True)
    print("valid len:", len(datasets["valid"]), flush=True)
    print("test len:", len(datasets["test"]), flush=True)

    loader_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )

    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(datasets["train"], **loader_kwargs)
    valid_loader = DataLoader(datasets["valid"], **loader_kwargs)
    test_loader = DataLoader(datasets["test"], **loader_kwargs)

    for r in range(rounds):
        print(f"\n######## ROUND {r} ########", flush=True)

        if which == "train":
            run_loader("train", train_loader, r)
        elif which in {"valid", "val"}:
            run_loader("valid", valid_loader, r)
        elif which == "test":
            run_loader("test", test_loader, r)
        elif which == "all":
            run_loader("train", train_loader, r)
            run_loader("valid", valid_loader, r)
            run_loader("test", test_loader, r)
        else:
            raise ValueError(f"Unknown WHICH={which!r}. Use train, valid, test, or all.")

    print("\nSTRESS TEST FINISHED CLEANLY", flush=True)


if __name__ == "__main__":
    output_path = os.environ.get(
        "OUTPUT_TXT",
        "/home/jiaray/mrBean/dataloader_stress_output.txt",
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", buffering=1) as f:
        with redirect_stdout(f), redirect_stderr(f):
            main()

    print(f"Wrote stress test output to: {output_path}", flush=True)