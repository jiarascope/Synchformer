import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

log_path = Path(sys.argv[1])

epoch_loss_re = re.compile(r"(train|valid) \((\d+)\): total_loss ([0-9.]+);")
metrics_re = re.compile(r"(train|valid) \((\d+)\) metrics: .*?'accuracy_1': ([0-9.]+).*?'accuracy_5': ([0-9.]+).*?'accuracy_1_tol1': ([0-9.]+)")

rows = []

for line in log_path.read_text(errors="ignore").splitlines():
    m = epoch_loss_re.search(line)
    if m:
        phase, epoch, loss = m.groups()
        rows.append({
            "phase": phase,
            "epoch": int(epoch),
            "loss": float(loss),
            "accuracy_1": None,
            "accuracy_5": None,
            "accuracy_1_tol1": None,
        })

    m = metrics_re.search(line)
    if m:
        phase, epoch, acc1, acc5, acc1tol1 = m.groups()
        epoch = int(epoch)
        for r in reversed(rows):
            if r["phase"] == phase and r["epoch"] == epoch:
                r["accuracy_1"] = float(acc1)
                r["accuracy_5"] = float(acc5)
                r["accuracy_1_tol1"] = float(acc1tol1)
                break

def series(phase, key):
    xs, ys = [], []
    for r in rows:
        if r["phase"] == phase and r.get(key) is not None:
            xs.append(r["epoch"])
            ys.append(r[key])
    return xs, ys

outdir = log_path.parent

plt.figure()
for phase in ["train", "valid"]:
    x, y = series(phase, "loss")
    if x:
        plt.plot(x, y, marker="o", label=phase)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over epochs")
plt.legend()
plt.grid(True)
plt.savefig(outdir / "loss_over_epochs.png", dpi=150, bbox_inches="tight")

plt.figure()
for phase in ["train", "valid"]:
    x, y = series(phase, "accuracy_1")
    if x:
        plt.plot(x, y, marker="o", label=phase)
plt.xlabel("Epoch")
plt.ylabel("Accuracy@1")
plt.title("Accuracy@1 over epochs")
plt.legend()
plt.grid(True)
plt.savefig(outdir / "accuracy1_over_epochs.png", dpi=150, bbox_inches="tight")

plt.figure()
for phase in ["train", "valid"]:
    x, y = series(phase, "accuracy_5")
    if x:
        plt.plot(x, y, marker="o", label=phase)
plt.xlabel("Epoch")
plt.ylabel("Accuracy@5")
plt.title("Accuracy@5 over epochs")
plt.legend()
plt.grid(True)
plt.savefig(outdir / "accuracy5_over_epochs.png", dpi=150, bbox_inches="tight")

print("Parsed rows:")
for r in rows:
    print(r)

print("Saved:")
print(outdir / "loss_over_epochs.png")
print(outdir / "accuracy1_over_epochs.png")
print(outdir / "accuracy5_over_epochs.png")
