# ============================================================
# Metriche.py — Robust academic plots (auto-detect columns)
# ============================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# -----------------------------
# GLOBAL STYLE (ACADEMIC)
# -----------------------------
mpl.rcParams.update({
    "figure.figsize": (8, 6),
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.5
})

# -----------------------------
# COLOR PALETTE (SIMPLE & CLEAR)
# -----------------------------
COLORS = {
    "CE": "#1f77b4",
    "CE+IoU": "#ff7f0e",
    "IoU": "#2ca02c",
    "Focal+Dice": "#9467bd"
}

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Helpers
# -----------------------------
def normalize_colname(s: str) -> str:
    # normalize for matching: lowercase, remove non-alphanum
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def pick_column(df: pd.DataFrame, candidates: list[str], metric_label: str) -> str:
    """
    Pick the first existing column among candidates, using case-insensitive/normalized matching.
    candidates: list of possible column names (human variants)
    """
    norm_to_real = {normalize_colname(c): c for c in df.columns}

    for cand in candidates:
        key = normalize_colname(cand)
        if key in norm_to_real:
            return norm_to_real[key]

    # try partial contains matching (e.g. "miou" inside "mioupresent")
    norm_cols = list(norm_to_real.keys())
    for cand in candidates:
        key = normalize_colname(cand)
        for nc in norm_cols:
            if key in nc:
                return norm_to_real[nc]

    raise KeyError(
        f"Non trovo una colonna per '{metric_label}'. "
        f"Ho provato: {candidates}\n"
        f"Colonne disponibili: {list(df.columns)}"
    )

def plot_confusion_matrix(csv_path, title, filename):
    cm_df = pd.read_csv(csv_path, index_col=0)
    cm = cm_df.values.astype(float)

    # normalize per-row (avoid div by zero)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    cm = cm / row_sums

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename))
    plt.close()

# ============================================================
# 0) LOAD SUMMARY + AUTO-DETECT METRIC COLUMNS
# ============================================================
summary_path = "ALL_losses_summary.csv"
df = pd.read_csv(summary_path)

# IMPORTANT: adjust/extend synonyms here if your CSV uses different names
miou_col = pick_column(
    df,
    candidates=["mIoU", "miou", "mean_iou", "meanIoU", "mIoU_present", "miou_present", "mIoU_present_only"],
    metric_label="mIoU"
)
mf1_col = pick_column(
    df,
    candidates=["mF1", "mf1", "mean_f1", "meanF1", "macro_f1", "macroF1"],
    metric_label="mF1"
)

# also auto-detect loss-name column (commonly: "loss", "Loss", "criterion", etc.)
loss_name_col = None
for cand in ["loss", "Loss", "criterion", "Criterion", "loss_name", "LossName"]:
    if normalize_colname(cand) in {normalize_colname(c) for c in df.columns}:
        loss_name_col = pick_column(df, [cand], "loss_name")
        break
if loss_name_col is None:
    # fallback: first column that looks like a name/category
    # (if your CSV has e.g. "run" or "experiment", add it above)
    loss_name_col = df.columns[0]

# normalize loss names to match COLORS keys (optional but helpful)
def normalize_loss_label(x: str) -> str:
    s = str(x).strip()
    s_norm = s.replace(" ", "").lower()
    mapping = {
        "ce": "CE",
        "crossentropy": "CE",
        "ce+iou": "CE+IoU",
        "ceiou": "CE+IoU",
        "iou": "IoU",
        "focal+dice": "Focal+Dice",
        "focaldice": "Focal+Dice",
        "focal_dice": "Focal+Dice",
    }
    return mapping.get(s_norm, s)

df["_loss_label"] = df[loss_name_col].apply(normalize_loss_label)

# ============================================================
# 1) GLOBAL METRICS COMPARISON (mIoU, mF1)
# ============================================================
metrics = [("mIoU", miou_col), ("mF1", mf1_col)]
x = np.arange(len(metrics))
width = 0.18

plt.figure(figsize=(8, 6))

for i, row in df.iterrows():
    label = row["_loss_label"]
    color = COLORS.get(label, None)  # if unknown label, matplotlib picks default
    values = [row[col] for _, col in metrics]

    plt.bar(
        x + i * width,
        values,
        width,
        label=label,
        color=color
    )

plt.xticks(x + width * (len(df)-1)/2, [m[0] for m in metrics])
plt.ylabel("Score")
plt.title("Global Metrics Comparison")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "global_metrics_comparison.png"))
plt.close()

# ============================================================
# 2) PER-CLASS IoU COMPARISON (CE vs CE+IoU) — Robust columns
# ============================================================
ce = pd.read_csv("CE_metrics.csv")
ceiou = pd.read_csv("CE_IOU_metrics.csv")

# auto-detect per-class columns
class_col_ce = pick_column(ce, ["class", "Class", "cls", "label"], "class")
iou_col_ce   = pick_column(ce, ["IoU", "iou", "class_iou", "per_class_iou"], "IoU")

class_col_c2 = pick_column(ceiou, ["class", "Class", "cls", "label"], "class")
iou_col_c2   = pick_column(ceiou, ["IoU", "iou", "class_iou", "per_class_iou"], "IoU")

# align by class (safe)
ce2 = ce[[class_col_ce, iou_col_ce]].rename(columns={class_col_ce: "class", iou_col_ce: "iou"})
c2  = ceiou[[class_col_c2, iou_col_c2]].rename(columns={class_col_c2: "class", iou_col_c2: "iou"})

merged = pd.merge(ce2, c2, on="class", how="inner", suffixes=("_CE", "_CEIOU"))

plt.figure(figsize=(11, 6))
plt.plot(merged["class"], merged["iou_CE"], marker="o", label="CE", color=COLORS["CE"])
plt.plot(merged["class"], merged["iou_CEIOU"], marker="s", label="CE+IoU", color=COLORS["CE+IoU"])
plt.xlabel("Class")
plt.ylabel("IoU")
plt.title("Per-class IoU Comparison (CE vs CE+IoU)")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "per_class_iou_comparison.png"))
plt.close()

# ============================================================
# 3) CONFUSION MATRICES (NORMALIZED)
# ============================================================
plot_confusion_matrix("CE_confusion_matrix.csv", "Normalized Confusion Matrix (CE)", "confusion_matrix_CE.png")
plot_confusion_matrix("CE_IOU_confusion_matrix.csv", "Normalized Confusion Matrix (CE + IoU)", "confusion_matrix_CE_IOU.png")
plot_confusion_matrix("IOU_confusion_matrix.csv", "Normalized Confusion Matrix (IoU)", "confusion_matrix_IOU.png")
plot_confusion_matrix("FOCAL_DICE_confusion_matrix.csv", "Normalized Confusion Matrix (Focal + Dice)", "confusion_matrix_FOCAL_DICE.png")

print("OK: figure salvate in:", os.path.abspath(OUT_DIR))
print("Usate colonne summary:", {"loss_name": loss_name_col, "mIoU": miou_col, "mF1": mf1_col})
