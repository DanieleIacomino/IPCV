import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
CSV_FILES = [
    "historyCE.csv",
    "historyCE+IOU.csv",
    "historyFOCAL+DICE.csv",
    "historyIOU.csv",
]

OUTDIR = "plots_out"  # cartella output
os.makedirs(OUTDIR, exist_ok=True)

# Output files
PNG_RAW_TV   = os.path.join(OUTDIR, "01_train_val_raw_same_scale.png")
PNG_NORM_TV  = os.path.join(OUTDIR, "02_train_val_normalized.png")
PNG_RAW_VAL  = os.path.join(OUTDIR, "03_val_only_raw_same_scale.png")
PNG_NORM_VAL = os.path.join(OUTDIR, "04_val_only_normalized.png")


# =========================
# UTILS
# =========================
def load_history(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {"epoch", "train_loss", "val_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: mancano colonne {missing}. Trovate: {list(df.columns)}")

    df = df.sort_values("epoch").reset_index(drop=True)
    return df

def title_from_filename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def minmax_norm(x: np.ndarray, x_min: float, x_max: float) -> np.ndarray:
    denom = (x_max - x_min)
    if denom == 0:
        return np.zeros_like(x, dtype=np.float64)
    return (x - x_min) / denom

def compute_global_ylim(histories, mode: str) -> tuple[float, float]:
    """
    mode: "train_val" oppure "val_only"
    Calcola ylim globale (non normalizzato) su tutti gli esperimenti.
    """
    vals = []
    for _, df in histories:
        if mode == "train_val":
            vals.append(df["train_loss"].to_numpy(dtype=np.float64))
            vals.append(df["val_loss"].to_numpy(dtype=np.float64))
        elif mode == "val_only":
            vals.append(df["val_loss"].to_numpy(dtype=np.float64))
        else:
            raise ValueError("mode non valido")

    allv = np.concatenate(vals)
    y_min = float(np.nanmin(allv))
    y_max = float(np.nanmax(allv))

    # margine 5%
    margin = 0.05 * (y_max - y_min) if y_max > y_min else 0.1
    return y_min - margin, y_max + margin

def plot_4panel(
    histories,
    mode: str,
    normalized: bool,
    outpath: str,
    suptitle: str,
    force_same_scale: bool = True,
):
    """
    mode:
      - "train_val": plotta train e val
      - "val_only" : plotta solo val
    normalized:
      - False: usa valori originali
      - True : min-max per esperimento (su train+val insieme o solo val)
    force_same_scale:
      - se True e normalized=False, imposta ylim globale comune.
      - se normalized=True, ylim comune fisso [0,1] (con piccolo margine).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=140)
    axes = axes.flatten()

    # Scala comune (solo per non-normalizzato) o standard per normalizzato
    if normalized:
        y_min_global, y_max_global = -0.03, 1.03
    else:
        y_min_global, y_max_global = compute_global_ylim(histories, mode)

    for ax, (fname, df) in zip(axes, histories):
        epochs = df["epoch"].to_numpy()

        # Best epoch e min val loss (su valori originali)
        val = df["val_loss"].to_numpy(dtype=np.float64)
        best_idx = int(np.nanargmin(val))
        best_epoch = epochs[best_idx]
        min_val = float(val[best_idx])

        # Serie da plottare
        if mode == "train_val":
            train = df["train_loss"].to_numpy(dtype=np.float64)

            if normalized:
                # Normalizzazione per-esperimento su train+val insieme
                local_min = float(np.nanmin(np.concatenate([train, val])))
                local_max = float(np.nanmax(np.concatenate([train, val])))
                train_p = minmax_norm(train, local_min, local_max)
                val_p = minmax_norm(val, local_min, local_max)
                min_val_p = float(minmax_norm(np.array([min_val]), local_min, local_max)[0])
            else:
                train_p = train
                val_p = val
                min_val_p = min_val

            ax.plot(epochs, train_p, label="Train loss")
            ax.plot(epochs, val_p, label="Val loss")

        elif mode == "val_only":
            if normalized:
                local_min = float(np.nanmin(val))
                local_max = float(np.nanmax(val))
                val_p = minmax_norm(val, local_min, local_max)
                min_val_p = float(minmax_norm(np.array([min_val]), local_min, local_max)[0])
            else:
                val_p = val
                min_val_p = min_val

            ax.plot(epochs, val_p, label="Val loss")

        else:
            raise ValueError("mode non valido")

        # Linee best epoch + min val
        ax.axvline(best_epoch, linestyle="--", linewidth=1, label="Best epoch (min val)")
        ax.axhline(min_val_p, linestyle=":", linewidth=1, label="Min val loss")

        # Titolo e annotazione
        ax.set_title(title_from_filename(fname), fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss" + (" (normalized)" if normalized else ""))

        # Scala
        if force_same_scale:
            ax.set_ylim(y_min_global, y_max_global)

        ax.grid(True, alpha=0.3)

        # Annotazione (in alto a sinistra del pannello)
        ax.text(
            0.02, 0.98,
            f"best epoch: {best_epoch}\nmin val: {min_val:.6g}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.15),
        )

        ax.legend(fontsize=8)

    # Se meno di 4, spegni extra
    for i in range(len(histories), 4):
        axes[i].axis("off")

    fig.suptitle(suptitle, fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    print(f"Salvato: {outpath}")
    plt.close(fig)


def main():
    histories = [(f, load_history(f)) for f in CSV_FILES]

    # 1) Train+Val raw (scala comune)
    plot_4panel(
        histories=histories,
        mode="train_val",
        normalized=False,
        outpath=PNG_RAW_TV,
        suptitle="Train + Val loss (scala Y comune, non normalizzato) + best epoch & min val",
        force_same_scale=True,
    )

    # 2) Train+Val normalized
    plot_4panel(
        histories=histories,
        mode="train_val",
        normalized=True,
        outpath=PNG_NORM_TV,
        suptitle="Train + Val loss (normalizzato min-max per esperimento) + best epoch & min val",
        force_same_scale=True,  # in normalized => ylim ~ [0,1]
    )

    # 3) Val-only raw (scala comune)
    plot_4panel(
        histories=histories,
        mode="val_only",
        normalized=False,
        outpath=PNG_RAW_VAL,
        suptitle="Solo Val loss (scala Y comune, non normalizzato) + best epoch & min val",
        force_same_scale=True,
    )

    # 4) Val-only normalized
    plot_4panel(
        histories=histories,
        mode="val_only",
        normalized=True,
        outpath=PNG_NORM_VAL,
        suptitle="Solo Val loss (normalizzato min-max per esperimento) + best epoch & min val",
        force_same_scale=True,
    )

    print("\nFatto. Trovi tutti i PNG in:", OUTDIR)

if __name__ == "__main__":
    main()
