import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def error_map(gt, pred):
    """Green = correct foreground (TP), red = extra (FP), blue = missed (FN)."""
    out = np.zeros((*gt.shape, 3), np.uint8)
    out[gt & pred] = (0, 200, 0)
    out[~gt & pred] = (230, 40, 40)
    out[gt & ~pred] = (40, 90, 255)
    return out


def save_panels(items, path, title=""):
    """items: dicts with img (RGB), gt (bool), pred (bool), title. Columns: image | GT | prediction | errors."""
    fig, ax = plt.subplots(len(items), 4, figsize=(13, 3.4 * len(items)), squeeze=False)
    for r, it in enumerate(items):
        panels = [it["img"], it["gt"].astype(np.uint8) * 255, it["pred"].astype(np.uint8) * 255,
                  error_map(it["gt"], it["pred"])]
        names = ["image", "ground truth", "prediction", "errors (TP / FP / FN)"]
        for c, (p, name) in enumerate(zip(panels, names)):
            ax[r, c].imshow(p, cmap="gray", vmin=0, vmax=255) if p.ndim == 2 else ax[r, c].imshow(p)
            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(name, fontsize=10)
        ax[r, 0].set_ylabel(it["title"], fontsize=9)
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=100)
    plt.close(fig)
