"""
P0 diagnostic: inspect fold 4 correct vs wrong patches.

Loads best_model_fold4.pth and evaluates on S14-248-1-3 val patches.
Samples N random patches the model got RIGHT (predicted gland) vs WRONG
(predicted non-gland) and saves them side-by-side.

If the wrong patches look like darker-stained areas and right ones look
brighter, that confirms the stain-shortcut hypothesis visually.

Output: Gland_Seg/Viz/fold4_correct_vs_wrong.png
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import Config
from dataset import PatchDataset, get_val_transforms
from model import create_model


N_SAMPLES = 10  # per category (correct / wrong)
RANDOM_SEED = 42
FOLD = 4
VAL_SLIDE = "S14-248-1-3"  # gland slide, the fold 4 val


@torch.no_grad()
def predict_all(model, dataset, device, batch_size=256):
    """Return per-sample predictions and the image paths, in dataset order."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    preds = []
    probs_gland = []
    model.eval()
    for images, _ in loader:
        images = images.to(device)
        out = model(images)
        p = torch.softmax(out, dim=1)
        preds.append(out.argmax(dim=1).cpu().numpy())
        probs_gland.append(p[:, 0].cpu().numpy())  # gland = label 0
    preds = np.concatenate(preds)
    probs_gland = np.concatenate(probs_gland)
    return preds, probs_gland


def load_rgb(path, size=256):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != size:
        img = cv2.resize(img, (size, size))
    return img


def plot_grid(paths, title, ax_list):
    for ax, p in zip(ax_list, paths):
        img = load_rgb(p)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
    if len(ax_list) > 0:
        ax_list[0].set_ylabel(title, fontsize=14, rotation=90, labelpad=10)


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(RANDOM_SEED)

    # Load val dataset
    val_dataset = PatchDataset(
        config.output_dir, [VAL_SLIDE], config.slides,
        transform=get_val_transforms(config.input_size),
    )
    print(f"Val dataset ({VAL_SLIDE}): {len(val_dataset)} patches")

    # Load model checkpoint
    ckpt_path = Path(config.checkpoint_dir) / f"best_model_fold{FOLD}.pth"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = create_model(num_classes=config.num_classes, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    print(f"  val_f1 at save: {ckpt.get('val_f1'):.4f}, "
          f"val_acc: {ckpt.get('val_acc'):.4f}")

    # Predict
    preds, probs_gland = predict_all(model, val_dataset, device)

    # Paths in the same order as dataset samples
    paths = [p for p, _ in val_dataset.samples]
    assert len(paths) == len(preds)

    # All patches are label 0 (gland) since S14-248-1-3 is a gland slide
    correct_idx = np.where(preds == 0)[0]
    wrong_idx = np.where(preds == 1)[0]
    print(f"Correct (predicted gland):   {len(correct_idx)}")
    print(f"Wrong   (predicted non-gland): {len(wrong_idx)}")

    # Sample
    correct_sample = rng.choice(correct_idx, size=min(N_SAMPLES, len(correct_idx)),
                                replace=False)
    wrong_sample = rng.choice(wrong_idx, size=min(N_SAMPLES, len(wrong_idx)),
                              replace=False)
    correct_paths = [paths[i] for i in correct_sample]
    wrong_paths = [paths[i] for i in wrong_sample]

    # Also: highest-confidence wrong (most strongly misclassified)
    confident_wrong_idx = wrong_idx[np.argsort(probs_gland[wrong_idx])[:N_SAMPLES]]
    confident_wrong_paths = [paths[i] for i in confident_wrong_idx]

    # Plot: 3 rows × N_SAMPLES cols, each tile ~2.2 inches
    tile = 2.2
    fig, axes = plt.subplots(3, N_SAMPLES, figsize=(N_SAMPLES * tile, 3 * tile + 1.2))
    plot_grid(correct_paths, "Correct\n(pred=gland)", axes[0])
    plot_grid(wrong_paths, "Wrong\n(pred=non-gland)", axes[1])
    plot_grid(confident_wrong_paths, "Most-confident\nwrong", axes[2])

    plt.suptitle(f"Fold {FOLD} — val = {VAL_SLIDE} (all true label = gland)\n"
                 f"Correct: {len(correct_idx)}, Wrong: {len(wrong_idx)} "
                 f"(acc = {len(correct_idx)/len(preds):.2%})",
                 fontsize=16)
    plt.tight_layout()
    out_path = Path(config.viz_dir) / "fold4_correct_vs_wrong.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    # Also print mean RGB of correct vs wrong patches as numeric evidence
    def mean_rgb(ps):
        acc = []
        for p in ps[:200]:  # first 200
            img = cv2.imread(str(p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            tissue = hsv[..., 1] > 20
            if tissue.sum() > 100:
                acc.append(img[tissue].mean(axis=0))
        return np.array(acc).mean(axis=0) if acc else np.array([np.nan]*3)

    correct_rgb = mean_rgb([paths[i] for i in correct_idx])
    wrong_rgb = mean_rgb([paths[i] for i in wrong_idx])
    print(f"\nMean RGB (tissue) of CORRECT patches: {correct_rgb.round(1)}")
    print(f"Mean RGB (tissue) of WRONG   patches: {wrong_rgb.round(1)}")
    print(f"Difference: {(correct_rgb - wrong_rgb).round(1)}")


if __name__ == "__main__":
    main()
