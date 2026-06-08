"""
Full-data training (no LOSO). All slides in config.slides used for training;
patch-level random 10% holdout for in-distribution validation. In addition,
each epoch runs **external test eval** on S14-2289-1-6 (using its XML parity
GT) so we can track WHEN training is best on the external slide.

Two checkpoints are saved:
    best_model_<bb>_full.pth         — best by internal val_f1 (early-stopping criterion)
    best_model_<bb>_full_byext.pth   — best by external macro-F1 on S14-2289-1-6

Usage (DDP, 2 GPU):
    PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0,1 \
      torchrun --standalone --nnodes=1 --nproc_per_node=2 train_full.py \
      2>&1 | tee /app/Gland_Seg/logs/train_full_<backbone>.log
"""

import os
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from lxml import etree
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, confusion_matrix,
    average_precision_score, roc_auc_score, matthews_corrcoef,
    balanced_accuracy_score,
)
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm

from config import Config
from dataset import PatchDataset, get_train_transforms, get_val_transforms
from model import create_model, freeze_early_layers, unfreeze_all
from train_cv import (
    setup_distributed, cleanup_distributed, is_main, barrier,
    _amp_dtype_of, train_one_epoch, validate,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ─────────────────────────────────────────────
# External eval (S14-2289-1-6) — rank 0 only
# ─────────────────────────────────────────────

EXT_SLIDE = "S14-2289-1-6"
# stride = patch_size (non-overlapping, no gap, full coverage).
# Computed at runtime as config.patch_size — see precompute_external_eval.
EXT_GT_THUMB_MAX = 4000

# Sub-epoch external eval frequency (iterations between evals within one epoch).
# Set to 0 to disable sub-epoch eval (eval only at epoch boundary).
SUB_EVAL_EVERY_ITERS = 100


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced classification.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    With alpha = inverse-frequency class_weights (same role as CE weight).
    gamma=0 reduces to weighted CE; gamma=2 is the standard focal value
    (Lin et al., 2017 — RetinaNet).
    """

    def __init__(self, alpha, gamma=2.0):
        super().__init__()
        self.alpha = alpha          # tensor [num_classes] (per-class weight)
        self.gamma = float(gamma)

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def _parse_aperio_polys(xml_path):
    tree = etree.parse(str(xml_path))
    polys = []
    for ann in tree.getroot().findall(".//Annotation"):
        for reg in ann.findall(".//Region"):
            verts = [(float(v.get("X")), float(v.get("Y")))
                     for v in reg.findall(".//Vertex")]
            if len(verts) >= 3:
                polys.append(np.array(verts))
    return polys


def precompute_external_eval(config, rank):
    """Run patch scan + GT once. Returns (patches_uint8, gt_array) or None.

    Only rank 0 produces data; other ranks get None.
    Uses the same imports as infer_external_slide for consistency.
    """
    if rank != 0:
        return None
    try:
        from infer_external_slide import scan_slide_patches  # lazy import
        import openslide

        if EXT_SLIDE not in getattr(config, "external_test_slides", {}):
            print(f"  [ext-eval] {EXT_SLIDE} not in config.external_test_slides — disabled",
                  flush=True)
            return None

        info = config.external_test_slides[EXT_SLIDE]
        svs_path = str(Path(config.svs_dir) / info["svs"])
        xml_path = str(Path(config.xml_dir) / info["xml"])

        target_rgb = None
        if config.stain_normalize:
            t = cv2.imread(config.stain_target_path)
            if t is None:
                raise FileNotFoundError(config.stain_target_path)
            target_rgb = cv2.cvtColor(t, cv2.COLOR_BGR2RGB)

        eval_stride = config.patch_size  # non-overlapping coverage
        print(f"  [ext-eval] scanning {EXT_SLIDE} (patch={config.patch_size}, "
              f"stride={eval_stride}, non-overlapping) ...", flush=True)
        results, (W, H) = scan_slide_patches(
            svs_path, target_rgb, config,
            stride=eval_stride, workers=24, verbose=False,
        )
        if not results:
            print(f"  [ext-eval] no tissue patches found — disabled", flush=True)
            return None

        xs = np.array([r[0] for r in results])
        ys = np.array([r[1] for r in results])
        patches = np.stack([r[2] for r in results])  # (N, input_size, input_size, 3) uint8

        # GT via XML polygon parity
        polys = _parse_aperio_polys(xml_path)
        scale = max(W, H) / EXT_GT_THUMB_MAX
        thumb_w = int(W / scale)
        thumb_h = int(H / scale)
        counter = np.zeros((thumb_h, thumb_w), dtype=np.int16)
        for p in polys:
            pts = (p / scale).round().astype(np.int32)
            m = np.zeros((thumb_h, thumb_w), dtype=np.uint8)
            cv2.fillPoly(m, [pts], 1)
            counter += m

        cx = ((xs + config.patch_size / 2) / scale).astype(int).clip(0, thumb_w - 1)
        cy = ((ys + config.patch_size / 2) / scale).astype(int).clip(0, thumb_h - 1)
        n_in = counter[cy, cx]
        gt = np.where(n_in == 0, -1, np.where(n_in == 1, 0, 1)).astype(np.int8)

        print(f"  [ext-eval] {len(patches)} tissue patches | "
              f"GT gland={int((gt==0).sum())}, non-gland={int((gt==1).sum())}, "
              f"no-GT={int((gt==-1).sum())}", flush=True)
        return patches, gt
    except Exception as e:
        print(f"  [ext-eval] precompute failed: {e}. External eval disabled.",
              flush=True)
        import traceback; traceback.print_exc()
        return None


def _save_byext_ckpt(model, config, epoch, sub_iter, ext_metrics, val_metrics_dict=None):
    """Save best-by-ext checkpoint. sub_iter=None for end-of-epoch saves."""
    ckpt_path = Path(config.checkpoint_dir) / f"best_model_{config.backbone}_full_byext{config.run_tag}.pth"
    raw_model = model.module if hasattr(model, "module") else model
    payload = {
        "backbone": config.backbone, "mode": "full_byext",
        "epoch": epoch,
        "sub_iter": sub_iter,                 # None or int
        "model_state_dict": raw_model.state_dict(),
        "ext_macro_f1": ext_metrics["macro_f1"],
        "ext_gland_f1": ext_metrics["gland_f1"],
        "ext_nongland_f1": ext_metrics["nongland_f1"],
    }
    if val_metrics_dict is not None:
        payload["val_f1"] = val_metrics_dict.get("f1")
        payload["val_acc"] = val_metrics_dict.get("acc")
    torch.save(payload, ckpt_path)


def train_one_epoch_with_subeval(
    model, loader, criterion, optimizer, device, epoch, epochs, rank, world_size,
    fold_tag, amp_dtype, config, test_data, best_ext_state,
    eval_every_iters=SUB_EVAL_EVERY_ITERS, sub_eval_log=None,
):
    """Drop-in replacement for train_one_epoch that also runs external eval
    every `eval_every_iters` minibatches (on rank 0). Saves best-by-ext as it
    finds new best.

    Returns: same train metrics dict (loss, acc).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = amp_dtype in (torch.bfloat16, torch.float16) and device.type == "cuda"

    if is_main(rank):
        pbar = tqdm(loader, desc=f"  {fold_tag} Epoch {epoch:2d}/{epochs} [Train]",
                    leave=True, dynamic_ncols=True, mininterval=1.0)
        iterator = pbar
    else:
        iterator = loader

    n_total = len(loader)
    for it, (images, labels) in enumerate(iterator):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
        )
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

        if is_main(rank):
            pbar.set_postfix(loss=f"{running_loss/max(total,1):.4f}",
                             acc=f"{correct/max(total,1):.4f}")

        # ── Sub-epoch external eval — same condition on all ranks (avoid DDP desync) ──
        done_iter = it + 1
        is_sub = (eval_every_iters > 0
                  and done_iter % eval_every_iters == 0
                  and done_iter < n_total)
        if is_sub:
            # ALL ranks sync first so rank 1 doesn't start next-iter backward() while rank 0 is in eval
            if world_size > 1:
                barrier(device)
            if is_main(rank) and test_data is not None:
                raw_model = model.module if hasattr(model, "module") else model
                ext = evaluate_external(raw_model, test_data[0], test_data[1],
                                        device, amp_dtype)
                msg = (f"  {fold_tag} SubEval ep{epoch} iter{done_iter:>5}/{n_total} | "
                       f"macro {ext['macro_f1']:.4f} | "
                       f"g {ext['gland_f1']:.4f} (P{ext['precision_gland']:.2f}/R{ext['recall_gland']:.2f}) | "
                       f"ng {ext['nongland_f1']:.4f} (P{ext['precision_nongland']:.2f}/R{ext['recall_nongland']:.2f}) | "
                       f"PR-AUC {ext['pr_auc_ng']:.3f} | MCC {ext['mcc']:.3f} | "
                       f"best-ext {best_ext_state['f1']:.4f}@e{best_ext_state['epoch']}"
                       f"{':i'+str(best_ext_state['iter']) if best_ext_state['iter'] else ''}")
                pbar.write(msg)
                if sub_eval_log is not None:
                    sub_eval_log.append((
                        epoch, done_iter, ext["macro_f1"],
                        ext["gland_f1"], ext["nongland_f1"],
                        ext["precision_gland"], ext["recall_gland"],
                        ext["precision_nongland"], ext["recall_nongland"],
                        ext["pr_auc_ng"], ext["roc_auc_ng"],
                        ext["mcc"], ext["balanced_acc"],
                        ext["tp_ng"], ext["fp_ng"], ext["fn_ng"], ext["tn_ng"],
                    ))
                if ext["macro_f1"] > best_ext_state["f1"]:
                    best_ext_state["f1"] = ext["macro_f1"]
                    best_ext_state["epoch"] = epoch
                    best_ext_state["iter"] = done_iter
                    _save_byext_ckpt(model, config, epoch, done_iter, ext)
                    pbar.write(f"  {fold_tag} >> Saved best-by-ext "
                               f"(macro-F1={ext['macro_f1']:.4f}) at epoch {epoch} iter {done_iter}")
                model.train()  # ensure back to train mode
            # ALL ranks sync after eval so non-rank-0 wait for rank 0 to finish before resuming training
            if world_size > 1:
                barrier(device)

    # Reduce per-rank stats
    stats = torch.tensor([running_loss, correct, total],
                          dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_sum, correct_sum, total_sum = stats.tolist()
    return {"loss": loss_sum / max(total_sum, 1),
            "acc": correct_sum / max(total_sum, 1)}


@torch.no_grad()
def evaluate_external(model, patches_u8, gt, device, amp_dtype, batch_size=128,
                       threshold_sweep=(0.3, 0.4, 0.5, 0.6, 0.7)):
    """Forward all test patches → softmax → per-class metrics + threshold sweep.

    Labels: 0=gland, 1=non-gland. Probabilities p_ng = softmax[:,1].
    Returns dict with:
      F1: macro_f1, gland_f1, nongland_f1
      Per-class P/R: precision_gland, recall_gland, precision_nongland, recall_nongland
      Clinical: sensitivity_ng (= recall_nongland), specificity_ng (= recall_gland)
      Probability-based: pr_auc_ng, roc_auc_ng
      Single robust: mcc, balanced_acc
      Counts: tp/fp/fn/tn (non-gland positive class)
      Threshold sweep: thr_<v>_recall_ng, thr_<v>_precision_ng (for v in threshold_sweep)
    """
    from infer_external_slide import imagenet_normalize  # lazy
    model.eval()
    all_probs = []
    use_amp = amp_dtype in (torch.bfloat16, torch.float16)
    for i in range(0, len(patches_u8), batch_size):
        batch = patches_u8[i:i + batch_size]
        x = torch.from_numpy(imagenet_normalize(batch)).to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                logits = model(x)
        else:
            logits = model(x)
        prob_ng = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        all_probs.append(prob_ng)
    probs_ng = np.concatenate(all_probs)            # (N,) prob of non-gland
    preds = (probs_ng >= 0.5).astype(np.int8)       # default threshold 0.5

    empty_thr_sweep = {f"thr_{v}_recall_ng": 0.0 for v in threshold_sweep}
    empty_thr_sweep.update({f"thr_{v}_precision_ng": 0.0 for v in threshold_sweep})

    mask = gt >= 0
    if int(mask.sum()) == 0:
        return {"macro_f1": 0.0, "gland_f1": 0.0, "nongland_f1": 0.0, "acc": 0.0,
                "n_eval": 0,
                "precision_gland": 0.0, "recall_gland": 0.0,
                "precision_nongland": 0.0, "recall_nongland": 0.0,
                "sensitivity_ng": 0.0, "specificity_ng": 0.0,
                "pr_auc_ng": 0.0, "roc_auc_ng": 0.0, "mcc": 0.0, "balanced_acc": 0.0,
                "tp_ng": 0, "fp_ng": 0, "fn_ng": 0, "tn_ng": 0,
                **empty_thr_sweep}

    gt_e = gt[mask].astype(np.int8)
    pred_e = preds[mask]
    probs_e = probs_ng[mask]

    # Basic
    acc = float((gt_e == pred_e).mean())
    macro = f1_score(gt_e, pred_e, average="macro", zero_division=0)
    f1_per = f1_score(gt_e, pred_e, labels=[0, 1], average=None, zero_division=0)

    # Per-class P/R
    prec_per, rec_per, _, _ = precision_recall_fscore_support(
        gt_e, pred_e, labels=[0, 1], zero_division=0)

    # Confusion matrix (non-gland as positive class)
    cm = confusion_matrix(gt_e, pred_e, labels=[0, 1])
    # cm[true][pred]: cm[0][0]=TN_ng (gland correctly), cm[0][1]=FP_ng (gland→non-gland),
    # cm[1][0]=FN_ng (non-gland missed), cm[1][1]=TP_ng (non-gland correctly)
    tn_ng, fp_ng = int(cm[0, 0]), int(cm[0, 1])
    fn_ng, tp_ng = int(cm[1, 0]), int(cm[1, 1])

    # Probability-based (need both classes present)
    try:
        pr_auc_ng = float(average_precision_score(gt_e, probs_e))
        roc_auc_ng = float(roc_auc_score(gt_e, probs_e))
    except ValueError:
        pr_auc_ng = 0.0; roc_auc_ng = 0.0

    # Single robust scores
    mcc = float(matthews_corrcoef(gt_e, pred_e))
    bal_acc = float(balanced_accuracy_score(gt_e, pred_e))

    # Threshold sweep — non-gland P/R at multiple decision thresholds
    thr_metrics = {}
    for v in threshold_sweep:
        pred_v = (probs_e >= v).astype(np.int8)
        prec_v, rec_v, _, _ = precision_recall_fscore_support(
            gt_e, pred_v, labels=[1], average=None, zero_division=0)
        thr_metrics[f"thr_{v}_precision_ng"] = float(prec_v[0])
        thr_metrics[f"thr_{v}_recall_ng"] = float(rec_v[0])

    return {
        "macro_f1": float(macro),
        "gland_f1": float(f1_per[0]),
        "nongland_f1": float(f1_per[1]),
        "acc": acc, "n_eval": int(mask.sum()),
        "precision_gland": float(prec_per[0]),
        "recall_gland": float(rec_per[0]),
        "precision_nongland": float(prec_per[1]),
        "recall_nongland": float(rec_per[1]),
        "sensitivity_ng": float(rec_per[1]),     # alias for clinical convention
        "specificity_ng": float(rec_per[0]),     # gland recall = non-gland specificity
        "pr_auc_ng": pr_auc_ng,
        "roc_auc_ng": roc_auc_ng,
        "mcc": mcc,
        "balanced_acc": bal_acc,
        "tp_ng": tp_ng, "fp_ng": fp_ng, "fn_ng": fn_ng, "tn_ng": tn_ng,
        **thr_metrics,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    config = Config()
    if int(os.environ.get("RANK", "0")) == 0:
        config.ensure_dirs()

    rank, world_size, local_rank, device = setup_distributed()
    if is_main(rank):
        print(f"DDP: rank={rank}, world_size={world_size}, device={device}", flush=True)
        print(f"Patches dir: {config.output_dir}", flush=True)
        print(f"Backbone: {config.backbone}", flush=True)
        print(f"Mode: FULL-TRAIN (all {len(config.slides)} slides), "
              f"patch-level 10% val for early stopping + external eval per epoch",
              flush=True)

    # ── Datasets ──
    all_slides = list(config.slides.keys())
    full_train_dataset = PatchDataset(
        config.output_dir, all_slides, config.slides,
        transform=get_train_transforms(config.input_size),
    )
    full_val_dataset = PatchDataset(
        config.output_dir, all_slides, config.slides,
        transform=get_val_transforms(config.input_size),
    )
    n_total = len(full_train_dataset)
    if is_main(rank):
        print(f"Total patches: {n_total}", flush=True)
        print(f"Class counts: {full_train_dataset.get_class_counts()}", flush=True)

    rng = np.random.default_rng(config.random_seed)
    perm = rng.permutation(n_total)
    n_val = max(1, int(0.1 * n_total))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()
    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_val_dataset, val_idx)
    if is_main(rank):
        print(f"Train: {len(train_dataset)}  Val (random patch-level 10%): {len(val_dataset)}",
              flush=True)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, sampler=train_sampler,
        num_workers=config.num_workers, pin_memory=True,
    )

    # ── Model ──
    if is_main(rank):
        print(f"  [setup] Creating model (backbone={config.backbone})...", flush=True)
    model = create_model(num_classes=config.num_classes, pretrained=True,
                        backbone=config.backbone,
                        head_type=getattr(config, "head_type", "linear"))
    freeze_early_layers(model, backbone=config.backbone)
    model = model.to(device)
    if world_size > 1:
        if is_main(rank):
            print(f"  [setup] DDP wrap...", flush=True)
        barrier(device)
        model = DDP(
            model, device_ids=[device.index], output_device=device.index,
            find_unused_parameters=False, gradient_as_bucket_view=True,
        )

    # ── Class weights ──
    train_labels_full = full_train_dataset.get_labels()
    train_labels = [train_labels_full[i] for i in train_idx]
    class_counts = np.bincount(train_labels).astype(float)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.FloatTensor(class_weights).to(device)

    # ── Loss selection (CE vs Focal) ──
    if getattr(config, "loss_type", "ce") == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=config.focal_gamma)
        if is_main(rank):
            print(f"  Loss: FocalLoss(gamma={config.focal_gamma}, alpha=class_weights)", flush=True)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if is_main(rank):
            print(f"  Loss: CrossEntropyLoss(weight=class_weights)", flush=True)

    # ── LR + optimizer ──
    effective_batch = config.batch_size * world_size
    scaled_lr = config.lr * (effective_batch / config.lr_scale_base)
    if is_main(rank):
        print(f"  World size {world_size}, per-GPU batch {config.batch_size}, "
              f"effective batch {effective_batch}, scaled lr {scaled_lr:.2e}", flush=True)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=scaled_lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    amp_dtype = _amp_dtype_of(config)
    if is_main(rank):
        print(f"  AMP dtype: {amp_dtype}", flush=True)

    # ── External eval pre-compute (rank 0 only, ~2-3 min one-time) ──
    test_data = precompute_external_eval(config, rank)
    if world_size > 1:
        barrier(device)

    # ── Train loop ──
    best_f1 = 0.0
    best_val_epoch = -1
    best_ext_state = {"f1": -1.0, "epoch": -1, "iter": None}
    epoch_log = []      # list of (epoch, val_f1, ext_macro, ext_gland, ext_nongland)
    sub_eval_log = []   # list of (epoch, iter, ext_macro, ext_gland, ext_nongland)
    patience_counter = 0
    fold_tag = f"[FullTrain {config.backbone}]"

    for epoch in range(1, config.epochs + 1):
        train_sampler.set_epoch(epoch)

        if epoch == config.unfreeze_epoch + 1:
            if is_main(rank):
                print(f"  {fold_tag} Phase 2: unfreeze_all", flush=True)
            raw_model = model.module if hasattr(model, "module") else model
            unfreeze_all(raw_model, backbone=config.backbone)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=scaled_lr / 10, weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config.epochs - epoch + 1,
            )

        train_metrics = train_one_epoch_with_subeval(
            model, train_loader, criterion, optimizer, device,
            epoch, config.epochs, rank, world_size,
            fold_tag=fold_tag, amp_dtype=amp_dtype,
            config=config, test_data=test_data,
            best_ext_state=best_ext_state,
            eval_every_iters=SUB_EVAL_EVERY_ITERS,
            sub_eval_log=sub_eval_log,
        )
        val_metrics = validate(model, val_dataset, criterion, device,
                               config.batch_size, config.num_workers,
                               epoch, config.epochs, rank, world_size,
                               fold_tag=fold_tag, amp_dtype=amp_dtype)
        scheduler.step()

        if is_main(rank):
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  {fold_tag} Epoch {epoch:3d}/{config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['acc']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['acc']:.4f} "
                f"F1: {val_metrics['f1']:.4f} | LR: {lr_now:.2e} | best F1 {best_f1:.4f}",
                flush=True,
            )

        # Save best by internal val_f1 (early stopping criterion)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_val_epoch = epoch
            patience_counter = 0
            if is_main(rank):
                ckpt_path = Path(config.checkpoint_dir) / f"best_model_{config.backbone}_full{config.run_tag}.pth"
                raw_model = model.module if hasattr(model, "module") else model
                torch.save({
                    "backbone": config.backbone, "mode": "full",
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "val_f1": best_f1, "val_acc": val_metrics["acc"],
                }, ckpt_path)
                print(f"  {fold_tag} >> Saved best-by-val (F1={best_f1:.4f}) at epoch {epoch}",
                      flush=True)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                if is_main(rank):
                    print(f"  {fold_tag} Early stopping at epoch {epoch} "
                          f"(patience {config.patience}, best val_f1 {best_f1:.4f}@{best_val_epoch})",
                          flush=True)
                break

        # End-of-epoch external eval (rank 0 only)
        if is_main(rank) and test_data is not None:
            raw_model = model.module if hasattr(model, "module") else model
            ext = evaluate_external(raw_model, test_data[0], test_data[1],
                                    device, amp_dtype)
            print(
                f"  {fold_tag} Ext (S14-2289-1-6, n={ext['n_eval']}): "
                f"macro {ext['macro_f1']:.4f} | "
                f"g {ext['gland_f1']:.4f}(P{ext['precision_gland']:.2f}/R{ext['recall_gland']:.2f}) | "
                f"ng {ext['nongland_f1']:.4f}(P{ext['precision_nongland']:.2f}/R{ext['recall_nongland']:.2f}) | "
                f"acc {ext['acc']:.4f} | bal-acc {ext['balanced_acc']:.4f} | "
                f"PR-AUC {ext['pr_auc_ng']:.3f} | ROC-AUC {ext['roc_auc_ng']:.3f} | MCC {ext['mcc']:.3f}",
                flush=True,
            )
            # Confusion matrix (non-gland positive class)
            print(
                f"  {fold_tag}   Confusion (ng+):  TP={ext['tp_ng']:>4}  FP={ext['fp_ng']:>4}  "
                f"FN={ext['fn_ng']:>4}  TN={ext['tn_ng']:>4}  | "
                f"Sens(ng)={ext['sensitivity_ng']:.3f} Spec(ng)={ext['specificity_ng']:.3f}",
                flush=True,
            )
            # Threshold sweep — non-gland P/R at multiple decision thresholds
            tsweep_parts = []
            for thr_v in (0.3, 0.4, 0.5, 0.6, 0.7):
                tsweep_parts.append(
                    f"thr{thr_v}:P{ext[f'thr_{thr_v}_precision_ng']:.2f}/R{ext[f'thr_{thr_v}_recall_ng']:.2f}"
                )
            print(f"  {fold_tag}   Thresh sweep (ng): {' | '.join(tsweep_parts)}", flush=True)

            print(
                f"  {fold_tag}   best-ext {best_ext_state['f1']:.4f}@e{best_ext_state['epoch']}"
                f"{':i'+str(best_ext_state['iter']) if best_ext_state['iter'] else ''}",
                flush=True,
            )
            epoch_log.append((
                epoch, val_metrics["f1"], ext["macro_f1"],
                ext["gland_f1"], ext["nongland_f1"],
                ext["precision_gland"], ext["recall_gland"],
                ext["precision_nongland"], ext["recall_nongland"],
                ext["pr_auc_ng"], ext["roc_auc_ng"],
                ext["mcc"], ext["balanced_acc"],
                ext["tp_ng"], ext["fp_ng"], ext["fn_ng"], ext["tn_ng"],
            ))

            if ext["macro_f1"] > best_ext_state["f1"]:
                best_ext_state["f1"] = ext["macro_f1"]
                best_ext_state["epoch"] = epoch
                best_ext_state["iter"] = None  # end-of-epoch marker
                _save_byext_ckpt(model, config, epoch, None, ext, val_metrics)
                print(f"  {fold_tag} >> Saved best-by-ext (macro-F1={ext['macro_f1']:.4f}) "
                      f"at epoch {epoch} (end)", flush=True)

            # ── Ext-based early stop (PRIMARY) ──
            # best_ext_state["epoch"] is updated by both sub-eval (inside train loop)
            # and end-of-epoch eval above. If best ext was N epochs ago, stop.
            ext_no_improve = epoch - best_ext_state["epoch"] if best_ext_state["epoch"] > 0 else 0
            if ext_no_improve >= config.ext_patience:
                print(f"  {fold_tag} Early stopping at epoch {epoch} "
                      f"(ext_patience {config.ext_patience}, "
                      f"best ext_f1 {best_ext_state['f1']:.4f}@e{best_ext_state['epoch']}"
                      f"{':i'+str(best_ext_state['iter']) if best_ext_state['iter'] else ''})",
                      flush=True)
                # Signal other ranks via a tensor broadcast or just rely on barrier+break.
                # Since only rank 0 runs ext eval, we need to broadcast the stop signal.
                _ext_stop = torch.tensor(1, device=device)
            else:
                _ext_stop = torch.tensor(0, device=device)
        else:
            _ext_stop = torch.tensor(0, device=device)

        if world_size > 1:
            dist.broadcast(_ext_stop, src=0)
        barrier(device)
        if int(_ext_stop.item()) == 1:
            break

    # ── Summary ──
    if is_main(rank):
        print(f"\n{fold_tag} ============ Summary ============", flush=True)
        print(f"  best val_f1 = {best_f1:.4f}  at epoch {best_val_epoch}", flush=True)
        iter_tag = f" iter{best_ext_state['iter']}" if best_ext_state['iter'] else " (end)"
        print(f"  best ext F1 = {best_ext_state['f1']:.4f}  at epoch {best_ext_state['epoch']}{iter_tag}",
              flush=True)

        if epoch_log:
            print(f"\n  Per-epoch log (epoch, val_f1, ext_macro, ext_gland, ext_nongland):",
                  flush=True)
            for e, v, em, eg, en in epoch_log:
                print(f"    {e:>3d}  val_f1={v:.4f}  ext={em:.4f}  g={eg:.4f}  ng={en:.4f}",
                      flush=True)
        if sub_eval_log:
            print(f"\n  Sub-epoch log (epoch, iter, ext_macro, ext_gland, ext_nongland):",
                  flush=True)
            for e, it, em, eg, en in sub_eval_log:
                print(f"    e{e}:i{it:>5}  ext={em:.4f}  g={eg:.4f}  ng={en:.4f}",
                      flush=True)

        try:
            import csv
            EXT_FULL_COLS = [
                "ext_macro_f1", "ext_gland_f1", "ext_nongland_f1",
                "precision_gland", "recall_gland",
                "precision_nongland", "recall_nongland",
                "pr_auc_ng", "roc_auc_ng", "mcc", "balanced_acc",
                "tp_ng", "fp_ng", "fn_ng", "tn_ng",
            ]
            if epoch_log:
                csv_path = Path(config.log_dir) / f"epoch_log_{config.backbone}_full{config.run_tag}.csv"
                with open(csv_path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["epoch", "val_f1"] + EXT_FULL_COLS)
                    w.writerows(epoch_log)
                print(f"  epoch log CSV: {csv_path}", flush=True)
            if sub_eval_log:
                sub_csv = Path(config.log_dir) / f"subeval_log_{config.backbone}_full{config.run_tag}.csv"
                with open(sub_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["epoch", "iter"] + EXT_FULL_COLS)
                    w.writerows(sub_eval_log)
                print(f"  sub-eval CSV: {sub_csv}", flush=True)
        except Exception as e:
            print(f"  [warn] CSV save failed: {e}", flush=True)

    cleanup_distributed()


if __name__ == "__main__":
    main()
