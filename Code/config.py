from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Paths ──
    base_dir: str = "/app/Gland_Seg"
    data_dir: str = "/app/Gland_Seg/Data"
    svs_dir: str = "/app/Gland_Seg/Data/S14/SVS"
    xml_dir: str = "/app/Gland_Seg/Data/S14/Annotation"
    output_dir: str = "/app/Gland_Seg/patches_raw_224_20x"
    checkpoint_dir: str = "/app/Gland_Seg/checkpoints"
    log_dir: str = "/app/Gland_Seg/logs"
    viz_dir: str = "/app/Gland_Seg/Viz"

    # ── Stain normalization ──
    stain_normalize: bool = False
    stain_target_path: str = "/app/Gland_Seg/Data/stain_target.png"  # target patch for Macenko.fit()

    # ── Slide-to-class mapping (label: gland=0, non-gland=solid=1) ──
    slides: dict = field(default_factory=lambda: {
        # non-gland (solid cancer) — 5 slides
        "S14-177-1-5":   {"xml": "S14-177-1-5_S.xml",   "svs": "S14-177-1-5.svs",   "class": "non-gland", "label": 1},
        "S14-1255-1-3":  {"xml": "S14-1255-1-3_S.xml",  "svs": "S14-1255-1-3.svs",  "class": "non-gland", "label": 1},
        "S14-1382-4":    {"xml": "S14-1382-4_S.xml",    "svs": "S14-1382-4.svs",    "class": "non-gland", "label": 1},
        "S14-1639-1-7":  {"xml": "S14-1639-1-7_S.xml",  "svs": "S14-1639-1-7.svs",  "class": "non-gland", "label": 1},
        "S14-2162-1-5":  {"xml": "S14-2162-1-5_S.xml",  "svs": "S14-2162-1-5.svs",  "class": "non-gland", "label": 1},
        # gland-forming cancer — 3 slides
        "S14-248-1-3":   {"xml": "S14-248-1-3_G.xml",   "svs": "S14-248-1-3.svs",   "class": "gland",     "label": 0},
        "S14-252-3":     {"xml": "S14-252-3_G.xml",     "svs": "S14-252-3.svs",     "class": "gland",     "label": 0},
        "S14-1720-6":    {"xml": "S14-1720-6_G.xml",    "svs": "S14-1720-6.svs",    "class": "gland",     "label": 0},
        # Additional non-gland (solid cancer) — round 2, 6 slides
        "S14-2476-1-3":  {"xml": "S14-2476-1-3_S.xml",  "svs": "S14-2476-1-3.svs",  "class": "non-gland", "label": 1},
        "S14-2478-1-6":  {"xml": "S14-2478-1-6_S.xml",  "svs": "S14-2478-1-6.svs",  "class": "non-gland", "label": 1},
        "S14-2503-1-9":  {"xml": "S14-2503-1-9_S.xml",  "svs": "S14-2503-1-9.svs",  "class": "non-gland", "label": 1},
        "S14-2571-1-8":  {"xml": "S14-2571-1-8_S.xml",  "svs": "S14-2571-1-8.svs",  "class": "non-gland", "label": 1},
        "S14-2635-3":    {"xml": "S14-2635-3_S.xml",    "svs": "S14-2635-3.svs",    "class": "non-gland", "label": 1},
        "S14-2991-1-5":  {"xml": "S14-2991-1-5_S.xml",  "svs": "S14-2991-1-5.svs",  "class": "non-gland", "label": 1},
    })

    # ── External evaluation (held-out, NOT used for training) ──
    # Professor's intent: pass these through the model and review predictions.
    # Annotation hierarchy: large positive boxes = ROI; small polygons inside = non-gland regions.
    # Outside small polygons but inside ROI = gland (mostly) + some normal.
    external_test_slides: dict = field(default_factory=lambda: {
        "S14-2289-1-6":  {"xml": "S14-2289-1-6.xml", "svs": "S14-2289-1-6.svs"},
    })

    # ── Patch extraction ──
    # Strategy: read 448x448 native @ L0 (40x, 0.252 um/px) → Macenko → downsample to input_size (224)
    # → save 224 PNG. Each saved patch represents a 448x448 L0 footprint = ~112 um FoV,
    # MPP_effective = 0.504 ≈ 20x. This matches Virchow2/UNI2/Phikon-v2 pretraining magnification.
    # SVS files have no true 20x level (L1 is a 1024px thumbnail), so we must read at L0 and downsample.
    patch_size: int = 448       # L0-pixel footprint of each patch (used by read_region, mask, stride loop, meta)
    stride: int = 224           # L0 pixels — 50% overlap of 448 native
    extraction_level: int = 0   # level 0 = 40x (0.252 um/px). L1 is thumbnail, NOT a 20x level.
    tissue_threshold: float = 0.7   # min tissue pixel fraction
    mask_threshold: float = 0.5     # min annotation mask fraction
    extract_workers: int = 48       # multiprocessing workers per slide; 0 or 1 = sequential

    # ── Backbone selection ──
    # See model_registry.REGISTRY for available names:
    #   "resnet18" | "uni2" | "virchow2" | "phikon-v2" | "h-optimus-0" | "uni"
    backbone: str = "hibou-l"
    head_type: str = "linear"      # "linear" | "mlp"
    amp_dtype: str = "bfloat16"    # "bfloat16" | "float16" | "float32". L40 supports bf16 natively.

    # If True, override `batch_size` with the registry's recommended_batch_size for the chosen backbone.
    auto_batch_size: bool = True

    # ── Training ──
    input_size: int = 224       # 224 for both ResNet18 and UNI2-h
    # per-GPU batch size. UNI2 ViT-H/14: 64 is safe on L40 48GB with bf16 autocast.
    # ResNet18: can go to 1024+. Adjust with backbone.
    batch_size: int = 64
    num_workers: int = 4        # per-GPU dataloader workers (fewer = less RAM per rank)
    epochs: int = 30            # UNI2 converges faster; ResNet18 used 50
    lr: float = 1e-4            # base lr @ batch 512 — scales linearly with effective batch via `lr_scale_base`
    lr_scale_base: int = 512    # effective-batch reference for linear-scaling rule
    weight_decay: float = 1e-4
    unfreeze_epoch: int = 999   # 0~4 LP / 5+ partial FT (last 4 block + norm + head, lr/10). 999=pure LP.
    patience: int = 5           # early stopping patience on val F1 (safety net; secondary)
    ext_patience: int = 3       # early stopping patience on external macro-F1 (PRIMARY).
                                # Stop if best_ext_state.epoch is >= ext_patience epochs behind current.
    random_seed: int = 42

    # ── Class info ──
    class_names: list = field(default_factory=lambda: ["gland", "non-gland"])
    num_classes: int = 2

    # ── Loss function ──
    # "ce"    : nn.CrossEntropyLoss(weight=class_weights)  — inverse-frequency weighted CE (default)
    # "focal" : FocalLoss(alpha=class_weights, gamma=focal_gamma) — focus on hard examples
    loss_type: str = "ce"
    focal_gamma: float = 2.0

    # ── Run tag ──
    # Appended to checkpoint filenames, external results dir, training log csv.
    # base run (256 px / 40x extraction): "" (empty)
    # 224_20x run (448 native → 224 disk, ~20x effective): "_224_20x"
    # Convention: leading underscore so paths read naturally (best_model_virchow2_full_224_20x.pth).
    run_tag: str = "_224_20x_aug"

    # ── Viz layout ──
    # Backbone → Viz subdir derived from model_registry.REGISTRY.
    # Layout: Viz/{backbone_subdir}/{category}/...
    # Categories: Annotation_Viz, Matrix_Viz, Prediction_Viz, Prediction_WSI

    def viz_dir_for(self, category: str):
        """Return Viz/{backbone_subdir}/{category}/ as a Path. Auto-creates."""
        from model_registry import REGISTRY
        sub = REGISTRY.get(self.backbone, {}).get("viz_subdir", self.backbone)
        d = Path(self.viz_dir) / sub / category
        d.mkdir(parents=True, exist_ok=True)
        return d

    def __post_init__(self):
        """Resolve backbone-dependent defaults (batch_size from registry)."""
        if self.auto_batch_size:
            try:
                from model_registry import REGISTRY
                rec = REGISTRY.get(self.backbone, {}).get("recommended_batch_size")
                if rec is not None:
                    self.batch_size = rec
            except Exception:
                pass  # registry import may fail at config-construction time; ignore

    def ensure_dirs(self):
        for d in [self.output_dir, self.checkpoint_dir, self.log_dir, self.viz_dir]:
            Path(d).mkdir(parents=True, exist_ok=True)
        # also pre-create per-backbone categories for current backbone
        for cat in ("Annotation_Viz", "Matrix_Viz", "Prediction_Viz", "Prediction_WSI"):
            self.viz_dir_for(cat)
