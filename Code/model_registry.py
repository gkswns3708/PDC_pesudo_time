"""
Pathology foundation model registry.

To add a new backbone:
    1. Define a `_load_<name>` function that returns (backbone_module, embed_dim).
       - For timm hub models: use `timm.create_model("hf-hub:org/repo", ...)` with
         `num_classes=0` to get a feature extractor.
       - For HuggingFace `transformers` models: use `AutoModel.from_pretrained(...)`
         and set `forward_kind="hf_cls"` so the wrapper extracts the CLS token.
    2. Call `register(name, loader=_load_<name>, embed_dim=..., viz_subdir=..., ...)`.
    3. (Optional) Update `config.batch_size` / `config.epochs` for the new model.

Then switch backbone via `config.backbone = "<name>"` — no code changes elsewhere.
"""

import torch.nn as nn


REGISTRY = {}  # name -> spec dict


def register(name, **spec):
    """Register a backbone. Required keys: loader, embed_dim, viz_subdir."""
    REGISTRY[name] = spec


# ─────────────────────────────────────────────────────────────
# Baseline — torchvision ResNet18 (not a foundation model)
# ─────────────────────────────────────────────────────────────
def _load_resnet18(num_classes, pretrained):
    from torchvision import models
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    m = models.resnet18(weights=weights)
    in_dim = m.fc.in_features
    m.fc = nn.Linear(in_dim, num_classes)
    return m, in_dim


register(
    "resnet18",
    loader=_load_resnet18,
    embed_dim=512,
    input_size=224,
    viz_subdir="Resnet18(scratch)",
    is_foundation=False,           # backbone+head bundled (m.fc replaced)
    recommended_batch_size=1024,
)


# ─────────────────────────────────────────────────────────────
# UNI2-h (MahmoodLab) — ViT-H/14, gated
# ─────────────────────────────────────────────────────────────
def _load_uni2(num_classes, pretrained):
    import timm
    from timm.layers import SwiGLUPacked
    m = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h", pretrained=pretrained,
        img_size=224, patch_size=14, depth=24, num_heads=24,
        init_values=1e-5, embed_dim=1536, mlp_ratio=2.66667 * 2,
        num_classes=0, no_embed_class=True,
        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU,
        reg_tokens=8, dynamic_img_size=True,
    )
    return m, 1536


register(
    "uni2",
    loader=_load_uni2,
    embed_dim=1536,
    input_size=224,
    viz_subdir="Uni",
    is_foundation=True,
    recommended_batch_size=64,
    hf_gated=True,
)


# ─────────────────────────────────────────────────────────────
# Virchow2 (Paige) — ViT-H/14, gated, 3.1M WSI pretrain
# ─────────────────────────────────────────────────────────────
def _load_virchow2(num_classes, pretrained):
    import timm
    from timm.layers import SwiGLUPacked
    m = timm.create_model(
        "hf-hub:paige-ai/Virchow2", pretrained=pretrained,
        img_size=224, init_values=1e-5,
        num_classes=0,
        mlp_layer=SwiGLUPacked, act_layer=nn.SiLU,
        reg_tokens=4, dynamic_img_size=True,
    )
    # Virchow2 returns full token sequence (B, 261, 1280) = CLS(1) + reg(4) + patches(256).
    # Paige official feature: cat(cls, mean(patch_tokens)) → 2560-dim.
    # Pooling done in FoundationClassifier via forward_kind="virchow".
    return m, 2560


register(
    "virchow2",
    loader=_load_virchow2,
    embed_dim=2560,
    input_size=224,
    viz_subdir="Virchow2",
    is_foundation=True,
    forward_kind="virchow",        # see model.py:_forward_backbone
    n_register_tokens=4,           # CLS(1) + reg(4) → patch tokens start at index 5
    recommended_batch_size=64,
    hf_gated=True,
)


# ─────────────────────────────────────────────────────────────
# Phikon v2 (Owkin) — ViT-L/16, NON-gated, easy access
# ─────────────────────────────────────────────────────────────
def _load_phikon_v2(num_classes, pretrained):
    # Loaded via HuggingFace transformers (not timm)
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained("owkin/phikon-v2")
    # ViT-L hidden_size = 1024
    return backbone, 1024


register(
    "phikon-v2",
    loader=_load_phikon_v2,
    embed_dim=1024,
    input_size=224,
    viz_subdir="Phikon-v2",
    is_foundation=True,
    forward_kind="hf_cls",         # signal to wrapper: extract last_hidden_state[:, 0, :]
    recommended_batch_size=128,
    hf_gated=False,
)


# ─────────────────────────────────────────────────────────────
# H-optimus-0 (Bioptimus) — ViT-G/14, OPEN weights, 500k WSI
# ─────────────────────────────────────────────────────────────
def _load_h_optimus_0(num_classes, pretrained):
    import timm
    m = timm.create_model(
        "hf-hub:bioptimus/H-optimus-0", pretrained=pretrained,
        init_values=1e-5, dynamic_img_size=False,
        num_classes=0,
    )
    return m, 1536  # ViT-G/14 cls dim


register(
    "h-optimus-0",
    loader=_load_h_optimus_0,
    embed_dim=1536,
    input_size=224,
    viz_subdir="H-optimus-0",
    is_foundation=True,
    recommended_batch_size=32,     # ViT-G is heavier
    hf_gated=True,                 # gated — access request required
)


# ─────────────────────────────────────────────────────────────
# Prov-GigaPath tile encoder (Microsoft) — ViT-G/14, gated, 86k WSI
# Note: GigaPath has both a tile encoder and a slide encoder. We use the tile
# encoder only for patch-level classification (slide encoder is for WSI tasks).
# ─────────────────────────────────────────────────────────────
def _load_prov_gigapath(num_classes, pretrained):
    import timm
    m = timm.create_model(
        "hf_hub:prov-gigapath/prov-gigapath",
        pretrained=pretrained,
        num_classes=0,
    )
    return m, 1536  # ViT-G/14 cls dim


register(
    "prov-gigapath",
    loader=_load_prov_gigapath,
    embed_dim=1536,
    input_size=224,
    viz_subdir="Prov-GigaPath",
    is_foundation=True,
    recommended_batch_size=32,     # ViT-G is heavy (~1.1B params)
    hf_gated=True,
)


# ─────────────────────────────────────────────────────────────
# Hibou-L (HistAI) — DINOv2 ViT-L, non-gated, 1.1M WSI pretrain
# Loaded via HuggingFace transformers (custom code requires trust_remote_code).
# ─────────────────────────────────────────────────────────────
def _load_hibou_l(num_classes, pretrained):
    from transformers import AutoModel
    backbone = AutoModel.from_pretrained(
        "histai/hibou-L",
        trust_remote_code=True,
    )
    return backbone, 1024  # ViT-L hidden_size


register(
    "hibou-l",
    loader=_load_hibou_l,
    embed_dim=1024,
    input_size=224,
    viz_subdir="Hibou-L",
    is_foundation=True,
    forward_kind="hf_cls",         # extract last_hidden_state[:, 0, :]
    recommended_batch_size=128,
    hf_gated=True,                 # gated as of 2025+ — access request required
)


# ─────────────────────────────────────────────────────────────
# UNI v1 (MahmoodLab) — ViT-L/16, gated, ablation/baseline
# ─────────────────────────────────────────────────────────────
def _load_uni(num_classes, pretrained):
    import timm
    m = timm.create_model(
        "hf-hub:MahmoodLab/uni", pretrained=pretrained,
        init_values=1e-5, dynamic_img_size=True,
        num_classes=0,
    )
    return m, 1024


register(
    "uni",
    loader=_load_uni,
    embed_dim=1024,
    input_size=224,
    viz_subdir="Uni-v1",
    is_foundation=True,
    recommended_batch_size=128,
    hf_gated=True,
)
