"""
Patch classification model factory (gland vs solid).

All foundation models are registered in `model_registry.py`. To swap:
    config.backbone = "uni2" | "virchow2" | "phikon-v2" | "h-optimus-0" | "uni" | "resnet18"

Phase 1 (linear probe): backbone fully frozen, only head learns. forward uses
  torch.no_grad() on backbone → activation buffer not retained.
Phase 2 (partial fine-tune): last N transformer blocks + final norm + head learn.
"""

import os
import torch
import torch.nn as nn

from model_registry import REGISTRY


# ─────────────────────────────────────────────────────────────
# Generic foundation classifier (works for any registered backbone)
# ─────────────────────────────────────────────────────────────
class FoundationClassifier(nn.Module):
    def __init__(self, backbone_name, num_classes=2, pretrained=True,
                 head_type="linear", unfreeze_last_n=4):
        super().__init__()
        if backbone_name not in REGISTRY:
            raise ValueError(f"Unknown backbone: {backbone_name!r}. "
                             f"Available: {list(REGISTRY)}")
        self.backbone_name = backbone_name
        spec = REGISTRY[backbone_name]
        self.spec = spec

        if spec.get("hf_gated") and pretrained and not (
            os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        ):
            print(f"WARN: {backbone_name} is HF gated. HF_TOKEN not set — "
                  f"if cached weights are missing, download will fail.", flush=True)

        self.backbone, embed_dim = spec["loader"](num_classes, pretrained)
        self.embed_dim = embed_dim
        self.forward_kind = spec.get("forward_kind", "default")

        self.head = self._build_head(head_type, embed_dim, num_classes)

        self.unfreeze_last_n = unfreeze_last_n
        self._train_backbone = False
        self.set_phase(1)  # default: linear probe

    @staticmethod
    def _build_head(kind, in_dim, num_classes):
        if kind == "linear":
            return nn.Linear(in_dim, num_classes)
        if kind == "mlp":
            return nn.Sequential(
                nn.Linear(in_dim, 512),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(512, num_classes),
            )
        raise ValueError(f"Unknown head_type: {kind!r}")

    def set_phase(self, phase):
        """phase=1: linear probe (backbone frozen).
           phase=2: partial fine-tune (last N blocks + norm + head)."""
        if phase == 1:
            for p in self.backbone.parameters():
                p.requires_grad = False
            for p in self.head.parameters():
                p.requires_grad = True
            self._train_backbone = False
        elif phase == 2:
            for p in self.backbone.parameters():
                p.requires_grad = False
            blocks = getattr(self.backbone, "blocks", None)
            if blocks is not None:
                for blk in blocks[-self.unfreeze_last_n:]:
                    for p in blk.parameters():
                        p.requires_grad = True
            norm = getattr(self.backbone, "norm", None)
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad = True
            for p in self.head.parameters():
                p.requires_grad = True
            self._train_backbone = True
        else:
            raise ValueError(f"Unknown phase {phase}")

    def _forward_backbone(self, x):
        if self.forward_kind == "hf_cls":
            # HuggingFace transformers model: extract CLS token from last hidden state
            out = self.backbone(x)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state[:, 0, :]
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                return out.pooler_output
            return out  # fallback
        if self.forward_kind == "virchow":
            # Virchow2 returns full sequence (B, 1+R+P, D) = CLS + register + patches.
            # Paige official feature: cat(cls, mean(patch_tokens)) → 2*D dim.
            n_reg = self.spec.get("n_register_tokens", 4)
            tokens = self.backbone(x)               # (B, 1+n_reg+P, D)
            cls = tokens[:, 0, :]                    # (B, D)
            patches = tokens[:, 1 + n_reg:, :]       # (B, P, D)
            return torch.cat([cls, patches.mean(dim=1)], dim=1)  # (B, 2D)
        # default: timm with num_classes=0 returns features directly
        return self.backbone(x)

    def forward(self, x):
        if self._train_backbone:
            feat = self._forward_backbone(x)
        else:
            with torch.no_grad():
                feat = self._forward_backbone(x)
            feat = feat.detach()
        return self.head(feat)


# ─────────────────────────────────────────────────────────────
# Public API (kept compatible with previous calls)
# ─────────────────────────────────────────────────────────────
_RESNET_EARLY = ("conv1", "bn1", "layer1", "layer2")


def create_model(num_classes=2, pretrained=True, backbone="uni2", head_type="linear"):
    """Build a model. Dispatcher driven by `model_registry.REGISTRY`."""
    if backbone not in REGISTRY:
        raise ValueError(f"Unknown backbone: {backbone!r}. Available: {list(REGISTRY)}")
    spec = REGISTRY[backbone]
    if not spec.get("is_foundation", True):
        # Non-foundation models (ResNet18) build backbone+head together
        m, _ = spec["loader"](num_classes, pretrained)
        return m
    return FoundationClassifier(
        backbone, num_classes=num_classes, pretrained=pretrained, head_type=head_type,
    )


def freeze_early_layers(model, backbone="resnet18"):
    """Phase 1 freeze."""
    if backbone == "resnet18":
        for name, p in model.named_parameters():
            if any(name.startswith(x) for x in _RESNET_EARLY):
                p.requires_grad = False
    else:
        # All foundation models go through FoundationClassifier
        raw = model.module if hasattr(model, "module") else model
        raw.set_phase(1)


def unfreeze_all(model, backbone="resnet18"):
    """Phase 2 unfreeze."""
    if backbone == "resnet18":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raw = model.module if hasattr(model, "module") else model
        raw.set_phase(2)
