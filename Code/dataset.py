"""
PyTorch Dataset with albumentations augmentation for patch classification.
Supports loading patches from multiple slide directories.
"""

from pathlib import Path

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def get_train_transforms(input_size=224):
    return A.Compose([
        A.RandomResizedCrop(size=(input_size, input_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.5),
        A.ColorJitter(
            brightness=0.2, contrast=0.2,
            saturation=0.2, hue=0.05, p=0.8
        ),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(20, 40),
            hole_width_range=(20, 40),
            p=0.3,
        ),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


def get_val_transforms(input_size=224):
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


class PatchDataset(Dataset):
    """Dataset that loads patches from per-slide directories.

    Expected structure:
        patches_dir/
            slide_name_1/
                gland/  or  solid/
                    *.png
            slide_name_2/
                ...

    Args:
        patches_dir: root patches directory
        slide_names: list of slide names to include
        slides_config: dict mapping slide_name -> {"class": ..., "label": ...}
        transform: albumentations transform
    """

    def __init__(self, patches_dir, slide_names, slides_config, transform=None):
        self.patches_dir = Path(patches_dir)
        self.transform = transform
        self.samples = []  # (path, label)

        for slide_name in slide_names:
            info = slides_config[slide_name]
            cls_name = info["class"]
            label = info["label"]
            cls_dir = self.patches_dir / slide_name / cls_name
            if not cls_dir.exists():
                continue
            for img_path in sorted(cls_dir.glob("*.png")):
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

    def get_labels(self):
        return [label for _, label in self.samples]

    def get_class_counts(self):
        labels = self.get_labels()
        return {"gland": labels.count(0), "non-gland": labels.count(1)}
