"""PyTorch dataset for PUWF-AV artery/vein segmentation."""

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


CLASS_COLORS_BGR = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [255, 0, 0],
    3: [0, 255, 0],
    4: [255, 255, 255],
}


class AVDataset(Dataset):
    """Load RGB images and convert color-coded BGR annotations to class indices."""

    def __init__(
        self, image_paths, annotation_paths, fov_paths=None, train=True, seed=123
    ):
        if len(image_paths) != len(annotation_paths):
            raise ValueError("The number of images and annotations must match.")
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        if fov_paths is not None and len(fov_paths) != len(image_paths):
            raise ValueError("The number of FOV masks and images must match.")
        self.fov_paths = fov_paths
        self.train = train

        transforms = []
        if self.train:
            transforms.extend(
                [
                    A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                    A.HorizontalFlip(p=0.5),
                ]
            )
        transforms.extend([A.ToFloat(max_value=255.0), ToTensorV2()])
        self.transform = A.Compose(transforms)
        if hasattr(self.transform, "set_random_seed"):
            self.transform.set_random_seed(seed)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        annotation_path = self.annotation_paths[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        annotation = cv2.imread(annotation_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if annotation is None:
            raise FileNotFoundError(f"Could not read annotation: {annotation_path}")
        if image.shape != annotation.shape:
            raise ValueError(
                f"Image and annotation shapes differ: {image_path}, {annotation_path}"
            )

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_mask = np.zeros(annotation.shape[:2], dtype=np.uint8)
        for class_index, color in CLASS_COLORS_BGR.items():
            class_mask[np.all(annotation == color, axis=-1)] = class_index

        transformed = self.transform(image=image, mask=class_mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"].unsqueeze(0).to(torch.long)
        if self.fov_paths is None:
            return image_tensor, mask_tensor

        fov_path = self.fov_paths[index]
        fov = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)
        if fov is None:
            raise FileNotFoundError(f"Could not read FOV mask: {fov_path}")
        if fov.shape != class_mask.shape:
            raise ValueError(
                f"Image and FOV shapes differ: {image_path}, {fov_path}"
            )
        fov_tensor = torch.from_numpy(fov > 0).unsqueeze(0)
        return image_tensor, mask_tensor, fov_tensor

    def __len__(self):
        return len(self.image_paths)
