"""Generate color-coded artery/vein segmentation masks."""

import argparse
import csv
import os

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from monai.inferers import sliding_window_inference
from tqdm import tqdm

from models import MODEL_NAMES, build_model
from utils import normalize_image


CLASS_COLORS_BGR = {
    0: [0, 0, 0],
    1: [0, 0, 255],
    2: [255, 0, 0],
    3: [0, 255, 0],
    4: [255, 255, 255],
}


def load_normalization_parameters(csv_path):
    """Read the three-channel training RGB parameters produced for one fold."""
    with open(csv_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 3 or not {"mean", "std"}.issubset(rows[0]):
        raise ValueError(
            "Normalization CSV must contain three rows and the columns 'mean' and 'std'."
        )
    means = [float(row["mean"]) for row in rows]
    standard_deviations = [float(row["std"]) for row in rows]
    return means, standard_deviations


class MaskPredictor:
    """Load one trained model and generate color-coded segmentation masks."""
    def __init__(
        self,
        model_path,
        model_name,
        image_path,
        fov_path,
        output_path,
        normalization_mean,
        normalization_std,
        image_size,
        device,
    ):
        self.image_path = image_path
        self.fov_path = fov_path
        self.output_path = output_path
        self.normalization_mean = normalization_mean
        self.normalization_std = normalization_std
        self.image_size = image_size
        self.device = device
        os.makedirs(self.output_path, exist_ok=True)

        self.model = build_model(model_name)
        state_dict = torch.load(
            model_path, map_location=self.device, weights_only=True
        )
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = A.Compose([A.ToFloat(max_value=255.0), ToTensorV2()])

    def predict(self):
        """Run sliding-window inference for every image in the input folder."""
        image_name_list = os.listdir(self.image_path)
        for image_name in tqdm(image_name_list, desc="Predicting"):
            image_file = os.path.join(self.image_path, image_name)
            fov_file = os.path.join(self.fov_path, image_name)
            image = cv2.imread(image_file, cv2.IMREAD_COLOR)
            fov = cv2.imread(fov_file, cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Could not read image: {image_file}")
            if fov is None:
                raise FileNotFoundError(f"Could not read FOV mask: {fov_file}")
            if image.shape != fov.shape:
                raise ValueError(
                    f"Image and FOV dimensions do not match: {image_file}, {fov_file}"
                )

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            fov = fov / 255.0
            image = image * fov
            image = normalize_image(
                image, self.normalization_mean, self.normalization_std
            )
            image = self.transform(image=image)["image"]
            image = image.unsqueeze(0).to(self.device, dtype=torch.float32)

            with torch.no_grad():
                prediction = sliding_window_inference(
                    inputs=image,
                    roi_size=self.image_size,
                    sw_batch_size=4,
                    predictor=self.model,
                    overlap=0.5,
                )
                prediction = F.softmax(prediction, dim=1).cpu().numpy()[0]
            prediction = np.argmax(prediction, axis=0).astype(np.uint8)

            color_mask = np.zeros((height, width, 3), dtype=np.uint8)
            for class_index, color in CLASS_COLORS_BGR.items():
                color_mask[prediction == class_index] = color
            color_mask = (color_mask * fov).astype(np.uint8)
            output_file = os.path.join(self.output_path, image_name)
            if not cv2.imwrite(output_file, color_mask):
                raise OSError(f"Could not write prediction: {output_file}")


def parse_args():
    """Parse checkpoint, preprocessing, input, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--normalization-csv", required=True)
    parser.add_argument("--image-path", default=os.path.join("PUWF-AV", "images"))
    parser.add_argument("--fov-path", default=os.path.join("PUWF-AV", "fov"))
    parser.add_argument("--output-path", default="predictions")
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    return parser.parse_args()


def main():
    """Load one checkpoint and generate predictions for an image folder."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    means, standard_deviations = load_normalization_parameters(
        args.normalization_csv
    )
    predictor = MaskPredictor(
        model_path=args.model_path,
        model_name=args.model,
        image_path=args.image_path,
        fov_path=args.fov_path,
        output_path=args.output_path,
        normalization_mean=means,
        normalization_std=standard_deviations,
        image_size=tuple(args.image_size),
        device=device,
    )
    predictor.predict()


if __name__ == "__main__":
    main()
