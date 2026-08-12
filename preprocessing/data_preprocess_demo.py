"""Demonstrate preprocessing one image and its corresponding JSON annotation."""

import argparse
import json
import os

import cv2
import numpy as np
from skimage import measure


class DatasetGenerator:
    """Convert one source image and polygon annotation into the release format."""

    def __init__(self, args):
        self.image_path = args.image_path
        self.annotation_path = args.annotation_path
        self.output_image_folder = os.path.join(args.output_folder, "images")
        self.output_annotation_folder = os.path.join(args.output_folder, "annotation")
        self.output_fov_folder = os.path.join(args.output_folder, "fov")

        for folder in (
            self.output_image_folder,
            self.output_annotation_folder,
            self.output_fov_folder,
        ):
            os.makedirs(folder, exist_ok=True)

    @staticmethod
    def generate_fov(image):
        """Estimate a circular field of view from image edges using RANSAC."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        y_coordinates, x_coordinates = np.nonzero(edges)
        points = np.column_stack((x_coordinates, y_coordinates))
        if len(points) < 3:
            raise ValueError("Insufficient edge points for FOV circle estimation.")

        circle_model, _ = measure.ransac(
            data=points,
            model_class=measure.CircleModel,
            min_samples=3,
            residual_threshold=3,
            max_trials=2000,
        )
        if circle_model is None or circle_model.params is None:
            raise RuntimeError("RANSAC failed to estimate the FOV circle.")
        center_x, center_y, radius = circle_model.params

        fov = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(
            fov,
            (int(center_x), int(center_y)),
            int(radius),
            255,
            thickness=-1,
        )
        return cv2.merge([fov, fov, fov])

    @staticmethod
    def generate_annotation(json_path):
        """Rasterize artery, vein, uncertain, and crossing polygon annotations."""
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        width = data["imageWidth"]
        height = data["imageHeight"]
        annotation = np.zeros((height, width, 3), dtype=np.uint8)
        artery = np.zeros((height, width), dtype=np.uint8)
        vein = np.zeros((height, width), dtype=np.uint8)
        uncertain = np.zeros((height, width), dtype=np.uint8)

        for shape in data["shapes"]:
            label = shape["label"]
            points = np.asarray(shape["points"], dtype=np.int32)
            if label == "artery":
                cv2.fillPoly(annotation, [points], (255, 0, 0))
                cv2.fillPoly(artery, [points], 255)
            elif label == "vein":
                cv2.fillPoly(annotation, [points], (0, 0, 255))
                cv2.fillPoly(vein, [points], 255)
            elif label == "uncertain":
                cv2.fillPoly(uncertain, [points], 255)

        crossing = (artery == 255) & (vein == 255)
        annotation[crossing] = (0, 255, 0)
        annotation[uncertain == 255] = (255, 255, 255)
        return annotation

    @staticmethod
    def pad_to_size(image, target_size):
        """Center-pad an image or mask with zeros."""
        target_width, target_height = target_size
        height, width = image.shape[:2]
        if width > target_width or height > target_height:
            raise ValueError("Target padding size is smaller than the source image.")

        left = (target_width - width) // 2
        right = target_width - width - left
        top = (target_height - height) // 2
        bottom = target_height - height - top
        return cv2.copyMakeBorder(
            image,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

    @staticmethod
    def apply_fov(image, fov):
        """Set all pixels outside the field of view to zero."""
        if image.shape != fov.shape:
            raise ValueError("Image and FOV shapes do not match.")
        return (image * (fov / 255.0)).astype(np.uint8)

    @staticmethod
    def _write_image(path, image):
        if not cv2.imwrite(path, image):
            raise OSError(f"Could not write image: {path}")

    def generate_dataset(self):
        """Generate processed images using the original image filename."""
        image = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read source image: {self.image_path}")

        annotation = self.generate_annotation(self.annotation_path)
        if image.shape != annotation.shape:
            raise ValueError("Image and annotation shapes differ.")
        fov = self.generate_fov(image)

        side_length = max(image.shape[:2])
        target_size = (side_length, side_length)
        image = self.pad_to_size(self.apply_fov(image, fov), target_size)
        annotation = self.pad_to_size(self.apply_fov(annotation, fov), target_size)
        fov = self.pad_to_size(fov, target_size)

        annotation = cv2.cvtColor(annotation, cv2.COLOR_RGB2BGR)
        output_name = os.path.basename(self.image_path)
        self._write_image(os.path.join(self.output_image_folder, output_name), image)
        self._write_image(
            os.path.join(self.output_annotation_folder, output_name), annotation
        )
        self._write_image(os.path.join(self.output_fov_folder, output_name), fov)


def parse_args():
    """Parse one source image, its JSON annotation, and the output folder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-path", required=True, help="Path to one source image.")
    parser.add_argument(
        "--annotation-path",
        required=True,
        help="Path to the JSON annotation corresponding to the source image.",
    )
    parser.add_argument(
        "--output-folder",
        default="PUWF-AV-demo",
        help="Output folder for the processed example.",
    )
    return parser.parse_args()


def main():
    """Generate one processed example from source files."""
    args = parse_args()
    DatasetGenerator(args).generate_dataset()


if __name__ == "__main__":
    main()
