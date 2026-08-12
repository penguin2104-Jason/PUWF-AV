"""Create reproducible five-fold datasets from the released PUWF-AV data."""

import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from utils import calculate_rgb_parameters, normalize_image


class FoldDataWriter:
    """Write the training patches and full-resolution validation/test images."""

    def __init__(
        self,
        dataset_folder,
        output_folder,
        train_files,
        val_files,
        test_files,
        patch_size,
        stride,
    ):
        self.image_folder = os.path.join(dataset_folder, "images")
        self.annotation_folder = os.path.join(dataset_folder, "annotation")
        self.fov_folder = os.path.join(dataset_folder, "fov")
        self.output_folder = output_folder
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files
        self.patch_size = patch_size
        self.stride = stride

        for split in ("train", "val", "test"):
            os.makedirs(
                os.path.join(self.output_folder, split, "images"), exist_ok=True
            )
            os.makedirs(
                os.path.join(self.output_folder, split, "av_masks"), exist_ok=True
            )
            if split in ("val", "test"):
                os.makedirs(
                    os.path.join(self.output_folder, split, "fov_masks"),
                    exist_ok=True,
                )

        training_images = [
            os.path.join(self.image_folder, name) for name in self.train_files
        ]
        training_fovs = [
            os.path.join(self.fov_folder, name) for name in self.train_files
        ]
        self.mean_rgb, self.std_rgb = calculate_rgb_parameters(
            training_images, training_fovs
        )
        parameters = pd.DataFrame({"mean": self.mean_rgb, "std": self.std_rgb})
        parameters.to_csv(
            os.path.join(self.output_folder, "train_rgb_parameters.csv"), index=False
        )

    def _load_and_preprocess(self, filename):
        """Load a matching image, annotation, and FOV mask and normalize the image."""
        image_path = os.path.join(self.image_folder, filename)
        annotation_path = os.path.join(self.annotation_folder, filename)
        fov_path = os.path.join(self.fov_folder, filename)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        annotation = cv2.imread(annotation_path, cv2.IMREAD_COLOR)
        fov = cv2.imread(fov_path, cv2.IMREAD_COLOR)
        for path, array in (
            (image_path, image),
            (annotation_path, annotation),
            (fov_path, fov),
        ):
            if array is None:
                raise FileNotFoundError(f"Could not read required file: {path}")
        if image.shape != annotation.shape or image.shape != fov.shape:
            raise ValueError(f"Image, annotation, and FOV shapes differ for {filename}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fov_scale = fov / 255.0
        image = image * fov_scale
        annotation = (annotation * fov_scale).astype(np.uint8)
        image = normalize_image(image, self.mean_rgb, self.std_rgb)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, annotation, fov

    @staticmethod
    def _write_image(path, image):
        if not cv2.imwrite(path, image):
            raise OSError(f"Could not write image: {path}")

    def _patch_starts(self, length):
        """Return regular starts plus a final window aligned to the boundary."""
        if length < self.patch_size:
            raise ValueError(
                f"Image dimension {length} is smaller than patch size {self.patch_size}."
            )
        final_start = length - self.patch_size
        starts = list(range(0, final_start + 1, self.stride))
        if starts[-1] != final_start:
            starts.append(final_start)
        return starts

    def generate_training_patches(self, filename):
        """Extract and save all full-size sliding-window patches from one image."""
        image, annotation, _ = self._load_and_preprocess(filename)
        height, width = image.shape[:2]
        source_name = os.path.splitext(filename)[0]
        patch_id = 0

        for y in self._patch_starts(height):
            for x in self._patch_starts(width):
                patch_name = f"{source_name}_{patch_id}.png"
                image_patch = image[y : y + self.patch_size, x : x + self.patch_size]
                annotation_patch = annotation[
                    y : y + self.patch_size, x : x + self.patch_size
                ]
                self._write_image(
                    os.path.join(
                        self.output_folder, "train", "images", patch_name
                    ),
                    image_patch,
                )
                self._write_image(
                    os.path.join(
                        self.output_folder, "train", "av_masks", patch_name
                    ),
                    annotation_patch,
                )
                patch_id += 1

    def save_full_image(self, filename, split):
        """Save one preprocessed full-resolution validation or test sample."""
        image, annotation, fov = self._load_and_preprocess(filename)
        self._write_image(
            os.path.join(self.output_folder, split, "images", filename), image
        )
        self._write_image(
            os.path.join(self.output_folder, split, "av_masks", filename),
            annotation,
        )
        self._write_image(
            os.path.join(self.output_folder, split, "fov_masks", filename), fov
        )

    def process_all_images(self):
        """Materialize all three subsets for the current fold."""
        for filename in tqdm(self.train_files, desc="Processing training images"):
            self.generate_training_patches(filename)
        for filename in tqdm(self.val_files, desc="Processing validation images"):
            self.save_full_image(filename, "val")
        for filename in tqdm(self.test_files, desc="Processing test images"):
            self.save_full_image(filename, "test")
        print(f"Data saved to {self.output_folder}")


class KFoldDataGenerator:
    """Load or create a split manifest and materialize every fold on disk."""

    MANIFEST_VERSION = 1

    def __init__(
        self,
        dataset_folder,
        output_folder,
        manifest_path,
        folds,
        val_size,
        patch_size,
        stride,
        seed,
    ):
        self.dataset_folder = dataset_folder
        self.output_folder = output_folder
        self.manifest_path = manifest_path
        self.folds = folds
        self.val_size = val_size
        self.patch_size = patch_size
        self.stride = stride
        self.seed = seed

    def create_splits(self, sample_list):
        """Create split lists once; subsequent runs reuse the saved manifest."""
        samples = np.asarray(sample_list)
        splitter = KFold(
            n_splits=self.folds, shuffle=True, random_state=self.seed
        )
        splits = []
        for train_val_indices, test_indices in splitter.split(samples):
            train_val = samples[train_val_indices].tolist()
            test = samples[test_indices].tolist()
            train, val = train_test_split(
                train_val,
                test_size=self.val_size,
                random_state=self.seed,
            )
            splits.append({"train": train, "val": val, "test": test})
        return splits

    def _validate_manifest(self, manifest, available_files):
        if manifest.get("version") != self.MANIFEST_VERSION:
            raise ValueError("Unsupported split manifest version.")
        if manifest.get("fold_count") != self.folds:
            raise ValueError(
                "The requested fold count does not match the split manifest."
            )
        if not np.isclose(
            manifest.get("validation_fraction", -1), self.val_size
        ):
            raise ValueError(
                "The requested validation fraction does not match the split manifest."
            )
        splits = manifest.get("splits", [])
        if len(splits) != self.folds:
            raise ValueError("The split manifest contains an invalid number of folds.")

        available = set(available_files)
        for fold_index, split in enumerate(splits, start=1):
            train = set(split.get("train", []))
            val = set(split.get("val", []))
            test = set(split.get("test", []))
            if (
                len(train) != len(split.get("train", []))
                or len(val) != len(split.get("val", []))
                or len(test) != len(split.get("test", []))
            ):
                raise ValueError(f"Fold {fold_index} contains duplicate filenames.")
            if train & val or train & test or val & test:
                raise ValueError(f"Fold {fold_index} contains overlapping splits.")
            if train | val | test != available:
                missing = sorted(available - (train | val | test))
                unknown = sorted((train | val | test) - available)
                raise ValueError(
                    f"Fold {fold_index} does not match the dataset. "
                    f"Missing: {missing}; unknown: {unknown}"
                )
        return splits

    def load_or_create_manifest(self, available_files):
        """Reuse a validated manifest or create and save one on the first run."""
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path, "r", encoding="utf-8") as file:
                manifest = json.load(file)
            splits = self._validate_manifest(manifest, available_files)
            print(
                f"Loaded split manifest: {self.manifest_path} "
                f"(generation seed={manifest.get('seed', 'unknown')})"
            )
            return splits

        splits = self.create_splits(available_files)
        manifest = {
            "version": self.MANIFEST_VERSION,
            "fold_count": self.folds,
            "validation_fraction": self.val_size,
            "seed": self.seed,
            "splits": splits,
        }
        manifest_folder = os.path.dirname(os.path.abspath(self.manifest_path))
        os.makedirs(manifest_folder, exist_ok=True)
        with open(self.manifest_path, "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
            file.write("\n")
        print(f"Created split manifest: {self.manifest_path} (seed={self.seed})")
        return splits

    def generate_dataset(self):
        """Validate inputs and materialize every fold from the fixed manifest."""
        image_folder = os.path.join(self.dataset_folder, "images")
        if not os.path.isdir(image_folder):
            raise FileNotFoundError(
                f"Dataset images were not found at {image_folder}. "
                "Download PUWF-AV from Figshare or pass --dataset-folder."
            )

        if os.path.isdir(self.output_folder):
            existing_paths = [
                os.path.abspath(entry.path) for entry in os.scandir(self.output_folder)
            ]
            allowed_paths = {os.path.abspath(self.manifest_path)}
            unexpected_paths = [
                path for path in existing_paths if path not in allowed_paths
            ]
            if unexpected_paths:
                raise FileExistsError(
                    f"Output folder contains generated data: {self.output_folder}. "
                    "Use an empty folder to prevent files from different splits being mixed."
                )

        available_files = sorted(
            name
            for name in os.listdir(image_folder)
            if name.lower().endswith(".png")
            and os.path.isfile(os.path.join(image_folder, name))
        )
        if not available_files:
            raise ValueError(f"No images found in {image_folder}.")
        splits = self.load_or_create_manifest(available_files)

        for fold_index, split in enumerate(splits, start=1):
            fold_folder = os.path.join(self.output_folder, f"fold_{fold_index}")
            writer = FoldDataWriter(
                dataset_folder=self.dataset_folder,
                output_folder=fold_folder,
                train_files=split["train"],
                val_files=split["val"],
                test_files=split["test"],
                patch_size=self.patch_size,
                stride=self.stride,
            )
            writer.process_all_images()


def parse_args():
    """Parse command-line options for split creation and data preparation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-folder",
        default="PUWF-AV",
        help="PUWF-AV root containing images, annotation, and fov directories.",
    )
    parser.add_argument("--output-folder", default="kfold_set")
    parser.add_argument(
        "--split-manifest",
        default=None,
        help=(
            "Optional split JSON path. By default, split_manifest.json is saved "
            "inside the output folder."
        ),
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=1 / 8)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument(
        "--seed", type=int, default=123, help="Split random seed (default: 123)."
    )
    return parser.parse_args()


def main():
    """Run cross-validation data preparation from command-line arguments."""
    args = parse_args()
    manifest_path = args.split_manifest or os.path.join(
        args.output_folder, "split_manifest.json"
    )
    generator = KFoldDataGenerator(
        dataset_folder=args.dataset_folder,
        output_folder=args.output_folder,
        manifest_path=manifest_path,
        folds=args.folds,
        val_size=args.val_size,
        patch_size=args.patch_size,
        stride=args.stride,
        seed=args.seed,
    )
    generator.generate_dataset()


if __name__ == "__main__":
    main()
