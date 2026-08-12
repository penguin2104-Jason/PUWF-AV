"""Shared reproducibility, normalization, timing, and metric utilities."""

import os
import random

import cv2
import numpy as np
import torch
from tqdm import tqdm


def seed_everything(seed):
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_data_loader_worker(worker_id):
    """Seed NumPy and Python inside a PyTorch DataLoader worker."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def epoch_time(start_time, end_time):
    """Convert elapsed seconds into integer minutes and seconds."""
    elapsed_time = end_time - start_time
    elapsed_minutes = int(elapsed_time / 60)
    elapsed_seconds = int(elapsed_time - elapsed_minutes * 60)
    return elapsed_minutes, elapsed_seconds


def calculate_rgb_parameters(image_paths, fov_paths):
    """Calculate RGB means and standard deviations after applying FOV masks.

    The calculation preserves the preprocessing used in the reported experiments:
    zero-valued pixels outside the FOV remain part of the image-level statistics.
    """
    if len(image_paths) != len(fov_paths) or not image_paths:
        raise ValueError("Matching, non-empty image and FOV path lists are required.")

    channel_sums = np.zeros(3, dtype=np.float64)
    channel_squared_sums = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for image_path, fov_path in tqdm(
        zip(image_paths, fov_paths),
        total=len(image_paths),
        desc="Calculating RGB statistics",
    ):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        fov = cv2.imread(fov_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        if fov is None:
            raise FileNotFoundError(f"Could not read FOV mask: {fov_path}")
        if image.shape != fov.shape:
            raise ValueError(f"Image and FOV shapes differ: {image_path}, {fov_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image *= fov / 255.0
        channel_sums += image.sum(axis=(0, 1))
        channel_squared_sums += np.square(image).sum(axis=(0, 1))
        pixel_count += image.shape[0] * image.shape[1]

    means = channel_sums / pixel_count
    variances = channel_squared_sums / pixel_count - np.square(means)
    standard_deviations = np.sqrt(np.maximum(variances, 0.0))
    mean_rgb = means.tolist()
    std_rgb = standard_deviations.tolist()
    return mean_rgb, std_rgb


def normalize_image(image, mean_rgb, std_rgb):
    """Apply channel standardization followed by image-level min-max scaling."""
    means = np.asarray(mean_rgb, dtype=np.float32)
    standard_deviations = np.asarray(std_rgb, dtype=np.float32)
    if means.shape != (3,) or standard_deviations.shape != (3,):
        raise ValueError("RGB means and standard deviations must each have 3 values.")
    if np.any(standard_deviations == 0):
        raise ValueError("RGB standard deviations must be non-zero.")

    normalized = (image - means) / standard_deviations
    value_range = normalized.max() - normalized.min()
    if value_range == 0:
        return np.zeros_like(normalized, dtype=np.uint8)
    normalized = (normalized - normalized.min()) / value_range
    return (normalized * 255).astype(np.uint8)


def safe_divide(numerator, denominator):
    """Return NaN when a metric is mathematically undefined."""
    return numerator / denominator if denominator != 0 else np.nan


def get_weighted_dice(prediction, target, class_count, ignored_classes=()):
    """Calculate foreground Dice weighted by target class pixel counts."""
    weighted_dice = 0.0
    evaluated_pixels = 0

    for class_index in range(class_count):
        if class_index in ignored_classes:
            continue
        target_class = target == class_index
        prediction_class = prediction == class_index
        target_pixels = np.sum(target_class)
        if target_pixels == 0:
            continue

        true_positive = np.sum(target_class & prediction_class)
        false_positive = np.sum(~target_class & prediction_class)
        false_negative = np.sum(target_class & ~prediction_class)
        class_dice = safe_divide(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        )
        evaluated_pixels += target_pixels
        weighted_dice += class_dice * target_pixels

    return safe_divide(weighted_dice, evaluated_pixels)


def evaluate(target, prediction):
    """Calculate binary segmentation metrics for one class.

    Undefined metrics are returned as NaN and omitted from aggregate statistics.
    """
    target_positive = target == 1
    prediction_positive = prediction == 1
    true_positive = np.sum(target_positive & prediction_positive)
    true_negative = np.sum(~target_positive & ~prediction_positive)
    false_positive = np.sum(~target_positive & prediction_positive)
    false_negative = np.sum(target_positive & ~prediction_positive)

    return {
        "dice": safe_divide(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        ),
        "jaccard": safe_divide(
            true_positive, true_positive + false_positive + false_negative
        ),
        "accuracy": safe_divide(
            true_positive + true_negative,
            true_positive + true_negative + false_positive + false_negative,
        ),
        "sensitivity": safe_divide(true_positive, true_positive + false_negative),
        "specificity": safe_divide(true_negative, true_negative + false_positive),
        "precision": safe_divide(true_positive, true_positive + false_positive),
        "NPV": safe_divide(true_negative, true_negative + false_negative),
    }
