"""Evaluate trained models over all cross-validation folds."""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from scipy import stats
from torch.utils.data import DataLoader

from dataset import AVDataset
from models import MODEL_NAMES, NUM_CLASSES, build_model
from utils import evaluate


CLASS_NAMES = [
    "Background",
    "Artery",
    "Vein",
    "Artery-vein crossing",
    "Uncertain",
    "Weighted (w/o background)",
]
METRICS = [
    "dice",
    "jaccard",
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "NPV",
]


class FoldTest:
    """Evaluate one architecture across all prepared cross-validation folds."""
    def __init__(
        self,
        dataset_path,
        loss_record_path,
        checkpoint_path,
        image_size,
        model_name,
        device,
        loss_selections=None,
    ):
        self.dataset_path = dataset_path
        self.loss_record_path = loss_record_path
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.model_name = model_name
        self.device = device
        self.loss_selections = loss_selections or {}

    def _resolve_loss_csv(self, loss_folder, fold_name):
        """Select the only CSV, or require an explicit manifest entry if ambiguous."""
        csv_files = [
            name
            for name in os.listdir(loss_folder)
            if name.lower().endswith(".csv")
            and os.path.isfile(os.path.join(loss_folder, name))
        ]
        if not csv_files:
            raise FileNotFoundError(f"No loss CSV found in {loss_folder}.")
        if len(csv_files) == 1:
            return os.path.join(loss_folder, csv_files[0])

        selected_name = self.loss_selections.get(fold_name)
        if selected_name is None:
            available = ", ".join(csv_files)
            raise RuntimeError(
                f"Multiple loss CSV files found in {loss_folder}: {available}. "
                "Create a loss-selection JSON file, pass it with "
                "--loss-selection-manifest, and specify the CSV for this model/fold."
            )
        if selected_name not in csv_files:
            raise ValueError(
                f"Loss CSV '{selected_name}' selected for {fold_name} was not found "
                f"in {loss_folder}. Available files: {', '.join(csv_files)}"
            )
        return os.path.join(loss_folder, selected_name)

    def foldtest(self):
        """Evaluate each fold and return cross-fold mean confidence intervals."""
        fold_list = [
            name
            for name in os.listdir(self.dataset_path)
            if name.startswith("fold_")
            and name.removeprefix("fold_").isdigit()
            and os.path.isdir(os.path.join(self.dataset_path, name))
        ]
        fold_list.sort(key=lambda name: int(name.removeprefix("fold_")))
        if not fold_list:
            raise FileNotFoundError(f"No fold directories found in {self.dataset_path}.")
        fold_results = []
        for fold_name in fold_list:
            test_set_path = os.path.join(self.dataset_path, fold_name)
            loss_folder = os.path.join(self.loss_record_path, fold_name)
            loss_csv_path = self._resolve_loss_csv(loss_folder, fold_name)
            model_folder_path = os.path.join(self.checkpoint_path, fold_name)
            result = self.test_onefold(
                test_set_path, loss_csv_path, model_folder_path
            )
            result["fold"] = fold_name
            fold_results.append(result)

        results = pd.concat(fold_results, axis=0, ignore_index=True)
        results = results.drop(columns=["pic"])
        id_columns = [
            column for column in results.columns if column not in ["fold", *METRICS]
        ]
        output = (
            results.groupby(id_columns, dropna=False)[METRICS]
            .agg(lambda values: self.mean_ci(values))
            .reset_index()
        )
        output["class"] = pd.Categorical(
            output["class"], categories=CLASS_NAMES, ordered=True
        )
        return output.sort_values("class")

    def test_onefold(self, test_set_path, loss_csv_path, model_folder_path):
        """Load the best-validation checkpoint and evaluate one test fold."""
        loss_frame = pd.read_csv(loss_csv_path)
        selected_epoch = int(
            loss_frame.loc[loss_frame["Val Dice"].idxmax(), "Epoch"]
        )
        checkpoint = os.path.join(
            model_folder_path, f"checkpoint_epoch{selected_epoch}.pth"
        )

        model = build_model(self.model_name)
        state_dict = torch.load(
            checkpoint, map_location=self.device, weights_only=True
        )
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()

        test_name = os.listdir(os.path.join(test_set_path, "test", "images"))
        images_path = [
            os.path.join(test_set_path, "test", "images", name) for name in test_name
        ]
        masks_path = [
            os.path.join(test_set_path, "test", "av_masks", name)
            for name in test_name
        ]
        fov_paths = [
            os.path.join(test_set_path, "test", "fov_masks", name)
            for name in test_name
        ]
        test_loader = DataLoader(
            AVDataset(images_path, masks_path, fov_paths=fov_paths, train=False),
            batch_size=1,
            shuffle=False,
        )

        evaluation_results = []
        with torch.no_grad():
            for file_index, (images, labels, fov_masks) in enumerate(test_loader):
                images = images.to(self.device, dtype=torch.float32)
                prediction = sliding_window_inference(
                    inputs=images,
                    roi_size=self.image_size,
                    sw_batch_size=4,
                    predictor=model,
                    overlap=0.5,
                )
                prediction = F.softmax(prediction, dim=1).cpu().numpy()[0]
                prediction = np.argmax(prediction, axis=0).astype(np.uint8)
                labels = labels[0, 0].cpu().numpy()
                valid_pixels = fov_masks[0, 0].cpu().numpy().astype(bool)
                labels_in_fov = labels[valid_pixels]
                prediction_in_fov = prediction[valid_pixels]

                for class_index in range(NUM_CLASSES):
                    label_class = (labels_in_fov == class_index).astype(np.uint8)
                    prediction_class = (
                        prediction_in_fov == class_index
                    ).astype(np.uint8)
                    result = evaluate(label_class, prediction_class)
                    result["class"] = class_index
                    result["pic"] = file_index
                    result["pixel_count"] = np.sum(label_class)
                    evaluation_results.append(result)

        result_frame = pd.DataFrame(evaluation_results)
        foreground = result_frame[result_frame["class"] != 0]
        weighted_rows = []
        for picture_index, picture_results in foreground.groupby("pic"):
            row = {"pic": picture_index}
            for metric in METRICS:
                valid = picture_results[metric].notna() & (
                    picture_results["pixel_count"] > 0
                )
                weights = picture_results.loc[valid, "pixel_count"]
                row[metric] = (
                    np.average(picture_results.loc[valid, metric], weights=weights)
                    if not weights.empty and weights.sum() > 0
                    else np.nan
                )
            weighted_rows.append(row)
        if weighted_rows:
            weighted = pd.DataFrame(weighted_rows).mean(
                numeric_only=True, axis=0
            ).to_frame().T
        else:
            weighted = pd.DataFrame([{metric: np.nan for metric in METRICS}])

        class_mean = result_frame.groupby("class", as_index=False).mean(
            numeric_only=True
        )
        output = pd.concat([class_mean, weighted], axis=0, ignore_index=True)
        output = output.drop(columns=["pixel_count"], errors="ignore")
        output["class"] = CLASS_NAMES
        return output

    @staticmethod
    def mean_ci(values, alpha=0.05):
        """Format a mean and two-sided Student-t confidence interval."""
        values = pd.to_numeric(values, errors="coerce").dropna()
        sample_count = values.size
        if sample_count == 0:
            return "NA"
        mean = values.mean()
        if sample_count == 1:
            return f"{mean:.3f}(NA, NA)"
        standard_error = values.std(ddof=1) / np.sqrt(sample_count)
        t_value = stats.t.ppf(1 - alpha / 2, df=sample_count - 1)
        lower = mean - t_value * standard_error
        upper = mean + t_value * standard_error
        return f"{mean:.3f}({lower:.3f}, {upper:.3f})"


def parse_args():
    """Parse evaluation paths, architectures, and loss-selection options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="kfold_set")
    parser.add_argument("--loss-record-path", default="loss_record")
    parser.add_argument("--checkpoint-path", default="checkpoint")
    parser.add_argument("--output-path", default="test_results")
    parser.add_argument(
        "--loss-selection-manifest",
        help="Optional JSON mapping model and fold names to selected loss CSV files.",
    )
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    return parser.parse_args()


def main():
    """Evaluate the requested architectures and save summary CSV files."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    os.makedirs(args.output_path, exist_ok=True)
    loss_selections = {}
    if args.loss_selection_manifest:
        with open(args.loss_selection_manifest, "r", encoding="utf-8") as file:
            loss_selections = json.load(file)
        if not isinstance(loss_selections, dict):
            raise ValueError("The loss-selection manifest must contain a JSON object.")

    for model_name in args.models:
        print(f"Evaluating {model_name}")
        model_loss_selections = loss_selections.get(model_name, {})
        if not isinstance(model_loss_selections, dict):
            raise ValueError(
                f"Loss selections for '{model_name}' must be a JSON object."
            )
        evaluator = FoldTest(
            dataset_path=args.dataset_path,
            loss_record_path=os.path.join(args.loss_record_path, model_name),
            checkpoint_path=os.path.join(args.checkpoint_path, model_name),
            image_size=tuple(args.image_size),
            model_name=model_name,
            device=device,
            loss_selections=model_loss_selections,
        )
        result = evaluator.foldtest()
        result.to_csv(
            os.path.join(args.output_path, f"test_result_{model_name}.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()
