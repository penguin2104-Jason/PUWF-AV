"""Train UNet, ResUNet, and AttentionUNet with five-fold cross-validation."""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AVDataset
from models import MODEL_NAMES, NUM_CLASSES, build_model
from utils import (
    epoch_time,
    get_weighted_dice,
    seed_data_loader_worker,
    seed_everything,
)


class Trainer:
    """Train one architecture on one prepared cross-validation fold."""
    def __init__(
        self,
        dataset_path,
        loss_record_path,
        checkpoint_path,
        image_size,
        model,
        lr,
        batch_size,
        num_epochs,
        patience,
        num_workers,
        use_amp,
        device,
        seed,
    ):
        self.dataset_path = dataset_path
        self.loss_record_path = loss_record_path
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.model = model.to(device)
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.early_stopping_patience = patience
        self.device = device
        self.seed = seed
        self.use_amp = use_amp and device.type == "cuda"
        self.pin_memory = device.type == "cuda"

        time_stamp = time.strftime("%Y%m%d_%H%M%S")
        self.loss_record_file = os.path.join(
            self.loss_record_path, f"loss_record_{time_stamp}.csv"
        )
        os.makedirs(self.loss_record_path, exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        train_name = os.listdir(os.path.join(self.dataset_path, "train", "images"))
        val_name = os.listdir(os.path.join(self.dataset_path, "val", "images"))
        images_path_train = [
            os.path.join(self.dataset_path, "train", "images", name)
            for name in train_name
        ]
        av_masks_path_train = [
            os.path.join(self.dataset_path, "train", "av_masks", name)
            for name in train_name
        ]
        images_path_val = [
            os.path.join(self.dataset_path, "val", "images", name)
            for name in val_name
        ]
        av_masks_path_val = [
            os.path.join(self.dataset_path, "val", "av_masks", name)
            for name in val_name
        ]
        fov_paths_val = [
            os.path.join(self.dataset_path, "val", "fov_masks", name)
            for name in val_name
        ]

        train_dataset = AVDataset(
            images_path_train, av_masks_path_train, train=True, seed=self.seed
        )
        val_dataset = AVDataset(
            images_path_val,
            av_masks_path_val,
            fov_paths=fov_paths_val,
            train=False,
            seed=self.seed,
        )
        loader_generator = torch.Generator()
        loader_generator.manual_seed(self.seed)
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_data_loader_worker,
            generator=loader_generator,
        )
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=num_workers > 0,
            worker_init_fn=seed_data_loader_worker,
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            betas=(0.5, 0.999),
            eps=1e-8,
            weight_decay=1e-4,
            amsgrad=False,
        )

        total_steps = len(self.train_loader) * self.num_epochs
        if total_steps < 2:
            raise ValueError("Training requires at least two optimizer steps.")
        warmup_steps = max(1, round(total_steps * 0.10))
        warmup_steps = min(warmup_steps, total_steps - 1)
        cosine_steps = total_steps - warmup_steps
        minimum_factor = 1e-8 / self.lr

        def _learning_rate_factor(step):
            if step < warmup_steps:
                progress = step / warmup_steps
                return 1e-2 + (1.0 - 1e-2) * progress
            progress = min((step - warmup_steps) / cosine_steps, 1.0)
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return minimum_factor + (1.0 - minimum_factor) * cosine_factor

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=_learning_rate_factor)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.loss_fn = DiceCELoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            lambda_dice=1.0,
            lambda_ce=1.0,
        )

    def train_epoch(self, epoch):
        """Run one optimization epoch and return the mean training loss."""
        epoch_loss = 0.0
        self.model.train()

        with tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {epoch + 1}/{self.num_epochs}",
            unit="batch",
        ) as progress:
            for images, labels in self.train_loader:
                images = images.to(
                    self.device, dtype=torch.float32, non_blocking=self.pin_memory
                )
                labels = labels.to(self.device, non_blocking=self.pin_memory)

                self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    predictions = self.model(images)
                    loss = self.loss_fn(predictions, labels)

                scale_before_update = self.scaler.get_scale()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scaler.get_scale() >= scale_before_update:
                    self.scheduler.step()

                epoch_loss += loss.item()
                progress.set_postfix({"Loss": epoch_loss / (progress.n + 1)})
                progress.update(1)

        return epoch_loss / len(self.train_loader)

    def validate(self, epoch):
        """Evaluate full validation images with sliding-window inference."""
        epoch_dice = 0.0
        valid_samples = 0
        self.model.eval()

        with torch.no_grad():
            with tqdm(
                total=len(self.val_loader),
                desc=f"Validation {epoch + 1}/{self.num_epochs}",
                unit="image",
            ) as progress:
                for images, labels, fov_masks in self.val_loader:
                    images = images.to(
                        self.device, dtype=torch.float32, non_blocking=self.pin_memory
                    )
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        prediction = sliding_window_inference(
                            inputs=images,
                            roi_size=self.image_size,
                            sw_batch_size=4,
                            predictor=self.model,
                            overlap=0.5,
                        )
                    prediction = F.softmax(prediction, dim=1).cpu().numpy()[0]
                    prediction = np.argmax(prediction, axis=0).astype(np.uint8)
                    labels = labels[0, 0].cpu().numpy()
                    valid_pixels = fov_masks[0, 0].cpu().numpy().astype(bool)

                    weighted_dice = get_weighted_dice(
                        prediction[valid_pixels],
                        labels[valid_pixels],
                        NUM_CLASSES,
                        (0,),
                    )
                    if not np.isnan(weighted_dice):
                        epoch_dice += weighted_dice
                        valid_samples += 1
                    average = epoch_dice / valid_samples if valid_samples else np.nan
                    progress.set_postfix({"Dice": average})
                    progress.update(1)

        if valid_samples == 0:
            raise ValueError("Validation set contains no evaluable foreground pixels.")
        return epoch_dice / valid_samples

    def train_model(self):
        """Train until all epochs finish or validation early stopping triggers."""
        with open(self.loss_record_file, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Train Loss", "Val Dice", "Elapsed Time"])

        patience = self.early_stopping_patience
        best_metric = -np.inf

        for epoch in range(self.num_epochs):
            start_time = time.time()
            train_loss = self.train_epoch(epoch)
            val_dice = self.validate(epoch)
            epoch_mins, epoch_secs = epoch_time(start_time, time.time())

            with open(
                self.loss_record_file, "a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(
                    [epoch + 1, train_loss, val_dice, f"{epoch_mins}m {epoch_secs}s"]
                )

            checkpoint_path = os.path.join(
                self.checkpoint_path, f"checkpoint_epoch{epoch + 1}.pth"
            )
            torch.save(self.model.state_dict(), checkpoint_path)

            print(f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
            print(f"        Train Loss: {train_loss:.5f}")
            print(f"        Val Dice: {val_dice:.5f}")

            if val_dice > best_metric:
                best_metric = val_dice
                patience = self.early_stopping_patience
            else:
                patience -= 1
            if patience <= 0:
                print("Early stop!")
                break


def parse_args():
    """Parse training paths, architectures, and hyperparameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="kfold_set")
    parser.add_argument("--loss-record-path", default="loss_record")
    parser.add_argument("--checkpoint-path", default="checkpoint")
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--image-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed-precision training and validation.",
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Training random seed (default: 123)."
    )
    return parser.parse_args()


def main():
    """Train the requested architectures over all requested folds."""
    args = parse_args()
    if args.num_epochs < 1:
        raise ValueError("--num-epochs must be at least 1.")
    if args.patience < 1:
        raise ValueError("--patience must be at least 1.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    for model_name in args.models:
        for fold_index in range(1, args.folds + 1):
            seed_everything(args.seed)
            print(f"\nTraining {model_name}, fold {fold_index}/{args.folds}")
            dataset_path = os.path.join(args.dataset_path, f"fold_{fold_index}")
            loss_path = os.path.join(
                args.loss_record_path, model_name, f"fold_{fold_index}"
            )
            checkpoint_path = os.path.join(
                args.checkpoint_path, model_name, f"fold_{fold_index}"
            )
            trainer = Trainer(
                dataset_path=dataset_path,
                loss_record_path=loss_path,
                checkpoint_path=checkpoint_path,
                image_size=tuple(args.image_size),
                model=build_model(model_name),
                lr=args.lr,
                batch_size=args.batch_size,
                num_epochs=args.num_epochs,
                patience=args.patience,
                num_workers=args.num_workers,
                use_amp=not args.no_amp,
                device=device,
                seed=args.seed,
            )
            trainer.train_model()


if __name__ == "__main__":
    main()
