"""
train.py
--------
Training loop, evaluation, and class-weight computation for ECGCNNBiLSTM.

Usage (from N5_Training.ipynb or standalone):
    from train import Trainer
    trainer = Trainer(device='cuda')
    trainer.run()
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from config import (
    NUM_CLASSES, CLASS_NAMES, BATCH_SIZE, LEARNING_RATE,
    MAX_EPOCHS, EARLY_STOP_PATIENCE, LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR, BEST_MODEL_PATH, OUTPUT_DATA_DIR,
    MONITOR_METRIC,
)
from model import ECGCNNBiLSTM, build_model
from dataset_builder import ECGWindowDataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: ECGWindowDataset) -> torch.Tensor:
    """Compute inverse-frequency class weights for CrossEntropyLoss.

    Uses balanced strategy: weight_k = n_samples / (n_active_classes * count_k).
    Handles missing classes (count_k == 0) gracefully without crashing by assigning
    a neutral weight of 1.0.

    Parameters
    ----------
    dataset : ECGWindowDataset
        Training split dataset.

    Returns
    -------
    torch.Tensor, shape (NUM_CLASSES,), dtype float32
        Class weights in class-index order (0–4).
    """
    labels = dataset.y.numpy()
    n_samples = len(labels)
    if n_samples == 0:
        return torch.ones(NUM_CLASSES, dtype=torch.float32)

    counts = np.bincount(labels, minlength=NUM_CLASSES)
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    active_classes = int(np.sum(counts > 0))

    if active_classes > 0:
        for c in range(NUM_CLASSES):
            if counts[c] > 0:
                weights[c] = n_samples / (active_classes * counts[c])
            else:
                weights[c] = 1.0  # Neutral weight for classes not present in split

    logger.info(f"Class weights: { {CLASS_NAMES[i]: f'{w:.3f}' for i, w in enumerate(weights)} }")
    return torch.tensor(weights, dtype=torch.float32)



@torch.no_grad()
def evaluate(
    model: ECGCNNBiLSTM,
    loader: DataLoader,
    criterion: nn.CrossEntropyLoss,
    device: str,
) -> dict:
    """Evaluate model on a dataloader and return metrics.

    Parameters
    ----------
    model : ECGCNNBiLSTM
    loader : DataLoader
    criterion : nn.CrossEntropyLoss
    device : str

    Returns
    -------
    dict with keys: loss (float), accuracy (float), macro_f1 (float),
                    confusion_matrix (np.ndarray, shape NUM_CLASSES x NUM_CLASSES)
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y_batch.cpu().numpy().tolist())

    avg_loss = total_loss / len(all_labels)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm       = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

    return {
        "loss":             avg_loss,
        "accuracy":         float(accuracy),
        "macro_f1":         float(macro_f1),
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Safe Checkpoint Helper
# ---------------------------------------------------------------------------

def save_checkpoint_if_not_exists(
    state: dict,
    filepath: Path,
    force: bool = False,
) -> bool:
    """Save a PyTorch model checkpoint only if file does not already exist.

    Parameters
    ----------
    state : dict
        Model/optimizer state dictionary.
    filepath : Path or str
        Destination path (.pt or .pth).
    force : bool
        If True, overwrite even if the file exists.

    Returns
    -------
    bool
        True if file was saved, False if skipped because it already exists.
    """
    path = Path(filepath)
    if not force and path.exists():
        logger.info(f"Checkpoint file already exists at {path}. Skipping save to conserve resources.")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info(f"Saved checkpoint to {path}")
    return True


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Orchestrates the full training pipeline for ECGCNNBiLSTM.

    Parameters
    ----------
    device : str, optional
        PyTorch device string. Auto-detects CUDA if available.
    batch_size : int, optional
        DataLoader batch size. Defaults to config.BATCH_SIZE.
    learning_rate : float, optional
        Initial Adam learning rate. Defaults to config.LEARNING_RATE.
    max_epochs : int, optional
        Maximum training epochs. Defaults to config.MAX_EPOCHS.
    patience : int, optional
        Early stopping patience (epochs without val improvement).
        Defaults to config.EARLY_STOP_PATIENCE.
    data_dir : Path, optional
        Directory containing .npz split files. Defaults to config.OUTPUT_DATA_DIR.
    save_path : Path, optional
        Where to save the best model checkpoint. Defaults to config.BEST_MODEL_PATH.
    skip_if_exists : bool, optional
        If True and save_path already exists, skips training to conserve resources.
        Defaults to True.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size:    int   = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        max_epochs:    int   = MAX_EPOCHS,
        patience:      int   = EARLY_STOP_PATIENCE,
        monitor:       str   = MONITOR_METRIC,
        data_dir:      Path  = OUTPUT_DATA_DIR,
        save_path:     Path  = BEST_MODEL_PATH,
        skip_if_exists: bool = True,
    ):
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.max_epochs    = max_epochs
        self.patience      = patience
        self.monitor       = monitor
        self.data_dir      = Path(data_dir)
        self.save_path     = Path(save_path)
        self.skip_if_exists = skip_if_exists

        # Configure monitoring mode and target metric
        if self.monitor in ("val_loss", "loss"):
            self.metric_key = "loss"
            self.mode = "min"
        elif self.monitor in ("val_acc", "val_accuracy", "accuracy"):
            self.metric_key = "accuracy"
            self.mode = "max"
        else:  # "val_f1", "val_macro_f1", or default
            self.metric_key = "macro_f1"
            self.mode = "max"

        logger.info(
            f"Trainer initialised | device={self.device} | max_epochs={self.max_epochs} | "
            f"patience={self.patience} | monitor={self.monitor} ({self.mode}) | skip_if_exists={self.skip_if_exists}"
        )

    def _build_loaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Load all three splits and return DataLoaders."""
        train_ds = ECGWindowDataset("train", self.data_dir)
        val_ds   = ECGWindowDataset("val",   self.data_dir)
        test_ds  = ECGWindowDataset("test",  self.data_dir)

        # Log class distribution
        for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            counts = ds.class_counts()
            readable = {CLASS_NAMES[k]: v for k, v in counts.items()}
            logger.info(f"{name} class distribution: {readable}")

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=(self.device != "cpu"))
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False,
                                  num_workers=0, pin_memory=(self.device != "cpu"))
        test_loader  = DataLoader(test_ds,  batch_size=self.batch_size, shuffle=False,
                                  num_workers=0)

        return train_loader, val_loader, test_loader, train_ds

    def run(self, force_retrain: bool = False) -> dict:
        """Run full training + evaluation pipeline.

        Parameters
        ----------
        force_retrain : bool, optional
            If True, retrains even if checkpoint file already exists. Default False.

        Returns
        -------
        dict
            Final test metrics: loss, accuracy, macro_f1, confusion_matrix.
        """
        # Guard: Check if model checkpoint already exists to conserve compute resources
        if self.skip_if_exists and not force_retrain and self.save_path.exists():
            logger.info(
                f"Checkpoint already exists at {self.save_path}. "
                "Skipping training to conserve compute resources. Evaluating existing checkpoint on test set..."
            )
            _, _, test_loader, train_ds = self._build_loaders()
            model = build_model(self.device)
            checkpoint = torch.load(self.save_path, map_location=self.device)
            model.load_state_dict(checkpoint["model_state"])
            if "val_metrics" in checkpoint:
                vm = checkpoint["val_metrics"]
                tm = checkpoint.get("train_metrics", {})
                logger.info(
                    f"Saved Checkpoint Performance (Epoch {checkpoint.get('epoch', '?')}):\n"
                    f"  Train: loss={tm.get('loss', 0.0):.4f} | acc={tm.get('accuracy', 0.0):.4f} | f1={tm.get('macro_f1', 0.0):.4f}\n"
                    f"  Val:   loss={vm.get('loss', 0.0):.4f} | acc={vm.get('accuracy', 0.0):.4f} | f1={vm.get('macro_f1', 0.0):.4f}"
                )
            test_metrics = evaluate(model, test_loader, criterion, self.device)
            logger.info(
                f"\nLoaded Checkpoint Test Results:\n"
                f"  Loss     : {test_metrics['loss']:.4f}\n"
                f"  Accuracy : {test_metrics['accuracy']:.4f}\n"
                f"  Macro-F1 : {test_metrics['macro_f1']:.4f}\n"
                f"Confusion Matrix:\n{test_metrics['confusion_matrix']}"
            )
            self._history = checkpoint.get("history", {})
            return test_metrics

        train_loader, val_loader, test_loader, train_ds = self._build_loaders()

        # Model
        model = build_model(self.device)
        logger.info(f"Model parameters: {model.count_parameters():,}")

        # Loss with class weights
        class_weights = compute_class_weights(train_ds).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimiser & scheduler
        optimizer = Adam(model.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE,
        )

        # Training loop
        best_metric = float("inf") if self.mode == "min" else -float("inf")
        epochs_no_improve = 0
        history = {
            "train_loss": [], "train_accuracy": [], "train_macro_f1": [],
            "val_loss": [], "val_accuracy": [], "val_macro_f1": [],
        }

        # Ensure destination directory exists
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.max_epochs + 1):
            # -- Train --
            model.train()
            train_loss = 0.0
            train_preds: list[int] = []
            train_labels: list[int] = []

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)
                loss.backward()
                # Gradient clipping prevents LSTM exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(y_batch)

                preds = logits.detach().argmax(dim=1)
                train_preds.extend(preds.cpu().numpy().tolist())
                train_labels.extend(y_batch.cpu().numpy().tolist())

            train_loss /= len(train_loader.dataset)
            train_acc  = float(np.mean(np.array(train_preds) == np.array(train_labels)))
            train_f1   = float(f1_score(train_labels, train_preds, average="macro", zero_division=0))

            # -- Validate --
            val_metrics = evaluate(model, val_loader, criterion, self.device)
            scheduler.step(val_metrics["loss"])

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["train_macro_f1"].append(train_f1)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])

            logger.info(
                f"Epoch {epoch:03d}/{self.max_epochs} | "
                f"Train: loss={train_loss:.4f} | acc={train_acc:.4f} | f1={train_f1:.4f} || "
                f"Val: loss={val_metrics['loss']:.4f} | acc={val_metrics['accuracy']:.4f} | f1={val_metrics['macro_f1']:.4f}"
            )

            # -- Early stopping & checkpoint --
            current_metric = val_metrics[self.metric_key]
            improved = (current_metric < best_metric) if self.mode == "min" else (current_metric > best_metric)

            if improved:
                best_metric = current_metric
                epochs_no_improve = 0
                train_metrics = {"loss": train_loss, "accuracy": train_acc, "macro_f1": train_f1}
                torch.save(
                    {
                        "epoch":         epoch,
                        "model_state":   model.state_dict(),
                        "optimizer":     optimizer.state_dict(),
                        "train_metrics": train_metrics,
                        "val_metrics":   val_metrics,
                        "history":       history,
                    },
                    self.save_path,
                )
                logger.info(
                    f"  ✓ Best model saved to {self.save_path} "
                    f"({self.monitor}={best_metric:.4f} | train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | val_f1={val_metrics['macro_f1']:.4f})"
                )
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch} (no improvement in {self.monitor} for {self.patience} epochs)")
                    break

        # -- Final test evaluation --
        checkpoint = torch.load(self.save_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = evaluate(model, test_loader, criterion, self.device)

        logger.info(
            f"\nTest Results:\n"
            f"  Loss     : {test_metrics['loss']:.4f}\n"
            f"  Accuracy : {test_metrics['accuracy']:.4f}\n"
            f"  Macro-F1 : {test_metrics['macro_f1']:.4f}\n"
            f"Confusion Matrix:\n{test_metrics['confusion_matrix']}"
        )

        self._history = history  # store for notebook plotting
        return test_metrics
