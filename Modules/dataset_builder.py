"""
dataset_builder.py
------------------
Builds the windowed ECG dataset from VitalDB annotations + on-demand waveform downloads.

Usage (from N3_Dataset_Builder.ipynb or standalone):
    from dataset_builder import VitalDBDatasetBuilder, ECGWindowDataset
    builder = VitalDBDatasetBuilder()
    builder.build_and_cache()          # downloads waveforms, writes .npz files
    ds = ECGWindowDataset(split='train')

Class label mapping (defined in config.RHYTHM_TO_CLASS):
    0 = Normal Sinus Rhythm
    1 = Atrial Fibrillation
    2 = Supraventricular Arrhythmia
    3 = Ventricular Arrhythmia
    4 = Noise / Artifact
"""

import time
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    import vitaldb
except ImportError:
    vitaldb = None  # graceful - some environments may not have it installed

from config import (
    ANNOTATION_DIR, METADATA_CSV, OUTPUT_DATA_DIR,
    SOURCE_FS, TARGET_FS, WINDOW_SAMPLES,
    STRIDE_SAMPLES, RHYTHM_TO_CLASS, MIN_VALID_FRAC,
    TRAIN_FRAC, VAL_FRAC, SPLIT_SEED,
)
from preprocessing import preprocess_ecg_window, is_valid_window

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------

def assign_label(rhythm_label: str, bad_signal_quality: bool) -> Optional[int]:
    """Map a single beat annotation to an integer class label.

    Priority:
      - bad_signal_quality==True  → class 4 (Noise/Artifact), regardless of rhythm_label
      - Otherwise, look up rhythm_label in RHYTHM_TO_CLASS
      - Unknown rhythm_label → returns None (window will be skipped)

    Parameters
    ----------
    rhythm_label : str
        The rhythm_label string from an annotation CSV row.
    bad_signal_quality : bool
        The bad_signal_quality flag from the same row.

    Returns
    -------
    int or None
        Integer class (0–4) or None if unmappable.
    """
    if bad_signal_quality:
        return 4
    clean_label = str(rhythm_label).strip()
    return RHYTHM_TO_CLASS.get(clean_label, None)


def get_window_label(
    window_start_sec: float,
    window_end_sec: float,
    annotations: pd.DataFrame,
) -> Optional[int]:
    """Derive a single class label for a 10-second ECG window.

    Strategy: majority-vote over all annotated beats whose R-peaks fall within
    the window. If no beats fall in the window, return None (skip window).
    Noise/Artifact votes that exceed 30% of total beats trigger class 4 override.

    Parameters
    ----------
    window_start_sec : float
        Start time of the window in seconds.
    window_end_sec : float
        End time of the window in seconds.
    annotations : pd.DataFrame
        Annotation dataframe for the current case with columns:
        [time_second, beat_type, rhythm_label, bad_signal_quality].

    Returns
    -------
    int or None
        Majority class label, or None if no beats in window.
    """
    mask = (
        (annotations["time_second"] >= window_start_sec) &
        (annotations["time_second"] <  window_end_sec)
    )
    beats = annotations[mask]

    if beats.empty:
        return None

    labels = []
    for _, row in beats.iterrows():
        lbl = assign_label(
            str(row["rhythm_label"]),
            bool(row["bad_signal_quality"]),
        )
        if lbl is not None:
            labels.append(lbl)

    if not labels:
        return None

    # Noise override: if >30% of beats are noise, label whole window as noise
    noise_frac = labels.count(4) / len(labels)
    if noise_frac > 0.30:
        return 4

    # Majority vote
    return max(set(labels), key=labels.count)


# ---------------------------------------------------------------------------
# Safe Persistence Helpers (Skip if exists to conserve resources)
# ---------------------------------------------------------------------------

def save_array_if_not_exists(
    filepath: Path,
    force: bool = False,
    **arrays,
) -> bool:
    """Save numpy arrays to compressed .npz only if file does not already exist.

    Parameters
    ----------
    filepath : Path or str
        Destination path (.npz).
    force : bool
        If True, overwrite even if the file exists.
    **arrays : np.ndarray
        Named arrays to save (e.g. X=X, y=y).

    Returns
    -------
    bool
        True if file was saved, False if skipped because it already exists.
    """
    path = Path(filepath)
    if not force and path.exists():
        logger.info(f"Array file already exists at {path}. Skipping save to conserve resources.")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    logger.info(f"Saved array data to {path}")
    return True


def save_dataframe_if_not_exists(
    df: pd.DataFrame,
    filepath: Path,
    force: bool = False,
    **kwargs,
) -> bool:
    """Save a pandas DataFrame to CSV or Parquet only if file does not already exist.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filepath : Path or str
        Destination path (.csv or .parquet).
    force : bool
        If True, overwrite even if the file exists.
    **kwargs
        Additional arguments passed to to_csv or to_parquet.

    Returns
    -------
    bool
        True if file was saved, False if skipped because it already exists.
    """
    path = Path(filepath)
    if not force and path.exists():
        logger.info(f"DataFrame file already exists at {path}. Skipping save to conserve resources.")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in (".parquet", ".pq"):
        df.to_parquet(path, **kwargs)
    else:
        df.to_csv(path, **kwargs)
    logger.info(f"Saved DataFrame to {path}")
    return True


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

class VitalDBDatasetBuilder:
    """Downloads VitalDB ECG waveforms and builds windowed .npz cache files.

    Parameters
    ----------
    annotation_dir : Path, optional
        Path to the Annotation_Files directory. Defaults to config.ANNOTATION_DIR.
    metadata_csv : Path, optional
        Path to metadata.csv. Defaults to config.METADATA_CSV.
    output_dir : Path, optional
        Directory to write .npz files. Defaults to config.OUTPUT_DATA_DIR.
    api_delay_sec : float, optional
        Seconds to sleep between VitalDB API calls. Default 0.5.
    """

    def __init__(
        self,
        annotation_dir: Path = ANNOTATION_DIR,
        metadata_csv: Path   = METADATA_CSV,
        output_dir: Path     = OUTPUT_DATA_DIR,
        api_delay_sec: float = 0.5,
    ):
        if vitaldb is None:
            raise ImportError("vitaldb package not installed. Run: pip install vitaldb")

        self.annotation_dir  = Path(annotation_dir)
        self.metadata_csv    = Path(metadata_csv)
        self.output_dir      = Path(output_dir)
        self.api_delay_sec   = api_delay_sec

        self.metadata = pd.read_csv(self.metadata_csv)
        logger.info(f"Loaded metadata: {len(self.metadata)} cases")

    # ------------------------------------------------------------------
    # Patient-wise split
    # ------------------------------------------------------------------

    def _split_case_ids(self) -> dict[str, list[int]]:
        """Split case_ids into train/val/test (patient-wise, stratified by dominant class).

        Returns
        -------
        dict with keys 'train', 'val', 'test', each mapping to a list of case_ids.
        """
        all_ids = self.metadata["case_id"].tolist()
        rng = np.random.default_rng(SPLIT_SEED)
        shuffled = rng.permutation(all_ids)

        n = len(shuffled)
        n_train = int(n * TRAIN_FRAC)
        n_val   = int(n * VAL_FRAC)

        return {
            "train": shuffled[:n_train].tolist(),
            "val":   shuffled[n_train:n_train + n_val].tolist(),
            "test":  shuffled[n_train + n_val:].tolist(),
        }

    # ------------------------------------------------------------------
    # Single-case processing
    # ------------------------------------------------------------------

    def _process_case(
        self,
        case_id: int,
        analysis_start: float,
        analysis_end: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Download ECG for one case and extract labelled windows.

        Parameters
        ----------
        case_id : int
        analysis_start : float
            Start time (seconds) of annotated analysis window.
        analysis_end : float
            End time (seconds) of annotated analysis window.

        Returns
        -------
        windows : np.ndarray, shape (N, WINDOW_SAMPLES)
        labels  : np.ndarray, shape (N,)  dtype int64
        """
        # Load annotation
        ann_path = self.annotation_dir / f"Annotation_file_{case_id}.csv"
        if not ann_path.exists():
            logger.warning(f"Case {case_id}: annotation file not found, skipping.")
            return np.empty((0, WINDOW_SAMPLES), dtype=np.float32), np.empty(0, dtype=np.int64)

        annotations = pd.read_csv(ann_path)
        annotations["bad_signal_quality"] = (
            annotations["bad_signal_quality"].astype(str).str.strip().str.lower() == "true"
        )

        # Download ECG waveform from VitalDB (500 Hz)
        try:
            raw_vals = vitaldb.load_case(case_id, ["SNUADC/ECG_II"], 1 / SOURCE_FS)
        except Exception as exc:
            logger.warning(f"Case {case_id}: VitalDB download failed ({exc}), skipping.")
            return np.empty((0, WINDOW_SAMPLES), dtype=np.float32), np.empty(0, dtype=np.int64)

        if raw_vals is None or raw_vals.size == 0:
            logger.warning(f"Case {case_id}: empty waveform, skipping.")
            return np.empty((0, WINDOW_SAMPLES), dtype=np.float32), np.empty(0, dtype=np.int64)

        # raw_vals shape: (total_samples, 1) — extract column 0
        ecg_full = raw_vals[:, 0].astype(np.float64)

        # Slice to analysis window (convert seconds to 500 Hz sample indices)
        start_sample = int(analysis_start * SOURCE_FS)
        end_sample   = int(analysis_end   * SOURCE_FS)
        ecg_analysis = ecg_full[start_sample:end_sample]

        # Sliding-window extraction
        windows_list: list[np.ndarray] = []
        labels_list:  list[int]        = []

        # Step through in TARGET_FS stride steps (convert stride from 250Hz domain to 500Hz)
        stride_source = int(STRIDE_SAMPLES * (SOURCE_FS / TARGET_FS))  # 1000 samples @ 500Hz
        win_len_source = int(WINDOW_SAMPLES * (SOURCE_FS / TARGET_FS))  # 5000 samples @ 500Hz

        total_len = len(ecg_analysis)
        pos = 0

        while pos + win_len_source <= total_len:
            raw_win = ecg_analysis[pos: pos + win_len_source]

            # Skip windows with too many NaN values (before preprocessing)
            if not is_valid_window(raw_win, MIN_VALID_FRAC):
                pos += stride_source
                continue

            # Determine label from annotations
            win_start_sec = analysis_start + pos / SOURCE_FS
            win_end_sec   = win_start_sec + WINDOW_SAMPLES / TARGET_FS  # 10 sec
            label = get_window_label(win_start_sec, win_end_sec, annotations)

            if label is None:
                pos += stride_source
                continue

            # Preprocess (resample 500->250, filter, z-score) → (2500,) float32
            try:
                processed = preprocess_ecg_window(raw_win, source_fs=SOURCE_FS, target_fs=TARGET_FS)
            except ValueError as exc:
                logger.debug(f"Case {case_id} pos={pos}: preprocess failed ({exc}), skipping.")
                pos += stride_source
                continue

            windows_list.append(processed)
            labels_list.append(label)
            pos += stride_source

        if not windows_list:
            return np.empty((0, WINDOW_SAMPLES), dtype=np.float32), np.empty(0, dtype=np.int64)

        return (
            np.stack(windows_list, axis=0),
            np.array(labels_list, dtype=np.int64),
        )

    # ------------------------------------------------------------------
    # Build all splits and cache to .npz
    # ------------------------------------------------------------------

    def build_and_cache(self, force_rebuild: bool = False) -> None:
        """Build train/val/test .npz files from all 482 cases.

        Skips building if .npz files already exist (use force_rebuild=True to override).

        Parameters
        ----------
        force_rebuild : bool
            If True, rebuild even if .npz files already exist.
        """
        split_ids = self._split_case_ids()
        split_files = {
            "train": self.output_dir / "windows_train.npz",
            "val":   self.output_dir / "windows_val.npz",
            "test":  self.output_dir / "windows_test.npz",
        }

        # Check if all splits already exist
        if not force_rebuild and all(p.exists() for p in split_files.values()):
            logger.info("All .npz cache files already exist. Pass force_rebuild=True to rebuild.")
            return

        split_windows: dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}
        split_labels:  dict[str, list[np.ndarray]] = {"train": [], "val": [], "test": []}

        for split_name, case_ids in split_ids.items():
            target_file = split_files[split_name]

            # Skip split entirely if its .npz already exists to avoid heavy downloads & processing
            if not force_rebuild and target_file.exists():
                logger.info(
                    f"Split '{split_name}' cache already exists at {target_file}. "
                    "Skipping downloading and processing to conserve resources."
                )
                continue

            logger.info(f"Processing split='{split_name}' ({len(case_ids)} cases)...")

            for i, case_id in enumerate(case_ids):
                row = self.metadata[self.metadata["case_id"] == case_id]
                if row.empty:
                    continue

                analysis_start = float(row.iloc[0]["analysis_start_time_sec"])
                analysis_end   = float(row.iloc[0]["analysis_end_time_sec"])

                wins, lbls = self._process_case(case_id, analysis_start, analysis_end)
                if len(wins) > 0:
                    split_windows[split_name].append(wins)
                    split_labels[split_name].append(lbls)

                if (i + 1) % 10 == 0:
                    logger.info(f"  [{split_name}] {i+1}/{len(case_ids)} cases done")

                time.sleep(self.api_delay_sec)

            # Concatenate and save if file doesn't already exist
            if not force_rebuild and target_file.exists():
                logger.info(f"File {target_file} already exists, skipping save.")
                continue

            if split_windows[split_name]:
                X = np.concatenate(split_windows[split_name], axis=0)
                y = np.concatenate(split_labels[split_name],  axis=0)
            else:
                X = np.empty((0, WINDOW_SAMPLES), dtype=np.float32)
                y = np.empty(0, dtype=np.int64)

            saved = save_array_if_not_exists(target_file, force=force_rebuild, X=X, y=y)
            if saved:
                # Log class distribution
                class_counts = {k: int(np.sum(y == k)) for k in range(5)}
                logger.info(
                    f"Saved {target_file.name}: "
                    f"{len(X)} windows | class dist: {class_counts}"
                )

        logger.info("Dataset build complete.")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ECGWindowDataset(Dataset):
    """PyTorch Dataset wrapping a pre-built .npz split file.

    Parameters
    ----------
    split : str
        One of 'train', 'val', 'test'.
    data_dir : Path, optional
        Directory containing windows_*.npz files. Defaults to config.OUTPUT_DATA_DIR.

    Attributes
    ----------
    X : torch.Tensor, shape (N, 1, 2500), dtype float32
        Preprocessed ECG windows with channel dimension added.
    y : torch.Tensor, shape (N,), dtype long
        Integer class labels.
    """

    def __init__(self, split: str, data_dir: Path = OUTPUT_DATA_DIR):
        assert split in ("train", "val", "test"), f"split must be train/val/test, got '{split}'"
        npz_path = Path(data_dir) / f"windows_{split}.npz"

        if not npz_path.exists():
            raise FileNotFoundError(
                f"{npz_path} not found. Run VitalDBDatasetBuilder.build_and_cache() first."
            )

        data = np.load(npz_path)
        self.X = torch.from_numpy(data["X"]).unsqueeze(1)  # (N, 1, 2500)
        self.y = torch.from_numpy(data["y"]).long()

        logger.info(f"ECGWindowDataset '{split}': {len(self.X)} samples loaded")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    def class_counts(self) -> dict[int, int]:
        """Return number of windows per class."""
        counts = {}
        for k in range(5):
            counts[k] = int((self.y == k).sum().item())
        return counts
