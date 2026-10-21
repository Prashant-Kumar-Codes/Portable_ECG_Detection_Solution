# Coding Rules

## General
- All paths via pathlib.Path, never os.path string concatenation
- All module functions must have numpy/google-style docstrings
- No bare except: - catch specific exceptions (ValueError, FileNotFoundError, etc.)
- All constants defined in Modules/config.py only; never hardcode in notebooks

## Data and ML
- preprocess_ecg_window() signature must never change between training and inference
- Always check for all-NaN windows before preprocessing (skip them, log warning)
- Batch dimension is always first: (B, C, L) for CNN, (B, L, F) for LSTM
- Model input dtype: torch.float32 | Labels dtype: torch.long
- Class weights go into CrossEntropyLoss, not into sampler

## Notebooks
- First cell: imports + sys.path.insert(0, str(PROJECT_ROOT / 'Modules'))
- Second cell: constants/paths (import from config)
- Cells must be idempotent (safely re-runnable without side effects)
- Clear all outputs before committing

## File Roles
- Modules/*.py = production-ready, importable, fully documented code
- Notebooks/N*.ipynb = exploration, validation, and visualisation only (not importable)
- Dataset/Output_Data/ = generated artifacts only, never raw source data

## Data Saving & Resource Persistence
- Whenever saving arrays (.npz/.npy), DataFrames (.csv/.parquet), or checkpoints (.pt/.pth), always check `Path.exists()` first.
- If the target file already exists, skip re-saving to conserve compute, bandwidth, and disk I/O resources.
- Provide an explicit `force=False` or `force_rebuild=False` flag for intentional overrides.
- Use `save_array_if_not_exists` and `save_dataframe_if_not_exists` from `Modules/dataset_builder.py` or `save_checkpoint_if_not_exists` from `Modules/train.py`.

