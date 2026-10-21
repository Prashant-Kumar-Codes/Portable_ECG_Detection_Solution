# Session Memory and Decisions

## Critical Bug Found (N1 Notebook)
vitaldb.load_case(case_id, ['SNUADC/ECG_II'], 1/500) returns shape (N, 1).
Correct indexing: ecg_data = vals[:, 0]  NOT vals[:, 1] (IndexError with size 1)

## VitalDB API Notes
- vdb.load_case(case_id, ['SNUADC/ECG_II'], 1/500) -> numpy array (N,1) at 500Hz
- Some case_ids may have no ECG_II track -> check for all-NaN arrays before processing
- Add 0.5s sleep between downloads to avoid rate limiting
- Annotations are time_second (R-peak timestamps in seconds), NOT sample indices

## Rhythm Label Abbreviations in metadata.csv rhythm_classes column
N=Normal Sinus Rhythm, AFIB/AFL=Atrial Fibrillation, SVTA=Supraventricular Tachyarrhythmia
VT=Ventricular Tachyarrhythmia, SND=Sinus Node Dysfunction, WAP/MAT=Wandering Atrial Pacemaker

## Annotation rhythm_label exact strings (from Annotation_Files CSVs)
VitalDB annotation CSV files use short abbreviations for beats:
- Class 0 (Normal Sinus Rhythm): 'N', 'Normal Sinus Rhythm'
- Class 1 (Atrial Fibrillation): 'AFIB/AFL', 'Atrial Fibrillation', 'Atrial fibrillation'
- Class 2 (Supraventricular Arrhythmia): 'SVTA', 'Patterned Atrial Ectopy', 'WAP/MAT', 'SND', 'AVB'
- Class 3 (Ventricular Arrhythmia): 'VT', 'Patterned Ventricular Ectopy'
- Class 4 (Noise / Artifact): 'Noise', 'Unclassifiable', or when bad_signal_quality == True

All mapped in config.RHYTHM_TO_CLASS.

## Class Weight Computation Fix
- sklearn.utils.class_weight.compute_class_weight raises ValueError if any class in `classes` has 0 occurrences in `y`.
- train.compute_class_weights() computes inverse-frequency weights using NumPy directly:
  weight_k = n_samples / (n_active_classes * count_k)
  For classes with count_k == 0, weight is set to 1.0 (neutral) to prevent division by zero or crashes.

## Safe Data Persistence Decision
- Always check Path.exists() before saving .npz arrays, DataFrames, or model checkpoints.
- Dataset builder checks target split files before processing/downloading VitalDB cases to conserve network and disk resources.
- Trainer checks best model checkpoint before running training epochs.
- Use force_rebuild=True or force_retrain=True to override when needed.

## Preprocessing Decisions
- resample_poly preferred over scipy.signal.resample (integer ratio 2:1, no frequency leakage)
- filtfilt (zero-phase) required for training+inference consistency
- Z-score per window, not per patient or global

## Dataset Split Decision
Split on case_id level (patient-wise) to prevent data leakage.
Do NOT split on individual windows - same patient windows must stay in same split.

