# Parking Research Agent v16.5.0

v16.5.0 is the **validation expansion release**. It does not silently promote the candidate: **v16.2 remains SAFE_BASELINE and v16.4 remains CANDIDATE**.

## Added

### 1-N CCTV / new-video support
- CCTV count: **1 to 12**.
- Dynamic ROI setup, preview, duplicate-slot linker, diagnostics and paper screenshots.
- Existing 3-CCTV setup remains compatible.

### Dataset Profiles + GT tool
- Save/load local Dataset Profiles.
- Dynamic count-GT CSV generator.
- Interactive GT Labeler with per-CCTV counts, total occupied, U and CUT marking.

### Episode / cut validation
- Configurable cut boundaries and post-cut warm-up exclusion.
- `episode_evaluation_mask.csv`
- `episode_metrics.csv`

### Repeated-source stability
For a repeated 4-minute source, use `repeat_period_sec=240`.
- `repeat_stability.csv`
- `REPEAT_STABILITY_REPORT.txt`
- This is reproducibility/stability testing, **not independent validation**.

### Automatic v16.4 evaluation
ALL-IN-ONE automatically evaluates v16.4 after v16.2 and packages candidate CSV/JSON/report files.

### Public external validation
Automatic **MetaPKLot / CNRPark-EXT** preparation:
- Quick / Standard selected official subsets.
- Full explicit opt-in shallow clone.
- resumable targeted downloads + size verification.
- automatic COCO parking-space annotation conversion.
- Accuracy / Precision / F1 / Occupied Recall / Empty Specificity / TP/TN/FP/FN.
- camera/weather breakdown.
- `EXTERNAL_VALIDATION_TO_CHATGPT.zip`.

Public-data results are labeled external **spatial** generalization, not continuous temporal validation.

### Cache visibility
The main UI now displays a compact CACHE audit summary.

## Protected research status
Retained development video:
- v16.2 SAFE: **85.29% Exact / 0.1471 MAE**
- v16.4 Candidate: **97.06% Exact / 0.0294 MAE**

No automatic SAFE_BASELINE promotion occurs.

## Update safety
`user_data/` remains preserved and now contains Dataset Profiles and downloaded public datasets.
