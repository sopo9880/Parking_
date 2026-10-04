# Parking Research Agent

Local research Agent for the Connect Hyundai parking-occupancy study.

Current release: **v16.4.0 CANDIDATE**  
Protected algorithm baseline: **v16.2 SAFE_BASELINE**

## Start on Windows
1. Extract the release ZIP.
2. Double-click `START.cmd` once. It creates `.venv` and installs dependencies.
3. Later launches can use `RUN.cmd`.

## Main workflow
- Select the test video and Ground Truth CSV.
- Configure 4-point CCTV ROI, slot points and duplicate-slot links when needed.
- Use **ALL-IN-ONE** for unattended cache/tuning/evidence/robustness/state evaluation and ZIP packaging.
- Use **Research History** to review the research lineage and previous decisions.
- Use **Check Update** to query the latest GitHub Release.

## v16.4 candidate transition refiner
v16.4.0 adds a narrow post-baseline candidate that targets transition timing without replacing the v16.2 SAFE_BASELINE.

After an ALL-IN-ONE run has produced `baseline_slot_timeseries.csv` and `baseline_count_timeseries.csv` in `output\`, run:

`RUN_V16_4_CANDIDATE.cmd`

The candidate writes:
- `candidate_v164_slot_timeseries.csv`
- `candidate_v164_global_slot_timeseries.csv`
- `candidate_v164_count_timeseries.csv`
- `candidate_v164_metrics.csv`
- `candidate_v164_transition_events.csv`
- `candidate_v164_event_balanced_metrics.csv`
- `candidate_v164_summary.json`
- `CANDIDATE_v16_4_REPORT.txt`

### Current retained-video result
- v16.2 SAFE_BASELINE TEST Exact: **85.29%**
- v16.4 candidate TEST Exact: **97.06%**
- v16.2 MAE: **0.1471**
- v16.4 candidate MAE: **0.0294**
- Max error: **1**
- Under-count: **0%**

This is **not** a SAFE_BASELINE promotion. The same retained validation video was used for error analysis, so an independent video is required before promotion.

## Auto Update
The updater reads the latest release from `sopo9880/Parking_`, downloads the matching ZIP and `.sha256`, verifies SHA-256, then installs only after the current app exits.

These paths are preserved during update:
- `settings.json`
- `slots.json`
- `rois.json`
- `ui_state.json`
- `sample_ground_truth.csv`
- `work/`
- `output/`
- `.venv/`
- `user_data/`

Startup update checking is enabled by default but installation requires confirmation.

## Version-lineage rule
`research_history.json` contains only the paper/research lineage. The separate field-operation application lineage that later used v20+ is intentionally not mixed into this timeline.
