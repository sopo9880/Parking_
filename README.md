# Parking Research Agent

Local research Agent for the Connect Hyundai parking-occupancy study.

Current release: **v16.3.0**  
Algorithm baseline: **v16.2 SAFE_BASELINE**

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
