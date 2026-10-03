# Parking Research Agent v16.3.0

This release turns the v16.2 research engine into a versioned local research Agent while deliberately keeping the **v16.2 SAFE_BASELINE algorithm unchanged**.

## Added
- Research History viewer with the paper/research lineage from v1 through v16.2.
- Local future-run journal that survives program updates.
- GitHub Releases update checker.
- SHA-256 verified update download and replacement helper.
- Persistent protection for settings, slots, ROI, GT, work/output, venv and user data.
- GitHub Actions release packaging.

## Baseline carried forward
- TEST Exact Match: **85.29%**
- MAE: **0.1471**
- Max error: **1**
- Under-count: **0%**

The rejected v16.2 candidates remain rejected: TRANSITION_GUARD 70.59%, SEG_ASSIST 70.59%, EMPTY_REF_ASSIST 0%.

## Next research target
The next algorithm experiment will target transition errors caused by track-ID switches using slot-level motion memory, inside this Agent shell.
