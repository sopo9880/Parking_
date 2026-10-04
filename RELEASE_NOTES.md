# Parking Research Agent v16.4.0

This is a **CANDIDATE algorithm release**. The protected **v16.2 SAFE_BASELINE remains unchanged**.

## Added
- `ENTRY_TRACK_SWITCH_HOLD`
  - Carries slot-level motion context across tracker-ID switches.
  - Prevents a newly assigned track from instantly looking like a stationary parked vehicle after a large maneuver.
- `LEAVING_WEAK_OWNER_RELEASE`
  - Only activates when the baseline itself enters LEAVING.
  - Requires weak mean confidence, meaningful motion, and an EMPTY sibling view for the same global slot.
  - Releases stale ghost occupancy after confirmation and requires a settle interval before reacquisition.
- Event-balanced transition evaluation outputs.
- Dedicated candidate runner: `RUN_V16_4_CANDIDATE.cmd`.
- Auditable transition-event CSV and candidate summary/report files.
- Release workflow now packages the current `main` source directly instead of reconstructing the old v16.3 bootstrap bundle.

## Retained validation result
### v16.2 SAFE_BASELINE
- TEST Exact Match: **85.29% (29/34)**
- MAE: **0.1471**
- Max error: **1**
- Under-count: **0%**

### v16.4 candidate
- TEST Exact Match: **97.06% (33/34)**
- MAE: **0.0294**
- Max error: **1**
- Under-count: **0%**
- Over-count: **2.94%**
- **4 of the previous 5 TEST errors removed**
- DEV metrics unchanged

The corrected retained TEST timestamps are 16:00, 16:10, 17:10 and 17:20. One over-count event at 17:00 remains.

## Important limitation
The rules were developed after analyzing the same retained validation video. Ground truth is **not used by the transition decisions**, but independent-video validation is still required before any SAFE_BASELINE promotion.

## Auto Update
GitHub Releases ZIP + SHA-256 verification remains enabled. User settings, slot/ROI configuration, outputs, venv and user data remain preserved during update.
