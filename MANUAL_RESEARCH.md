# Manual initialization research

This experimental branch keeps the existing SAFE baseline. The validation center can select a CSV or a folder of CSV snapshots. The editor proposes a separate filename for each time; existing multi-time CSV files remain supported. Rows contain `time_sec,global_slot_id,initial_state` with `O/E/U`. Missing slots are UNKNOWN. Duplicate IDs at a selected time are rejected. Count GT is not a slot snapshot.

`fresh_manual_experiment.py` makes an isolated copy of a completed original or alternate fit, creates an **unverified pseudo snapshot from archived full-frame detection at exactly the selected start**, then reruns that interval from video on CUDA. Positive detection at confidence >= 0.25 is OCCUPIED; all other slots are UNKNOWN. This is not human manual GT and it is not based on future count labels. The two conditions do not have the same initialization information. The fitted parameters and continuous traces are reused, and newly extracted restart observations are shared among restart variants. Continuous reference traces are historical, not re-extracted under the new runtime. This is the same historically reviewed video, not independent validation.

```
python fresh_manual_experiment.py --video VIDEO --gt COUNT_GT --fit SPLIT_FIT --app APP_ROOT --start 900 --output NEW_OUTPUT_FOLDER
python analyze_manual_experiment.py --restart COMPLETED_RESTART_RUN --start 900 --output NEW_COMPARISON_CSV
python inspect_unknown_slots.py --video VIDEO --run FRESH_OUTPUT_FOLDER --output NEW_PRIVATE_IMAGE_FOLDER
```

Supported starts: 0, 330, 660, 900 seconds. The existing evaluator also includes the split's TEST start even when a different time is specified; missing-time pseudo snapshots are skipped. Preserve the private image folder locally. Saved snapshot provenance and hashes are in `manifest.json` and `manual_init_audit.json`.

UNKNOWN must not be counted as EMPTY. Normal exact/MAE are withheld when a scope contains unknown states; the occupancy bounds and fully resolved sample count remain available. The initial checkpoint is reported separately from later checkpoints. Slot-level correctness or transition errors cannot be established from count GT alone.

`research_manager.py` offers the existing fixed replay/diagnosis queue and an optional fresh GPU job, with bounded retries, disk checks, per-attempt logs, checkpointed state, and sampled process memory and whole-GPU utilization. A JSON config supplies `video`, `gt`, `fit`, `app`, and optional `start`; pass it with `--fresh-config`. Use a CUDA-enabled Python environment for a fresh GPU job. `--max-seconds 0` means no deadline. Create `STOP` in the run directory to stop the current job; remove it to resume. Completed jobs are reused only when the input/code fingerprint matches. A forced process termination can leave a lock file: verify the owner has stopped before removing it. GPU metrics are whole-device measurements, not process attribution.

The manager is not a general autonomous research agent. Windows Task Scheduler, logout behavior, hardware-failure recovery, independent video validation, and automatic creation of trustworthy human slot labels are not implemented or validated.
