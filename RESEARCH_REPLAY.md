# Restart evidence replay (research only)

This tool replays saved, fresh-restart detection evidence against the protected SAFE engine and three fixed startup-recovery windows. It makes no changes to the operational baseline or to the original evidence. This is an offline, historically reviewed, same-video analysis, not independent validation. Occupancy is evaluated against count GT, not per-slot GT. Selection is fixed using DEV before TEST is scored; both partitions have previously been reviewed. The experimental window sweep must not be used to claim a generalization result.

Run with a Python environment containing the project requirements:

```
python research_replay.py --source PATH_TO_SPLIT_EXPERIMENTS --gt PATH_TO_GT.csv --output NEW_OUTPUT_DIRECTORY
python research_diagnose.py --source PATH_TO_SPLIT_EXPERIMENTS --output NEW_DIAGNOSTIC.csv
```

The source must contain completed original and alternate split fits and completed restart runs with evidence. Inputs are read only. The replay requires a new output folder, records input hashes, writes partial results and failure state, and freezes each DEV selection before scoring TEST. `research_diagnose.py` counts local slots whose recovery gate is ready at the first eligible checkpoint and whose SAFE state at that same checkpoint is occupied. This overlap describes that checkpoint; it does not establish slot-level correctness or a causal explanation for the whole trace.

For a sequential, bounded local queue with logs and restartable completed steps:

```
python research_manager.py --source PATH_TO_SPLIT_EXPERIMENTS --gt PATH_TO_GT.csv --run-dir NEW_RUN_DIRECTORY --max-seconds 21600 --min-free-gb 10
```

The queue runs replay then diagnosis, records state and per-attempt logs, and leaves failed attempts in place. A `STOP` file inside the run directory stops the current subprocess. Removing that file permits a restart; a process killed externally may leave `manager.lock`, which must be inspected before manually removing it. Completed steps are skipped only for the same input/code fingerprint. Retry count and resource limits are configurable. Paths and logs are local and may contain private file locations: do not publish them.

This is a bounded fixed research queue, not an autonomous hypothesis generator, scheduler, GPU inference pipeline, or operating system service. It does not create manual labels, retrain YOLO, or promote a candidate to production. It has not been validated for Windows logout or power-loss recovery.
