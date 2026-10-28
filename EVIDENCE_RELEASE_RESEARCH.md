# Evidence-based release research (not operational)

SAFE and its count outputs remain unchanged. The optional `evidence_release_candidate.py` consumes a completed baseline global trace and the corresponding local observations. When the baseline proposes OCCUPIED→EMPTY, a current confident FULL detection or strong AUX detection from any linked camera may retain the existing occupied state. AUX protection uses the existing additive criteria: confidence >= .45, or confidence >= .20 with at least two supporting scales. Otherwise, without corroborating exit evidence, the candidate emits UNKNOWN. UNKNOWN is never counted as EMPTY. A later baseline OCCUPIED transition can resolve it. A candidate EMPTY transition requires both outward-motion and visible-empty cues at the same time; a current confident FULL or strong AUX detection in any camera for the same global ID vetoes it. The first state is the supplied snapshot. The policy operates chronologically and does not read GT.

`causal_empty_cues.py` and `extract_causal_empty_cues.py` provide an **uncalibrated experimental proxy**, not a validated empty-space classifier. An operator-marked EMPTY initial crop is required; a later occupied reference needs three confident detections while the baseline says OCCUPIED. The proxy looks for a sustained return to that empty appearance, difference from the occupied reference, and a recent outward radial movement of one track. It is sensitive to image alignment, lighting, occlusion, erroneous initial labels, poor tracking, and nearby traffic. Initially occupied slots have no verified empty reference and cannot obtain an empty cue this way. Fixed thresholds are exploratory, not tuned or validated on independent data. Do not treat a cue as a verified departure or deploy this policy.

Run local research with the existing completed fresh GPU traces and their human comparisons:

```
python audit_release_visual.py --fresh FRESH_RUN --comparison COMPARISON --video VIDEO --output NEW_CSV
python audit_slot_mapping.py --fit FIT --video VIDEO --gt GT --snapshot-dir VERIFIED_DIR --output NEW_DIR
python extract_causal_empty_cues.py --fresh FRESH_RUN --comparison COMPARISON --snapshot VERIFIED_CSV --video VIDEO --output NEW_DIR
python run_evidence_release_replay.py --fresh FRESH_RUN --comparison COMPARISON --gt GT --cues CUES_DIR --output NEW_DIR
python run_release_dev_exploration.py --fit ORIGINAL_FIT --pseudo-dir SAVED_PSEUDO_DIR --gt GT --output NEW_DIR
```

The two human-initialized experiments reuse saved fresh-GPU detections, including human-supplied information at time zero. The DEV exploration uses pseudo snapshots at 330 and 660 seconds, not human GT; the 660 window is truncated before 900. Same-video windows have been historically reviewed and are not independent validation. Count-GT containment inside an UNKNOWN range is not count accuracy. Exact/MAE and stable recovery are withheld for scopes with unresolved slots. Per-slot false release and actual exit delay require subsequent verified per-slot labels; no such labels are silently fabricated.

Mapping and visual diagnostics produce local private CCTV images or tables. Do not publish these, human snapshots, count GT, local paths, or original video. A clear next step is to obtain a small number of independent human event labels and an independent video, then pre-register evidence rules on a development set before evaluating untouched footage.
