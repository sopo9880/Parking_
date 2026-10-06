# Causal Parking-Space Occupancy Estimation from Re-recorded Multi-CCTV Video: Parking Research Agent Living Paper

Living Paper | v16.5.6 | Research revision temporal-restart-evidence-selection-r6

[한국어](paper_ko.md) | [English](paper_en.md)

Research draft. Retained results are separated from recomputed user-supplied v16.5.4/v16.5.5 prediction CSVs. Actual v16.5.5 UNKNOWN restarts and failed startup recovery are included. Full-video performance of revised v16.5.6 candidates remains unmeasured. Authors and affiliation are unspecified.

<!-- section:abstract -->

## Abstract

This paper documents the development of a research agent for estimating occupied parking spaces from phone recordings of a CCTV monitor. The setting combines shake, perspective distortion, low resolution, sun glare, monitor stripes, and missed vehicle detections. The method combines fixed manual slot points, perspective correction, Voronoi spatial constraints, FULL-first detection with selective AUX recovery, and causal temporal state tracking. Retained research records report TEST Exact 85.29% and MAE 0.1471 for v16.2 SAFE_BASELINE, and 97.06% and MAE 0.0294 for v16.4 CANDIDATE after track-switch entry holding and weak-owner departure release. Candidate rules were developed after reviewing errors on the same validation video; these results therefore cannot establish independent generalization. Archived CNRPark-EXT predictions were recomputed across nine cameras, three weather conditions, and 4,073 images. Removing 35,127 duplicate rows leaves 165,530 unique image–space pairs, with Accuracy 95.0770% and F1 95.4401%. Image-level occupied-space Count Exact is 33.9553% and MAE is 1.440953. Re-inference on 20 selected images reproduced archived decisions. Spatial results do not establish temporal-transition generalization, and parameters remain frozen as an External Baseline.

Keywords: parking occupancy, re-recorded CCTV, causal state tracking, multi-CCTV, robustness, Living Paper.

Retained-video cross-window recalculation found Candidate gains only in the final of four windows; full-trace Exact was SAFE 34.96% and Candidate 38.21%. This is stability evaluation, not independent validation.

<!-- section:introduction -->

## 1. Introduction

When a direct digital CCTV stream is unavailable, a recording of the monitor can serve as the research input. Better detection alone does not resolve slot identity, completed parking, or departure. A vehicle visible in multiple CCTV views also requires a distinction between per-view detections and physical occupied-space counts.

The research questions are: (1) Can weak detections be assigned consistently while keeping manual slot points authoritative? (2) Can maneuvers and stable occupancy be distinguished without future frames? (3) Do improvements on the retained video persist on new videos and degraded inputs? This paper organizes the progression of spatial assignment and temporal decisions, reports unsuccessful experiments, and identifies independent validation as the next step.

<!-- section:related -->

## 2. Related work and scope

Ultralytics YOLO inference [R3] supplies vehicle observations. Comparisons in this study concern retained internal versions; no ranking or state-of-the-art claim against papers using different protocols is made. Internal experiments compare full-view detection, slot crops, Hungarian one-to-one assignment, fixed manual spatial constraints, and temporal state tracking.

MetaPKLot [R4] provides annotations and evaluation resources for existing parking datasets. The v16.5 CNRPark-EXT path evaluates occupancy in individual images. Its spatial evaluation cannot substitute for entry/departure decisions in continuous video. External data remain evaluation targets without parameter tuning from target-environment ground truth.

<!-- section:environment -->

## 3. Environment and problem definition

The initial input is a phone recording of a CCTV monitor [R1]. Four-corner ROIs correct tilted and trapezoidal views. Manual slot points accommodate obscured parking lines and initial full occupancy. Operators link views of the same physical parking space using GLOBAL SLOT identities. The original configuration has three CCTV views; v16.5 supports profiles with 1-12 views.

Let y(t) denote the ground-truth occupied-space count and ŷ(t) the prediction. Exact is the fraction of evaluated timestamps satisfying y(t)=ŷ(t); MAE averages |y(t)-ŷ(t)|. Occupied-space count, unique-vehicle count, and per-CCTV count are separate quantities. Technical state names EMPTY, MANEUVERING, OCCUPIED, LEAVING, UNKNOWN and SAFE_BASELINE/CANDIDATE remain canonical for data and code compatibility.

![Figure 1. Actual re-recorded Hyundai parking CCTV monitor environment. Multiple CCTV views are shown on one monitor, with parking-space inference overlays on each ROI.](figures/hyundai/environment_overview.svg)

**Figure 1. Actual re-recorded Hyundai parking CCTV monitor environment.** The input is a phone recording of the control monitor rather than a direct digital CCTV stream. Oblique viewpoints, low resolution, reflections, monitor artifacts, and different perspective distortions coexist. This figure is a real paper-ready capture from an ALL-IN-ONE run, not a synthetic illustration.

![Figure 2. Manual parking-space reference points and Voronoi spatial partitioning for CCTV2.](figures/hyundai/voronoi_cctv2.svg)

**Figure 2. Manual parking-space reference points and Voronoi spatial partitioning for CCTV2.** Operator-defined reference points remain the authoritative slot locations, while the spatial partition constrains vehicle-to-slot association. Learned auxiliary information does not replace the manual points and is used to stabilize assignment under strong perspective distortion and occlusion in the real CCTV view.

<!-- section:methods -->

## 4. Proposed method

### 4.1 Spatial assignment and FULL-first observations

After perspective correction, YOLO observations are assigned under Voronoi constraints fixed by manual slot points. Learned anchors do not replace those points and are restricted by shift and sample conditions. Duplicate suppression and one-to-one assignment reduce multiple-slot assignment of a physical vehicle. FULL detection is authoritative. Selective CROP/AUX runs when primary observations are weak, missing, visually suspicious, or due for sparse auditing. AUX may recover missed FULL evidence but cannot erase valid FULL evidence.

### 4.2 Causal temporal states

Online tracking and a rolling ten-second history distinguish maneuvers from stable occupancy. The first ten seconds are warm-up; future frames are excluded. Long-stationary candidate slots are advisory until operator review and are not automatically added to the physical slot count. Cached geometry and appearance-template metadata are restored together; missing templates or implausible evidence distributions trigger rejection and recomputation.

### 4.3 Track switches and departure candidates

v16.4 ENTRY_TRACK_SWITCH_HOLD retains recent slot-motion context across tracker-ID switches. A newly stationary-looking track following substantial motion cannot immediately confirm OCCUPIED before sufficient stabilization. LEAVING_WEAK_OWNER_RELEASE requires the baseline's LEAVING transition, weak mean owner confidence, meaningful movement, and an EMPTY vote from a linked CCTV before releasing ghost occupancy. A reacquisition settling interval blocks immediate re-entry. These rules are CANDIDATE postprocessing and neither automatically promote nor modify SAFE_BASELINE [R2].

### 4.4 External spatial baseline and unique samples

Still-image evaluation uses the unchanged detector/polygon-intersection decision without temporal tracking. Annotation conversion and evaluation normalize path separators and enforce a unique (image_path, spot_id) key. Conflicting GT, polygons, or predictions for a key halt evaluation instead of arbitrarily keeping one value. Audits record per-camera duplicates and input/unique sizes. Image counts sum OCCUPIED over unique annotated spaces, distinct from off-slot vehicles or cross-camera unique-vehicle totals. The External Baseline saves initial parameters/model hashes and does not apply subsequent setting changes to evaluation.

### 4.5 Restart and startup recovery candidate

Restart experiments retain fitted detectors, anchors and templates but create new trackers, motion history, AUX memory, segmentation recovery frame history and occupancy state. Initial occupancy is UNKNOWN; unlabeled appearance similarity is not treated as occupied evidence. RESET_RECOVERY is experimental: at least 10 seconds of observation, at least eight samples in a 10-second window with at least 75% repeated FULL detections, stationary center/speed bounds, and multi-camera agreement or stronger single-camera confidence can initialize occupancy within the first 30 seconds. SAFE temporal rules follow. It uses current/past images and receives no TEST GT. No 10–30 second recovery guarantee or commercial readiness is claimed. Default SAFE behavior and CANDIDATE status remain unchanged.

v16.5.6 revision: startup recovery adds corroborated entry without suppressing normal SAFE entry. The old first-ten-second blocking candidate remains a separate comparator.

<!-- section:protocol -->

## 5. Experimental protocol and evidence status

The retained protocol selects parameters on DEV before 15:00, then evaluates TEST from 15:00 to 20:30 after selection. The first ten seconds are treated as warm-up. The record contains 34 TEST checkpoints, with possible temporal correlation between adjacent samples. Maximum error, under/over-count rates, and event-balanced transition metrics complement Exact and MAE.

Numbers here summarize reported results in research_history.json and change records. This documentation patch does not rerun the original video or recover complete runtime artifacts. sample_ground_truth.csv is a reference, not a replacement for the source video, parameters, and predictions. Reproduction must retain version, input hashes, settings, ROIs, slots, GT, DEV selection, TEST predictions, cache audits, and environment. Because v16.4 rules were designed after reviewing TEST errors on the same video, its absolute TEST score is not presented as an untouched holdout result.

External evidence [R6] is the original prediction CSV verified identical to the user-supplied EXTERNAL_VALIDATION_TO_CHATGPT.zip. ZIP, prediction, GT, settings, and model hashes are retained. GT reconciliation and conflict checks precede metric recomputation from unchanged predictions. A separate original-run settings snapshot was unavailable; current supplied settings and original weights were used to re-infer 20 representative images. All slot decisions and detection counts in these images matched archived predictions and were accepted as overlay evidence. This is the official recalculation of a frozen prediction baseline, not a new inference pass over all 4,073 images. Screenshots are deterministically chosen for error analysis rather than random sampling.

### Fresh DEV/TEST split experiment design (v16.5.4)

Separate from frozen cross-window statistics, Original (DEV 0–900 s / TEST 900–1230 s) and Alternate (DEV 330–1230 s / TEST 0–330 s) run in fresh processes. Alternate intervals are configurable and must be disjoint. Manual points, operator-supplied initial states, pretrained YOLO weights and the search grid remain fixed inputs. Existing detector selections, learned directories and caches are not reused. Detector configurations are selected using DEV labels; anchors/templates and quality quantiles use DEV frames. Temporal variants are selected using DEV labels only; slot-event annotations are excluded from selection. This does not retrain YOLO network weights.

The selected configuration hash is frozen before TEST labels evaluate SAFE_BASELINE, TRANSITION_GUARD, SEG_ASSIST, EMPTY_REF, SELECTED and CANDIDATE. Intervals include their start and exclude their end, except the evaluation horizon endpoint. The original 10 s startup warm-up remains; state is not reset at split boundaries. Unlike the legacy learner's boundary-frame inclusion, fresh fitting follows explicit DEV membership. Inference proceeds chronologically from zero, so past online state history can span intervals. The reverse split applies templates fitted on later DEV frames to earlier TEST frames: it is offline evaluation, not causal future prediction.

Implementation and functional validation are complete in v16.5.4, with full real-CCTV split experiments pending at publication. Archived predictions from a subsequently supplied v16.5.4 run are now recalculated. Tests perturb TEST labels while checking invariant actual state selection and changed evaluation metrics, inspect actual detector/geometry DEV frame access, and verify boundaries and failures. Functional validation is not performance evidence. Actual v16.5.5 restart/recovery scores, pending at publication, are now reported in Section 6.3. Full-video scores of revised v16.5.6 candidates remain unreported. See the [protocol record](data/split_protocol_v1654.json); runtime outputs include split_final_metrics.csv and split_audit.json. Historically reviewed same-video intervals cannot establish independent validation.

### Restart evaluation protocol (v16.5.5)

After freezing each fresh split DEV selection, inference starts anew at 0/330/660/900 seconds and the TEST start, for up to 330 seconds. Continuous SAFE/Candidate are compared with RESET_SAFE, RESET_CANDIDATE and RESET_RECOVERY. Resetting state on sliced precomputed evidence is insufficient: detector histories and recovery frame histories are freshly reconstructed from the restart timestamp. The Validation Center can configure starts or reuse a completed actual fit directory for restart inference only. Unique output directories contain audits and explicit failure status. Restarts use UNKNOWN initial occupancy while existing continuous runs use operator initialization, so this is an operational comparison, not an ablation isolating temporal history alone.

Metrics include startup-inclusive Exact/MAE; first 30/60/120-second errors; first exact; the beginning and confirmation time of three consecutive available exact GT checkpoints; post-stability Exact; never-stabilized flag; and continuous/reset slot-state differences. The first-10-second exclusion is a separate row. Three exact count checkpoints do not prove permanent stability or individual-slot correctness, nor guarantee correctness between GT samples. Actual v16.5.5 results are reported in Section 6.3. v16.5.6 adds relapse metrics and legacy/revised recovery and DEV-selected research candidates; a new full-video evaluation remains pending.

<!-- section:results -->

## 6. Results

Table 1 summarizes retained occupied-count results [R1, R2], preserving original MAE rounding. v12 and v13 share Exact but differ in MAE and maximum error. v16.2's 29/34 and v16.4's 33/34 differ by approximately 11.77 percentage points; MAE decreases from 0.1471 to 0.0294. Both report maximum error 1 and zero under-count. The candidate reports 2.94% over-count, while DEV metrics are recorded as unchanged.

Accuracy on new Hyundai continuous videos remains unreported. CNRPark-EXT spatial results use separate denominators and tables in Section 10. These historical numbers must not be treated as evidence of a newly completed experiment. Early versions lacking quantitative records are not assigned fabricated values.

| Version / method | TEST Exact (%) | MAE | Max error | Evidence |
| --- | --- | --- | --- | --- |
| v7 | 82.35 | 0.3235 | - | retained_record |
| v8 | 32.35 | 1.294 | - | retained_record |
| v9 | 67.65 | 0.441 | - | retained_record |
| v10/v10.1 | 79.41 | 0.265 | - | retained_record |
| v12 FULL | 85.29 | ~0.206 | 2 | retained_record |
| v13 SAFE_BASELINE | 85.29 | 0.147 | 1 | retained_record |
| v16.2 SAFE_BASELINE | 85.29 | 0.1471 | 1 | retained_record |
| v16.4 CANDIDATE | 97.06 | 0.0294 | 1 | same_video_post_hoc |
| v16.5 external/new video | - | - | - | pending_at_v16.5_release |
| v16.5.1 | - | - | - | presentation_patch_no_new_measurement |
| v16.5.2 external spatial | - | - | - | separate_spatial_metrics_section_10 |

<!-- section:windows -->

## 6.1 Multi-window evaluation of the retained video

v16.5.3 lets the Validation Center specify multiple windows and compare saved SAFE/Candidate predictions. It performs no training, parameter selection, or inference. Defaults are 0–330, 330–660, 660–900, 900–1230 seconds and the full trace. Starts are inclusive and ends exclusive, except the largest configured end is included. The original continuous state history is preserved without resetting at window starts. Initial 10-second warm-up, configured post-cut warm-up, invalid values, and predictions missing from either algorithm are excluded from a common denominator, with exclusion counts recorded.

The two count traces from the retained 2026-10-04 run were copied read-only and recalculated. Both methods have Exact 0% in the first two windows, with MAE 1.5 and 2.0. The third window has Exact 58.33% and MAE 0.6667 for both. The final window reproduces SAFE 85.29%/0.1471 and Candidate 97.06%/0.0294. Improvement is observed in 1/4 windows, with no gain in the first three. Across all 123 samples, Exact is 34.96% versus 38.21% and MAE 1.0976 versus 1.0650. The final-window 97.06% cannot be generalized to the full video. Initialization problems are a possible explanation for early errors requiring further analysis, not an established cause.

ALL counts each sample once. Unweighted window Exact mean/population standard deviation is SAFE 35.91%/37.15 percentage points and Candidate 38.85%/41.19 points; these differ from full-sample Exact. Both have worst W1 and best W4 (ties use the first window). Custom overlapping windows are permitted and flagged; ALL is excluded from macro summaries. Early windows were historical DEV and the final window informed error refinement, so this is not independent hold-out validation. Rules stay frozen with no SAFE promotion.

### Window results on common paired samples

| Window | N | SAFE Exact % | Candidate Exact % | Delta pp | SAFE MAE | Candidate MAE |
| --- | --- | --- | --- | --- | --- | --- |
| 0:00–5:30 | 32 | 0.00 | 0.00 | +0.00 | 1.5000 | 1.5000 |
| 5:30–11:00 | 33 | 0.00 | 0.00 | +0.00 | 2.0000 | 2.0000 |
| 11:00–15:00 | 24 | 58.33 | 58.33 | +0.00 | 0.6667 | 0.6667 |
| 15:00–20:30 | 34 | 85.29 | 97.06 | +11.76 | 0.1471 | 0.0294 |
| ALL | 123 | 34.96 | 38.21 | +3.25 | 1.0976 | 1.0650 |

![Figure 12. Frozen predictions across retained-video windows. Improvement is observed only in the final window (1/4).](figures/v16_5_3/retained_window_comparison.png)

Figure 12. Frozen predictions across retained-video windows. Improvement is observed only in the final window (1/4).

<!-- section:fresh_splits -->

## 6.2 Supplied fresh-split results and restart sensitivity

Per-split DEV/TEST predictions and selected-configuration hashes in the supplied v16.5.4 UPLOAD_TO_CHATGPT.zip were recalculated; no full inference was rerun. Original TEST evaluates 900–1230 seconds after continuous execution from zero, a warm-state condition. Alternate TEST evaluates 0–330 seconds after fitting settings/geometry on later DEV frames. Operator initial occupancy labels were retained, distinguishing it from the new UNKNOWN restart protocol.

Original TEST reproduces SAFE Exact 85.2941%/MAE 0.147059 and Candidate 97.0588%/0.029412. Alternate TEST (N=32) yields SAFE 9.3750%/1.187500 and Candidate 9.3750%/1.218750; Guard/SEG_ASSIST yield 46.8750%/0.687500. SAFE under-count rate is 90.6250%. Equal Exact with worse Candidate MAE cannot support promotion. Both SAFE and Guard first reach exact count at 170 seconds, with three consecutive exact GT checkpoints confirmed at 190 seconds. These are count-level observations, not slot correctness or permanent recovery.

Fitting, scene content and initialization conditions differ together, so these results cannot prove cold start as the sole cause of early errors. The table/figure are historical same-video diagnostic evidence, not scores of the new recovery candidate or independent generalization.

### Recomputed supplied v16.5.4 TEST metrics

| Split | Variant | N | Exact (%) | MAE | Max error | Under (%) | Over (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| original | SAFE_BASELINE | 34 | 85.2941 | 0.147059 | 1 | 0.0000 | 14.7059 |
| original | TRANSITION_GUARD | 34 | 70.5882 | 0.294118 | 1 | 0.0000 | 29.4118 |
| original | SEG_ASSIST | 34 | 70.5882 | 0.294118 | 1 | 0.0000 | 29.4118 |
| original | SELECTED | 34 | 85.2941 | 0.147059 | 1 | 0.0000 | 14.7059 |
| original | EMPTY_REF | 34 | 0.0000 | 2.970588 | 4 | 0.0000 | 100.0000 |
| original | CANDIDATE | 34 | 97.0588 | 0.029412 | 1 | 0.0000 | 2.9412 |
| alternate | SAFE_BASELINE | 32 | 9.3750 | 1.187500 | 3 | 90.6250 | 0.0000 |
| alternate | TRANSITION_GUARD | 32 | 46.8750 | 0.687500 | 2 | 53.1250 | 0.0000 |
| alternate | SEG_ASSIST | 32 | 46.8750 | 0.687500 | 2 | 53.1250 | 0.0000 |
| alternate | SELECTED | 32 | 9.3750 | 1.187500 | 3 | 90.6250 | 0.0000 |
| alternate | EMPTY_REF | 32 | 43.7500 | 0.718750 | 2 | 53.1250 | 3.1250 |
| alternate | CANDIDATE | 32 | 9.3750 | 1.218750 | 3 | 90.6250 | 0.0000 |

![Figure 13. Archived v16.5.4 results under different DEV fits/TEST windows. Not a matched cold/warm comparison; recovery-candidate scores remain unreported.](figures/v16_5_5/fresh_split_results.png)

Figure 13. Archived v16.5.4 results under different DEV fits/TEST windows. Not a matched cold/warm comparison; recovery-candidate scores remain unreported.

<!-- section:restart_results -->

## 6.3 Actual restart results and post-agreement relapses

The v16.5.5 UPLOAD_TO_CHATGPT.zip supplied on 2026-10-06 contains actual original/alternate restarts with new trackers at 0/330/660/900 seconds, as well as continuous split traces. The log places fresh split/restart execution between 00:26:58 and 09:26:53. Restarts use UNKNOWN initialization without prior tracking or occupancy history; detectors and slot geometry are frozen from each DEV fit. Tables report INCLUSIVE_STARTUP over 330-second windows, including the final 1230-second endpoint. Sample counts differ from legacy TEST because the video's first-ten-second exclusion is not applied here. Separate matched AFTER_RESTART_WARMUP metrics are preserved and recomputed [R8].

With the original DEV fit, continuous SAFE at 900 seconds has N=34, Exact 85.2941% and MAE 0.147059, versus 0% and 1.352941 for UNKNOWN RESET_SAFE. The old v16.5.5 RESET_RECOVERY is worse at 0% and 2.088235. Neither RESET_SAFE nor RESET_RECOVERY reaches count agreement in any of the four original-fit restart windows. With the alternate fit, RESET_SAFE at 660 seconds achieves 69.6970%/0.333333 versus continuous SAFE 33.3333%/0.666667. At 900 seconds, RESET_SAFE achieves 14.7059%/1.176471 and legacy RESET_RECOVERY 0%/1.882353. Restarts do not uniformly degrade results, and the legacy recovery candidate does not establish improvement.

These observations establish strong operational restart sensitivity. Continuous comparators retain operator-supplied initial OCCUPIED/EMPTY priors, whereas resets use UNKNOWN; detector/AUX request paths are recomputed with fresh history. Differences cannot be attributed solely to temporal memory. Additional controls must match slot GT, initialization information and detector observations. This candidate does not achieve the 10-30 second recovery objective.

In the alternate legacy split with original initialization priors, SAFE and Guard first reach count agreement at 170 seconds; three checkpoints confirm it at 190 seconds. SAFE is wrong again at the 200-second checkpoint. Over 16 checkpoints at 170-320 seconds, SAFE has Exact 18.75%/MAE 0.8750 while Guard has 93.75%/0.0625. This interval was chosen after inspecting results and is explanatory, not a new TEST score. EMPTY-to-OCCUPIED transitions for G002/G027/G019 occur at 33/53/163 seconds; G002 returns to EMPTY at 192 seconds. Without per-slot GT these slot decisions cannot be confirmed as correct.

v16.5.6 treats three exact-count checkpoints as provisional agreement and adds first relapse time, relapse episode count, post-agreement MAE, longest exact run and terminal exact run length. The revised recovery candidate preserves normal SAFE entry while adding corroborated startup entry, correcting the old blanket first-ten-second entry suppression. Replaying the archived original 900-910 second observations produces 17 occupied spaces at both endpoints under SAFE and the revision, versus 0 initially and 15 at 910 seconds under legacy recovery. This is a startup regression check, not full-video reinference or per-slot validation. RESET_RECOVERY_V1655 retains the legacy comparator, and operational SAFE defaults remain unchanged.

### Actual restarts with original DEV fitting (v16.5.5)

| Start (s) | Variant | N | Exact (%) | MAE | First 30s MAE | First 60s MAE | First Exact (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | CONTINUOUS_SAFE | 33 | 0.0000 | 1.5152 | 2.0000 | 2.0000 | not reached |
| 0 | RESET_SAFE | 33 | 0.0000 | 2.1818 | 4.0000 | 4.0000 | not reached |
| 0 | RESET_RECOVERY | 33 | 0.0000 | 2.6970 | 9.3333 | 6.8333 | not reached |
| 330 | CONTINUOUS_SAFE | 33 | 0.0000 | 2.0000 | 2.0000 | 2.0000 | not reached |
| 330 | RESET_SAFE | 33 | 0.0000 | 2.0303 | 2.3333 | 2.1667 | not reached |
| 330 | RESET_RECOVERY | 33 | 0.0000 | 2.6364 | 9.0000 | 5.5000 | not reached |
| 660 | CONTINUOUS_SAFE | 33 | 63.6364 | 0.5455 | 2.0000 | 2.0000 | 100.0000 |
| 660 | RESET_SAFE | 33 | 0.0000 | 1.2727 | 2.0000 | 2.0000 | not reached |
| 660 | RESET_RECOVERY | 33 | 0.0000 | 1.7879 | 7.6667 | 4.8333 | not reached |
| 900 | CONTINUOUS_SAFE | 34 | 85.2941 | 0.1471 | 0.0000 | 0.0000 | 0.0000 |
| 900 | RESET_SAFE | 34 | 0.0000 | 1.3529 | 1.0000 | 1.1667 | not reached |
| 900 | RESET_RECOVERY | 34 | 0.0000 | 2.0882 | 8.0000 | 5.1667 | not reached |

### Actual restarts with alternate DEV fitting (v16.5.5)

| Start (s) | Variant | N | Exact (%) | MAE | First 30s MAE | First 60s MAE | First Exact (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | CONTINUOUS_SAFE | 33 | 9.0909 | 1.2424 | 3.0000 | 2.6667 | 170.0000 |
| 0 | RESET_SAFE | 33 | 0.0000 | 2.1818 | 4.3333 | 4.0000 | not reached |
| 0 | RESET_RECOVERY | 33 | 0.0000 | 2.6061 | 9.0000 | 6.3333 | not reached |
| 330 | CONTINUOUS_SAFE | 33 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | not reached |
| 330 | RESET_SAFE | 33 | 0.0000 | 1.1212 | 2.3333 | 1.6667 | not reached |
| 330 | RESET_RECOVERY | 33 | 0.0000 | 1.7273 | 9.0000 | 5.0000 | not reached |
| 660 | CONTINUOUS_SAFE | 33 | 33.3333 | 0.6667 | 1.0000 | 1.0000 | 70.0000 |
| 660 | RESET_SAFE | 33 | 69.6970 | 0.3333 | 1.0000 | 1.0000 | 70.0000 |
| 660 | RESET_RECOVERY | 33 | 69.6970 | 0.8788 | 7.0000 | 4.0000 | 70.0000 |
| 900 | CONTINUOUS_SAFE | 34 | 73.5294 | 0.2647 | 0.0000 | 0.1667 | 0.0000 |
| 900 | RESET_SAFE | 34 | 14.7059 | 1.1765 | 0.3333 | 0.1667 | 10.0000 |
| 900 | RESET_RECOVERY | 34 | 0.0000 | 1.8824 | 7.0000 | 4.0000 | not reached |

![Figure 14. Actual v16.5.5 continuous/UNKNOWN restart counts, including failed startup recovery. Different initialization priors prevent isolating history alone.](figures/v16_5_6/restart_count_traces.png)

Figure 14. Actual v16.5.5 continuous/UNKNOWN restart counts, including failed startup recovery. Different initialization priors prevent isolating history alone.

![Figure 15. Alternate split count agreement at 170 seconds can relapse under SAFE. This is a post-hoc segment analysis, not slot-level correctness.](figures/v16_5_6/post_recovery_relapse.png)

Figure 15. Alternate split count agreement at 170 seconds can relapse under SAFE. This is a post-hoc segment analysis, not slot-level correctness.

<!-- section:experimental_selection -->

## 6.4 Protected SAFE and DEV-selected research candidate

The existing protection rule prevents automatic promotion when DEV count results are under-count-confounded. In alternate v16.5.5 DEV, SAFE has Exact 31.8681% versus Guard/SEG 73.6264%, yet SAFE remains selected. v16.5.6 retains PRODUCTION_SAFE_SELECTED as the SAFE baseline and exports DEV_SELECTED_EXPERIMENTAL separately. Eligible variants are SAFE/Guard/SEG, ranked by DEV Exact descending, then MAE, over-rate and maximum error ascending; exact ties prefer SAFE, then Guard, then SEG. EMPTY_REF and the v16.4 post-hoc transition refiner are excluded. Parameters and their hash are frozen before TEST scores are computed.

Applying this fixed ranking to archived v16.5.5 DEV tables selects Guard for both fits, mapping to archived TEST scores of 70.5882%/MAE 0.294118 for original and 46.8750%/0.687500 for alternate. Original Guard is worse than SAFE TEST 85.2941%, so uniform improvement is not claimed. The ranking itself was introduced after these historical TEST results were reviewed: this is retrospective analysis, not independent validation. Future evaluation must freeze it before observing new-video outcomes. Research selection does not promote operational SAFE; RESET_DEV_SELECTED and RESET_TRANSITION_GUARD are evaluated separately at restart.

<!-- section:failures -->

## 7. Unsuccessful experiments and ablation analysis

Table 2 records alternatives toward the same goal without treating them as a fully controlled ablation study. Version conditions and additional observation costs differ, so individual causal contributions cannot be isolated from one number. Large CROP/HYBRID regressions and long EMPTY lock-ins under ZONE_MEMORY show the risk of letting auxiliary observations or memory replace primary evidence. TRANSITION_GUARD, SEG_ASSIST, and EMPTY_REF underperformed the safe baseline and were not adopted. v15.1 records approximately 7,451 additional segmentation inferences with zero resulting state changes, motivating lazy and selective scheduling.

These explanations are interpretations of retained error records rather than general laws validated on external datasets. Unsuccessful results and the reasons for rejection remain in the appendix.

| Experiment | TEST Exact (%) | Observation | Source |
| --- | --- | --- | --- |
| CROP | 14.71 | Less stable than FULL | research_history.json:v11 |
| HYBRID | 23.53 | Not adopted | research_history.json:v11 |
| ZONE_MEMORY | 14.71 | Long EMPTY lock-ins | research_history.json:v14 |
| TRANSITION_GUARD | 70.59 | Rejected below SAFE | research_history.json:v16.2 |
| SEG_ASSIST | 70.59 | Rejected below SAFE | research_history.json:v16.2 |
| EMPTY_REF (EMPTY_REF_ASSIST) | 0 | Rejected below SAFE | research_history.json:v16.2 |

<!-- section:transitions -->

## 8. Transition-error analysis

The five v16.2 TEST errors concentrate in two transition events. During entry, a tracker-ID change may reset motion history and make a maneuvering vehicle appear stationary. During departure, weak evidence may remain associated with an old owner and preserve occupancy after the vehicle leaves. The candidate selectively addresses these families through slot-motion retention and linked-CCTV EMPTY disagreement.

Four of five previous errors were removed in the record, but one over-count at 17:00 remains. Correlated checkpoints within a transition cannot be counted as multiple independent successes. Entry delay, departure delay, false-occupancy duration, ID-switch frequency, and event-level success should be reported separately. Unreported transition delays are not added without the underlying CSV evidence.

> **Figure placeholder 3. Early OCCUPIED after ID switch and ghost occupancy errors**
> No actual image file was found in the repository. Review GT-review or paper-ready artifacts from a future run before adding this figure. Do not substitute synthetic photographs or unverified before/after results.

> **Figure placeholder 4. SAFE and v16.4 Candidate comparison at matched timestamps**
> No actual image file was found in the repository. Review GT-review or paper-ready artifacts from a future run before adding this figure. Do not substitute synthetic photographs or unverified before/after results.

<!-- section:robustness -->

## 9. Degradation robustness

Robustness experiments address LOW_RES, SUN_GLARE, and MONITOR_STRIPES. v15.3 introduced conservative resampling/unsharp processing, highlight tone compression, and a causal temporal median over five past/current frames. v15.4 added per-CCTV percentile-based relative condition classification and lazy segmentation. From v16 onward, sampling balance, immediate O/E/U persistence, exclusion of U from binary metrics, Accuracy/Precision/F1/Occupied Recall/Empty Specificity/FP/FN evaluation, and sample repair were strengthened.

The record qualitatively describes reduced stripe energy and improved detection retention with temporal median, but complete condition-specific numerical tables are unavailable. No per-condition accuracy gains or external robustness numbers are invented. OCCUPIED-heavy GT and omitted supplemental review images remain limitations. Further experiments must report per-condition O/E sample sizes, excluded U counts, real versus synthetic degradation, RAW/ADAPTIVE cost, and actual state-change effects.

> **Figure placeholder 5. RAW versus ADAPTIVE under LOW_RES / SUN_GLARE / MONITOR_STRIPES**
> No actual image file was found in the repository. Review GT-review or paper-ready artifacts from a future run before adding this figure. Do not substitute synthetic photographs or unverified before/after results.

<!-- section:external -->

## 10. External validation and repeated-source stability design

### 10.1 Independent temporal validation and repeated-source stability

The Validation Center configures camera counts, GT, and post-cut warm-up to generate episode_evaluation_mask.csv and episode_metrics.csv for new local videos. repeat_stability.csv tests accumulated-state reproducibility, without increasing independent evidence. Hyundai SAFE 85.29%/0.1471 and Candidate 97.06%/0.0294 remain temporal/transition results on the same retained video. Evidence for promotion on independent continuous videos is still unavailable.

### 10.2 CNRPark-EXT External Baseline

The retained Full run contained 200,657 rows, including 15,820 duplicate camera1 rows and 19,307 duplicate camera4 rows. The same annotations were loaded from different extraction directories. This code removes 35,127 rows and evaluates 165,530 unique image–space pairs over 4,073 images, nine cameras, and three weather conditions, with no conflicting duplicate GT/predictions. Recomputed Accuracy is 95.0770%, Precision 93.4883%, Occupied Recall 97.4751%, Empty Specificity 92.3886%, F1 95.4401%, and Balanced Accuracy 94.9319%. TP/TN/FP/FN are 85,280/72,101/5,940/2,209. Thresholds were not adjusted after inspecting these results.

### 10.3 Slot classification versus image occupied-space counts

Image-level occupied-space Count Exact is 33.9553%, MAE 1.440953, median absolute error 1, P90 4, P95 5, and maximum 14. Over-count rate is 51.0435%, under-count 15.0012%, and Bias +0.916032. Slot accuracy near 95.08% and image-count Exact near 33.96% use different evaluation units and are compatible. Without GT for off-slot vehicles, these metrics cannot be described as total unique-vehicle counting accuracy.

### 10.4 Descriptive camera/weather observations

The camera Accuracy range is approximately 9.8594 percentage points (camera1 87.7118%–camera2 97.5711%); the weather range is approximately 1.0253 points. FP is approximately 2.689 times FN. camera1 has FP/FN 1,758/186, Count MAE 3.623894, and Bias +3.477876; camera9 has 1,054/148, MAE 2.084257, and Bias +2.008869. camera2 has Count Exact 79.6499% and MAE 0.229759. Neighboring detections intersecting slot geometry in real FP examples provide hypotheses for error analysis. Camera/weather groups differ in composition and sampling, so these ranges do not prove a causal geometry effect. Any future improvement informed by these external results requires a separate holdout.

Tables and real figures link to the CSVs, hashes, and selection history in [R6]. The montage combines camera1/camera9 FP, camera2 success, and TP/TN/FN. Real panels show detections, parking polygons, and selected-space GT/Pred.

### External spatial and image-count metrics

| Metric | Value |
| --- | --- |
| Images | 4073 |
| Unique image/spot pairs | 165530 |
| Duplicate rows removed | 35127 |
| accuracy | 95.0770% |
| precision | 93.4883% |
| occupied_recall | 97.4751% |
| empty_specificity | 92.3886% |
| f1 | 95.4401% |
| balanced_accuracy | 94.9319% |
| TP | 85280 |
| TN | 72101 |
| FP | 5940 |
| FN | 2209 |
| count_exact | 33.9553% |
| count_mae | 1.440953 |
| count_median_abs_error | 1.000000 |
| count_p90_error | 4.000000 |
| count_p95_error | 5.000000 |
| count_max_error | 14.000000 |
| over_count_rate | 51.0435% |
| under_count_rate | 15.0012% |
| count_bias | 0.916032 |

### Slot evaluation by camera

| Camera | Slot pairs | Accuracy % | F1 % | Recall % | Specificity % |
| --- | --- | --- | --- | --- | --- |
| camera1 | 15820 | 87.71 | 90.24 | 97.97 | 73.57 |
| camera2 | 4570 | 97.57 | 98.11 | 99.07 | 94.93 |
| camera3 | 10396 | 96.07 | 96.46 | 94.74 | 97.79 |
| camera4 | 19307 | 96.22 | 96.52 | 97.93 | 94.25 |
| camera5 | 24970 | 96.53 | 96.69 | 97.66 | 95.30 |
| camera6 | 25025 | 95.93 | 95.92 | 95.80 | 96.06 |
| camera7 | 24115 | 95.46 | 95.52 | 97.59 | 93.36 |
| camera8 | 24640 | 96.87 | 97.08 | 98.42 | 95.12 |
| camera9 | 16687 | 92.80 | 93.42 | 98.30 | 86.83 |

### Slot evaluation by weather

| Weather | Slot pairs | Accuracy % | F1 % | Recall % | Specificity % |
| --- | --- | --- | --- | --- | --- |
| OVERCAST | 50367 | 95.39 | 95.60 | 98.18 | 92.49 |
| RAINY | 42858 | 95.58 | 95.49 | 98.07 | 93.32 |
| SUNNY | 72305 | 94.56 | 95.32 | 96.74 | 91.63 |

### Image occupied-space counts by camera

| Camera | Images | Exact % | MAE | Bias | Max error |
| --- | --- | --- | --- | --- | --- |
| camera1 | 452 | 4.65 | 3.6239 | 3.4779 | 14 |
| camera2 | 457 | 79.65 | 0.2298 | 0.1247 | 2 |
| camera3 | 452 | 48.67 | 0.7367 | -0.4624 | 5 |
| camera4 | 449 | 40.31 | 0.9933 | 0.6726 | 6 |
| camera5 | 454 | 36.56 | 1.2004 | 0.5749 | 8 |
| camera6 | 455 | 33.63 | 1.1495 | -0.0725 | 6 |
| camera7 | 455 | 28.35 | 1.6418 | 1.1363 | 6 |
| camera8 | 448 | 27.90 | 1.3214 | 0.7991 | 7 |
| camera9 | 451 | 5.32 | 2.0843 | 2.0089 | 7 |

### Image occupied-space counts by weather

| Weather | Images | Exact % | MAE | Bias | Max error |
| --- | --- | --- | --- | --- | --- |
| OVERCAST | 1240 | 34.52 | 1.4153 | 1.1169 | 12 |
| RAINY | 1053 | 35.42 | 1.4046 | 1.0513 | 14 |
| SUNNY | 1780 | 32.70 | 1.4803 | 0.6961 | 11 |

![Spatial occupancy accuracy by camera and weather](figures/v16_5_2/external_accuracy_by_camera_weather.png)

Spatial occupancy accuracy by camera and weather

![Occupied-space count MAE by camera and signed image-level errors](figures/v16_5_2/external_count_error.png)

Occupied-space count MAE by camera and signed image-level errors

![Actual camera1/camera9 FP, camera2 success, and TP/TN/FN cases](figures/v16_5_2/external_validation_montage.jpg)

Actual camera1/camera9 FP, camera2 success, and TP/TN/FN cases

![camera1 FP case: spot 21000000876, GT=0, Pred=1](figures/v16_5_2/camera1_fp.jpg)

camera1 FP case: spot 21000000876, GT=0, Pred=1

![camera9 FP case: spot 29000012580, GT=0, Pred=1](figures/v16_5_2/camera9_fp.jpg)

camera9 FP case: spot 29000012580, GT=0, Pred=1

![camera2 TP case: spot 22000004350, GT=1, Pred=1](figures/v16_5_2/camera2_tp.jpg)

camera2 TP case: spot 22000004350, GT=1, Pred=1

<!-- section:history -->

## 11. Research lineage and reproducibility

v1-v4 explored detection/stabilization, smaller decision zones, high-resolution/tiled inference, and inference-frequency optimization. v4 is a partial proposal record without confirmed quantitative results. v5-v6 established points, duplicate linking, and perspective correction. v7-v10 progressed through baseline measurement, unsuccessful learned anchors, manual Voronoi constraints, and causal temporal tracking. v11-v14 compared FULL/CROP/HYBRID, protected auxiliary evidence, selective caching, and unsuccessful memory. v15-v16.2 strengthened transitions, segmentation/degradation conditions, GT balance, evidence-cache health, and regression checks. v16.3 added history/update tooling, v16.4 transition candidates, v16.5 expanded validation, v16.5.1 UI localization and this paper structure, and v16.5.2 deduplicated external spatial evidence.

Goals, changes, results, issues, and decisions for each version are retained in the [full research-history appendix](appendix/full_experiment_history.md). New evidence is integrated by method, result, error, and limitation rather than appended as release diaries. Release ZIPs and SHA-256 support source-distribution reproducibility but do not automatically freeze input videos or model weights.

v16.5.3 adds frozen-prediction cross-window comparison and observed retained-video limitations.

<!-- section:limitations -->

## 12. Limitations and future work

The central limitations are a single retained video, a small TEST set, temporal correlation around transitions, post-hoc candidate error analysis, and incomplete quantitative robustness evidence. Implementing v16.5 features does not establish success on new data. Public-image spatial evaluation alone cannot establish temporal-transition stability or field-operation safety. Failed-experiment numbers are not overinterpreted as perfectly controlled ablations.

Priorities are event-balanced evaluation on independent continuous videos, balanced O/E robustness samples, under-count and false-occupancy costs, runtime/cache costs, camera-count/viewpoint changes, and real before/after figures reviewed for privacy. SAFE_BASELINE promotion is considered only after independent validation; regressions keep the method a candidate with documented evidence.

External recalculation uses retained binary predictions and is distinct from a complete inference rerun. Reproducing 20 selected images partially audits the missing original-run settings snapshot but does not prove full reconstruction of that run. Camera and weather conditions are confounded, and group-range comparisons are descriptive. camera1/9 over-occupancy and overall image Count Exact 33.96% remain limitations. Spatial external results establish neither transition-rule generalization nor SAFE promotion.

Cross-window results show that final-window refinement did not resolve early errors. They compare retained count traces rather than a fresh run on another video. The original settings hash is retained; the complete settings file containing local video paths is not published.

Fresh split/restart experiments are computationally expensive. Actual v16.5.4/v16.5.5 results are archived separately; full-video v16.5.6 candidate results remain unreported. Future results must be reviewed with input/configuration hashes, sample sizes and failure status before integration into results tables. CANDIDATE is not promoted without independent-video temporal evidence.

Restart diagnosis requires fresh inference on the same fit/window and separation of initialization-prior, detector/tracker history and scene effects. Future evaluation needs slot GT, wrong-startup occupancy, arrivals/departures, missed detections and reconnection cases. This release does not blindly restore persisted occupancy state.

Actual v16.5.5 restart results expose initialization sensitivity and a failed recovery candidate. Different initial priors prevent isolating temporal history alone. Full-video v16.5.6 results remain unavailable; the archived 11-second replay only verifies the startup suppression regression.

<!-- section:conclusion -->

## 13. Conclusion

The retained research progressed toward protecting manual slot identities, combining spatial constraints with causal temporal decisions, and limiting auxiliary-detection authority. SAFE 85.29% improved to Candidate 97.06% on the retained video, while independent temporal generalization remains unverified. Spatial Accuracy 95.0770% is recomputed over 165,530 unique CNRPark-EXT pairs, with camera-specific FP and image-count errors reported separately. The bilingual Living Paper preserves identical evidence and limitations, integrating meaningful new methods, tables, and figures into the appropriate sections as releases evolve. Patches without research changes need not increase paper length.

Actual v16.5.5 restarts reveal initialization and persistence failures, with no established improvement from legacy recovery. v16.5.6 documents these failures, removes startup entry suppression and separates research selection and relapse evaluation. Operational recovery performance still requires full-video and independent validation.

<!-- section:references -->

## References and evidence

- [R1] [Canonical research history](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/research_history.json), retained records v1-v16.5.
- [R2] [v16.4 change record](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/CHANGES_v16_4.txt) and [v16.2 change record](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/CHANGES_v16_2.txt).
- [R3] [Ultralytics YOLO prediction documentation](https://docs.ultralytics.com/modes/predict/).
- [R4] [DSBD-Research MetaPKLot dataset and evaluation resources](https://github.com/DSBD-Research/MetaPKLot-Dataset), accessed 2026-10-04.
- [R5] [v16.5 implementation record](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/CHANGES_v16_5.txt).

- [R6] [Frozen external baseline, recalculated evidence and source hashes](data/external_v1652/external_summary.json), [source package](data/external_v1652/EXTERNAL_VALIDATION_TO_CHATGPT.zip), and [source/license attribution](data/external_v1652/SOURCE_AND_LICENSE.txt), recomputed 2026-10-05.

- [R7] [Retained-video cross-window evidence](data/windows_v1653/window_evaluation_summary.json), [frozen predictions and recalculation ZIP](data/windows_v1653/WINDOW_EVALUATION_TO_CHATGPT.zip), recalculated 2026-10-05.

- [R8] [Actual v16.5.5 restart predictions and provenance](data/restart_v1655/provenance.json), [40 recomputed startup-inclusive rows](data/restart_v1655/recomputed_restart_metrics.csv), recomputed from supplied artifacts on 2026-10-06.
