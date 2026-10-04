# Causal Parking-Space Occupancy Estimation from Re-recorded Multi-CCTV Video: Parking Research Agent Living Paper

Living Paper | v16.5.1 | Research revision retained-v16.5-r1

[한국어](paper_v16.5.1_ko.md) | [English](paper_v16.5.1_en.md)

Research draft. Numbers below are reported values from retained records, not newly measured in this release. Authors, affiliation, and source-video distribution permissions are unspecified.

<!-- section:abstract -->

## Abstract

This paper documents the development of a research agent for estimating occupied parking spaces from phone recordings of a CCTV monitor. The setting combines shake, perspective distortion, low resolution, sun glare, monitor stripes, and missed vehicle detections. The method combines fixed manual slot points, perspective correction, Voronoi spatial constraints, FULL-first detection with selective AUX recovery, and causal temporal state tracking. Retained research records report TEST Exact 85.29% and MAE 0.1471 for v16.2 SAFE_BASELINE, and 97.06% and MAE 0.0294 for v16.4 CANDIDATE after track-switch entry holding and weak-owner departure release. Candidate rules were developed after reviewing errors on the same validation video; these results therefore cannot establish independent generalization. v16.5 provides tools for new-video, episode, repeated-source, and public spatial evaluation without reporting new accuracy. v16.5.1 adds bilingual UI and synchronized Living Papers as a presentation/documentation patch, without a new algorithm or performance claim.

Keywords: parking occupancy, re-recorded CCTV, causal state tracking, multi-CCTV, robustness, Living Paper.

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

> **Figure placeholder 1. Actual re-recorded CCTV monitor environment**
> No actual image file was found in the repository. Review GT-review or paper-ready artifacts from a future run before adding this figure. Do not substitute synthetic photographs or unverified before/after results.

> **Figure placeholder 2. Manual slot points, perspective correction, and Voronoi regions**
> No actual image file was found in the repository. Review GT-review or paper-ready artifacts from a future run before adding this figure. Do not substitute synthetic photographs or unverified before/after results.

<!-- section:methods -->

## 4. Proposed method

### 4.1 Spatial assignment and FULL-first observations

After perspective correction, YOLO observations are assigned under Voronoi constraints fixed by manual slot points. Learned anchors do not replace those points and are restricted by shift and sample conditions. Duplicate suppression and one-to-one assignment reduce multiple-slot assignment of a physical vehicle. FULL detection is authoritative. Selective CROP/AUX runs when primary observations are weak, missing, visually suspicious, or due for sparse auditing. AUX may recover missed FULL evidence but cannot erase valid FULL evidence.

### 4.2 Causal temporal states

Online tracking and a rolling ten-second history distinguish maneuvers from stable occupancy. The first ten seconds are warm-up; future frames are excluded. Long-stationary candidate slots are advisory until operator review and are not automatically added to the physical slot count. Cached geometry and appearance-template metadata are restored together; missing templates or implausible evidence distributions trigger rejection and recomputation.

### 4.3 Track switches and departure candidates

v16.4 ENTRY_TRACK_SWITCH_HOLD retains recent slot-motion context across tracker-ID switches. A newly stationary-looking track following substantial motion cannot immediately confirm OCCUPIED before sufficient stabilization. LEAVING_WEAK_OWNER_RELEASE requires the baseline's LEAVING transition, weak mean owner confidence, meaningful movement, and an EMPTY vote from a linked CCTV before releasing ghost occupancy. A reacquisition settling interval blocks immediate re-entry. These rules are CANDIDATE postprocessing and neither automatically promote nor modify SAFE_BASELINE [R2].

<!-- section:protocol -->

## 5. Experimental protocol and evidence status

The retained protocol selects parameters on DEV before 15:00, then evaluates TEST from 15:00 to 20:30 after selection. The first ten seconds are treated as warm-up. The record contains 34 TEST checkpoints, with possible temporal correlation between adjacent samples. Maximum error, under/over-count rates, and event-balanced transition metrics complement Exact and MAE.

Numbers here summarize reported results in research_history.json and change records. This documentation patch does not rerun the original video or recover complete runtime artifacts. sample_ground_truth.csv is a reference, not a replacement for the source video, parameters, and predictions. Reproduction must retain version, input hashes, settings, ROIs, slots, GT, DEV selection, TEST predictions, cache audits, and environment. Because v16.4 rules were designed after reviewing TEST errors on the same video, its absolute TEST score is not presented as an untouched holdout result.

<!-- section:results -->

## 6. Results

Table 1 summarizes retained occupied-count results [R1, R2], preserving original MAE rounding. v12 and v13 share Exact but differ in MAE and maximum error. v16.2's 29/34 and v16.4's 33/34 differ by approximately 11.77 percentage points; MAE decreases from 0.1471 to 0.0294. Both report maximum error 1 and zero under-count. The candidate reports 2.94% over-count, while DEV metrics are recorded as unchanged.

New-video and public-data accuracy for v16.5 and v16.5.1 is unreported. These historical numbers must not be treated as evidence of a newly completed experiment. Early versions lacking quantitative records are not assigned fabricated values.

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
| v16.5 external/new video | - | - | - | pending_runtime_results |
| v16.5.1 | - | - | - | presentation_patch_no_new_measurement |

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

The v16.5 Validation Center supports Dataset Profiles for new local videos, camera counts, dynamic count-GT templates, and interactive labeling. Cut times and post-cut warm-up exclusions generate episode_evaluation_mask.csv and episode_metrics.csv. For a repeated four-minute source, repeat_period_sec=240 produces repeat_stability.csv and REPEAT_STABILITY_REPORT.txt. Repetition tests state-accumulation reproducibility and stability; it does not increase independent evidence.

The external spatial path prepares selected official MetaPKLot/CNRPark-EXT images and slot annotations, then reports occupancy metrics by camera and weather. Quick/Standard use restricted samples; Full is explicitly selected for broader preparation. Actual files, splits, exclusions, model settings, and sample sizes must accompany results. Independent temporal evaluation must freeze the algorithm/settings first, collect distinct dates, cameras, videos, and real entry/departure events apart from the error-analysis video, then compare SAFE and Candidate under one protocol. Execution results remain unreported, and Candidate promotion requirements are not claimed as satisfied.

<!-- section:history -->

## 11. Research lineage and reproducibility

v1-v4 explored detection/stabilization, smaller decision zones, high-resolution/tiled inference, and inference-frequency optimization. v4 is a partial proposal record without confirmed quantitative results. v5-v6 established points, duplicate linking, and perspective correction. v7-v10 progressed through baseline measurement, unsuccessful learned anchors, manual Voronoi constraints, and causal temporal tracking. v11-v14 compared FULL/CROP/HYBRID, protected auxiliary evidence, selective caching, and unsuccessful memory. v15-v16.2 strengthened transitions, segmentation/degradation conditions, GT balance, evidence-cache health, and regression checks. v16.3 added history/update tooling, v16.4 transition candidates, v16.5 expanded validation, and v16.5.1 UI localization and this paper structure.

Goals, changes, results, issues, and decisions for each version are retained in the [full research-history appendix](../appendix/full_experiment_history.md). New evidence is integrated by method, result, error, and limitation rather than appended as release diaries. Release ZIPs and SHA-256 support source-distribution reproducibility but do not automatically freeze input videos or model weights.

<!-- section:limitations -->

## 12. Limitations and future work

The central limitations are a single retained video, a small TEST set, temporal correlation around transitions, post-hoc candidate error analysis, and incomplete quantitative robustness evidence. Implementing v16.5 features does not establish success on new data. Public-image spatial evaluation alone cannot establish temporal-transition stability or field-operation safety. Failed-experiment numbers are not overinterpreted as perfectly controlled ablations.

Priorities are event-balanced evaluation on independent continuous videos, balanced O/E robustness samples, under-count and false-occupancy costs, runtime/cache costs, camera-count/viewpoint changes, and real before/after figures reviewed for privacy. SAFE_BASELINE promotion is considered only after independent validation; regressions keep the method a candidate with documented evidence.

<!-- section:conclusion -->

## 13. Conclusion

The retained research progressed toward protecting manual slot identities, combining spatial constraints with causal temporal decisions, and limiting auxiliary-detection authority. SAFE 85.29% improved to Candidate 97.06% on the retained video, while generalization remains unverified. The bilingual Living Paper preserves identical evidence and limitations, integrating meaningful new methods, tables, and figures into the appropriate sections as releases evolve. Patches without research changes need not increase paper length.

<!-- section:references -->

## References and evidence

- [R1] [Canonical research history](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/research_history.json), retained records v1-v16.5.
- [R2] [v16.4 change record](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/CHANGES_v16_4.txt) and [v16.2 change record](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/CHANGES_v16_2.txt).
- [R3] [Ultralytics YOLO prediction documentation](https://docs.ultralytics.com/modes/predict/).
- [R4] [DSBD-Research MetaPKLot dataset and evaluation resources](https://github.com/DSBD-Research/MetaPKLot-Dataset), accessed 2026-10-04.
- [R5] [v16.5 implementation record](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/CHANGES_v16_5.txt).

