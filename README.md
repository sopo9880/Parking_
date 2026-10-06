# Parking Research Agent

Local research Agent for the Connect Hyundai parking-occupancy study.

Current release: **v16.5.7**
Protected algorithm baseline: **v16.2 SAFE_BASELINE**
Transition candidate: **v16.4 CANDIDATE**


## v16.5.7: 선택 시점 수동 초기화 / Exact-time manual initialization

1. 영상과 GT를 선택하고 검증 센터에서 **영상 보며 O/E/U 초기 상태 입력**을 누릅니다. 주차면 설정 JSON(현재 `slots.json` 또는 완료된 fit의 파일)을 선택합니다. 점 좌표는 변경하지 않습니다.
2. `0:00`, `5:30`, `11:00`, `15:00` 중 입력할 시각과 카메라를 선택하고 **해당 시점 초기값 / 영상 보기**를 누릅니다. 목록의 슬롯을 선택해 O(차량 있음), E(빈자리), U(미확정)를 지정합니다. 같은 global 슬롯을 보는 카메라는 한 값으로 초기화합니다.
3. **재시작 초기 상태 저장**으로 CSV를 저장합니다. 여러 시점은 같은 파일에 저장할 수 있습니다. 다른 시점 행은 보존합니다. CSV 열은 `time_sec,global_slot_id,initial_state`; 시각은 초, 상태는 O/E/U 또는 OCCUPIED/EMPTY/UNKNOWN입니다. 총수 GT는 개별 슬롯 상태를 알 수 없어 사용할 수 없습니다.
4. **재시작 실험에서만 해당 시점의 수동 초기 상태 사용**을 켜고 저장합니다. 직접 입력은 `operator_snapshot`, 슬롯별 정답의 해당 시점 입력은 `per_slot_gt_snapshot`으로 출처를 구분합니다. GT 기반 초기화는 논문에서 assisted initialization으로 보고합니다.
5. ALL-IN-ONE 또는 **완료된 분할 학습으로 재시작 비교 실행**을 실행합니다. 검출기·주차면 DEV 학습과 설정 선택에는 수동 값이 들어가지 않습니다. 입력 없는 시점은 수동 대조군만 생략하고 감사 파일에 남깁니다.

같은 새 관측으로 `RESET_UNKNOWN`(진짜 UNKNOWN), `RESET_MANUAL_INIT`, 기존 `CONTINUOUS_SAFE`를 비교합니다. 기존 `RESET_SAFE` 및 파일 식별자는 유지합니다. O는 움직임 없이 시작하며 첫 temporal window(기본 10초) 동안 초기 O의 즉시 EMPTY 해제를 막은 뒤 정상 SAFE 입·출차 판정을 적용합니다. E/U도 첫 프레임은 입력 그대로입니다. U는 검출 누락만으로 EMPTY로 바꾸지 않으며 긍정 입차 근거로 해결합니다. 시작 상태를 전체 구간에 고정하지 않습니다.

`manual_init_snapshot.csv`와 `manual_init_audit.json`에는 사용한 시각·출처·해시·O/E/U 수와 미래 GT 미사용 여부를 기록합니다. `*_occupancy_bounds.csv`는 confirmed/possible occupancy와 UNKNOWN 수를 제공합니다. UNKNOWN이 있는 범위의 일반 Exact/MAE·복구 시간은 비워 두고 lower-bound count 통계와 범위 포함률을 구분합니다. 범위 포함은 정확한 예측이 아닙니다.

**실제 v16.5.6 결과:** 두 학습 분할 × 네 재시작에서 수정 복구와 SAFE의 count·state/phase/score·전환이 모두 동일했습니다. [144개 scoped 결과 재계산](paper/data/restart_v1656/recomputed_metrics.csv)을 한·영 논문에 추가했습니다. 15분 재시작 후 10초를 제외한 original RESET_SAFE/복구는 N=33, Exact 0%, MAE 1.363636입니다. 기존 주 TEST의 N=34 점수와 표본 범위를 구분합니다. **v16.5.7 수동 초기화 전체 성능은 아직 미측정입니다.** SAFE 자동 승격은 없습니다.

Manual snapshots are optional and off by default. Only the exact restart-time slot dictionary enters frozen restart inference. Future GT scores outputs only. Count-only GT is rejected; missing slots remain UNKNOWN; missing-time manual runs are explicitly skipped. Initial-O clearing is deferred for one temporal window, then ordinary automatic SAFE logic applies. This assisted-information condition is not an equal-information ablation or independent validation. Stale-state persistence is deferred. Both language papers, immutable release snapshots and PDFs follow the same evidence.

## v16.5.6: 실제 재시작 결과와 연구 후보 선택

- 한·영 Living Paper에 실제 v16.5.5 재시작 결과, 실패한 복구 후보, G002/G027/G019 전환과 재오류 분석을 추가했습니다. [재계산 근거](paper/data/restart_v1655/provenance.json)를 제공합니다. Original 설정의 15분 연속 SAFE는 85.29%, UNKNOWN 재시작 SAFE는 0%이며, 초기 사전 정보 차이도 함께 기록합니다.
- `PRODUCTION_SAFE_SELECTED`는 SAFE 기준선입니다. `DEV_SELECTED_EXPERIMENTAL`은 SAFE/Guard/SEG를 DEV Exact·MAE·과대율·최대 오차로 선택하고 TEST 평가 전에 별도 설정을 동결합니다. 운영 SAFE 자동 승격은 아닙니다. 기존 `SELECTED` 파일은 유지합니다.
- `RESET_RECOVERY`는 정상 SAFE 입차 판정을 막지 않는 수정 후보입니다. 기존 차단 후보는 `RESET_RECOVERY_V1655`, Guard는 `RESET_TRANSITION_GUARD`, DEV 후보는 `RESET_DEV_SELECTED`로 함께 비교합니다. 모든 새 후보는 같은 재시작 관측을 사용합니다.
- 3회 count Exact 뒤의 첫 재오류 시점, 재오류 횟수, 이후 MAE, 최장/마지막 연속 Exact 표본 수를 기록합니다. 3회 일치는 영구 복구나 슬롯 GT 정확도를 뜻하지 않습니다.
- **이미 v16.5.5를 실행했다면** 아래의 완료된 분할 학습 버튼에서 실제 `original` 또는 `alternate` 폴더를 선택해 검출기/공간 학습을 반복하지 않고 다음 재시작 비교를 실행하세요. 새 DEV 후보 선택도 저장된 DEV 표만 사용합니다.

31개 단위/UI 테스트, 실제 저장 관측 900–910초의 회귀 재생과 v16.5.5 재시작 80행 재검산을 수행합니다. 수정 후보의 전체 영상 성능은 미측정이며, 기존 결과와 구분합니다.

## 재시작 / Cold start (v16.5.5부터 지원)

검증 센터의 **새 분할 실험: 연속 실행·재시작·초기 복구 비교**가 기본 활성화됩니다. ALL-IN-ONE의 각 fresh split 학습이 끝나면 설정을 동결한 채 0:00 / 5:30 / 11:00 / 15:00 및 TEST 시작 지점에서 새 추적·상태로 추론합니다. 재시작 시점을 변경할 수 있으며 각 구간은 최대 5분 30초입니다.

**이미 v16.5.4 학습을 완료했다면**, 영상과 GT를 선택하고 검증 센터의 **완료된 분할 학습으로 재시작 비교 실행**을 누른 뒤 `.../split_experiments/original` 또는 `alternate` 폴더를 선택하세요. 학습된 템플릿이 포함된 실제 실행 폴더가 필요합니다. 업로드 ZIP은 템플릿을 생략하므로 이 용도로 쓸 수 없습니다. 검출기 탐색·공간 학습을 반복하지 않고 재시작 추론만 수행합니다.

Restart extraction starts at the requested source timestamp with new detector trackers, motion history, AUX memory, and segmentation recovery frame history. Learned geometry/templates and fitted detector settings remain frozen; the selected-parameter hash is checked. Initial occupancy is UNKNOWN, and unlabeled appearance is not assumed to mean OCCUPIED. Continuous and reset modes therefore represent different operational initialization conditions, not a pure history-only causal ablation.

`CONTINUOUS_SAFE`, `CONTINUOUS_CANDIDATE`, `RESET_SAFE`, `RESET_CANDIDATE`, `RESET_RECOVERY` are reported separately. `RESET_RECOVERY` is an experimental causal startup candidate: repeated stationary FULL detections over at least 10 seconds, with multi-camera support or stronger single-camera confidence, can initialize occupancy during the first 30 seconds. Normal SAFE temporal rules follow; existing SAFE defaults remain unchanged. No production promotion or guaranteed 10–30 second recovery is claimed.

Each unique `restart_experiments/run_.../` includes `restart_metrics.csv`, per-start count/slot traces, state differences and audits. The standalone button creates `RESTART_EVALUATION_TO_CHATGPT.zip`; ALL-IN-ONE includes restart outputs in its normal upload ZIP. Metrics include startup-inclusive Exact/MAE, first 30/60/120 s errors, first exact, three consecutive available GT checkpoints for stable exact (with confirmation time), never-stabilized flag and post-stability Exact. A separate matched post-restart 10 s warm-up row is included; initial errors are not removed from startup metrics. Count stability does not establish slot correctness or permanent stability. Failures preserve partial outputs and are flagged.

The supplied v16.5.4 run has been recomputed: original continuous TEST SAFE 85.29% / Candidate 97.06%; alternate startup TEST SAFE 9.375% / MAE 1.1875, Candidate 9.375% / MAE 1.21875, guard 46.875% / MAE 0.6875. Those windows also differ in fitting and content, so they suggest a restart/initialization problem without proving its sole cause. Those formerly pending scores are now archived, alongside actual v16.5.6 results in the latest paper. Same-video results are not independent validation.

## ALL-IN-ONE: 실제 DEV/TEST 분할 실험 / Fresh split experiments

**검증 센터 → 기존·대체 분할을 각각 새로 학습·평가**가 기본 활성화됩니다. 설정을 저장한 뒤 ALL-IN-ONE을 실행하면 기존 파이프라인·강건성·구간별 통계에 이어 두 개의 별도 실행을 수행합니다.

| 분할 | DEV | TEST |
| --- | --- | --- |
| Original | 0:00–15:00 | 15:00–20:30 |
| Alternate (기본값) | 5:30–20:30 | 0:00–5:30 |

대체 DEV/TEST 구간은 검증 센터에서 변경하고 프로필과 함께 저장할 수 있습니다. 구간은 겹치면 안 되며 영상 평가 길이 안에 있어야 합니다. 각 실행은 공유 캐시와 기존 검출기 선택을 사용하지 않고, DEV 영상으로 검출기 설정을 선택하고 주차면 앵커·템플릿을 다시 학습합니다. YOLO 가중치 자체를 재학습하지는 않습니다. 상태 변형 선택에는 DEV 정답만 전달하고 슬롯 이벤트 정답은 제외합니다. 설정을 확정·기록한 뒤 TEST 정답으로 SAFE_BASELINE / TRANSITION_GUARD / SEG_ASSIST / EMPTY_REF / SELECTED / CANDIDATE를 평가합니다.

Each split uses a fresh subprocess and separate detector/geometry/evidence directories. Detector configuration and temporal-variant selection receive DEV labels only. Adaptive quality thresholds are fitted on DEV crops; TEST crops are only classified against them. Operator-supplied manual points, initial states, pretrained weights and the search grid are shared fixed inputs. Chronological inference still begins at time zero; online state history can span intervals. The reverse split is an **offline same-video evaluation**, using historically reviewed data, not an independent future-video hold-out.

`output/run_..._ALL_IN_ONE/split_experiments/` contains `original/`, `alternate/`, `split_experiment_comparison.csv` and `suite_summary.json`. Each child has `split_final_metrics.csv`, DEV-only selection tables, `selection_freeze.json`, `split_audit.json`, full traces and a worker log. Exact, MAE, maximum error and Over/Under are reported separately for DEV/TEST. The upload ZIP includes results and audits, excluding the private worker request and learned template images. A failed child preserves partial outputs and marks the suite failed.

새 분할은 두 번의 전체 검출기 탐색·학습·추론으로 시간이 더 걸립니다. 기존 **저장된 SAFE/Candidate 구간 비교** 버튼은 빠른 통계 재계산 기능으로 계속 유지됩니다. 제공된 v16.5.4 실행 결과는 현재 논문에서 별도 재검산 표로 보고합니다. 새로운 v16.5.5 재시작·복구 점수는 실행 후 생성됩니다.

## 한국어 / English UI

기본 언어는 한국어입니다. 메인 화면의 **설정 → 언어**에서 `ko` 또는 `en`을 선택하면 열린 메인 UI와 검증 센터가 즉시 바뀝니다. `settings.json`의 `ui.language`에 저장되어 다음 실행과 자동 업데이트 후에도 유지됩니다. 영상 경로와 입력 폼은 유지되며 `SAFE_BASELINE`, `CANDIDATE`, `OCCUPIED/EMPTY/MANEUVERING` 및 CSV 열·결과 파일명은 그대로 사용합니다.

The default language is Korean. Choose **Settings → Language → ko/en** to update open windows immediately. The preference persists across restarts and updates. Advanced settings remain available as JSON. Machine identifiers, CSV schemas, filenames, research logs and image-overlay identifiers stay canonical.

## Living Paper

[한국어 논문](paper/paper_ko.md) | [English paper](paper/paper_en.md) | [유지 규칙 / Maintenance](paper/README.md) | [전체 이력](paper/appendix/full_experiment_history.md)

릴리스가 발전하면 새 방법·결과·오류 분석·표·그림을 기존 본문에 통합합니다. 두 언어는 공통 결과 CSV와 짝지어진 원고를 사용하며 manifest/workflow가 동기화를 검사합니다. 연구 변경이 없는 patch는 분량을 늘리지 않아도 됩니다. 매 릴리스의 한·영 snapshot은 보존하고 수정하지 않습니다.

Research changes grow the appropriate sections; patches without research changes need no padding. Update both reviewed language paragraphs, evidence tables, figure provenance and `paper_manifest.json`, then generate and check both papers before publishing. The release workflow builds Korean/English PDFs, Markdown snapshots, a source ZIP and SHA-256 assets.

CNRPark-EXT frozen predictions were recalculated after removing **35,127** duplicate image/spot pairs: **4,073 images / 165,530 unique pairs**, occupancy Accuracy **95.08%**, F1 **95.44%**; image Count Exact **33.96%**, MAE **1.441**. Count means occupied annotated parking spaces. These are external spatial results, separate from retained SAFE **85.29% / MAE 0.1471** and Candidate **97.06% / MAE 0.0294** temporal results. Independent temporal validation remains pending.

실제 TP/TN/FP/FN, 카메라별·날씨별 이미지 33장, camera1/9 FP와 camera2 성공 사례, montage 및 그래프를 한·영 논문에 추가했습니다. 전체 4,073장 추론을 다시 실행한 것은 아니며, 기존 예측 재계산과 대표 20장 재추론 일치 확인을 구분해 기록했습니다. 외부 결과로 임계값을 조정하지 않습니다.

[Frozen external evidence](paper/data/external_v1652/external_summary.json) | [Auditable evidence ZIP](paper/data/external_v1652/EXTERNAL_VALIDATION_TO_CHATGPT.zip). This reviewed derivative contains public CNRPark-EXT benchmark annotations, predictions and selected overlays, with source/license attribution; original private CCTV imagery is unavailable and remains a placeholder.

## Verification

`python self_test.py`, `python -m unittest discover -s tests -v`, and `python tools/build_paper.py --check`, and `python tools/check_external_evidence.py` validate source behavior, language persistence, open-form preservation, identifier/cache compatibility, and paper synchronization. On Linux use `xvfb-run -a` for UI tests. PDF build instructions are in [paper/README.md](paper/README.md).

## Start on Windows
1. Extract the release ZIP.
2. Double-click `START.cmd` once. It creates `.venv` and installs dependencies.
3. Later launches can use `RUN.cmd`.

## Main workflow
- Select the test video and Ground Truth CSV.
- Open **Validation Center** for a new video/environment.
- Set CCTV count (1-12), optional cut boundaries, and repeated-source settings.
- Configure 4-point ROI, slot points and duplicate-slot links.
- Run **ALL-IN-ONE**.
- ALL-IN-ONE evaluates the protected v16.2 SAFE_BASELINE and then automatically evaluates v16.4 CANDIDATE.
- Upload `UPLOAD_TO_CHATGPT.zip` for analysis.

## Validation Center

### New/local video
Dataset Profiles can save:
- video / GT / optional slot-GT paths
- CCTV count
- GT sampling interval
- cut boundaries + post-cut warm-up
- repeat period/count

**Create Count-GT Template** creates dynamic `cctv1_count ... cctvN_count` columns.
**Open GT Labeler** shows each sampled frame and lets you enter CCTV counts, total occupied count, notes, U and CUT directly.

### Concatenated clips
For separate clips concatenated into one file, enter cut boundaries such as:

`4:00, 8:00, 12:00, 16:00`

v16.5 excludes the configured post-cut warm-up from episode metrics.

### Same source clip repeated
For the same 4-minute source repeated five times:
- repeat period = `240`
- repeat count = `5`

This produces a reproducibility/state-accumulation report. It is **not** independent validation.

## Automatic v16.4 evaluation
The separate `RUN_V16_4_CANDIDATE.cmd` remains available, but ALL-IN-ONE now runs the candidate automatically.

Retained development-video result:
- v16.2 SAFE: **85.29% Exact / 0.1471 MAE**
- v16.4 Candidate: **97.06% Exact / 0.0294 MAE**

v16.4 remains CANDIDATE because its rules were developed after retained-video error analysis.

## Public external validation
Validation Center can automatically prepare **MetaPKLot / CNRPark-EXT** from the official upstream repository.

Modes:
- **Quick**: small selected subset for fast smoke/generalization testing.
- **Standard**: more cameras, dates and weather.
- **Full**: explicit opt-in full upstream shallow clone; may require several GB.

The Agent downloads/resumes files, extracts MetaPKLot spot annotations, converts them to `external_gt_spots.csv`, runs occupancy validation, reports overall/camera/weather metrics, and creates `EXTERNAL_VALIDATION_TO_CHATGPT.zip`.

This public benchmark measures **spatial/environment generalization**. It does not replace continuous Hyundai-video temporal-transition validation.

Source: MetaPKLot / CNRPark-EXT. CNRPark-EXT is distributed under ODbL v1.0; consult the upstream README before redistribution.

## Cache behavior
The bottom status area includes a CACHE summary. Warm-cache runs can reuse detector tuning, slot learning, evidence, segmentation and robustness results. A fast warm-cache run therefore means expensive evidence was safely reused; it does not imply the detector itself became tens of times faster.

## Auto Update
GitHub Releases ZIP + SHA-256 verification remains enabled.

Preserved during update:
- `settings.json`
- `slots.json`
- `rois.json`
- `ui_state.json`
- `sample_ground_truth.csv`
- `work/`
- `output/`
- `.venv/`
- `user_data/` (profiles/public datasets included)

## Version-lineage rule
`research_history.json` contains only the paper/research lineage. The separate field-operation v20+ lineage is intentionally not mixed into this timeline.

## Evaluation windows / 다중 구간 평가

검증 센터의 평가 구간에 `0:00-5:30; 5:30-11:00; 11:00-15:00; 15:00-20:30`처럼 입력하고 설정/프로필을 저장하세요. **저장된 SAFE / Candidate 구간 비교**에서 두 결과가 있는 실행 폴더를 선택하면 학습·추론 없이 구간별 Exact/MAE/최대 오차/과대·과소 추정과 개선폭, 평균·표준편차·최상/최악 구간을 저장합니다. 새 실행에서는 Candidate 평가 후 자동 생성하며 UPLOAD_TO_CHATGPT.zip에도 포함됩니다.

Outputs: `window_metrics.csv`, `window_comparison.csv`, `window_evaluation_summary.json`, `WINDOW_EVALUATION_REPORT.txt`, `WINDOW_EVALUATION_TO_CHATGPT.zip`. CLI: `python tools/evaluate_windows.py RUN_FOLDER --windows "0:00-5:30;5:30-11:00"`. Evaluation windows do not change DEV tuning boundaries or model parameters. Continuous saved state is preserved; paired samples use the same validity/warm-up mask. Intervals include starts and exclude ends except the largest end. ALL uses the full trace once. Macro summaries exclude ALL and flag overlaps. Historical DEV/error-reviewed windows are stability evaluation, not independent hold-out.

The retained run shows gains in **1/4 windows**: the final window retains 85.29% → 97.06%, while the first three show no improvement. Full-trace Exact is **34.96% → 38.21%** (123 pairs). These are recalculated saved predictions, not new inference or SAFE promotion. See the bilingual paper for early-window limitations and evidence hashes.
