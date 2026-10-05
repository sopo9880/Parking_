# Parking Research Agent

Local research Agent for the Connect Hyundai parking-occupancy study.

Current release: **v16.5.4**
Protected algorithm baseline: **v16.2 SAFE_BASELINE**
Transition candidate: **v16.4 CANDIDATE**


## ALL-IN-ONE: 실제 DEV/TEST 분할 실험 / Fresh split experiments

**검증 센터 → 기존·대체 분할을 각각 새로 학습·평가**가 기본 활성화됩니다. 설정을 저장한 뒤 ALL-IN-ONE을 실행하면 기존 파이프라인·강건성·구간별 통계에 이어 두 개의 별도 실행을 수행합니다.

| 분할 | DEV | TEST |
| --- | --- | --- |
| Original | 0:00–15:00 | 15:00–20:30 |
| Alternate (기본값) | 5:30–20:30 | 0:00–5:30 |

대체 DEV/TEST 구간은 검증 센터에서 변경하고 프로필과 함께 저장할 수 있습니다. 구간은 겹치면 안 되며 영상 평가 길이 안에 있어야 합니다. 각 실행은 공유 캐시와 기존 검출기 선택을 사용하지 않고, DEV 영상으로 검출기 설정을 선택하고 주차면 앵커·템플릿을 다시 학습합니다. YOLO 가중치 자체를 재학습하지는 않습니다. 상태 변형 선택에는 DEV 정답만 전달하고 슬롯 이벤트 정답은 제외합니다. 설정을 확정·기록한 뒤 TEST 정답으로 SAFE_BASELINE / TRANSITION_GUARD / SEG_ASSIST / EMPTY_REF / SELECTED / CANDIDATE를 평가합니다.

Each split uses a fresh subprocess and separate detector/geometry/evidence directories. Detector configuration and temporal-variant selection receive DEV labels only. Adaptive quality thresholds are fitted on DEV crops; TEST crops are only classified against them. Operator-supplied manual points, initial states, pretrained weights and the search grid are shared fixed inputs. Chronological inference still begins at time zero; online state history can span intervals. The reverse split is an **offline same-video evaluation**, using historically reviewed data, not an independent future-video hold-out.

`output/run_..._ALL_IN_ONE/split_experiments/` contains `original/`, `alternate/`, `split_experiment_comparison.csv` and `suite_summary.json`. Each child has `split_final_metrics.csv`, DEV-only selection tables, `selection_freeze.json`, `split_audit.json`, full traces and a worker log. Exact, MAE, maximum error and Over/Under are reported separately for DEV/TEST. The upload ZIP includes results and audits, excluding the private worker request and learned template images. A failed child preserves partial outputs and marks the suite failed.

새 분할은 두 번의 전체 검출기 탐색·학습·추론으로 시간이 더 걸립니다. 기존 **저장된 SAFE/Candidate 구간 비교** 버튼은 빠른 통계 재계산 기능으로 계속 유지됩니다. v16.5.4는 기능·누수 방지 검증을 완료한 배포이며, 실제 전체 CCTV 재실험 수치는 실행 후 생성됩니다. 새 점수는 아직 논문에 보고하지 않았습니다.

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
