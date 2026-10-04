# Parking Research Agent

Local research Agent for the Connect Hyundai parking-occupancy study.

Current release: **v16.5.1**
Protected algorithm baseline: **v16.2 SAFE_BASELINE**
Transition candidate: **v16.4 CANDIDATE**


## 한국어 / English UI

기본 언어는 한국어입니다. 메인 화면의 **설정 → 언어**에서 `ko` 또는 `en`을 선택하면 열린 메인 UI와 검증 센터가 즉시 바뀝니다. `settings.json`의 `ui.language`에 저장되어 다음 실행과 자동 업데이트 후에도 유지됩니다. 영상 경로와 입력 폼은 유지되며 `SAFE_BASELINE`, `CANDIDATE`, `OCCUPIED/EMPTY/MANEUVERING` 및 CSV 열·결과 파일명은 그대로 사용합니다.

The default language is Korean. Choose **Settings → Language → ko/en** to update open windows immediately. The preference persists across restarts and updates. Advanced settings remain available as JSON. Machine identifiers, CSV schemas, filenames, research logs and image-overlay identifiers stay canonical.

## Living Paper

[한국어 논문](paper/paper_ko.md) | [English paper](paper/paper_en.md) | [유지 규칙 / Maintenance](paper/README.md) | [전체 이력](paper/appendix/full_experiment_history.md)

릴리스가 발전하면 새 방법·결과·오류 분석·표·그림을 기존 본문에 통합합니다. 두 언어는 공통 결과 CSV와 짝지어진 원고를 사용하며 manifest/workflow가 동기화를 검사합니다. 연구 변경이 없는 patch는 분량을 늘리지 않아도 됩니다. 매 릴리스의 한·영 snapshot은 보존하고 수정하지 않습니다.

Research changes grow the appropriate sections; patches without research changes need no padding. Update both reviewed language paragraphs, evidence tables, figure provenance and `paper_manifest.json`, then generate and check both papers before publishing. The release workflow builds Korean/English PDFs, Markdown snapshots, a source ZIP and SHA-256 assets.

Current image evidence is unavailable, so figures use explicit descriptions instead of fabricated images. Retained SAFE **85.29% / MAE 0.1471** and Candidate **97.06% / MAE 0.0294** are historical, same-video results. Independent temporal validation and v16.5 external runtime results remain pending.

## Verification

`python self_test.py`, `python -m unittest discover -s tests -v`, and `python tools/build_paper.py --check` validate source behavior, language persistence, open-form preservation, identifier/cache compatibility, and paper synchronization. On Linux use `xvfb-run -a` for UI tests. PDF build instructions are in [paper/README.md](paper/README.md).

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
