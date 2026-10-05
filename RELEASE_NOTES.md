# Parking Research Agent v16.5.3

- 검증 센터에서 다중 평가 구간 지정·설정/프로필 유지, 저장된 SAFE/Candidate 결과 재비교 버튼 추가. 학습·파라미터 선택·추론 없이 비교합니다. 새 실행 후 자동 생성하고 업로드 ZIP에 포함합니다.
- 구간별 Exact, MAE, 최대 오차, Over/Under, 공통 표본 N, 누락 표본 수, Candidate 개선폭, 구간 평균·모표준편차·최상/최악 구간과 개선 구간 수를 저장합니다.
- 기본 0–5:30 / 5:30–11:00 / 11:00–15:00 / 15:00–20:30 및 전체. 시작 포함·끝 제외, 가장 큰 끝은 포함. 원래 연속 상태와 초기/cut warm-up을 유지합니다. ALL은 표본을 한 번씩 집계하며 겹치는 사용자 구간은 표시합니다.
- 실제 보존 영상 결과: 앞 두 구간 Exact 0%, 셋째 58.33%로 두 방법 동일. 마지막 SAFE 85.29% → Candidate 97.06%. 개선 1/4; 전체 123개 표본 Exact 34.96% → 38.21%, MAE 1.0976 → 1.0650. 한·영 Living Paper에 실제 표·그래프·근거를 반영하고 이전 snapshot은 보존했습니다.

Evaluation-only cross-window stability on frozen retained predictions, not new inference. Historical DEV/error-reviewed intervals are not independent hold-out. No threshold changes or SAFE promotion; v16.5.2 external baseline evidence remains frozen. Full settings with local video paths are omitted from evidence packages; hashes and evaluation masks are recorded.

Validation: 17 unit/UI tests, protected engine self-test, frozen external evidence recalculation, complete window evidence recalculation, bilingual paper/immutable snapshots, PDF rendering and release checksums.

---

# Parking Research Agent v16.5.2

## 한국어 / English

- 외부 주석의 동일 image/spot 중복을 변환·평가 단계에서 제거하고, 충돌과 제거 개수를 기록합니다. 제공된 기존 예측 200,657행에서 중복 35,127개를 제거하여 165,530개 고유 표본으로 재계산했습니다.
- CNRPark-EXT: 4,073 images; Accuracy 95.08%, Precision 93.49%, Recall 97.48%, Specificity 92.39%, F1 95.44%, Balanced Accuracy 94.93%. TP 85,280 / TN 72,101 / FP 5,940 / FN 2,209.
- Image Count Exact 33.96%, MAE 1.441, median 1, P90 4, P95 5, max 14; over 51.04%, under 15.00%. Count는 이미지별 점유된 주석 주차면 수입니다.
- 실제 GT/Pred/polygon/detection overlay 33장, TP/TN/FP/FN·카메라·날씨별 사례, camera1/9 FP, camera2 성공 사례, montage와 그래프를 저장합니다.
- EXTERNAL_BASELINE의 설정·모델 해시를 동결하고 외부 정답에 맞춘 임계값 튜닝을 막습니다. 한·영 Living Paper와 v16.5.2 Markdown/PDF snapshot, 재계산 근거 ZIP·체크섬을 공개합니다. 기존 snapshot은 보존합니다.
- 업데이트 프로그램은 정확한 버전의 프로그램 ZIP을 선택하므로 외부 근거 ZIP이 함께 있어도 잘못 설치하지 않습니다.

This release recomputes the complete supplied frozen prediction set after deduplication and verifies matching inference on 20 selected images. It does **not** rerun inference on all 4,073 images. The original full-run settings snapshot is unavailable; selected replay settings and model fingerprint are documented. Selected failure illustrations are deterministic, not a random sample. The external result evaluates spatial occupancy; independent temporal validation remains pending. v16.2 SAFE_BASELINE (85.29%, MAE 0.1471) and v16.4 CANDIDATE (97.06%, MAE 0.0294) remain unchanged.

## Validation

Application self-test, 13 unit/UI tests, complete frozen-archive integrity and metric recalculation, paired manuscript/table/figure/manifest checks, immutable snapshots, and bilingual PDF glyph/text/page rendering checks gate the Release. Public CNRPark-EXT evidence includes source and ODbL attribution. Original user files remain untouched.

---

# Parking Research Agent v16.5.1

## 한국어 릴리스

- 한국어/영어 UI 추가. 설정에서 `ko`/`en`을 전환하면 열린 창의 표시 문구가 갱신됩니다. 언어 선택은 재시작·자동 업데이트 이후에도 유지됩니다.
- 메인 UI, 검증 센터, GT 라벨링, 주요 도움말·버튼·상태·메시지박스 현지화. 입력 폼, 기술 식별자, CSV 스키마, 결과 파일명과 연구 캐시 호환성을 유지합니다.
- 한·영 Living Paper, 공통 결과표, 전체 v1-v16.5.1 연구 부록, 근거 manifest, 실제 이미지가 없는 그림의 설명 자리표시자 추가.
- v16.5.1 한·영 Markdown snapshot과 한국어/영어 PDF Release asset 자동 빌드. 스냅샷은 수정하지 않습니다.
- 연구 릴리스의 새 내용은 방법·결과·오류·강건성·한계에 통합합니다. 연구 변경이 없는 patch는 논문 길이를 늘리지 않아도 됩니다.

보호 상태: **v16.2 SAFE_BASELINE 85.29% / MAE 0.1471**, **v16.4 CANDIDATE 97.06% / MAE 0.0294**. 이 숫자는 동일 보존 영상의 기존 보고값입니다. 이번 릴리스에서 새 정확도 실험을 수행하거나 Candidate를 승격하지 않았습니다. 실제 그림과 독립 시간 검증 및 외부 실행 결과는 확보 후 추가합니다.

## English release

Adds persistent Korean/English presentation, synchronized bilingual Living Papers, shared evidence tables, complete research-history appendix, explicit missing-figure descriptions, immutable v16.5.1 snapshots, and Markdown-based bilingual PDF release automation. Research changes integrate into existing sections; patches require no artificial length growth.

v16.2 remains SAFE_BASELINE; v16.4 remains CANDIDATE. Historical retained-video scores are unchanged, with no new accuracy or independent-generalization claim. Source ZIP, individual ZIP checksum, both PDF/Markdown assets and SHA256SUMS.txt accompany the release. The workflow gates publication on application, UI, paper and PDF validation.

## Validation

- Existing state/evidence-cache self-test.
- Six localization/UI tests: restart persistence, update merge, open-form preservation, localized message boxes, canonical identifiers/filenames and unchanged research cache signatures.
- Paired sections/common tables, current version, manifest hashes and immutable snapshot checks.
- Both PDFs: embedded Korean/Latin glyph coverage, required result text, page bounds, full-page renders and preview artifacts.

---

# Parking Research Agent v16.5.0

v16.5.0 is the **validation expansion release**. It does not silently promote the candidate: **v16.2 remains SAFE_BASELINE and v16.4 remains CANDIDATE**.

## Added

### 1-N CCTV / new-video support
- CCTV count: **1 to 12**.
- Dynamic ROI setup, preview, duplicate-slot linker, diagnostics and paper screenshots.
- Existing 3-CCTV setup remains compatible.

### Dataset Profiles + GT tool
- Save/load local Dataset Profiles.
- Dynamic count-GT CSV generator.
- Interactive GT Labeler with per-CCTV counts, total occupied, U and CUT marking.

### Episode / cut validation
- Configurable cut boundaries and post-cut warm-up exclusion.
- `episode_evaluation_mask.csv`
- `episode_metrics.csv`

### Repeated-source stability
For a repeated 4-minute source, use `repeat_period_sec=240`.
- `repeat_stability.csv`
- `REPEAT_STABILITY_REPORT.txt`
- This is reproducibility/stability testing, **not independent validation**.

### Automatic v16.4 evaluation
ALL-IN-ONE automatically evaluates v16.4 after v16.2 and packages candidate CSV/JSON/report files.

### Public external validation
Automatic **MetaPKLot / CNRPark-EXT** preparation:
- Quick / Standard selected official subsets.
- Full explicit opt-in shallow clone.
- resumable targeted downloads + size verification.
- automatic COCO parking-space annotation conversion.
- Accuracy / Precision / F1 / Occupied Recall / Empty Specificity / TP/TN/FP/FN.
- camera/weather breakdown.
- `EXTERNAL_VALIDATION_TO_CHATGPT.zip`.

Public-data results are labeled external **spatial** generalization, not continuous temporal validation.

### Cache visibility
The main UI now displays a compact CACHE audit summary.

## Protected research status
Retained development video:
- v16.2 SAFE: **85.29% Exact / 0.1471 MAE**
- v16.4 Candidate: **97.06% Exact / 0.0294 MAE**

No automatic SAFE_BASELINE promotion occurs.

## Update safety
`user_data/` remains preserved and now contains Dataset Profiles and downloaded public datasets.
