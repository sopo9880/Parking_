# 전체 연구 이력 / Full research history

보존된 정식 연구 이력과 한국어 요약. 수치가 없는 버전은 미측정/미보존 상태를 유지한다. / Canonical retained history with Korean summaries; missing metrics remain unreported.

## v1

**한국어 요약:** 초기 YOLO와 수동 다각형 점유 판정. 재촬영 흔들림과 단일 기준점 판단이 취약했다. 신뢰할 정량 성능 기록 없음.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** Initial YOLO + polygon occupancy MVP

**Goal:** Detect vehicles and decide whether three manually defined parking slots are occupied.

### Changes

- YOLO vehicle detection

- Manual parking-slot polygons

- Bottom-center based slot inclusion test

- Real-time OCCUPIED/EMPTY visualization

- Stabilization was required because the source was a phone recording of a CCTV monitor

### Results

- Working MVP; no reliable quantitative benchmark recorded.

### Issues

- Shaky re-recorded CCTV caused coordinate drift

- Low-quality detections

- Initial full occupancy made slot definition difficult

- Single-point bbox judgment was fragile

**Decision:** Superseded by v2.

## v2

**한국어 요약:** 안정화 우선, 작은 판정 영역, BBox 하단 겹침과 단일 최적 주차면 대응. 동결된 DEV/TEST 수치 없음.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** Stabilized decision-zone occupancy

**Goal:** Reduce slot-boundary mistakes and make setup practical on low-quality footage.

### Changes

- Display-scale and frame-navigation GUI

- Smaller occupancy decision zones

- BBox lower-area vs decision-zone overlap

- Best-slot single assignment

- YOLOv8s baseline, confidence around 0.15

- Stabilization-first pipeline

### Results

- Functional v2 artifact was produced; no frozen DEV/TEST benchmark recorded.

### Issues

- Distant/black vehicles still weak

- Accuracy-speed tradeoff remained

**Decision:** Moved to stronger detection in v3.

## v3

**한국어 요약:** imgsz 1280과 선택적 타일 추론 및 UNKNOWN 도입. 검출 개선은 기록되었지만 타일 추론이 느렸다.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** High-resolution / tiled detection + UNKNOWN

**Goal:** Recover missed distant vehicles, especially the left-side black vehicle.

### Changes

- imgsz 1280

- Optional tile inference

- confidence around 0.08

- UNKNOWN state added

### Results

- Detection recall improved in difficult regions, but tiled inference was very slow.

### Issues

- Tile inference ran YOLO multiple times per frame

- Fast 640/no-tile mode missed vehicles and was rejected

**Decision:** Keep detection quality and optimize execution frequency in the v4 direction.

## v4

**한국어 요약:** 검출 간격 조절과 이전 검출 재사용을 제안한 부분 기록. 확정 수치 없음.

**Status:** PARTIAL_RECORD | **Evidence:** partial

**Title:** Detection-interval optimization direction

**Goal:** Keep high-resolution detection quality while reducing runtime.

### Changes

- Proposed detect-interval mode

- Reuse previous detections between YOLO runs

- Suggested YOLOv8s + imgsz 960/1280 + detect interval

### Results

- No frozen quantitative result was found in the retained research record.

### Issues

- This version is recorded mainly as a proposed optimization step.

**Decision:** Research later shifted toward explicit slot/temporal state modeling.

## v5

**한국어 요약:** 기준점 주차면, 중복 연결, 장기 정지 후보 도입. 보존된 동결 수치 없음.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** Point-based slots + linked duplicates

**Goal:** Make slot setup less dependent on visible parking-line polygons and support multiple CCTV views of the same physical slot.

### Changes

- Point-based slot identity

- Duplicate-slot linking

- Long-stationary vehicle candidates

- DEV state-condition search followed by TEST verification

### Results

- No preserved frozen metric was found for v5.

### Issues

- Geometry and perspective handling still needed improvement.

**Decision:** Perspective-normalized setup introduced in v6.

## v6

**한국어 요약:** 꼭짓점 4개 원근 보정. 시각 검증 기록만 있고 동결 정량 수치 없음.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** 4-point perspective-corrected CCTV

**Goal:** Normalize tilted CCTV views before slot setup and detection.

### Changes

- 4-point perspective ROI replacing rectangular crop

- Automatic point ordering

- Drag/delete/preview UI

- Slot setup and YOLO operate on warped CCTV

### Results

- Perspective crop visually verified; no frozen quantitative metric recorded.

### Issues

- Counting errors were still concentrated around slot identity and temporal transitions.

**Decision:** Established the geometry base used by later versions.

## v7

**한국어 요약:** TEST Exact 82.35% (28/34), MAE 0.3235. 전이 오류와 중복 대응 및 출차 누락.

**Status:** BASELINE_ARCHIVE | **Evidence:** confirmed

**Title:** Pre-anchor baseline

**Goal:** Establish a measurable DEV/TEST baseline before learned slot anchors.

### Changes

- Frozen DEV/TEST evaluation

- Error review focused on duplicate assignment and missed departures

### Results

- TEST Exact Match 82.35% (28/34)

- TEST MAE 0.3235

### Issues

- Errors concentrated in transitions

- Duplicate slot assignment

- Missed departures

**Decision:** Attempt learned anchor matching in v8.

## v8

**한국어 요약:** Hungarian 일대일 및 학습 앵커. DEV 31.11%, TEST 32.35%, MAE 1.294. 앵커 붕괴와 식별자 이동으로 거절.

**Status:** REJECTED | **Evidence:** confirmed

**Title:** Hungarian 1:1 + learned anchors

**Goal:** Improve per-slot identity assignment using learned anchor centers and one-to-one matching.

### Changes

- Hungarian 1:1 matching

- Anchor-center learning

- Overlap warnings

- Slot-GT support

### Results

- DEV Exact 31.11%

- TEST Exact 32.35%

- TEST MAE 1.294

### Issues

- Anchor collapse / identity drift

- Weak detections amplified matching errors

**Decision:** Rejected. Return authority to manual slot points in v9.

## v9

**한국어 요약:** 수동 기준점 고정 Voronoi와 제한된 앵커. DEV 72.22%, TEST 67.65%, MAE 0.441.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** Manual-point Voronoi + constrained anchor

**Goal:** Prevent learned geometry from overriding manual slot identity.

### Changes

- Manual-point-fixed Voronoi slots

- Constrained Hungarian matching

- Anchor drift/sample rejection

- Warped-CCTV YOLO parameter sweep

- CONF_MAX/CONF_MEAN fusion

- Stale-learning guard

### Results

- DEV Exact 72.22%

- TEST Exact 67.65%

- TEST MAE 0.441

### Issues

- Ghost occupancy and temporal persistence remained

**Decision:** Add causal temporal state tracking in v10.

## v10/v10.1

**한국어 요약:** 최근 10초 인과 이력과 시간 상태 추적. TEST 79.41%, MAE 0.265.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** Causal temporal parking-state tracker

**Goal:** Distinguish maneuvering vehicles from stable parked vehicles without future frames.

### Changes

- Rolling 10-second causal history

- Track IDs

- EMPTY / MANEUVERING / OCCUPIED / LEAVING states

- No future-frame leakage

- ALL-IN-ONE automated pipeline

### Results

- TEST Exact 79.41%

- TEST MAE 0.265

### Issues

- Detector mode and recovery strategy still needed comparison

**Decision:** Compare FULL/CROP/HYBRID evidence in v11.

## v11

**한국어 요약:** FULL 85.29%, CROP 14.71%, HYBRID 23.53%. FULL 채택; crop은 복구 보조로 제한.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** FULL vs CROP vs HYBRID

**Goal:** Determine whether slot crops outperform the full-CCTV detector.

### Changes

- FULL detector path

- Per-slot CROP detector path

- HYBRID comparison

### Results

- FULL TEST 85.29%

- CROP TEST 14.71%

- HYBRID TEST 23.53%

### Issues

- CROP/HYBRID were much less stable than FULL

**Decision:** FULL adopted; crop relegated to auxiliary recovery.

## v12

**한국어 요약:** FULL 권한 보호. TEST 85.29%, MAE 약 0.206, 최대 오차 2. HYBRID 후보 미채택.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** FULL-protected auxiliary recovery

**Goal:** Use crop evidence only when it cannot erase a valid FULL detection.

### Changes

- FULL is authoritative

- CROP becomes AUX evidence

- HYBRID_ADD experiment

- Protection against AUX deleting FULL evidence

### Results

- Adopted FULL TEST 29/34 = 85.29%

- MAE about 0.206

- Max error 2

- HYBRID candidate generalized poorly and was not adopted

### Issues

- Runtime remained high

- Low-confidence repeated false positives remained

**Decision:** Selective AUX + cache optimization in v13.

## v13

**한국어 요약:** 선택적 AUX와 교차 실행 캐시. TEST 85.29%, MAE 0.147, 최대 오차 1, 과소 집계 0%. CROP 7,184/39,392 주차면-프레임 (18.2%); 기록 실행시간 약 7시간33분에서 1시간52분.

**Status:** SAFE_BASELINE | **Evidence:** confirmed

**Title:** Selective AUX + cache safe baseline

**Goal:** Keep v12 accuracy while reducing runtime and max error.

### Changes

- Selective CROP/AUX invocation

- Cross-run caching

- Low-confidence repeated-detection suppression

### Results

- TEST Exact 85.29%

- TEST MAE 0.147

- Max error 1

- Under-count 0%

- Selective CROP 7,184 / 39,392 slot-frames (18.2%)

- Runtime about 7h33m -> 1h52m

### Issues

- Remaining errors were mainly transition timing

**Decision:** Became the SAFE_BASELINE carried forward.

## v14

**한국어 요약:** ZONE_MEMORY TEST 14.71%. 주차면 EMPTY 고착으로 거절. SAFE 85.29%/0.147 유지.

**Status:** REJECTED | **Evidence:** confirmed

**Title:** ZONE_MEMORY experiment

**Goal:** Use recent slot memory to stabilize missed detections.

### Changes

- ZONE_MEMORY variant

- Automatic comparison against SAFE_BASELINE

- Automatic fallback on regression

### Results

- ZONE_MEMORY TEST 14.71%

- SAFE_BASELINE remained 85.29% / MAE 0.147

### Issues

- Some slots became stuck EMPTY for long periods

- Memory was too authoritative

**Decision:** Rejected; temporal memory should assist transitions, not replace state evidence.

## v15

**한국어 요약:** 전이에서만 기억 보조. 안전 기준 보호, 더 우수한 동결 후보 채택 없음.

**Status:** EXPERIMENTAL | **Evidence:** confirmed

**Title:** Transition-only temporal guard

**Goal:** Keep v13 SAFE_BASELINE and use memory only around actual state changes.

### Changes

- v13 SAFE_BASELINE protected

- Entry requires real movement evidence

- Departure requires FULL miss + AUX miss + visual change

- Weak YOLO cannot initialize OCCUPIED by itself

- Manual point remains immutable unless dragged

### Results

- Baseline protection retained; no superior frozen candidate adopted.

### Issues

- Transition rules still needed more targeted evidence

**Decision:** Test segmentation and degradation robustness without changing baseline authority.

## v15.1

**한국어 요약:** SEG_ASSIST 약 7,451회 추가 추론, 상태 변경 0회. 적극적인 분할 미채택.

**Status:** REJECTED_EXPERIMENT | **Evidence:** confirmed

**Title:** SEG_ASSIST first pass

**Goal:** Use segmentation only as additional evidence on difficult frames.

### Changes

- Selective segmentation assist

### Results

- About 7,451 additional segmentation inferences

- State changes caused by SEG: 0

### Issues

- High compute cost with no useful state change

**Decision:** Do not use eager segmentation; make it lazy and condition-specific.

## v15.2

**한국어 요약:** RAW/enhanced 열화 실험의 부분 기록. 채택된 상태 엔진 개선 없음.

**Status:** EXPERIMENTAL | **Evidence:** partial

**Title:** RAW / enhanced degradation experiments

**Goal:** Test whether preprocessing can recover low-resolution, glare, and monitor-stripe cases.

### Changes

- RAW vs enhanced segmentation inputs

- Synthetic/bounded degradation robustness experiments

### Results

- No adopted state-engine gain recorded.

### Issues

- Generic enhancement was not condition-aware enough

**Decision:** Move to condition-specific adaptive preprocessing in v15.3.

## v15.3

**한국어 요약:** 조건별 저해상도/태양광/모니터 줄무늬 복구와 5프레임 인과 median. SAFE 85.29%/0.147/최대1 유지. 줄무늬 에너지와 검출 유지의 정성 개선만 기록.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** Condition-specific adaptive robustness

**Goal:** Use different recovery logic for LOW_RES, SUN_GLARE and MONITOR_STRIPES while preserving SAFE_BASELINE.

### Changes

- LOW_RES conservative resample/unsharp

- SUN_GLARE highlight tone compression

- Causal 5-frame temporal median for monitor stripes

- Robustness GT label UI

- Shared cross-version learning cache

- Four timing fields: Total Elapsed / Step Elapsed / Step ETA / Total Remaining

### Results

- SAFE_BASELINE TEST 85.29%

- MAE 0.147

- Max error 1

- Temporal median reduced stripe energy and improved retention in the recorded stress test

### Issues

- Robustness GT initially had no useful balanced labels

- Condition classifier/forced-slot issues remained

**Decision:** Improve classifier, cache migration and lazy scheduling in v15.4.

## v15.4

**한국어 요약:** CCTV별 상대 열화 분류, 지연 분할, 캐시 이관 및 강제 주차면 잘림 보정. GT의 OCCUPIED 편중은 남음.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** Relative degradation classifier + lazy SEG

**Goal:** Make degradation classification relative per CCTV and keep expensive processing sparse.

### Changes

- Per-CCTV percentile degradation classifier

- Operational causal temporal median

- Model-specific recovery detector for glare/stripe

- Forced-slot truncation fix

- Cross-version learning-cache migration fix

- Real Step Elapsed tracking

### Results

- Baseline preserved; robustness instrumentation improved.

### Issues

- Human robustness labels were overwhelmingly OCCUPIED

**Decision:** Redesign robustness sampling in v16.

## v16

**한국어 요약:** 균형 강건성 GT, O/E 이진 지표와 전이 시점 분할, EMPTY 참조 실험. UNKNOWN 저장 및 중복 열 버그 발생.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** Balanced robustness GT + transition-time SEG

**Goal:** Measure both false-empty and false-occupied behavior and only schedule expensive segmentation near meaningful transitions.

### Changes

- Balanced-likelihood robustness GT sampling

- Accuracy/Precision/F1/Recall/Specificity/FP/FN metrics

- Cross-preprocess fusion tables

- Transition-time SEG scheduling

- Experimental EMPTY-reference assist

- Pinned progress/timing UI

### Results

- SAFE_BASELINE still protected; robustness evaluation became more complete.

### Issues

- UNKNOWN label persistence and state-search CSV collision bugs

**Decision:** Stability fixes in v16.1.

## v16.1

**한국어 요약:** O/E/U 문자열 저장과 즉시 저장, U 제외, 상태 비교 중복 열 수정. 알고리즘/캐시 서명 유지.

**Status:** SUPERSEDED | **Evidence:** confirmed

**Title:** O/E/U + state-search stability patch

**Goal:** Make human GT labeling and experiment tables robust.

### Changes

- O/E/U persisted as explicit strings

- U excluded from binary metrics

- Immediate save after each label

- Duplicate-column state-search repair

- Pinned progress UI preserved

### Results

- Labeling/state-search crashes repaired without changing detector/evidence cache signatures.

### Issues

- Shared learned-template metadata could still be missing after cache reuse

**Decision:** Repair learned metadata/evidence cache in v16.2.

## v16.2

**한국어 요약:** 학습 슬롯 메타데이터·증거 건강 점검·회귀 점검. SAFE 85.29%, MAE 0.1471, 최대1, 과소0%. 가드/분할 70.59%, EMPTY 참조 0% 거절. TEST 오류5개는 이벤트2개.

**Status:** CURRENT_BASELINE | **Evidence:** confirmed

**Title:** Evidence-cache repair + non-regression audit

**Goal:** Prevent corrupt appearance evidence while preserving the known safe result.

### Changes

- learned_slots_snapshot metadata cache

- Template/region health checks

- Evidence sanity rejection/recompute

- NON_REGRESSION_CHECK

- Minority-class robustness sample repair

- Existing O/E/U labels preserved

### Results

- SAFE_BASELINE TEST 85.29%

- MAE 0.1471

- Max error 1

- Under-count 0%

- TRANSITION_GUARD TEST 70.59% -> rejected

- SEG_ASSIST TEST 70.59% -> rejected

- EMPTY_REF_ASSIST TEST 0% -> rejected

- Five TEST errors collapsed into two transition events

### Issues

- Tracker ID switch can reset motion history and make a maneuvering vehicle look stationary

- Supplemental balance-review images were not packaged in one run

**Decision:** Keep v16.2 algorithm as SAFE_BASELINE; move development workflow into an Agent app.

## v16.3.0

**한국어 요약:** 연구 이력과 GitHub 자동 업데이트 및 SHA-256 배포 도구. v16.2 알고리즘 유지.

**Status:** AGENT_RELEASE | **Evidence:** confirmed

**Title:** Parking Research Agent + GitHub Auto Update

**Goal:** Turn the research scripts into a versioned local research Agent without changing the v16.2 SAFE_BASELINE algorithm.

### Changes

- Research History viewer

- Canonical v1-v16.2 timeline

- Future run journal

- GitHub Releases update checker

- SHA-256 verified update package

- Persistent settings/slots/ROI/work/output preservation

- GitHub Actions release packaging

### Results

- Algorithm baseline intentionally unchanged from v16.2.

### Issues

- Slot-level motion memory remains a next experiment, not silently merged into this infrastructure release.

**Decision:** Use this as the new development shell for v16.3+ experiments.

## v16.4.0

**한국어 요약:** ID 전환 입차 지연 및 약한 소유자 출차 해제. 동일 영상 TEST 97.06% (33/34), MAE0.0294, 최대1, 과소0%, 과대2.94%. 이전 오류4개 제거, DEV 유지. 17:00 오류 남음; 독립 검증 전 후보 유지.

**Status:** CANDIDATE | **Evidence:** confirmed_on_retained_validation_only

**Title:** Track-switch entry hold + weak-owner ghost release

**Goal:** Target the two remaining transition-error families without modifying the v16.2 SAFE_BASELINE.

### Changes

- ENTRY_TRACK_SWITCH_HOLD preserves recent slot-level motion across tracker-ID switches

- New tracks that suddenly appear stationary after large motion cannot immediately create OCCUPIED

- LEAVING_WEAK_OWNER_RELEASE acts only on baseline LEAVING transitions with weak owner confidence and duplicate-camera EMPTY disagreement

- Reacquisition settle interval prevents immediate ghost re-entry

- Event-balanced transition metrics added

- Candidate runs are auditable and never auto-promote the SAFE_BASELINE

### Results

- Retained TEST Exact Match 97.06% (33/34)

- Retained TEST MAE 0.0294

- Max error 1

- Under-count 0%

- Over-count 2.94%

- Four of five previous TEST errors removed

- DEV metrics unchanged

### Issues

- Rules were developed after analyzing errors on the same retained validation video

- Independent-video validation is required before SAFE_BASELINE promotion

- One TEST over-count event at 17:00 remains

**Decision:** Release as CANDIDATE only; keep v16.2 as SAFE_BASELINE until independent validation.

## v16.5.0

**한국어 요약:** 1-12 CCTV 검증 센터, 프로필/GT/구간/반복 및 외부 공간 평가 도구. 새로운 정확도 주장 없음. 반복 원본은 독립 검증이 아님.

**Status:** VALIDATION_RELEASE | **Evidence:** implementation_validated_release_pending_runtime_dataset_results

**Title:** Multi-CCTV Validation Center + public external benchmark

**Goal:** Validate the protected SAFE and v16.4 candidate across new local layouts, repeated-source stability tests, episode-aware concatenated clips, and a public external parking benchmark without tuning on the external target.

### Changes

- Configurable 1-12 CCTV local profiles

- Dataset Profile save/load

- Dynamic count-GT template and interactive GT labeler

- Cut-boundary / episode evaluation mask and per-episode metrics

- Repeated-source reproducibility/stability metrics

- Automatic v16.4 candidate evaluation inside ALL-IN-ONE

- MetaPKLot/CNRPark-EXT Quick/Standard/Full downloader

- Automatic COCO parking-space annotation conversion

- External occupancy metrics by camera/weather

- External validation ChatGPT ZIP packaging

- Cache audit summary in the main UI

### Results

- No new accuracy claim is assigned before v16.5 is run on new/public validation data.

- Retained v16.2 SAFE and v16.4 Candidate results remain unchanged and separately labeled.

### Issues

- Repeated copies of one source clip remain a stability test, not independent evidence.

- MetaPKLot/CNRPark-EXT is a still-image spatial benchmark and does not validate continuous maneuvering transitions.

- v16.4 SAFE promotion still requires genuinely independent temporal evidence.

**Decision:** Release validation tooling while keeping v16.2 SAFE_BASELINE and v16.4 CANDIDATE statuses unchanged.

## v16.5.1

**한국어 요약:** 한·영 UI와 Living Paper 및 PDF 자동화. 신규 정확도 측정 없음. SAFE와 Candidate 상태 유지.

**Status:** DOCUMENTATION_RELEASE | **Evidence:** self_test_and_six_localization_ui_tests_passed_no_new_accuracy_claim

**Title:** Korean/English UI + bilingual Living Paper

**Goal:** Persist language preferences and publish synchronized growing research papers.

### Changes

- Presentation-only Korean/English localization

- Persistent ui.language setting

- Paired paper source, evidence tables, figure placeholders and immutable snapshots

- Bilingual PDF release automation

### Results

- No new dataset or accuracy measurement. Retained SAFE and CANDIDATE metrics remain unchanged.

### Issues

- Independent temporal validation and actual figure artifacts remain pending.

**Decision:** Keep v16.2 SAFE_BASELINE and v16.4 CANDIDATE; grow papers only when research evidence changes.
