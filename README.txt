v16.2 PATCH NOTES
- v16/v16.1의 visual_diff_initial=1.0 전체 오염 원인을 수정했습니다.
- shared slot-learning cache가 template 이미지뿐 아니라 learned slot metadata snapshot도 함께 보관/복구합니다.
- 기존 오염 Evidence cache는 자동 sanity check 후 폐기/재계산합니다. 사용자가 cache를 직접 지울 필요가 없습니다.
- 현재 검증 GT에서는 SAFE_BASELINE 85.29% / MAE 0.1471 기준으로 evaluation-only non-regression audit를 남깁니다.
- 기존 robustness GT 라벨은 그대로 유지하고, 실제 사람 GT에서 부족한 클래스(현재는 EMPTY) 후보만 추가합니다.
- U/UNKNOWN 저장, state-search CSV 복구, 하단 Progress 고정 UI는 v16.1 동작을 그대로 유지합니다.

Parking Slot Engine v16.2
=======================

핵심 목표
---------
v16.2는 v15 계열의 안정적인 FULL/AUX + 시간축 판정은 그대로 보존하면서,
1) 열악한 CCTV 조건에서 실제로 어떤 보조 입력이 도움이 되는지 빈자리까지 포함한 GT로 검증하고,
2) SEG를 수천 프레임 선계산하지 않고 상태가 바뀌려는 순간에만 호출하고,
3) 고정 CCTV의 장점인 EMPTY reference를 G029 같은 난해 슬롯에 실험적으로 활용하는 버전입니다.

가장 중요한 안전 원칙
--------------------
- SAFE_BASELINE이 계속 기준입니다.
- TEST 성능을 보고 모델/상태 엔진을 자동 선택하지 않습니다.
- SEG/보정 Detector/EMPTY reference 하나만으로 새 OCCUPIED를 확정하지 않습니다.
- Manual Point는 자동 이동하지 않습니다.
- 같은 물리 차량/SEG mask가 같은 CCTV 프레임에서 여러 슬롯을 동시에 먹지 못하도록 1:1/Voronoi 보호를 유지합니다.

UI / Progress
-------------
처음 프로그램을 켰을 때 창 크기를 따로 키우지 않아도 하단에 다음 항목이 항상 보이도록 레이아웃을 바꿨습니다.
- Progress bar
- Status
- Total Elapsed
- Step Elapsed
- Step ETA
- Total Remaining

Controls 설명은 스크롤 가능한 영역으로 바꿔서 글이 더 늘어나도 Progress 영역을 밀어내지 않습니다.
초기 창 크기는 화면 해상도에 맞춰 자동 조절됩니다.

Balanced Robustness GT
----------------------
v15.4의 72개 라벨이 전부 OCCUPIED였기 때문에 빈자리 오탐을 평가할 수 없었습니다.
첫 GT가 없을 때는 기본 96개 CLEAN crop을 다음과 같이 뽑습니다.
- likely OCCUPIED 후보
- likely EMPTY 후보
- C2_S003 / C2_S012(G018) / C3_S009(G029) 등 forced hard slot
- 부족하면 시간/슬롯 diversity sample

중요: likely OCCUPIED / likely EMPTY는 샘플을 고르는 힌트일 뿐 정답으로 사용하지 않습니다.
최종 GT는 사용자가 직접 O/E/U로 라벨링한 값만 사용합니다.

라벨링 키
---------
- O 또는 1 = OCCUPIED
- E 또는 0 = EMPTY
- U = UNKNOWN
- B = 이전
- S/Q/Esc = 저장 후 종료

라벨 파일은 %LOCALAPPDATA%\ParkingSlotEngine\labels\robustness_gt.csv 에 유지됩니다.
기존 sample_id가 다시 나오면 예전 라벨을 재사용합니다.

v16.2부터 이미 사람 라벨이 존재하면 그 96개를 다시 시키지 않습니다. 실제 O/E 비율을 보고 minority class가 40% 미만이면 부족한 클래스 후보만 최대 48개 추가합니다. 새 ALL-IN-ONE 후 Label 버튼을 누르면 기존 O/E는 건너뛰고 새 U 후보부터 이어서 라벨링합니다.

진짜 GT 기반 Robustness 지표
---------------------------
robustness/degradation_ground_truth_metrics.csv
- Accuracy
- Precision
- F1
- False Empty rate
- False Occupied rate
- Occupied Recall
- Empty Specificity
- TP/TN/FP/FN

robustness/degradation_fusion_metrics.csv
- YOLO RAW OR ADAPTIVE
- SEG RAW OR ADAPTIVE
- FUSION ANY 4
- FUSION 2-of-4
- FUSION CONSERVATIVE
- FUSION CONDITION RULED

열화별 입력 정책
----------------
CLEAN
- RAW YOLO
- RAW SEG

LOW_RES
- RAW 입력 유지
- 이전 프레임만 사용하는 causal multi-frame median
- 이후 conservative resize/resample + mild unsharp
- YOLO/SEG 보조 입력으로 비교

SUN_GLARE / highlight-heavy
- RAW 입력 유지
- highlight tone-compressed 입력을 병렬로 비교
- RAW를 보정본으로 대체하지 않음

MONITOR_STRIPES / stripe-heavy
- RAW YOLO 유지
- causal temporal median YOLO 추가
- RAW SEG 유지
- temporal-median SEG는 사용하지 않음

Operational SEG = Transition-time only
--------------------------------------
v15.4처럼 WEAK_FULL/RECENT_MISS를 이유로 SEG를 대량 선계산하지 않습니다.
v16.2는 주로 다음 순간만 SEG를 요청합니다.
- SAFE baseline이 EMPTY -> OCCUPIED로 바뀌려는 순간
- SAFE baseline이 OCCUPIED -> EMPTY로 바뀌려는 순간
- initial EMPTY 슬롯에서 EMPTY-reference는 크게 변했는데 FULL은 놓치고 Crop evidence는 있는 경우(희소 간격)

이렇게 해서 SEG 호출 수와 실행 시간을 줄이고 실제 상태 전환에 직접 관련된 증거를 모읍니다.

EMPTY Reference experiment
--------------------------
초기 상태가 EMPTY인 슬롯은 startup template을 빈자리 기준 모습으로 사용할 수 있습니다.
v16.2는 warm-up 구간의 visual difference를 이용해 슬롯별 EMPTY reference threshold를 자동 보정합니다.

실험 규칙:
- EMPTY reference 변화만으로 OCCUPIED 금지
- EMPTY reference 변화 + FULL/Crop/SEG/Adaptive Detector 중 독립 차량 증거가 함께 있어야 함
- 짧은 causal streak가 유지되어야 실험적 OCCUPIED 허용
- 이미 OCCUPIED인데 EMPTY reference가 계속 크게 다르면 잘못된 EMPTY 전환을 막는 보조 증거로 사용

이 variant는 EMPTY_REF_ASSIST로 별도 결과를 출력하지만 기본 SAFE selection을 자동으로 덮어쓰지 않습니다.
G029(C3_S009) 같은 원본부터 탐지가 어려운 슬롯을 연구하기 위한 실험입니다.

주요 결과 파일
--------------
- EVIDENCE_CACHE_SANITY.txt
- NON_REGRESSION_CHECK.txt
- temporal_variant_comparison.csv
- metrics_dev_test.csv
- baseline_* / transition_guard_* / seg_assist_*
- empty_ref_assist_slot_timeseries.csv
- empty_ref_assist_global_slot_timeseries.csv
- empty_ref_assist_state_transitions.csv
- EMPTY_REF_EXPERIMENT.txt
- segmentation_summary.csv
- segmentation_condition_thresholds.csv
- segmentation_condition_counts.csv
- robustness/robustness_sample_manifest.csv
- robustness/degradation_robustness_samples.csv
- robustness/degradation_ground_truth_metrics.csv
- robustness/degradation_fusion_metrics.csv
- robustness/degradation_robustness_summary.csv
- robustness/degradation_adaptive_recovery.csv
- robustness/gt_review/

Cache
-----
공유 cache: %LOCALAPPDATA%\ParkingSlotEngine\cache

정상 FULL/AUX evidence는 구조가 같으면 그대로 재사용합니다.
v16/v16.1에서 visual_diff_initial이 거의 전부 1.0인 오염 cache가 발견되면 자동으로 거부하고 evidence만 재계산합니다.
slot-learning cache는 learned_slots_snapshot.json까지 포함하므로 새 버전 폴더에서도 template/region metadata가 같이 복구됩니다.

Candidate 확정 정보
-------------------
- CCTV3 약 (812,205): CAND_001 = FALSE / 기둥
- CCTV3 약 (915,330): CAND_002 = DUPLICATE / C2_S004(G010)과 같은 물리적 주차면
- DUPLICATE는 새 Global Slot을 만들지 않음
- GT를 모델 결과로 자동 수정하지 않음

실행
----
처음: START.cmd
이후: RUN.cmd
자체 검사: SELF_TEST.cmd

권장 순서
---------
1. v16.2 ALL-IN-ONE 1회 (오염 cache면 자동 복구)
2. Label / repair robustness GT balance 버튼으로 새로 추가된 U 샘플만 라벨링
3. 새 샘플을 라벨링했다면 ALL-IN-ONE 한 번 더 실행
4. 생성된 UPLOAD_TO_CHATGPT.zip 전송
