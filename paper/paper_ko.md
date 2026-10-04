# 재촬영 다중 CCTV 영상의 인과적 주차면 점유 추정: Parking Research Agent Living Paper

Living Paper | v16.5.1 | Research revision retained-v16.5-r1

[한국어](paper_ko.md) | [English](paper_en.md)

연구 초안. 아래 수치는 보존 기록의 보고값이며 이번 릴리스에서 새로 측정한 값이 아니다. 저자·소속·원본 영상 배포 권한은 미기재 상태다.

<!-- section:abstract -->

## 초록

본 연구는 CCTV 모니터를 휴대전화로 재촬영한 영상에서 주차면 점유 수를 추정하는 연구 에이전트의 개발 과정을 정리한다. 흔들림, 원근 왜곡, 저해상도, 태양광 반사, 모니터 줄무늬와 차량 검출 누락이 동시에 존재하는 환경을 대상으로 한다. 수동 주차면 기준점, 원근 보정, Voronoi 기반 공간 제약, FULL 우선 검출과 선택적 AUX 복구, 인과적 시간 상태 추적을 결합한다. 보존된 연구 기록에서 v16.2 SAFE_BASELINE은 TEST Exact 85.29%, MAE 0.1471이며, 추적 ID 전환 시 입차 지연과 약한 출차 소유자 해제를 적용한 v16.4 CANDIDATE는 97.06%, MAE 0.0294이다. 다만 후보 규칙은 동일 검증 영상의 오류 분석 후 개발되었으므로 독립 일반화 성능으로 해석할 수 없다. v16.5는 새 영상, 구간, 반복 원본 및 공개 데이터 공간 평가 도구를 제공하며 새 정확도 결과는 아직 보고하지 않는다. v16.5.1은 한·영 UI와 동기화된 Living Paper를 추가하는 표현·문서 패치로, 연구 알고리즘이나 성능 주장을 추가하지 않는다.

핵심어: 주차면 점유, CCTV 재촬영, 인과적 상태 추적, 다중 CCTV, 강건성, Living Paper.

<!-- section:introduction -->

## 1. 서론

직접 디지털 CCTV 스트림을 확보하기 어려운 환경에서는 모니터 재촬영 영상이 연구 입력이 된다. 이때 검출 정확도를 높이는 것만으로는 차량이 어느 주차면에 대응하는지, 실제로 주차했는지, 출차했는지를 해결하기 어렵다. 같은 차량이 여러 CCTV에 보이는 경우에는 화면별 검출 수와 실제 점유 주차면 수를 구분해야 한다.

연구 질문은 세 가지다. (1) 수동 기준점을 유지하면서 약한 검출을 주차면에 안정적으로 대응시킬 수 있는가? (2) 미래 프레임 없이 주차 동작과 안정적 점유를 구분할 수 있는가? (3) 보존 영상에서의 개선을 새로운 영상과 열화 조건에서도 확인할 수 있는가? 본문은 공간 대응과 시간 상태 판단의 발전을 설명하고, 실패한 실험을 함께 보고하며, 독립 검증을 다음 단계로 명시한다.

<!-- section:related -->

## 2. 관련 연구와 연구 범위

Ultralytics YOLO의 객체 검출 추론 [R3]을 차량 관측 계층으로 사용한다. 본 연구의 비교 대상은 보존된 내부 버전이며, 다른 논문과 동일 프로토콜로 비교한 순위나 최첨단 성능을 주장하지 않는다. 전체 화면 검출, 주차면 crop, Hungarian 일대일 대응, 수동 기준점 공간 제약과 시간 상태 추적을 내부 실험으로 비교한다.

MetaPKLot [R4]은 기존 주차 데이터의 주석과 평가 체계를 제공한다. v16.5의 CNRPark-EXT 경로는 이미지별 주차면 점유 평가용이다. 이 구현의 공간 평가 결과를 연속 영상의 입차·출차 판정 결과로 대체할 수 없다. 외부 데이터셋은 목표 환경의 정답으로 튜닝하지 않는 검증 대상으로 유지한다.

<!-- section:environment -->

## 3. 연구 환경 및 문제 정의

초기 연구 입력은 CCTV 모니터의 휴대전화 재촬영 영상이다 [R1]. 원래 화면의 기울기와 사다리꼴 왜곡은 꼭짓점 4개 ROI로 보정한다. 가려진 주차선과 초기 만차 상태를 고려하여 주차면은 수동 기준점으로 정의한다. 서로 다른 화면에서 동일한 실제 주차면을 사용자가 GLOBAL SLOT으로 연결한다. 초기 설정은 3 CCTV이며 v16.5는 1-12 CCTV 프로필을 지원한다.

시각 t의 정답 점유 수를 y(t), 예측 수를 ŷ(t)라 하면 Exact는 평가 시점 중 y(t)=ŷ(t)의 비율이고 MAE는 |y(t)-ŷ(t)|의 평균이다. 점유 수, 고유 차량 수, CCTV별 수는 서로 다른 항목이다. 기술 상태 EMPTY, MANEUVERING, OCCUPIED, LEAVING, UNKNOWN과 SAFE_BASELINE/CANDIDATE 이름은 데이터 및 코드 호환성을 위해 그대로 유지한다.

> **그림 자리표시자 1. 실제 CCTV 모니터 재촬영 환경**
> 저장소에서 실제 이미지 파일을 확인하지 못했다. 향후 실행 결과의 GT review 또는 paper-ready 이미지를 검토하여 추가한다. 합성 사진이나 미검증 전후 결과로 대체하지 않는다.

> **그림 자리표시자 2. 수동 주차면 기준점과 원근 보정 및 Voronoi 영역**
> 저장소에서 실제 이미지 파일을 확인하지 못했다. 향후 실행 결과의 GT review 또는 paper-ready 이미지를 검토하여 추가한다. 합성 사진이나 미검증 전후 결과로 대체하지 않는다.

<!-- section:methods -->

## 4. 제안 방법

### 4.1 공간 대응과 FULL 우선 관측

ROI 원근 보정 후 YOLO 관측을 수동 기준점으로 고정한 Voronoi 영역과 대응한다. 학습 앵커는 기준점을 대체하지 않으며 이동 범위와 표본 조건으로 제한한다. 중복 검출 억제와 일대일 대응으로 동일 차량의 다중 주차면 할당을 줄인다. FULL 검출은 주 관측이다. 주 관측이 약하거나 누락되거나 외관 변화가 의심스러울 때, 또는 드문 점검 시점에 선택적 CROP/AUX를 호출한다. AUX는 FULL 누락을 복구할 수 있지만 유효한 FULL 관측을 지우지 못한다.

### 4.2 인과적 시간 상태

온라인 추적과 최근 10초 이력을 사용하여 주차 동작과 정지 점유를 구분한다. 첫 10초는 준비 구간이며 미래 프레임을 사용하지 않는다. 장기 정지 후보는 사용자 검토용이며 실제 주차면 수에 자동으로 추가하지 않는다. 캐시에서 학습 기하와 외관 템플릿 메타데이터를 함께 복원하고, 템플릿 누락이나 비현실적인 증거 분포는 거절 후 재계산한다.

### 4.3 추적 ID 전환과 출차 후보

v16.4 ENTRY_TRACK_SWITCH_HOLD는 ID가 바뀌어도 최근 주차면 움직임 맥락을 유지한다. 큰 움직임 직후 갑자기 정지로 보이는 새 추적은 충분한 시간 동안 안정화되기 전 OCCUPIED로 확정하지 않는다. LEAVING_WEAK_OWNER_RELEASE는 기준 엔진의 LEAVING 상태, 약한 소유자 평균 신뢰도, 유의미한 움직임, 연결 CCTV의 EMPTY 의견이 함께 있는 경우에만 유령 점유를 해제한다. 재획득 안정화 간격으로 즉시 재입차를 방지한다. 두 규칙은 CANDIDATE 후처리이며 SAFE_BASELINE을 자동 승격하거나 수정하지 않는다 [R2].

<!-- section:protocol -->

## 5. 실험 구성 및 근거 수준

보존 프로토콜은 15:00 이전 DEV로 파라미터를 선택하고, 선택 후 15:00-20:30 TEST를 평가한다. 전체 실험에서 최초 10초를 준비 구간으로 처리한다. 기록의 TEST 표본 수는 34개이며 인접 표본은 시간적으로 상관될 수 있다. Exact와 MAE 외에 최대 오차, 과소/과대 집계, 전이 이벤트별 지표를 구분한다.

이 문서의 수치는 research_history.json과 변경 기록에 보고된 결과를 정리한 것이다. 이번 문서 패치에서 원본 영상이나 전체 실행 결과를 확보하여 재실험하지 않았다. sample_ground_truth.csv는 참고 파일이며 원본 영상·실행 파라미터·전체 예측 파일을 대체하지 않는다. 재현 시 버전, 입력 해시, 설정, ROI, 슬롯, GT, DEV 선택 결과, TEST 예측, 캐시 검증과 실행 환경을 함께 보존해야 한다. v16.4 규칙을 동일 영상의 TEST 오류 분석 후 설계했으므로 TEST의 절대 수치를 미사용 holdout 결과로 표현하지 않는다.

<!-- section:results -->

## 6. 실험 결과

표 1은 보존 기록의 버전별 점유 수 성능이다 [R1, R2]. MAE 반올림 정밀도는 원래 기록을 유지한다. v12와 v13은 Exact가 같아도 MAE와 최대 오차가 다르다. v16.2의 29/34와 v16.4의 33/34는 약 11.77%p 차이이며, MAE는 0.1471에서 0.0294로 낮아졌다. 두 방법 모두 최대 오차 1과 과소 집계 0%를 보고한다. 후보의 과대 집계는 2.94%이며 DEV 지표는 변하지 않았다는 기록이다.

v16.5 및 v16.5.1의 새 영상·공개 데이터 정확도는 미보고이다. 문서의 숫자를 새 실험 완료의 증거로 사용해서는 안 된다. 연구 원시 결과가 없는 초기 버전에는 수치를 부여하지 않는다.

| 버전/방법 | TEST Exact (%) | MAE | 최대 오차 | 근거 |
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

## 7. 실패 실험 및 절제 분석

표 2의 실험은 동일 목적을 향한 대안을 기록하지만 완전한 통제 절제 실험으로 간주하지 않는다. 버전별 조건과 추가 관측 비용이 달라 인과적 기여를 단일 숫자로 분리할 수 없다. CROP/HYBRID의 큰 회귀와 ZONE_MEMORY의 장기간 EMPTY 고착은 보조 관측과 기억이 주 관측을 대체할 때의 위험을 보여준다. TRANSITION_GUARD, SEG_ASSIST, EMPTY_REF는 안전 기준보다 낮아 채택하지 않았다. v15.1 분할은 약 7,451회의 추가 추론에도 상태 변경이 0회로 기록되어, 지연·선택적 분할을 개발하는 근거가 되었다.

원인 설명은 기록에 근거한 오류 해석이며 외부 데이터셋에서 검증된 일반 법칙이 아니다. 실패한 실험도 결과와 결정의 근거를 부록에 남긴다.

| 실험 | TEST Exact (%) | 관찰 | 출처 |
| --- | --- | --- | --- |
| CROP | 14.71 | FULL보다 불안정 | research_history.json:v11 |
| HYBRID | 23.53 | 채택하지 않음 | research_history.json:v11 |
| ZONE_MEMORY | 14.71 | 장기간 EMPTY 고착 | research_history.json:v14 |
| TRANSITION_GUARD | 70.59 | SAFE보다 낮아 거절 | research_history.json:v16.2 |
| SEG_ASSIST | 70.59 | SAFE보다 낮아 거절 | research_history.json:v16.2 |
| EMPTY_REF (EMPTY_REF_ASSIST) | 0 | SAFE보다 낮아 거절 | research_history.json:v16.2 |

<!-- section:transitions -->

## 8. 전이 오류 분석

v16.2의 TEST 오차 5개는 전이 이벤트 2개에 집중되었다. 입차 시 추적 ID가 바뀌면 움직임 이력이 초기화되어 조작 중 차량이 정지 차량으로 보일 수 있다. 출차에서는 약한 관측이 기존 소유자에 남아 실제 출차 후 점유가 지속될 수 있다. 후보는 주차면 수준 움직임 보존과 연결 CCTV EMPTY 의견을 이용해 두 유형을 제한적으로 보정한다.

기록상 이전 오차 5개 중 4개가 제거되었지만 17:00의 과대 집계 1개는 남았다. 프레임/시점 정확도가 높아져도 서로 상관된 전이 오류를 독립 성공 사례 여러 개로 계산할 수 없다. 이벤트별 입차 지연, 출차 지연, 거짓 점유 지속, ID 전환 빈도와 이벤트별 성공률을 별도로 보고해야 한다. 원시 CSV가 확보되기 전에는 미보고 전이 지연 수치를 추가하지 않는다.

> **그림 자리표시자 3. ID 전환 후 조기 OCCUPIED 및 유령 점유 오류**
> 저장소에서 실제 이미지 파일을 확인하지 못했다. 향후 실행 결과의 GT review 또는 paper-ready 이미지를 검토하여 추가한다. 합성 사진이나 미검증 전후 결과로 대체하지 않는다.

> **그림 자리표시자 4. 동일 시각 SAFE와 v16.4 Candidate 전후 비교**
> 저장소에서 실제 이미지 파일을 확인하지 못했다. 향후 실행 결과의 GT review 또는 paper-ready 이미지를 검토하여 추가한다. 합성 사진이나 미검증 전후 결과로 대체하지 않는다.

<!-- section:robustness -->

## 9. 영상 열화 강건성

강건성 실험은 LOW_RES, SUN_GLARE, MONITOR_STRIPES 조건을 다룬다. v15.3은 보수적 재샘플링/언샤프, 하이라이트 톤 압축, 과거·현재 5프레임의 인과적 temporal median을 도입했다. v15.4는 CCTV별 백분위 기반 상대 조건 분류와 지연 분할을 추가했다. v16 이후 GT 균형 보정, O/E/U 즉시 저장, U의 이진 지표 제외, Accuracy/Precision/F1/점유 Recall/빈 주차면 Specificity/FP/FN 평가와 표본 보정을 강화했다.

기록에는 temporal median의 stripe energy 감소와 검출 유지 개선이 정성적으로 남아 있지만 완전한 조건별 수치 표는 확보하지 못했다. 따라서 열화별 정확도 개선이나 외부 강건성 수치를 만들지 않는다. OCCUPIED에 치우친 GT와 일부 보충 검토 이미지의 패키징 누락은 한계다. 다음 실험은 조건별 O/E 표본 수, U 제외 수, 원본과 합성 열화 구분, RAW/ADAPTIVE 비용 및 상태 변화 효과를 함께 보고해야 한다.

> **그림 자리표시자 5. LOW_RES / SUN_GLARE / MONITOR_STRIPES의 RAW와 ADAPTIVE 비교**
> 저장소에서 실제 이미지 파일을 확인하지 못했다. 향후 실행 결과의 GT review 또는 paper-ready 이미지를 검토하여 추가한다. 합성 사진이나 미검증 전후 결과로 대체하지 않는다.

<!-- section:external -->

## 10. 외부 검증 및 반복 안정성 설계

v16.5 검증 센터는 새 로컬 영상별 Dataset Profile, CCTV 개수, 동적 count-GT 양식과 대화형 라벨링을 지원한다. 장면 전환 시각과 전환 후 준비 구간을 기록하고 episode_evaluation_mask.csv 및 episode_metrics.csv로 전환 주변을 평가에서 제외한다. 4분 원본을 반복한 경우 repeat_period_sec=240으로 repeat_stability.csv와 REPEAT_STABILITY_REPORT.txt를 만든다. 반복 원본은 상태 누적의 재현성·안정성 실험이며 독립 표본 증가가 아니다.

외부 공간 평가 경로는 MetaPKLot/CNRPark-EXT의 선택된 공식 이미지와 주차면 주석을 준비하고 카메라·날씨별 점유 지표를 보고한다. Quick/Standard는 제한된 표본이고 Full은 직접 선택하는 전체 준비 경로다. 실제 준비한 파일과 분할, 제외 사유, 모델 설정, 표본 수를 결과에 기록해야 한다. 독립 시간 검증에서는 알고리즘과 설정을 먼저 고정하고, 기존 오류 영상과 분리된 날짜·카메라·영상 및 실제 입출차 사건을 확보하여 SAFE와 Candidate를 같은 프로토콜로 비교한다. 이 설계의 실행 결과는 아직 미보고이며 Candidate 승격 조건도 아직 충족했다고 주장하지 않는다.

<!-- section:history -->

## 11. 연구 이력의 통합 및 재현성

v1-v4는 검출과 안정화, 작은 판정 영역, 고해상도/타일 추론과 실행 빈도 최적화를 탐색했다. v4는 제안 중심의 부분 기록이며 확정 실험 수치가 없다. v5-v6은 기준점, 중복 연결, 원근 보정을 구축했다. v7-v10은 기준 평가, 실패한 학습 앵커, 수동 Voronoi 제약, 인과적 시간 추적으로 발전했다. v11-v14는 FULL/CROP/HYBRID 비교, 보조 관측 보호, 선택적 캐시 최적화와 실패한 기억 실험을 다뤘다. v15-v16.2는 전이와 분할·열화 조건, GT 균형, 증거 캐시 건전성과 회귀 점검을 강화했다. v16.3은 연구 이력/자동 업데이트 도구, v16.4는 전이 후보, v16.5는 검증 확대, v16.5.1은 UI 현지화와 이 문서 구조를 추가한다.

각 버전의 목표·변경·결과·문제·결정은 [전체 연구 이력 부록](appendix/full_experiment_history.md)에 보존한다. 본문은 출시 순서가 아니라 방법·결과·오류·한계에 맞게 새 근거를 흡수한다. release ZIP과 SHA-256은 소스 배포 재현성을 지원하지만 입력 영상과 모델 가중치까지 자동으로 고정하는 것은 아니다.

<!-- section:limitations -->

## 12. 한계 및 향후 연구

가장 큰 한계는 단일 보존 영상, 작은 TEST 표본 수, 전이 시점의 시간 상관, 후보의 사후 오류 분석, 정량 근거가 불완전한 강건성 결과다. v16.5의 기능 구현은 새로운 데이터에서의 성공을 의미하지 않는다. 공개 이미지의 공간 검증만으로 시간 전이 안정성이나 현장 운영 안전성을 주장할 수 없다. 실패 실험의 수치도 조건이 완전히 동일한 절제 결과로 과해석하지 않는다.

향후 연구는 독립 연속 영상의 이벤트 균형 평가, O/E 균형 강건성 표본, 과소 집계와 거짓 점유 비용, 처리 시간과 캐시 재사용 비용, 카메라 수 및 시점 변화, 개인정보를 검토한 실제 전후 그림을 우선한다. 독립 검증 후에만 SAFE_BASELINE 승격을 논의하며, 성능이 회귀하면 후보로 유지하고 실패 근거를 남긴다.

<!-- section:conclusion -->

## 13. 결론

본 연구 기록은 수동 주차면 식별자를 보호하고 공간 제약과 인과적 시간 판단을 결합하며 보조 검출의 권한을 제한하는 방향으로 발전했다. 보존 영상에서 SAFE 85.29%에서 Candidate 97.06%로 개선되었지만 일반화는 미확인이다. 한·영 Living Paper는 동일한 연구 근거와 한계를 유지하고, 이후 릴리스에서 의미 있는 새 방법·표·그림을 적절한 본문 위치에 통합한다. 연구 변화가 없는 patch는 길이를 늘리지 않아도 된다.

<!-- section:references -->

## 참고 문헌 및 근거 자료

- [R1] [Canonical research history](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/research_history.json), retained records v1-v16.5.
- [R2] [v16.4 change record](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/CHANGES_v16_4.txt) and [v16.2 change record](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/CHANGES_v16_2.txt).
- [R3] [Ultralytics YOLO prediction documentation](https://docs.ultralytics.com/modes/predict/).
- [R4] [DSBD-Research MetaPKLot dataset and evaluation resources](https://github.com/DSBD-Research/MetaPKLot-Dataset), accessed 2026-10-04.
- [R5] [v16.5 implementation record](https://github.com/sopo9880/Parking_/blob/3b8fe54bdcc2a08570fe84bbaace44347470b673/CHANGES_v16_5.txt).

