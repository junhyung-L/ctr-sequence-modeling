# CTR 예측: 정형 피처와 행동 시퀀스를 함께 다룬 DCN-V2 실험

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## 한눈에 보기

광고 클릭률(CTR) 예측에서는 사용자·광고 인벤토리·시간대의 정형 정보와 사용자가 최근 상호작용한 행동 시퀀스를 함께 읽어야 한다. 이 프로젝트는 DCN-V2(Deep & Cross Network v2) cross/deep 타워에 DIN(Deep Interest Network), DIEN(Deep Interest Evolution Network), BST(Behavior Sequence Transformer) 인코더를 교체 가능하게 붙여 두 신호를 함께 실험한다.

검증에서 DCN-V2 + DIEN은 ROC-AUC 0.7413, PR-AUC(정밀도-재현율 곡선 아래 면적) 0.0792를 기록했다. 같은 클릭률에서 무작위 순위가 받는 PR-AUC 0.0191의 약 4배다. DUSIN이라고 표기된 확장 실험은 ROC-AUC 0.7417, PR-AUC 0.0780이어서 최고 설정은 지표에 따라 달라진다. cross 타워 없는 DIN 모델도 0.7412 / 0.0779를 기록했다. 다음 실험에 가장 쓸모 있는 발견은 이번 실행에서 cross 타워의 뚜렷한 이득이 없다는 점이다.

## 해결하려 한 문제

두 문제를 다룬다. 정형 피처 사이의 비선형 상호작용 학습, 그리고 길이가 제각각인 행동 이력을 고정된 메모리 안에서 표현하는 일이다. 학습 파일은 크고 불균형하다. `notebooks/baseline.ipynb`의 저장된 출력 기준 10,704,179행·119열, 클릭 204,179건(약 1.91%)이다.

CrossNetMix는 저차원·게이팅 expert로 명시적 교차항을 만들고, 병렬 MLP는 더 일반적인 비선형 패턴을 학습한다. 두 출력은 마지막 예측 헤드에서만 시퀀스 관심 벡터와 결합된다. 시퀀스 인코더는 같은 정형 골격 안에서 교체한다.

- **DIN**은 현재 후보에서 과거 아이템 임베딩으로 attention을 계산한다.
- **DIEN**은 이력을 GRU(gated recurrent unit)로 통과시킨 뒤 변화하는 관심 상태에 attention을 건다.
- **BST**는 attention pooling 전에 positional encoding과 Transformer encoder를 더한다.

## 주요 결정

**1. 행동 시퀀스 표현.**
선택지: 쉼표로 구분된 값을 숫자 그대로 LSTM에 넣는 방식(첫 노트북 `baseline.ipynb`), 모든 아이템 ID를 임베딩하는 방식, 최근 50개로 자르고 ID를 262,144개 버킷으로 해시하는 방식.
결정: 절단과 해싱. 왼쪽 padding으로 배치 길이를 고정하고, 빈 시퀀스의 attention에서 `NaN`이 생기지 않도록 막았다.
이유: 아이템은 크기가 아니라 범주이므로 임베딩이 필요하다. 전체 ID 어휘는 1,070만 행 파일에서 메모리 상한이 없지만 해싱은 상한이 있다.
비용: 해시 충돌로 서로 다른 아이템이 합쳐지고 50개보다 오래된 이력은 버려진다. 두 영향 모두 측정하지 않았다.

**2. 1.91% 클릭률 처리.**
선택지: 음성 표본 축소(기준 노트북은 클릭 전부와 그 두 배의 미클릭만 남긴 612,537행), 또는 전체 행을 유지하고 손실에서 양성에 가중치를 주는 방식.
결정: 전체 데이터에 `BCEWithLogitsLoss(pos_weight = negative / positive)`(학습 분할 기준 51.4).
이유: 검증 분할이 실제 클릭률을 유지하므로 PR-AUC를 인위적인 33%가 아닌 실제 기준선에 대해 읽을 수 있다.
비용: 전체 파일 학습 때문에 실행을 3 epoch로 제한했고, 가중 손실이 예측 확률을 이동시키므로 점수는 보정된 클릭 확률이 아니라 순위다.

**3. 인코더의 공정한 비교.**
선택지: 구조마다 따로 튜닝하는 방식, 또는 정형 골격·분할·seed·학습 예산을 고정하고 시퀀스 인코더만 바꾸는 방식.
결정: 골격을 고정한 교체 비교, 그리고 cross 타워를 뺀 실행 하나를 비교 기준점으로 추가.
이유: 같은 정형 피처 위에서 각 이력 표현이 더하는 것을 분리할 수 있다.
비용: 설정마다 seed 하나(한계 참고).

**4. 판단 지표.**
선택지: 지표 하나로 우승 모델을 정하는 방식, 또는 ROC-AUC와 PR-AUC를 나란히 보고하는 방식.
결정: 둘 다 보고하고, early stopping은 ROC-AUC로, 양성 순위의 기준은 PR-AUC로 삼았다.
이유: 지표마다 최고 설정이 다르고, 전체 순위와 상위 후보 정밀도 중 무엇이 중요한지는 활용 목적에 달려 있다.
비용: 하나의 추천 모델을 제시하지 않는다.

## 데이터와 학습 설계

입력: `clicked`(라벨), `seq`(행동 시퀀스. 대괄호·쉼표 형식과 숫자 문자열을 모두 파싱), `ID`(제출 식별자. 제출 파일에 보존). 연속형은 `float32`로 바꾸고 결측치는 0으로 채운다. 범주 매핑은 학습 분할에서만 만들고, 검증·테스트에서 처음 보는 값은 OOV(out-of-vocabulary) 인덱스로 처리한다.

CLI(`src/train.py`)는 seed 42의 층화 85/15 학습·검증 분할, `pos_weight = negative / positive`를 준 `BCEWithLogitsLoss`, AdamW, cosine annealing, ROC-AUC 기반 early stopping을 사용하고 예측과 실행 메타데이터를 저장한다. 저장된 주요 실행은 전체 학습 데이터(`sample_subset=1.0`), batch size 512, 학습률 0.001, dropout 0.2, 3 epoch 조건이다.

## 검증 결과

| 저장된 실행 | ROC-AUC | PR-AUC | 해석 |
|---|---:|---:|---|
| 무작위 순위(참고) | 0.5 | 0.0191 | 클릭률과 같음 |
| cross 타워 없는 DIN | 0.7412 | 0.0779 | DCN-V2 없는 비교 기준점 |
| DCN-V2 + DIN | 0.7402 | 0.0775 | attention 기반 시퀀스 인코더 |
| DCN-V2 + auto-BST | 0.7403 | 0.0783 | Transformer 기반 시퀀스 인코더 |
| DCN-V2 + DIEN | 0.7413 | **0.0792** | PR-AUC 최고 |
| DCN-V2 + DIEN + DUSIN (full) | **0.7417** | 0.0780 | ROC-AUC 최고 |

PR-AUC 기준의 참고 설정은 DIEN이며, 최종 모델 선택은 운영 목표를 따라야 한다. cross 타워 없는 DIN 실행은 DCN-V2 + DIEN과 ROC-AUC 0.0001 이내다.

## 한계

- 검증 지표만 있으며 리더보드·외부 테스트·온라인 A/B 결과는 없다.
- 원본 Parquet와 데이터 설명서는 포함하지 않았고, 출처와 수집 기간은 적지 않았다.
- 설정마다 seed 하나이며 약 0.001 수준의 ROC-AUC 차이는 편차 범위다. 신뢰구간·calibration·순위 기반(@K) 비즈니스 지표는 없다.
- 시간 순서가 아닌 무작위 층화 분할이므로 이후 시점 트래픽에서의 성능은 검증하지 않았다.
- cross 타워 없는 DIN 실행은 피처 유형 구분이 달라 깨끗한 ablation이 아니라 비교 기준점이다.
- 결과는 노트북에서 나왔고, `src/train.py`는 그 코드를 리팩터링한 것으로 다시 실행하지 않았다. auto-BST와 DUSIN은 노트북에만 있다.
- 여러 DCN-V2 노트북 로그에서 검증 loss가 `Val nan`으로 출력되지만 ROC-AUC·PR-AUC는 정상 계산된다. 원인은 추적하지 않았다.
- `main.py`는 `src/train.py`와 별개인 LightGBM 피처 엔지니어링 스텁이다.

## 다음 단계

데이터 버전·분할별 건수·패키지 버전·장비·산출물 경로를 담은 실험 매니페스트. 광고 의사결정을 뒷받침할 시간 순서 평가·반복 seed·calibration·Precision/Recall@K. DCN-V2가 비용만큼의 값을 하는지 가리기 위한, 피처 유형을 맞추고 cross 타워 유무를 여러 seed로 비교하는 ablation.

## 근거 파일

- 학습 파이프라인: [`src/train.py`](src/train.py)
- 데이터·시퀀스 처리: [`src/data.py`](src/data.py)
- DCN-V2 및 DIN/DIEN/BST 구현: [`src/models.py`](src/models.py)
- 결과 메타데이터: [`results/dcnv2_dien_meta.json`](results/dcnv2_dien_meta.json), [`results/dcnv2_dien_dusin_full_meta.json`](results/dcnv2_dien_dusin_full_meta.json), [`results/din_meta.json`](results/din_meta.json)
- 행 수·클릭 수: [`notebooks/baseline.ipynb`](notebooks/baseline.ipynb) 앞부분 셀의 저장된 출력
- 실행 매니페스트: [`research/RUN_MANIFEST.md`](research/RUN_MANIFEST.md)
