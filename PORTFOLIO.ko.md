# DCN-V2와 행동 시퀀스를 활용한 CTR 예측

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## 한눈에 보기

정형 정보와 사용자의 최근 행동 시퀀스를 함께 써서 광고 클릭률(CTR)을 예측하는 모델이다. DCN-V2(Deep & Cross Network v2) cross/deep 타워에 DIN, DIEN, BST 시퀀스 인코더 중 하나를 바꿔 끼운다.

**문제.** 학습 파일은 10,704,179행·119열이고 클릭은 204,179건(1.9%)이다. 정형 피처 사이의 비선형 상호작용을 배우면서, 길이가 제각각인 행동 이력을 정해진 메모리 안에서 표현해야 한다. 평가는 드문 클릭을 얼마나 위로 올리느냐로 한다.

**결과.** 층화 85/15 검증 분할(seed 42, 설정별 1회 실행)에서 DCN-V2 + DIEN은 ROC-AUC 0.7413, PR-AUC 0.0792를 기록했다. 무작위 순위의 PR-AUC 0.0191보다 약 4배 높다. cross 타워 없는 DIN도 0.7412 / 0.0779여서, 이번 단일 seed 실행에서는 cross 타워의 이득이 확인되지 않았다.

[README](README.ko.md) · [코드](src/) · [실행 메타데이터](results/)

**모델.** CrossNetMix는 저차원·게이팅 expert로 명시적 교차항을 만들고, 병렬 MLP는 더 일반적인 비선형 패턴을 배운다. 두 출력은 마지막 예측 헤드에서만 시퀀스 관심 벡터와 합쳐진다. 시퀀스 인코더는 같은 정형 골격 위에서 바꿔 끼운다.

- **DIN**(Deep Interest Network)은 현재 후보에서 과거 아이템 임베딩으로 attention을 건다.
- **DIEN**(Deep Interest Evolution Network)은 이력을 GRU(gated recurrent unit)에 통과시킨 뒤, 변해 가는 관심 상태에 attention을 건다.
- **BST**(Behavior Sequence Transformer)는 attention pooling 전에 positional encoding과 Transformer encoder를 더한다.

**데이터와 학습 설정.** 입력은 `clicked`(라벨), `seq`(행동 시퀀스), `ID`(제출 식별자)다. 연속형은 `float32`로 바꾸고 결측은 0으로 채운다. 범주 매핑은 학습 분할에서만 만들고, 처음 보는 값은 OOV(out-of-vocabulary) 인덱스로 처리한다. 모든 실행은 전체 데이터 학습, batch size 512, 학습률 0.001, dropout 0.2, 3 epoch, AdamW, cosine annealing, ROC-AUC 기반 early stopping을 쓴다.

## 주요 결정

**1. 행동 시퀀스 표현.**
선택지: 쉼표로 구분된 값을 숫자 그대로 LSTM에 넣기(첫 노트북 `baseline.ipynb`), 모든 아이템 ID를 임베딩하기, 최근 50개로 자르고 ID를 262,144개 버킷으로 해시하기.
선택: 자르기와 해싱. 왼쪽 padding으로 배치 길이를 맞추고, 빈 시퀀스에 attention을 걸 때 `NaN`이 나오지 않게 막았다.
이유: 아이템은 크기가 아니라 범주라서 임베딩이 필요하다. 전체 ID 어휘는 1,070만 행 파일에서 메모리 상한이 없지만, 해싱은 상한이 정해진다.
대가: 해시 충돌로 서로 다른 아이템이 섞이고, 50개보다 오래된 이력은 버려진다. 두 영향 모두 재 보지 않았다.

**2. 1.9% 클릭률 다루기.**
선택지: 음성 표본 줄이기(기준 노트북은 클릭 전부와 그 두 배의 미클릭만 남겨 612,537행), 또는 전체 행을 두고 손실에서 양성에 가중치 주기.
선택: 전체 데이터에 `BCEWithLogitsLoss(pos_weight = negative / positive)`(학습 분할 기준 51.4).
이유: 검증 분할이 실제 클릭률을 유지하므로, PR-AUC를 인위적인 33%가 아니라 실제 기준선과 비교해 읽을 수 있다.
대가: 전체 파일로 학습하느라 실행을 3 epoch로 제한했다. 가중 손실은 예측 확률을 이동시키므로 점수는 보정된 클릭 확률이 아니라 순위로만 읽어야 한다.

**3. 인코더 비교 방식.**
선택지: 구조마다 따로 튜닝하기, 또는 정형 골격·분할·seed·학습 예산을 고정하고 시퀀스 인코더만 바꾸기.
선택: 골격을 고정한 교체 비교, 그리고 cross 타워를 뺀 실행 하나를 비교점으로 추가.
이유: 같은 정형 피처 위에서 이력 표현마다 무엇이 더해지는지 따로 볼 수 있다.
대가: 설정마다 seed가 하나이고, cross 타워를 뺀 실행은 피처 유형 구분이 다르다(한계 참고).

**4. 판단 지표.**
선택지: 지표 하나로 우승 모델을 정하기, 또는 ROC-AUC와 PR-AUC를 나란히 보고하기.
선택: 둘 다 보고한다. early stopping은 ROC-AUC로, 양성 순위 판단은 PR-AUC로 한다.
이유: 지표마다 최고 설정이 다르고, 전체 순위와 상위 후보의 정밀도 중 무엇이 중요한지는 쓰임새에 달려 있다.
대가: 추천 모델을 하나로 정하지 않는다.

## 결과

| 실행 | 검증 ROC-AUC | 검증 PR-AUC | 해석 |
|---|---:|---:|---|
| 무작위 순위(참고) | 0.5000 | 0.0191 | 클릭률과 같음 |
| cross 타워 없는 DIN | 0.7412 | 0.0779 | DCN-V2 없는 비교점 |
| DCN-V2 + DIN | 0.7402 | 0.0775 | attention 기반 시퀀스 인코더 |
| DCN-V2 + 자동 튜닝 BST | 0.7403 | 0.0783 | Transformer 기반 시퀀스 인코더 |
| DCN-V2 + DIEN | 0.7413 | **0.0792** | PR-AUC 최고 |
| DCN-V2 + DIEN + 세그먼트 관심 모듈(DUSIN, full) | **0.7417** | 0.0780 | ROC-AUC 최고 |

PR-AUC 기준으로는 DIEN이 기준 설정이고, 최종 모델은 운영 목표에 맞춰 골라야 한다. cross 타워 없는 DIN 실행은 DCN-V2 + DIEN과 ROC-AUC 차이가 0.0001 이내다.

## 한계

- 검증 지표뿐이며 리더보드·외부 테스트·온라인 A/B 결과는 없다.
- 설정마다 seed가 하나라서, 0.001 안팎의 ROC-AUC 차이는 실행 간 편차 범위다.
- 시간 순서가 아닌 무작위 층화 분할이므로 이후 시점 트래픽에서의 성능은 알 수 없다.
- 비교 조건이 맞지 않는다. cross 타워 없는 DIN 실행은 피처 유형 구분이 달라(연속 111 / 범주 6, DCN-V2 실행은 85 / 32) 조건을 맞춘 ablation이 아니라 참고용 비교점이다.

## 다음 단계

- 피처 유형을 맞추고 cross 타워 유무를 여러 seed로 비교하는 ablation으로, DCN-V2가 비용만큼의 값을 하는지 가린다.
- 광고 의사결정에 쓸 수 있도록 시간 순서 평가, calibration, Precision/Recall@K를 더한다.
- 데이터 버전·분할별 건수·패키지 버전·장비를 담은 실험 매니페스트를 만든다.

## 근거 파일

- [학습 파이프라인](src/train.py)
- [데이터·시퀀스 처리](src/data.py)
- [DCN-V2, DIN, DIEN, BST 구현](src/models.py)
- [DIEN 메타데이터](results/dcnv2_dien_meta.json), [DUSIN 메타데이터](results/dcnv2_dien_dusin_full_meta.json), [cross 타워 없는 DIN 메타데이터](results/din_meta.json)
- [행 수·클릭 수](notebooks/baseline.ipynb)
- [실행 매니페스트](research/RUN_MANIFEST.md)
