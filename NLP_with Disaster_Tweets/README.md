# 🚢 NLP with Disaster Tweets

> Kaggle 대회 — 트윗이 실제 재난 상황인지 아닌지 분류하는 이진 분류 태스크  
> **목표: F1 Score 0.85 돌파** | 최종 결과: **F1 0.845**

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 태스크 | 트윗 텍스트 이진 분류 (재난 O / X) |
| 평가 지표 | F1 Score |
| 데이터 | Kaggle NLP Getting Started |
| 최종 모델 | RoBERTa-base, 3-fold 앙상블 |
| 최고 점수 | **0.845** (test set) |

---

## 🔑 핵심 인사이트 요약

1. **도메인 일치가 성능을 결정한다** — Twitter 데이터로 사전학습된 BERTweet가 Wikipedia 기반 BERT 대비 일관되게 우세
2. **데이터 노이즈가 실질적 병목** — OOF 분석 결과 FN(False Negative) 샘플의 confidence가 0.75~0.9 구간에 집중, 라벨 자체의 노이즈 가능성 확인
3. **Pseudo Labeling은 베이스 모델이 충분히 강할 때 효과** — 노이즈가 많은 환경에서 17% 데이터 증량은 오히려 성능 하락 유발

---

## 📊 주요 실험 결과

| 모델 | 설정 | Val Score | Test Score | 비고 |
|------|------|-----------|------------|------|
| bert-base | max_len=128, lr=2e-5, epoch=4 | 0.796 | 0.830 | 베이스라인 |
| bertweet | max_len=128, lr=2e-5, epoch=4 | 0.803 | **0.843** | 도메인 일치 효과 |
| roberta-base | seed=42, cosine, epoch=4 | 0.8084 | 0.832 | - |
| roberta-clean-base | 3-fold, cosine, epoch=4 | 0.7925 ±0.0088 | **0.845** | 최고 성능 |
| roberta-clean-pseudo | pseudo labeling 추가 | 0.8269 ±0.0044 | 0.837 | 성능 하락 |

---

## 🧪 실험 로그

### Test 1 — 베이스라인 모델 선정

**목적**: 태스크에 적합한 사전학습 모델 선정

**핵심 판단**: 모델 선택 기준을 사전학습 데이터의 도메인 일치도에 둠

| 모델 | 사전학습 데이터 | Test Score |
|------|--------------|------------|
| BERT | Wikipedia + Books | 0.830 |
| BERTweet | Twitter | **0.843** |
| RoBERTa-base | 대규모 영어 코퍼스 | - |

**레슨런**: BERTweet가 Twitter 도메인 데이터로 학습되어 확실한 우위. **사전학습 데이터와 타깃 도메인의 일치가 파인튜닝 성능에 직접적인 영향**을 미침.

---

### Test 2 — LR Scheduling 도입

**목적**: warmup + cosine decay 적용으로 학습 안정화

**결과**: 성능 수치 차이는 크지 않으나 학습 안정성 향상, 이후 실험 기본 설정으로 채택

---

### Test 3 — FGM (Fast Gradient Method) 적용

**목적**: epoch 1 이후 val loss 상승 → 더 이상 학습할 게 없다고 판단, 일반화 방향으로 전환

**방법**: embedding에 adversarial noise를 추가한 샘플을 같이 학습

**결과**:

| epsilon | Val Score | Test Score |
|---------|-----------|------------|
| 0.5 | 0.8025 | 0.834 |
| **1.0** | **0.8051** | **0.843** |
| 1.5 | 0.8075 | - |
| 2.0 | 0.8125 | 0.835 |
| 5.0 | 0.8137 | 0.835 |

**레슨런**: epsilon=1.0이 최적. **epsilon이 커질수록 val은 올라가지만 test는 오히려 고정 또는 하락** — val-test 간 분포 차이가 있음을 시사.

---

### Test 4 — Val 비율 조정

**목적**: val이 test 경향을 못 따라가는 문제 해결 시도 (train:val = 0.9:0.1 → 0.8:0.2)

**결과**: 개선 없음

**레슨런**: val set 비율 조정으로는 val-test 분포 차이를 해소할 수 없음. 근본 원인은 **데이터 노이즈**일 가능성이 높다고 판단, 이후 실험 방향 전환.

---

### Test 5 — Label Smoothing

**목적**: 데이터 노이즈가 심각해 보여 label smoothing 적용

**결과**: 경향성 없음, 성능 변화 미미

**레슨런**: label smoothing은 노이즈 데이터에 대한 근본적 해결책이 아님. **노이즈 원인을 직접 분석하는 방향(→ Test 7)이 필요**.

---

### Test 6 — K-Fold 앙상블 도입

**목적**: val-test 경향 맞추기, 앙상블을 통한 분산 감소

**추가 발견**: RoBERTa에 BERTweet 방식 전처리를 적용했더니 성능 향상

**레슨런**: **전처리 방식만으로도 도메인 일치도를 높일 수 있음**. 모델 교체 없이 전처리만으로 성능 개선 가능.

---

### Test 7 — OOF 기반 데이터 노이즈 분석 ⭐

**목적**: 전체 train set에 대한 confidence 추출로 노이즈 후보 식별

**분석 방법**:
- **고확신 오답** (모델 confidence 높은데 라벨 반대) = 노이즈 후보
- **저확신 샘플** (풍자, 비유, 문맥 부족) = 본질적으로 어려운 샘플

**주요 발견**:

```
"well it feels like im on fire."          target=0  → 비유적 표현
"The Lightning out here is something serious"  target=1  → 실제인지 비유인지 불명확
"They sky was ablaze tonight in Los Angeles"   target=0  → 노을 묘사
"PRAY! For EAST COAST FOREST FIRES!"      target=1  → 실제 산불인지 불명확
```

FN(Disaster인데 non-disaster로 예측) 쪽 confidence가 0.75~0.9 구간에 집중 → **라벨 자체의 노이즈가 상당한 수준**

**실험 결과**: 고확신 오답 144개 중 34개/82개 제거 → 성능 하락 또는 차이 없음

**레슨런**: 노이즈 제거 방향은 맞으나 34~82개 수준의 제거로는 영향이 작음. 또한 **저확신 샘플(풍자, 비유)은 모델 문제가 아니라 태스크 자체의 난이도**로 봐야 함.

---

### Test 8 — Pseudo Labeling

**목적**: test set의 고확신 예측을 train에 추가해 데이터 증강

**방법**:
- 베이스 모델: RoBERTa-base, 3-fold 앙상블
- 비대칭 임계값 적용 (class 0: ≥0.90, class 1: ≥0.95)
- 추가 샘플 1,303개 (train 대비 +17%)
- MC Dropout (Bayesian uncertainty) 버전도 실험

**결과**:

| 모델 | Val Score | Test Score |
|------|-----------|------------|
| roberta-clean-base | 0.7925 ±0.0088 | **0.845** |
| roberta-clean-pseudo | 0.8269 ±0.0044 | 0.837 (-0.008) |
| roberta-clean-pseudo-mc | 0.8207 ±0.0084 | 0.845 |

**레슨런**: 
- **Pseudo Labeling은 베이스 모델이 충분히 강할 때 효과적** — 베이스 모델이 목표 성능에 미치지 못한 상태에서의 pseudo label은 노이즈를 증폭시킴
- 3-fold 앙상블 기준 고확신 샘플에도 노이즈 포함 가능성 존재
- train/test 간 분포 차이가 있는 상태에서 17% 증량은 오히려 역효과

---

## 🔍 전체 회고

**잘된 점**
- 단순 하이퍼파라미터 튜닝에 머물지 않고 OOF 분석으로 데이터 품질 문제를 정량적으로 확인한 것
- 실험 실패 시 원인을 분포 차이, 노이즈 수준 등으로 구체적으로 분석한 것

**아쉬운 점**
- K-Fold 확립 이전에 FGM 효과를 측정해 신뢰도가 낮았음
- 노이즈 원인 분석(Test 7) 전에 label smoothing(Test 5)을 먼저 시도 — 분석 선행 후 해결책 적용 순서가 이상적

**목표(F1 0.85) 미달성 원인**
- 데이터 자체의 라벨 노이즈가 상당 수준으로, 모델/학습 기법 개선만으로는 한계가 있었음
- val-test 분포 차이로 인해 val score를 신뢰하기 어려운 환경이었음

---

## ⚙️ 실험 환경

```
- Python 3.x
- PyTorch + HuggingFace Transformers
- 주요 모델: roberta-base, bertweet, deberta-v3-base
- 공통 설정: max_len=128, lr=2e-5, epoch=4, batch_size=16/32
```
