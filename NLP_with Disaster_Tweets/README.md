# Disaster Tweets Classification

> 이 프로젝트는 단순한 leaderboard score 개선뿐만 아니라, NLP 분류 문제에서 validation 신뢰도, label noise, 일반화 성능을 분석하는 데 초점을 두었습니다.

Kaggle의 **Natural Language Processing with Disaster Tweets** 대회를 기반으로, 트윗이 실제 재난과 관련된 내용인지 아닌지를 분류하는 Binary Text Classification 프로젝트입니다.

- Competition: Natural Language Processing with Disaster Tweets
- Task: Tweet Binary Classification
- Metric: F1 Score
- Goal: Public Leaderboard F1 Score 0.85 근접 또는 달성
- Main Models: BERT, RoBERTa, DeBERTa, BERTweet

---

## 1. Project Overview

이 프로젝트는 재난 관련 트윗 분류 문제를 해결하기 위해 다양한 pretrained language model과 학습 기법을 실험한 과정입니다.

초기에는 BERT, RoBERTa, DeBERTa, BERTweet 등 여러 pretrained model을 비교했고, 이후 validation score와 test score의 괴리를 줄이기 위해 K-Fold validation, FGM, label smoothing, OOF error analysis, pseudo labeling 등을 실험했습니다.

---

## 2. Dataset

Kaggle에서 제공하는 Disaster Tweets 데이터셋을 사용했습니다.

각 샘플은 다음과 같은 정보를 포함합니다.

| Column | Description |
|---|---|
| `id` | Tweet ID |
| `keyword` | Tweet keyword |
| `location` | Tweet location |
| `text` | Tweet text |
| `target` | 1: real disaster, 0: not disaster |

---
## 3. Experiment Result

| 모델 | 설정 | Val Score | Test Score | 비고 |
|------|------|-----------|------------|------|
| bert-base | max_len=128, lr=2e-5, epoch=4 | 0.796 | 0.830 | 베이스라인 |
| bertweet | max_len=128, lr=2e-5, epoch=4 | 0.803 | **0.843** | 도메인 일치 효과 |
| roberta-base | cosine, epoch=4 | 0.8084 | 0.832 | - |
| roberta-clean-base | 3-fold, cosine, epoch=4 | 0.7925 ±0.0088 | **0.845** | 최고 성능 |
| roberta-clean-pseudo | pseudo labeling 추가 | 0.8269 ±0.0044 | 0.84462 | 성능 하락 |

---

## 4. Experiment Flow

전체 실험은 아래와 같은 흐름으로 진행했습니다.

```text
Baseline Model Comparison
→ Learning Rate Scheduler
→ FGM
→ Validation Strategy / K-Fold
→ OOF Error Analysis
→ Label Noise Handling
→ Label Smoothing
→ Pseudo Labeling
```
---

## 5. Experiments

### 5.1 Baseline Model Comparison

먼저 여러 pretrained language model을 비교했습니다.

| Model | Test Score | Notes |
|---|---:|---|
| BERT-base | 0.830 | Baseline |
| BERTweet | 0.843 | Twitter domain pretrained  |
| RoBERTa-base | 0.832 | Stable baseline |
| DeBERTa-v3-base | 0.819 | Lower performance in this setting |

초기 실험에서는 Twitter 데이터로 사전학습된 **BERTweet**이 가장 좋은 성능을 보였습니다. 이를 통해 pretrained model을 선택할 때 모델 구조뿐 아니라, **사전학습 데이터와 현재 task 데이터의 domain 유사성**도 중요하다는 것을 확인했습니다.

---

### 5.2 Learning Rate Scheduler

Warmup과 cosine decay scheduler를 적용했습니다.

| Experiment | Result |
|---|---|
| Warmup + Cosine Scheduler | 성능 차이는 크지 않았으나 이후 실험의 기본 설정으로 사용 |

Scheduler 적용만으로 큰 성능 향상은 없었지만, 학습률을 안정적으로 조절하기 위해 이후 실험에 포함했습니다.

---

### 5.3 FGM: Fast Gradient Method

초기 모델에서 epoch 1 이후 validation loss가 증가하는 경향이 관찰되었습니다. 이를 overfitting 가능성으로 보고, 일반화 성능을 개선하기 위해 FGM을 적용했습니다.

FGM은 embedding에 작은 adversarial noise를 추가해 모델이 더 robust하게 학습되도록 하는 방법입니다.

| FGM Epsilon | Validation Score | Test Score |
|---:|---:|---:|
| 0.5 | 0.8025 | 0.83358 |
| 1.0 | 0.8051 | 0.84308 |
| 1.5 | 0.8075 | - |
| 2.0 | 0.8125 | 0.835 |
| 5.0 | 0.8137 | 0.835 |

Validation score만 보면 epsilon이 커질수록 좋아지는 것처럼 보였지만, test score 기준으로는 `epsilon=1.0`이 가장 좋은 결과를 보였습니다.

---

### 5.4 Validation Strategy and K-Fold

초기 실험에서는 validation score와 public leaderboard score의 경향이 일치하지 않는 문제가 있었습니다. 이를 해결하기 위해 validation split 비율을 조정하고, Stratified K-Fold를 적용했습니다.

| Model | K-Fold | Validation Score | Test Score |
|---|---:|---:|---:|
| RoBERTa-clean-base | 3 | 0.7925 ± 0.0088 | 0.845 |
| BERTweet-clean-cosine | 3 | 0.7990 ± 0.0057 | 0.840 |
| BERTweet-clean-cosine-FGM | 3 | 0.8027 ± 0.0132 | 0.84370 |

`RoBERTa-clean-base`가 가장 높은 test score를 기록했습니다.

초기 baseline에서는 BERTweet이 강했지만, K-Fold와 전처리 실험 이후에는 RoBERTa 계열이 더 좋은 결과를 보였습니다.

---

### 5.5 OOF Error Analysis

K-Fold 모델의 OOF prediction을 활용해 train set의 예측 confidence를 분석했습니다.

분석 목적은 다음과 같았습니다.

- 모델이 높은 confidence로 틀린 샘플 확인
- label noise 후보 탐색
- 모델이 헷갈려 하는 ambiguous sample 분석

예시:

| Text | Target | Comment |
|---|---:|---|
| `well it feels like im on fire.` | 0 | 비유적 표현 |
| `The Lightning out here is something serious` | 1 | 실제인지 비유인지 불명확 |
| `They sky was ablaze tonight in Los Angeles` | 0 | 노을 묘사 |
| `Advice from Noah: Don't go running in a thunderstorm` | 0 | 유머성 표현 |
| `PRAY! For EAST COAST FOREST FIRES!` | 1 | 실제 산불인지 불명확 |

OOF 분석 결과, Disaster Tweets 데이터에는 실제 재난과 비유적 표현이 섞여 있어 label noise와 ambiguous sample이 성능에 큰 영향을 줄 수 있음을 확인했습니다.

---

### 5.6 Noisy Sample Removal

OOF 분석을 통해 고확신 오답 샘플을 찾고, 모델 확신도 높은데 라벨이 반대인 샘플들(noisy sample)을 제거하는 실험을 진행했습니다.

| Experiment | Validation Score | Test Score |
|---|---:|---:|
| RoBERTa-clean-base | 0.7925 ± 0.0088 | 0.845 |
| Remove 34 noisy samples | 0.7916 ± 0.0215 | 0.841 |
| Remove 82 noisy samples | 0.8090 ± 0.0055 | 0.84002 |

고확신 오답 샘플을 제거했지만 test score는 오히려 하락했습니다.

---

### 5.7 Label Smoothing

데이터 노이즈를 완화하기 위해 label smoothing을 적용했습니다.

| Experiment | Validation Score | Test Score |
|---|---:|---:|
| BERTweet-clean-cosine-FGM | 0.8288 | 0.84247 |
| Label Smoothing = 0.1 | - | 0.84033 |
| Label Smoothing = 0.15 | - | 0.8299 |
| Label Smoothing = 0.2 | - | 0.83726 |

Label smoothing은 뚜렷한 성능 향상으로 이어지지 않았습니다. 이를 통해 label noise가 존재한다고 해서 label smoothing이 항상 효과적인 것은 아니라는 점을 확인했습니다.

---
### 5.8 Pseudo Labeling

RoBERTa 3-Fold ensemble 모델을 활용해 test set에 대한 예측 확률을 구하고,  
confidence가 높은 샘플을 pseudo label로 추가했습니다.

사용한 threshold는 다음과 같습니다.

| Class | Threshold |
|---|---:|
| Class 0 | `prob_class0 >= 0.90` |
| Class 1 | `prob_class1 >= 0.95` |

추가된 pseudo-labeled sample 수는 다음과 같습니다.

| Class | Count |
|---|---:|
| Class 0 | 676 |
| Class 1 | 627 |
| Total | 1,303 |

이는 train set 대비 약 17%에 해당하는 규모였습니다.

추가로 `roberta-clean-pseudo-mc` 실험에서는  
**Uncertainty-aware Self-training for Text Classification with Few Labels** 논문을 참고해  
베이지안 관점의 uncertainty-aware sampling을 적용했습니다.

단순히 confidence가 높은 샘플을 추가하는 방식은 pseudo label의 불확실성을 충분히 반영하지 못할 수 있다고 판단했습니다.  
따라서 Monte Carlo 기반 sampling을 활용해 예측 불확실성을 고려하고,  
상대적으로 신뢰도가 높은 pseudo-labeled sample을 선별하는 방향으로 실험했습니다.

| Experiment | Validation Score | Test Score | Notes |
|---|---:|---:|---|
| RoBERTa-clean-base | 0.7925 ± 0.0088 | 0.845 | Baseline |
| RoBERTa-clean-pseudo | 0.8269 ± 0.0044 | 0.837 | Confidence-based pseudo labeling |
| RoBERTa-clean-pseudo-mc | 0.8207 ± 0.0084 | 0.84462 | Uncertainty-aware sampling |
| RoBERTa-clean-pseudo-mc-v2 | - | 0.83512 | Modified pseudo labeling setting |

Pseudo labeling을 적용했지만 최종적으로 baseline test score를 넘지는 못했습니다.

특히 단순 confidence-based pseudo labeling은 test score가 크게 하락했고,  
uncertainty-aware sampling을 적용한 `roberta-clean-pseudo-mc`는 baseline에 근접한 성능을 보였습니다.

이를 통해 confidence가 높은 pseudo label에도 noise가 포함될 수 있으며,  
pseudo labeling에서는 단순 확률값뿐 아니라 예측 불확실성을 함께 고려하는 것이 중요하다는 점을 확인했습니다.

---

## 6. Final Result

가장 좋은 test score를 기록한 모델은 다음과 같습니다.

| Model | Validation Score | Test Score |
|---|---:|---:|
| RoBERTa-clean-base | 0.7925 ± 0.0088 | 0.845 |

초기 baseline에서는 BERTweet이 좋은 성능을 보였지만, K-Fold 및 전처리 실험 이후에는 RoBERTa-clean-base가 가장 좋은 public leaderboard score를 기록했습니다.

---


## 8. Lesson Learned

### 8.1 Validation score improvement does not always mean better generalization

이번 프로젝트에서 가장 크게 배운 점은 validation score와 실제 test generalization 성능이 항상 같은 방향으로 움직이지 않는다는 점이었다.

FGM, label smoothing, pseudo labeling 등 여러 실험에서 validation score는 상승했지만, public leaderboard score는 오히려 하락하는 경우가 반복적으로 발생했다.

이를 통해 단일 validation split이나 단일 score만 보고 모델을 선택하는 것은 매우 위험하다는 점을 확인했다.

특히 NLP classification에서는 validation set의 분포와 실제 unseen data distribution 사이에 차이가 존재할 수 있기 때문에, K-Fold validation, fold variance, public score를 함께 확인해야 했다.

---

### 8.2 Model architecture was less important than data interpretation

초기에는 BERT, RoBERTa, DeBERTa, BERTweet 등 pretrained model 자체를 바꾸는 방향으로 성능 개선을 시도했다.

하지만 실험을 반복하면서 실제 성능에 더 큰 영향을 주는 요소는 모델 구조보다 다음과 같은 데이터 관련 문제라는 점을 확인했다.

- ambiguous tweet
- label noise
- confidence mismatch
- domain-specific 표현
- validation distribution mismatch

특히 OOF prediction 분석 과정에서 모델이 높은 confidence로 틀리는 샘플들을 확인하면서, 단순히 더 강한 모델을 사용하는 것보다 데이터를 어떻게 해석하고 검증하는지가 더 중요하다는 점을 배웠다.

---

### 8.3 More advanced techniques do not automatically improve performance

FGM, label smoothing, noisy sample removal, pseudo labeling 등 다양한 성능 개선 기법을 적용했지만, 대부분의 기법은 validation score 상승과 달리 test score 개선으로 이어지지 않았다.

특히 pseudo labeling은 confidence가 높은 샘플만 추가했음에도 성능이 하락했다.

이를 통해 다음과 같은 점을 배웠다.

- high confidence ≠ correct prediction
- pseudo label can amplify existing noise
- regularization methods are highly data-dependent
- stronger techniques can also overfit validation distribution

즉, 고급 기법을 적용하는 것 자체보다 현재 데이터와 validation 환경에서 왜 해당 기법이 필요한지 먼저 정의하는 것이 중요했다.

---

### 8.4 Uncertainty estimation matters in pseudo labeling

초기 pseudo labeling은 단순 confidence threshold 기반으로 진행했다.

하지만 이후 uncertainty-aware self-training 논문을 참고해 Monte Carlo 기반 sampling을 적용하면서, prediction probability 자체보다 prediction uncertainty가 더 중요할 수 있다는 점을 이해하게 되었다.

특히 confidence가 높은 prediction도 실제로는 noisy pseudo label일 수 있기 때문에, pseudo labeling에서는 uncertainty estimation이 중요한 역할을 한다는 점을 배웠다.

---

### 8.5 Experiment tracking is essential in iterative NLP research

이번 프로젝트에서는 모델, seed, scheduler, fold 수, FGM epsilon, pseudo labeling threshold 등을 지속적으로 기록하며 비교했다.

실험이 많아질수록 단순히 최고 점수만 기록하는 것이 아니라:

- 왜 해당 실험을 했는지
- 어떤 문제를 해결하려 했는지
- validation과 test score가 왜 다르게 움직였는지

를 함께 기록하는 것이 중요하다는 점을 느꼈다.

특히 이번 프로젝트에서는 성능 개선보다 “왜 특정 실험이 실패했는가”를 이해하는 과정이 더 큰 학습이 되었다.

---

