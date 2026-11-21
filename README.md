# AI Study Notes

> Machine Learning → Deep Learning → AI Agent
> 개념 정리와 직접 실행한 실습 코드로 쌓아 가는 AI 학습 기록

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

각 주제는 **개념 노트 → 실습 코드 → 실행 결과·그래프** 순서로 구성했습니다.
핵심 알고리즘은 라이브러리에 기대기 전에 **NumPy로 먼저 직접 구현**해 동작 원리를 확인하는 것을 원칙으로 했습니다.

## Contents

| 단계 | 주제 | 실습 하이라이트 |
|---|---|---|
| [01 머신러닝](01_머신러닝/README.md) | 편향-분산, 전처리와 데이터 누수, 회귀·분류, 앙상블, 비지도 학습, 모델 평가, 실전 워크플로우 | 경사하강법 직접 구현, 데이터 누수 재현 실험, ROC/PR·확률 보정 곡선, 교차검증 기반 모델 비교 |
| [02 딥러닝](02_딥러닝/README.md) | 신경망, 최적화·정규화, CNN, RNN, Transformer, 생성 모델, LLM, PyTorch 실전 | NumPy로 MLP 역전파·합성곱·LSTM·Attention 구현, Mini-GPT, Autoencoder·Diffusion, LoRA 미세조정 |
| [03 AI Agent](03_AI_Agent/README.md) | 에이전트 구조, 프롬프트, Tool calling, RAG, 메모리·계획, 멀티에이전트, MCP, 평가·안전 | 프레임워크 없이 에이전트 루프 구현, 도구 레지스트리, RAG 파이프라인, ReAct, 평가 하네스, MCP 서버 |

## Selected Figures

| Bias–Variance | Attention Weights |
|---|---|
| <img src="01_머신러닝/code/figures/ml01_poly_fits.png" width="400"> | <img src="02_딥러닝/code/figures/dl05_attention.png" width="400"> |

## Structure

```
ai-study-notes/
├── 01_머신러닝/   노트 8개 · code/ml01–ml08 · figures · outputs
├── 02_딥러닝/     노트 8개 · code/dl01–dl08 · figures · outputs
└── 03_AI_Agent/   노트 9개 · code/ag01–ag09 · outputs
```

- `code/outputs/` 에 각 실습의 실행 로그를, `code/figures/` 에 생성한 그래프를 함께 두어 코드를 돌리지 않고도 결과를 확인할 수 있습니다.

## Run

```bash
pip install -r requirements.txt
cd 01_머신러닝/code && python ml01_overfitting_bias_variance.py
```

- `*_torch.py`, `*_hf.py`, `dl08_*`: PyTorch 필요 (GPU 권장)
- `ag01`–`ag08`: API 키 없이 동작하도록 규칙 기반 Mock LLM으로 작성
- `ag07`: `pip install "mcp[cli]"` · `ag09`: `pip install anthropic` 후 환경 변수 `ANTHROPIC_API_KEY` 설정 (키는 코드에 넣지 않음)

## Roadmap

- [x] 머신러닝 기초 ~ 실전 워크플로우
- [x] 딥러닝 기초 ~ LLM
- [x] AI Agent 기초 ~ 평가·안전
- [ ] 각 단원 실습을 실제 데이터셋 / Kaggle 문제로 확장
- [ ] Agent 실습을 LangGraph 기반으로 재구현
