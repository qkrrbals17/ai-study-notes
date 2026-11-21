# Transformer

> "Attention Is All You Need" (Vaswani et al., 2017) — 현대 AI(LLM, ViT, 멀티모달)의 핵심 구조

## 1. 전체 구조
```
입력 토큰 → 임베딩 + 위치 인코딩
   → [인코더 블록 × N]  ──┐
                          ▼
출력 토큰 → 임베딩 + 위치 → [디코더 블록 × N] → Linear → Softmax
```
블록 구성 (Pre-LN 기준, 현대 표준):
```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

## 2. Self-Attention (Scaled Dot-Product)
각 토큰이 시퀀스의 **모든 토큰을 보고** 자신에게 필요한 정보를 가중합

- Q = XW_Q (Query: 내가 찾는 것)
- K = XW_K (Key: 내가 가진 꼬리표)
- V = XW_V (Value: 실제 전달할 내용)

**Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V**

- √d_k로 나누는 이유: 내적 값이 커져 softmax가 포화되는 것 방지
- 계산량 O(T²·d) → 긴 문맥이 비싼 이유

## 3. Multi-Head Attention
- 어텐션을 h개 헤드로 나눠 병렬 수행 → 서로 다른 관계(문법, 의미, 지시어 등)를 학습
- 결과를 concat 후 선형 변환
- 변형: **MQA / GQA** (K,V 헤드를 공유해 추론 메모리 절약 — 최근 LLM 표준)

## 4. 마스킹
- **Padding mask**: 패딩 토큰 무시
- **Causal(look-ahead) mask**: 디코더가 미래 토큰을 못 보게 → 자기회귀 생성

## 5. 위치 정보 (Positional Encoding)
어텐션은 순서를 모름 → 위치 정보 추가 필요
- 사인/코사인 고정 인코딩 (원 논문)
- 학습형 위치 임베딩 (BERT, GPT-2)
- **RoPE** (회전 위치 임베딩, 최근 LLM 표준), ALiBi → 길이 외삽에 유리

## 6. Feed-Forward Network
- 토큰별 독립 2층 MLP, 보통 차원 4배 확장
- 최근 LLM: **SwiGLU** 활성화
- **MoE(Mixture of Experts)**: FFN을 여러 전문가로 두고 토큰마다 일부만 활성 → 파라미터는 크게, 연산은 적게

## 7. 세 가지 계열
| 계열 | 구조 | 사전학습 목표 | 대표 | 용도 |
|---|---|---|---|---|
| Encoder-only | 양방향 어텐션 | MLM(빈칸 맞추기) | BERT, RoBERTa | 분류, 임베딩, 검색 |
| Decoder-only | Causal 어텐션 | 다음 토큰 예측 | GPT 계열, Llama, Claude 등 대부분 LLM | 생성, 대화, 에이전트 |
| Encoder-Decoder | 둘 다 + Cross-attn | Span corruption 등 | T5, BART | 번역, 요약 |

## 8. 다른 도메인으로의 확장
- **ViT**: 이미지를 16×16 패치로 잘라 토큰처럼 처리
- **CLIP**: 이미지·텍스트 임베딩을 대조학습으로 정렬
- 음성(Whisper), 멀티모달 LLM, 시계열, 단백질 구조 등

## 9. 효율화 기법
- **FlashAttention**: GPU 메모리 접근 최적화로 정확한 어텐션을 빠르게
- **KV Cache**: 생성 시 이전 토큰의 K,V 재사용
- Sparse/Sliding-window attention, Linear attention

## 10. 미니 구현
```python
import torch, torch.nn as nn, torch.nn.functional as F, math

class CausalSelfAttention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
    def forward(self, x):                         # (B, T, d)
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = [t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v)]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, d))

class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CausalSelfAttention(d, h)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))
```
👉 추천 실습: Andrej Karpathy의 "Let's build GPT" / nanoGPT 따라 하기

## ✅ 체크리스트
- [ ] Q, K, V의 의미와 어텐션 수식을 설명할 수 있다
- [ ] √d_k 스케일링, causal mask의 이유를 안다
- [ ] BERT와 GPT의 구조·학습 목표 차이를 안다
- [ ] 작은 GPT를 직접 구현해 텍스트를 생성해 봤다
