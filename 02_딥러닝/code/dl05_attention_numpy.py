"""
[실습 05] Self-Attention / Multi-Head / Causal Mask / 위치 인코딩을 NumPy로 구현하고 성질 확인
"""
import numpy as np
from _plot_setup import plt, save
rng = np.random.default_rng(0)

def softmax(z, axis=-1):
    z = z - z.max(axis, keepdims=True); e = np.exp(z); return e / e.sum(axis, keepdims=True)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(d_k)      # (.., T, T)
    if mask is not None: scores = np.where(mask, scores, -1e9)
    A = softmax(scores)
    return A @ V, A

# 1) 간단한 예: 문장 토큰 6개
tokens = ["나는", "어제", "서울에서", "친구를", "만났다", "."]
T, d = len(tokens), 16
X = rng.normal(size=(T, d))
Wq, Wk, Wv = (rng.normal(0, 1 / np.sqrt(d), (d, d)) for _ in range(3))
out, A = attention(X @ Wq, X @ Wk, X @ Wv)
causal = np.tril(np.ones((T, T), bool))
out_c, A_c = attention(X @ Wq, X @ Wk, X @ Wv, mask=causal)
print("[어텐션 가중치] 각 행의 합 =", A.sum(1).round(3))
print("[Causal mask] 위 삼각(미래) 가중치 최대값 =", A_c[~causal].max())

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for a, M, t in [(ax[0], A, "Self-Attention (양방향, BERT식)"), (ax[1], A_c, "Causal Self-Attention (GPT식)")]:
    im = a.imshow(M, cmap="Blues"); a.set_xticks(range(T)); a.set_yticks(range(T))
    a.set_xticklabels(tokens, rotation=45); a.set_yticklabels(tokens); a.set_title(t); a.set_xlabel("Key (참조 대상)"); a.set_ylabel("Query")

# 2) √d_k 스케일링이 왜 필요한가
print("\n[스케일링 효과] d_k별 softmax 최대 가중치 평균 (1에 가까우면 포화 → 기울기 소실)")
for dk in (4, 64, 512):
    q, k = rng.normal(size=(200, dk)), rng.normal(size=(200, dk))
    raw = softmax(q @ k.T).max(1).mean(); scaled = softmax(q @ k.T / np.sqrt(dk)).max(1).mean()
    print(f"  d_k={dk:3d}: 스케일링 X {raw:.3f}  |  스케일링 O {scaled:.3f}")

# 3) Multi-Head Attention
def multi_head(X, h, Wq, Wk, Wv, Wo, mask=None):
    T, d = X.shape; dk = d // h
    split = lambda M: M.reshape(T, h, dk).transpose(1, 0, 2)      # (h, T, dk)
    o, A = attention(split(X @ Wq), split(X @ Wk), split(X @ Wv), mask)
    return o.transpose(1, 0, 2).reshape(T, d) @ Wo, A
Wo = rng.normal(0, 1 / np.sqrt(d), (d, d))
mh_out, mh_A = multi_head(X, 4, Wq, Wk, Wv, Wo, causal)
print(f"\n[Multi-Head] 입력 {X.shape} → 출력 {mh_out.shape}, 헤드별 어텐션 {mh_A.shape}")

# 4) 순서 정보가 없다는 성질: 입력 순서를 섞으면 출력도 똑같이 섞일 뿐 (permutation equivariance)
perm = rng.permutation(T)
o1, _ = attention(X @ Wq, X @ Wk, X @ Wv)
o2, _ = attention(X[perm] @ Wq, X[perm] @ Wk, X[perm] @ Wv)
print(f"[순서 불변성] 섞은 입력의 출력 == 원래 출력을 섞은 것? {np.allclose(o1[perm], o2)}  → 그래서 위치 인코딩이 필요")

# 5) 사인/코사인 위치 인코딩
def sinusoidal_pe(T, d):
    pos = np.arange(T)[:, None]; i = np.arange(d // 2)[None]
    ang = pos / 10000 ** (2 * i / d)
    pe = np.zeros((T, d)); pe[:, 0::2] = np.sin(ang); pe[:, 1::2] = np.cos(ang); return pe
PE = sinusoidal_pe(100, 64)
im = ax[2].imshow(PE.T, aspect="auto", cmap="RdBu"); ax[2].set_title("사인/코사인 위치 인코딩"); ax[2].set_xlabel("위치"); ax[2].set_ylabel("차원")
save(fig, "dl05_attention.png")
sim = PE @ PE[50]
print(f"[위치 인코딩] 위치 50과의 내적: 50→{sim[50]:.1f}, 51→{sim[51]:.1f}, 60→{sim[60]:.1f}, 90→{sim[90]:.1f}  (가까울수록 큼)")

# 6) RoPE: 회전 위치 임베딩 → q·k 가 '상대 위치'에만 의존
def rope(x, pos):
    d = x.shape[-1]; theta = pos / 10000 ** (np.arange(0, d, 2) / d)
    x1, x2 = x[0::2], x[1::2]; c, s = np.cos(theta), np.sin(theta)
    out = np.empty_like(x); out[0::2] = x1 * c - x2 * s; out[1::2] = x1 * s + x2 * c; return out
q, k = rng.normal(size=64), rng.normal(size=64)
print(f"[RoPE] q@pos3·k@pos1 = {rope(q,3)@rope(k,1):.4f},  q@pos10·k@pos8 = {rope(q,10)@rope(k,8):.4f}  (상대거리 2로 같으면 값도 같음)")

# 7) 계산량: 시퀀스 길이에 따른 어텐션 행렬 메모리
for L in (1_000, 8_000, 128_000):
    print(f"  시퀀스 {L:>7,} 토큰: 어텐션 행렬 {L*L:,}칸 = fp16 기준 {L*L*2/1e9:,.3f} GB (헤드 1개, 층 1개)")
