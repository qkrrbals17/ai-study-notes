"""
[실습 07] LLM 핵심 요소를 작게 직접 구현
1) BPE 토크나이저 학습
2) 디코딩 전략: temperature / top-k / top-p
3) LoRA: 저랭크 업데이트로 '적은 파라미터'로 미세조정 (NumPy 선형층)
4) 양자화(INT8/INT4)와 오차, KV 캐시 메모리 계산
"""
import re, numpy as np
from collections import Counter
from _plot_setup import plt, save
rng = np.random.default_rng(0)

# ---------- 1) BPE ----------
corpus = ("low lower lowest newer newest wider widest " * 5 + "딥러닝 딥러닝은 러닝 머신러닝 머신러닝은 " * 4).split()
def train_bpe(words, n_merges):
    vocab = Counter(tuple(w) + ("</w>",) for w in words)
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for w, c in vocab.items():
            for a, b in zip(w, w[1:]): pairs[(a, b)] += c
        if not pairs: break
        best = pairs.most_common(1)[0][0]; merges.append(best)
        new = {}
        for w, c in vocab.items():
            out, i = [], 0
            while i < len(w):
                if i < len(w) - 1 and (w[i], w[i+1]) == best: out.append(w[i] + w[i+1]); i += 2
                else: out.append(w[i]); i += 1
            new[tuple(out)] = c
        vocab = new
    return merges
def tokenize(word, merges):
    w = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(w) - 1:
            if w[i] == a and w[i+1] == b: w[i:i+2] = [a + b]
            else: i += 1
    return w
merges = train_bpe(corpus, 20)
print("[BPE] 처음 10개 병합 규칙:", ["".join(m) for m in merges[:10]])
for w in ["lowest", "newer", "slowest", "머신러닝은", "강화러닝"]:
    print(f"  {w:8s} → {tokenize(w, merges)}")
print("  → 처음 보는 단어(slowest, 강화러닝)도 아는 조각으로 분해됨 (OOV 문제 해결)")

# ---------- 2) 디코딩 전략 ----------
vocab = ["사과", "바나나", "포도", "딸기", "수박", "키위", "망고", "자두"]
logits = np.array([3.2, 2.9, 2.1, 1.5, 0.8, 0.3, -0.5, -1.0])
def softmax(z): e = np.exp(z - z.max()); return e / e.sum()
def top_k(p, k): q = np.where(p >= np.sort(p)[-k], p, 0); return q / q.sum()
def top_p(p, thr):
    idx = np.argsort(p)[::-1]; cum = np.cumsum(p[idx]); keep = idx[:np.searchsorted(cum, thr) + 1]
    q = np.zeros_like(p); q[keep] = p[keep]; return q / q.sum()
fig, ax = plt.subplots(1, 3, figsize=(15, 3.4))
for T_ in (0.5, 1.0, 1.5): ax[0].plot(softmax(logits / T_), "o-", label=f"T={T_}")
ax[0].set_title("Temperature"); ax[0].legend()
ax[1].bar(vocab, top_k(softmax(logits), 3)); ax[1].set_title("Top-k (k=3)")
ax[2].bar(vocab, top_p(softmax(logits), 0.8)); ax[2].set_title("Top-p (p=0.8)")
for a in ax: a.set_xticks(range(len(vocab))); a.set_xticklabels(vocab, fontsize=8)
save(fig, "dl07_decoding.png")
ent = lambda p: -np.sum(p[p > 0] * np.log(p[p > 0]))
for T_ in (0.2, 0.7, 1.0, 1.5):
    p = softmax(logits / T_); s = rng.choice(len(vocab), 1000, p=p)
    print(f"[Temperature {T_}] 엔트로피={ent(p):.2f}, 1000번 샘플 중 서로 다른 단어 {len(set(s))}개, 최빈='{vocab[np.bincount(s).argmax()]}'")
print(f"[Top-p 0.8] 후보로 남은 단어: {[v for v, q in zip(vocab, top_p(softmax(logits), .8)) if q > 0]}")

# ---------- 3) LoRA ----------
d_in, d_out, r = 256, 256, 4
W0 = rng.normal(0, 1 / np.sqrt(d_in), (d_out, d_in))                      # '사전학습' 가중치 (고정)
dW_true = rng.normal(size=(d_out, 4)) @ rng.normal(size=(4, d_in)) * 0.05 # 새 작업이 요구하는 변화 (저랭크라고 가정)
X = rng.normal(size=(2000, d_in)); Y = X @ (W0 + dW_true).T
A = rng.normal(0, 0.01, (r, d_in)); B = np.zeros((d_out, r)); alpha = 8; scale = alpha / r   # B=0 초기화 → 시작 시 원래 모델과 동일
lr, hist = 0.02, []
for step in range(1500):
    idx = rng.integers(0, len(X), 128); x, yt = X[idx], Y[idx]
    pred = x @ (W0 + scale * B @ A).T; err = pred - yt
    gW = err.T @ x / len(x)                                                 # dL/dW (W0는 업데이트 안 함)
    gB, gA = scale * gW @ A.T, scale * B.T @ gW
    B -= lr * gB; A -= lr * gA
    hist.append(np.mean(err ** 2))
base_mse = np.mean((X @ W0.T - Y) ** 2)
print(f"\n[LoRA] 학습 파라미터: {A.size + B.size:,}개 (전체 미세조정이면 {W0.size:,}개 → {100*(A.size+B.size)/W0.size:.1f}%)")
print(f"       원래 모델 MSE {base_mse:.4f} → LoRA 학습 후 MSE {hist[-1]:.2e}")
for name, d, L in [("7B급 (d=4096, 32층)", 4096, 32), ("70B급 (d=8192, 80층)", 8192, 80)]:
    full = 4 * d * d * L; lora = 4 * 2 * 16 * d * L    # q,k,v,o 투영에 r=16
    print(f"  {name}: 어텐션 투영 전체 {full/1e9:.2f}B vs LoRA(r=16) {lora/1e6:.1f}M 파라미터")

# ---------- 4) 양자화 & KV 캐시 ----------
w = rng.normal(0, 0.02, 102400).astype(np.float32)
for bits in (8, 4):
    qmax = 2 ** (bits - 1) - 1; s = np.abs(w).max() / qmax
    wq = np.clip(np.round(w / s), -qmax - 1, qmax) * s
    print(f"[양자화 INT{bits}] 상대오차 {np.linalg.norm(w-wq)/np.linalg.norm(w):.3%}, 메모리 {32//bits}배 절약 (FP32 대비)")
# 그룹 단위 스케일(블록 양자화)이 오차를 줄이는 이유
g = w.reshape(-1, 64); s = np.abs(g).max(1, keepdims=True) / 7
wq = (np.clip(np.round(g / s), -8, 7) * s).ravel()
print(f"[INT4 + 64개 그룹별 스케일] 상대오차 {np.linalg.norm(w-wq)/np.linalg.norm(w):.3%}")
def kv_cache_gb(layers, kv_heads, head_dim, seq, batch, bytes_=2):
    return 2 * layers * kv_heads * head_dim * seq * batch * bytes_ / 1e9
print(f"[KV 캐시] 32층, KV헤드 32개(MHA), head_dim 128, 32K 토큰: {kv_cache_gb(32, 32, 128, 32768, 1):.1f} GB")
print(f"          같은 조건, KV헤드 8개(GQA)            : {kv_cache_gb(32, 8, 128, 32768, 1):.1f} GB  ← GQA가 쓰이는 이유")
