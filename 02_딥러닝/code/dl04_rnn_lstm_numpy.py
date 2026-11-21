"""
[실습 04] RNN / LSTM 셀을 NumPy로 구현 + 기울기 소실 실험 + 문자 단위 RNN 언어모델 학습(BPTT 직접 구현)
"""
import numpy as np
from _plot_setup import plt, save
rng = np.random.default_rng(0)
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# 1) 기울기 소실: ∂h_T/∂h_t 의 크기가 시간 거리에 따라 어떻게 변하나
def rnn_grad_norms(T=60, H=32, scale=1.0):
    Wh = rng.normal(0, scale / np.sqrt(H), (H, H)); Wx = rng.normal(0, 1 / np.sqrt(H), (H, H))
    hs, h = [], np.zeros(H)
    for t in range(T):
        h = np.tanh(Wx @ rng.normal(size=H) + Wh @ h); hs.append(h)
    J, norms = np.eye(H), []
    for t in reversed(range(T)):           # 역방향으로 야코비안 곱 누적
        norms.append(np.linalg.norm(J))
        J = J @ (np.diag(1 - hs[t] ** 2) @ Wh)
    return norms[::-1]

def lstm_cell_grad_norms(T=60, f_bias=3.0):
    """셀 상태 경로 ∂c_T/∂c_t = Π f_t (forget gate). 편향을 크게 주면 f≈1 → 기울기 유지"""
    f = sigmoid(f_bias + rng.normal(0, 0.5, T))
    return [np.prod(f[t:]) for t in range(T)]

fig, ax = plt.subplots(figsize=(7, 3.5))
for s in (0.5, 1.0, 2.0):
    ax.plot(rnn_grad_norms(scale=s), label=f"vanilla RNN, W 스케일={s}")
ax.plot(lstm_cell_grad_norms(), "k--", label="LSTM 셀 상태 경로 (forget bias=3)")
ax.set_yscale("log"); ax.set_xlabel("시점 t (오른쪽이 마지막 시점 T)"); ax.set_ylabel("||∂h_T / ∂h_t||")
ax.set_title("시간 거리에 따른 기울기 크기"); ax.legend(fontsize=8)
save(fig, "dl04_vanishing_gradient.png")
n = rnn_grad_norms(scale=1.0)
print(f"[기울기 소실] vanilla RNN: 1스텝 전 {n[-2]:.2e}, 30스텝 전 {n[-31]:.2e}, 59스텝 전 {n[0]:.2e}")
l = lstm_cell_grad_norms()
print(f"             LSTM 셀 경로: 30스텝 전 {l[-31]:.2e}, 59스텝 전 {l[0]:.2e}")

# 2) LSTM 셀 순전파 (게이트 계산 확인)
def lstm_step(x, h, c, W, b):
    z = W @ np.r_[x, h] + b
    H = len(h)
    i, f, o, g = sigmoid(z[:H]), sigmoid(z[H:2*H]), sigmoid(z[2*H:3*H]), np.tanh(z[3*H:])
    c = f * c + i * g                  # 덧셈 경로!
    h = o * np.tanh(c)
    return h, c, dict(i=i.mean(), f=f.mean(), o=o.mean())
D, H = 4, 8
W = rng.normal(0, 0.3, (4 * H, D + H)); b = np.zeros(4 * H); b[H:2*H] = 1.0   # forget bias=1 (흔한 트릭)
h = c = np.zeros(H)
for t in range(3):
    h, c, gates = lstm_step(rng.normal(size=D), h, c, W, b)
    print(f"[LSTM t={t}] 평균 게이트값 input={gates['i']:.2f} forget={gates['f']:.2f} output={gates['o']:.2f}, |c|={np.linalg.norm(c):.3f}")

# 3) 문자 단위 RNN 언어모델 (BPTT 직접 구현)
text = ("딥러닝은 데이터로부터 표현을 학습한다. 순환 신경망은 순서가 있는 데이터를 처리한다. "
        "트랜스포머는 어텐션으로 순서를 처리한다. 딥러닝은 재미있다. ") * 3
chars = sorted(set(text)); V = len(chars); ix = {c: i for i, c in enumerate(chars)}
data = np.array([ix[c] for c in text])
Hd, seq, lr = 64, 25, 0.05
P = {"Wxh": rng.normal(0, .01, (Hd, V)), "Whh": rng.normal(0, .01, (Hd, Hd)), "Why": rng.normal(0, .01, (V, Hd)),
     "bh": np.zeros(Hd), "by": np.zeros(V)}
M = {k: np.zeros_like(v) for k, v in P.items()}       # Adagrad 메모리

def step(inputs, targets, hprev):
    xs, hs, ps = {}, {-1: hprev}, {}; loss = 0
    for t in range(len(inputs)):
        xs[t] = np.zeros(V); xs[t][inputs[t]] = 1
        hs[t] = np.tanh(P["Wxh"] @ xs[t] + P["Whh"] @ hs[t-1] + P["bh"])
        y = P["Why"] @ hs[t] + P["by"]; ps[t] = np.exp(y - y.max()); ps[t] /= ps[t].sum()
        loss -= np.log(ps[t][targets[t]])
    g = {k: np.zeros_like(v) for k, v in P.items()}; dhnext = np.zeros(Hd)
    for t in reversed(range(len(inputs))):           # BPTT
        dy = ps[t].copy(); dy[targets[t]] -= 1
        g["Why"] += np.outer(dy, hs[t]); g["by"] += dy
        dh = P["Why"].T @ dy + dhnext
        draw = (1 - hs[t] ** 2) * dh
        g["bh"] += draw; g["Wxh"] += np.outer(draw, xs[t]); g["Whh"] += np.outer(draw, hs[t-1])
        dhnext = P["Whh"].T @ draw
    for k in g: np.clip(g[k], -5, 5, out=g[k])       # gradient clipping (폭발 방지)
    return loss / len(inputs), g, hs[len(inputs) - 1]

def sample(seed_char, n=40):
    h = np.zeros(Hd); x = np.zeros(V); x[ix[seed_char]] = 1; out = seed_char
    for _ in range(n):
        h = np.tanh(P["Wxh"] @ x + P["Whh"] @ h + P["bh"]); y = P["Why"] @ h + P["by"]
        p = np.exp(y - y.max()); p /= p.sum(); k = rng.choice(V, p=p)
        x = np.zeros(V); x[k] = 1; out += chars[k]
    return out

hist, hprev, pos = [], np.zeros(Hd), 0
print(f"\n[문자 RNN] 어휘 크기 {V}, 텍스트 길이 {len(data)}")
for it in range(3001):
    if pos + seq + 1 >= len(data): pos, hprev = 0, np.zeros(Hd)
    loss, g, hprev = step(data[pos:pos+seq], data[pos+1:pos+seq+1], hprev)
    for k in P: M[k] += g[k] ** 2; P[k] -= lr * g[k] / np.sqrt(M[k] + 1e-8)
    hist.append(loss); pos += seq
    if it in (0, 500, 3000):
        print(f"iter {it:4d} loss={loss:.3f}  샘플: {sample('딥')}")
fig, ax = plt.subplots(figsize=(6, 3)); ax.plot(np.convolve(hist, np.ones(50) / 50, "valid"))
ax.set_title("문자 RNN 학습 손실 (이동평균)"); ax.set_xlabel("iteration")
save(fig, "dl04_char_rnn_loss.png")
