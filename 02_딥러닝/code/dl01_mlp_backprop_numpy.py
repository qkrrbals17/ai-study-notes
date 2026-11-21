"""
[실습 01] NumPy만으로 2층 신경망(MLP) + 역전파 직접 구현
- 순전파 / 역전파 / 수치 미분으로 기울기 검증(gradient check) / 학습 / 결정경계
- 활성화 함수가 없으면(선형) 초승달 데이터를 못 푼다는 것도 확인
"""
import numpy as np
from sklearn.datasets import make_moons
from _plot_setup import plt, save

rng = np.random.default_rng(0)
X, y = make_moons(600, noise=0.2, random_state=0)
Y = np.eye(2)[y]                                       # one-hot

def init(n_in, n_h, n_out):
    return {"W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, n_h)), "b1": np.zeros(n_h),   # He 초기화
            "W2": rng.normal(0, np.sqrt(2 / n_h), (n_h, n_out)), "b2": np.zeros(n_out)}

def softmax(z):
    z = z - z.max(1, keepdims=True)                    # 수치 안정성
    e = np.exp(z); return e / e.sum(1, keepdims=True)

def forward(P, X, act="relu"):
    z1 = X @ P["W1"] + P["b1"]
    h = np.maximum(0, z1) if act == "relu" else z1     # act="linear"이면 비선형성 없음
    p = softmax(h @ P["W2"] + P["b2"])
    return p, (X, z1, h, p)

def loss_fn(p, Y):
    return -np.mean(np.sum(Y * np.log(p + 1e-12), 1))  # 교차엔트로피

def backward(P, cache, Y, act="relu"):
    X, z1, h, p = cache; n = len(X)
    dz2 = (p - Y) / n                                  # softmax+CE의 미분이 이렇게 깔끔해짐
    g = {"W2": h.T @ dz2, "b2": dz2.sum(0)}
    dh = dz2 @ P["W2"].T
    dz1 = dh * (z1 > 0) if act == "relu" else dh       # ReLU 미분 = 0 또는 1
    g["W1"] = X.T @ dz1; g["b1"] = dz1.sum(0)
    return g

# 1) Gradient check: 해석적 기울기 vs 수치 미분
P = init(2, 8, 2)
p, cache = forward(P, X[:20]); g = backward(P, cache, Y[:20])
eps, max_rel = 1e-5, 0
for k in P:
    it = np.nditer(P[k], flags=["multi_index"])
    for _ in it:
        i = it.multi_index; old = P[k][i]
        P[k][i] = old + eps; lp = loss_fn(forward(P, X[:20])[0], Y[:20])
        P[k][i] = old - eps; lm = loss_fn(forward(P, X[:20])[0], Y[:20])
        P[k][i] = old
        num = (lp - lm) / (2 * eps)
        max_rel = max(max_rel, abs(num - g[k][i]) / (abs(num) + abs(g[k][i]) + 1e-12))
print(f"[Gradient check] 최대 상대오차 = {max_rel:.2e}  (1e-6 이하면 역전파 구현이 맞음)")

# 2) 학습: ReLU MLP vs 선형(활성화 없음)
def train(act, hidden=32, lr=0.5, epochs=2000):
    P = init(2, hidden, 2); hist = []
    for ep in range(epochs):
        p, cache = forward(P, X, act)
        hist.append(loss_fn(p, Y))
        g = backward(P, cache, Y, act)
        for k in P: P[k] -= lr * g[k]
    acc = np.mean(forward(P, X, act)[0].argmax(1) == y)
    return P, hist, acc

P_relu, h_relu, acc_relu = train("relu")
P_lin, h_lin, acc_lin = train("linear", lr=0.1)
print(f"[학습 결과] ReLU MLP 정확도 = {acc_relu:.3f},  활성화 없는(선형) 망 정확도 = {acc_lin:.3f}")
print(f"초기 손실 = {h_relu[0]:.3f} (이론값 ln2 = {np.log(2):.3f} 근처면 정상),  최종 손실 = {h_relu[-1]:.3f}")

xx, yy = np.meshgrid(np.linspace(-2, 3, 250), np.linspace(-1.5, 2, 250))
grid = np.c_[xx.ravel(), yy.ravel()]
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].plot(h_relu, label="ReLU MLP"); ax[0].plot(h_lin, label="선형 망"); ax[0].set_title("학습 손실"); ax[0].set_xlabel("epoch"); ax[0].legend()
for a, P_, t in [(ax[1], P_relu, f"ReLU MLP (acc={acc_relu:.2f})"), (ax[2], P_lin, f"활성화 없음 (acc={acc_lin:.2f})")]:
    a.contourf(xx, yy, forward(P_, grid, "relu" if P_ is P_relu else "linear")[0][:, 1].reshape(xx.shape), 20, cmap="coolwarm", alpha=0.6)
    a.scatter(X[:, 0], X[:, 1], c=y, s=6, cmap="coolwarm"); a.set_title(t)
save(fig, "dl01_mlp_numpy.png")

# 3) 은닉 뉴런 수에 따른 표현력
for hdim in (2, 4, 16, 64):
    print(f"hidden={hdim:3d} → train acc={train('relu', hidden=hdim, epochs=1500)[2]:.3f}")
