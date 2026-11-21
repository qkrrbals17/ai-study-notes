"""
[실습 02] 옵티마이저(SGD/Momentum/RMSProp/Adam) 직접 구현 & 비교,
          가중치 초기화가 깊은 망의 활성값에 미치는 영향, Dropout 동작 확인, LR 스케줄
"""
import numpy as np
from _plot_setup import plt, save

# 1) Rosenbrock 함수 최적화 (최솟값 (1,1)) - 좁고 휘어진 계곡이라 옵티마이저 차이가 잘 보임
f = lambda w: (1 - w[0]) ** 2 + 100 * (w[1] - w[0] ** 2) ** 2
grad = lambda w: np.array([-2 * (1 - w[0]) - 400 * w[0] * (w[1] - w[0] ** 2), 200 * (w[1] - w[0] ** 2)])

class SGD:
    def __init__(s, lr): s.lr = lr
    def step(s, w, g): return w - s.lr * g
class Momentum:
    def __init__(s, lr, beta=0.9): s.lr, s.b, s.v = lr, beta, 0
    def step(s, w, g): s.v = s.b * s.v + g; return w - s.lr * s.v
class RMSProp:
    def __init__(s, lr, beta=0.99): s.lr, s.b, s.s = lr, beta, 0
    def step(s, w, g): s.s = s.b * s.s + (1 - s.b) * g ** 2; return w - s.lr * g / (np.sqrt(s.s) + 1e-8)
class Adam:
    def __init__(s, lr, b1=0.9, b2=0.999): s.lr, s.b1, s.b2, s.m, s.v, s.t = lr, b1, b2, 0, 0, 0
    def step(s, w, g):
        s.t += 1
        s.m = s.b1 * s.m + (1 - s.b1) * g; s.v = s.b2 * s.v + (1 - s.b2) * g ** 2
        mh, vh = s.m / (1 - s.b1 ** s.t), s.v / (1 - s.b2 ** s.t)     # 편향 보정
        return w - s.lr * mh / (np.sqrt(vh) + 1e-8)

opts = {"SGD": SGD(1e-3), "Momentum": Momentum(1e-4 * 2), "RMSProp": RMSProp(1e-2), "Adam": Adam(5e-2)}
paths = {}
print("옵티마이저 | 3000스텝 후 위치      | f(w)")
for name, o in opts.items():
    w = np.array([-1.5, 2.0]); path = [w]
    for _ in range(3000):
        w = o.step(w, np.clip(grad(w), -100, 100)); path.append(w)
    paths[name] = np.array(path)
    print(f"{name:9s} | ({w[0]: .4f}, {w[1]: .4f}) | {f(w):.2e}")

xx, yy = np.meshgrid(np.linspace(-2, 2, 300), np.linspace(-1, 3, 300))
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].contour(xx, yy, np.log1p(f(np.array([xx, yy]))), 25, cmap="gray", alpha=0.5)
for name, p in paths.items(): ax[0].plot(p[:, 0], p[:, 1], lw=1.2, label=name)
ax[0].scatter([1], [1], c="r", marker="*", s=200, zorder=5); ax[0].legend(); ax[0].set_title("Rosenbrock 위 옵티마이저 궤적")
for name, p in paths.items(): ax[1].plot([f(w) for w in p], label=name)
ax[1].set_yscale("log"); ax[1].set_xlabel("step"); ax[1].set_title("손실 감소"); ax[1].legend()
save(fig, "dl02_optimizers.png")

# 2) 초기화 실험: 10층 tanh/ReLU 망의 층별 활성값 표준편차
rng = np.random.default_rng(0)
x = rng.normal(size=(1000, 256))
print("\n[초기화별 10층 통과 후 활성값 표준편차]")
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
for i, (act, inits) in enumerate([("tanh", {"N(0,1)": 1.0, "N(0,0.01)": 0.01, "Xavier": np.sqrt(1 / 256)}),
                                  ("relu", {"N(0,0.01)": 0.01, "Xavier": np.sqrt(1 / 256), "He": np.sqrt(2 / 256)})]):
    for name, std in inits.items():
        h, stds = x, []
        for _ in range(10):
            z = h @ rng.normal(0, std, (256, 256))
            h = np.tanh(z) if act == "tanh" else np.maximum(0, z)
            stds.append(h.std())
        ax[i].plot(range(1, 11), stds, "o-", label=name)
        sat = f", 포화(|h|>0.99) 비율 {np.mean(np.abs(h) > 0.99):.0%}" if act == "tanh" else ""
        print(f"{act:4s} + {name:9s}: 1층 {stds[0]:.3e} → 10층 {stds[-1]:.3e}{sat}")
    ax[i].set_yscale("log"); ax[i].set_title(f"{act} 망의 층별 활성값 std"); ax[i].set_xlabel("층"); ax[i].legend()
save(fig, "dl02_init.png")
print("→ N(0,1)은 std가 커 보여도 대부분 ±1에 포화(기울기≈0), 0.01은 신호가 사라짐")
print("→ tanh엔 Xavier, ReLU엔 He 초기화가 신호 크기를 유지한다")

# 3) Dropout (inverted dropout) 동작 확인
def dropout(h, p, train=True):
    if not train: return h
    mask = (rng.random(h.shape) > p) / (1 - p)        # 살아남은 뉴런을 1/(1-p)배 → 기댓값 유지
    return h * mask
h = np.ones((10000, 100))
print(f"\n[Dropout p=0.5] 학습 모드 평균={dropout(h, .5).mean():.3f}, 0인 비율={np.mean(dropout(h, .5) == 0):.3f}, 추론 모드 평균={dropout(h, .5, False).mean():.3f}")

# 4) BatchNorm / LayerNorm 정규화 축 비교
B = rng.normal(5, 3, (4, 6))
bn = (B - B.mean(0)) / B.std(0); ln = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
print(f"[BatchNorm] 열(특성)별 평균 ≈ {np.round(bn.mean(0), 3)}")
print(f"[LayerNorm] 행(샘플)별 평균 ≈ {np.round(ln.mean(1), 3)}")

# 5) 학습률 스케줄: Warmup + Cosine
total, warm, base = 1000, 100, 3e-4
lr = [base * s / warm if s < warm else base * 0.5 * (1 + np.cos(np.pi * (s - warm) / (total - warm))) for s in range(total)]
step_lr = [base * (0.5 ** (s // 300)) for s in range(total)]
fig, ax = plt.subplots(figsize=(6, 3)); ax.plot(lr, label="Warmup + Cosine"); ax.plot(step_lr, label="Step decay")
ax.set_xlabel("step"); ax.set_ylabel("learning rate"); ax.legend(); ax.set_title("학습률 스케줄")
save(fig, "dl02_lr_schedule.png")
