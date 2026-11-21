"""
[실습 01] 과적합 vs 과소적합, 편향-분산 트레이드오프 직접 확인하기
- sin 곡선 + 노이즈 데이터에 다항 회귀 차수를 1~15로 바꿔가며 train/test 오차 비교
- 부트스트랩으로 편향^2 / 분산을 실제로 분해해 봄
"""
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from _plot_setup import plt, save

rng = np.random.default_rng(42)
f = lambda x: np.sin(2 * np.pi * x)          # 진짜 함수 (보통은 모름)

def make_data(n):
    x = rng.uniform(0, 1, n)
    return x.reshape(-1, 1), f(x) + rng.normal(0, 0.25, n)

X_tr, y_tr = make_data(30)
X_te, y_te = make_data(500)

# 1) 차수별 train / test 오차
degrees = range(1, 16)
tr_err, te_err = [], []
for d in degrees:
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    tr_err.append(mean_squared_error(y_tr, m.predict(X_tr)))
    te_err.append(mean_squared_error(y_te, m.predict(X_te)))
    if d in (1, 3, 15):
        print(f"degree={d:2d}  train MSE={tr_err[-1]:.4f}  test MSE={te_err[-1]:.4f}")
best = list(degrees)[int(np.argmin(te_err))]
print(f"→ test MSE가 가장 낮은 차수: {best}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
xs = np.linspace(0, 1, 300).reshape(-1, 1)
for ax, d, title in zip(axes, (1, best, 15), ("과소적합", "적절", "과적합")):
    m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
    ax.scatter(X_tr, y_tr, s=15, c="k", label="train")
    ax.plot(xs, f(xs), "g--", label="진짜 함수")
    ax.plot(xs, np.clip(m.predict(xs), -2, 2), "r", label=f"degree={d}")
    ax.set_title(f"{title} (degree={d})"); ax.set_ylim(-2, 2); ax.legend(fontsize=7)
save(fig, "ml01_poly_fits.png")

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(degrees, tr_err, "o-", label="train MSE")
ax.plot(degrees, te_err, "s-", label="test MSE")
ax.set_yscale("log"); ax.set_xlabel("다항식 차수 (모델 복잡도)"); ax.set_ylabel("MSE")
ax.axvline(best, ls=":", c="gray"); ax.legend(); ax.set_title("모델 복잡도에 따른 오차")
save(fig, "ml01_complexity_curve.png")

# 2) 편향-분산 분해 (훈련셋을 200번 새로 뽑아 반복 학습)
x0 = np.linspace(0.05, 0.95, 50).reshape(-1, 1)
print("\n차수 | bias^2  | variance | bias^2+var")
for d in (1, 3, 9, 15):
    preds = []
    for _ in range(200):
        Xb, yb = make_data(30)
        m = make_pipeline(PolynomialFeatures(d), StandardScaler(), LinearRegression()).fit(Xb, yb)
        preds.append(m.predict(x0))
    preds = np.array(preds)
    bias2 = np.mean((preds.mean(0) - f(x0.ravel())) ** 2)
    var = np.mean(preds.var(0))
    print(f"{d:4d} | {bias2:.4f}  | {var:8.4f} | {bias2 + var:.4f}")
print("→ 차수↑ : 편향↓ 분산↑  (트레이드오프 확인)")
