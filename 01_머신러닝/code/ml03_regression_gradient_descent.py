"""
[실습 03] 선형회귀를 3가지 방법으로 풀어보고 비교 + Ridge/Lasso 규제 경로
- 정규방정식, NumPy 경사하강법(직접 구현), scikit-learn
- diabetes 데이터셋 (sklearn 내장, 442명 x 10개 특성)
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from _plot_setup import plt, save

X, y = load_diabetes(return_X_y=True, as_frame=True)
names = X.columns.tolist()
X_tr, X_te, y_tr, y_te = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
sc = StandardScaler().fit(X_tr)
X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)
Xb_tr = np.c_[np.ones(len(X_tr)), X_tr]          # 편향(절편) 열 추가
Xb_te = np.c_[np.ones(len(X_te)), X_te]

# (1) 정규방정식  w = (X^T X)^-1 X^T y
w_ne = np.linalg.solve(Xb_tr.T @ Xb_tr, Xb_tr.T @ y_tr)

# (2) 배치 경사하강법
w = np.zeros(Xb_tr.shape[1]); lr = 0.1; hist = []
for epoch in range(20000):   # s1,s2 특성이 강하게 상관(다중공선성) → 수렴이 느려서 반복을 넉넉히
    err = Xb_tr @ w - y_tr
    grad = 2 / len(Xb_tr) * Xb_tr.T @ err
    w -= lr * grad
    hist.append(np.mean(err ** 2))
w_gd = w

# (2-b) 미니배치 SGD
w_sgd = np.zeros_like(w); rng = np.random.default_rng(0); hist_sgd = []
for epoch in range(100):
    idx = rng.permutation(len(Xb_tr))
    for s in range(0, len(idx), 32):
        b = idx[s:s + 32]
        w_sgd -= 0.01 * 2 / len(b) * Xb_tr[b].T @ (Xb_tr[b] @ w_sgd - y_tr[b])
    hist_sgd.append(np.mean((Xb_tr @ w_sgd - y_tr) ** 2))

# (3) scikit-learn
sk = LinearRegression().fit(X_tr, y_tr)
w_sk = np.r_[sk.intercept_, sk.coef_]

print("방법          | 절편    | 첫 3개 계수                 | test RMSE")
for name, ww in [("정규방정식", w_ne), ("경사하강법", w_gd), ("미니배치SGD", w_sgd), ("sklearn", w_sk)]:
    rmse = mean_squared_error(y_te, Xb_te @ ww) ** 0.5
    print(f"{name:12s} | {ww[0]:7.2f} | {np.round(ww[1:4], 2)!s:27s} | {rmse:.2f}")
print("최대 계수 차이 (정규방정식 vs GD):", np.abs(w_ne - w_gd).max().round(4))

p = sk.predict(X_te)
print(f"\n[평가지표] MAE={mean_absolute_error(y_te, p):.2f}  RMSE={mean_squared_error(y_te, p)**.5:.2f}  R2={r2_score(y_te, p):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(hist, label="Batch GD (lr=0.1)"); ax[0].plot(np.arange(100) * 200, hist_sgd, label="Mini-batch SGD (epoch 단위)")
ax[0].set_yscale("log"); ax[0].set_xscale("log"); ax[0].set_xlabel("iteration"); ax[0].set_ylabel("train MSE"); ax[0].legend(); ax[0].set_title("학습 곡선")
for lr_try, c in [(0.001, "b"), (0.1, "g"), (1.05, "r")]:
    ww = np.zeros(Xb_tr.shape[1]); h = []
    for _ in range(60):
        ww -= lr_try * 2 / len(Xb_tr) * Xb_tr.T @ (Xb_tr @ ww - y_tr); h.append(np.mean((Xb_tr @ ww - y_tr) ** 2))
    ax[1].plot(h, c, label=f"lr={lr_try}")
ax[1].set_yscale("log"); ax[1].set_title("학습률에 따른 수렴/발산"); ax[1].legend()
save(fig, "ml03_gd_curves.png")

# 규제 경로
alphas = np.logspace(-2, 3, 60)
ridge_c = np.array([Ridge(alpha=a).fit(X_tr, y_tr).coef_ for a in alphas])
lasso_c = np.array([Lasso(alpha=a / 100, max_iter=20000).fit(X_tr, y_tr).coef_ for a in alphas])
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for j in range(len(names)):
    ax[0].plot(alphas, ridge_c[:, j], label=names[j]); ax[1].plot(alphas / 100, lasso_c[:, j])
ax[0].set_xscale("log"); ax[0].set_title("Ridge (L2): 계수가 0에 '가까워짐'"); ax[0].set_xlabel("alpha")
ax[1].set_xscale("log"); ax[1].set_title("Lasso (L1): 계수가 '정확히 0'이 됨"); ax[1].set_xlabel("alpha")
ax[0].legend(fontsize=7, ncol=2)
save(fig, "ml03_regularization_path.png")

lasso = Lasso(alpha=1.0).fit(X_tr, y_tr)
print("\nLasso(alpha=1) 에서 0이 된 특성:", [n for n, c in zip(names, lasso.coef_) if c == 0])
for a in (0.1, 1, 10, 100):
    print(f"Ridge alpha={a:<5} test RMSE={mean_squared_error(y_te, Ridge(alpha=a).fit(X_tr, y_tr).predict(X_te))**.5:.2f}")
