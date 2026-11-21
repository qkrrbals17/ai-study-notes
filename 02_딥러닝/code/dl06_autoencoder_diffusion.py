"""
[실습 06] 생성 모델 원리 실습 (scikit-learn + NumPy)
1) 오토인코더: MLP로 X→X 를 학습 (병목 2차원) → 재구성 / 잠재공간 시각화
2) 미니 Diffusion(DDPM): 2D 초승달 분포에 노이즈를 넣고, MLP가 노이즈를 예측하도록 학습 → 순수 노이즈에서 샘플 생성
"""
import numpy as np, warnings
from sklearn.datasets import load_digits, make_moons
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from _plot_setup import plt, save
warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)

# ---------------- 1) Autoencoder ----------------
X, y = load_digits(return_X_y=True)
Xs = MinMaxScaler().fit_transform(X)
ae = MLPRegressor(hidden_layer_sizes=(256, 64, 2, 64, 256), activation="relu", max_iter=1500,   # 64→256→64→[2]→64→256→64
                  random_state=0, learning_rate_init=1e-3)
ae.fit(Xs, Xs)

def encode(model, X):                                # 병목층(2차원)까지만 순전파
    h = X
    for W, b in list(zip(model.coefs_, model.intercepts_))[:3]:
        h = np.maximum(0, h @ W + b)
    return h

rec = ae.predict(Xs)
pca = PCA(2).fit(Xs); rec_pca = pca.inverse_transform(pca.transform(Xs))
print(f"[오토인코더] 잠재 2차원 재구성 MSE = {np.mean((rec - Xs)**2):.4f}")
print(f"[PCA 2차원 ] 재구성 MSE        = {np.mean((rec_pca - Xs)**2):.4f}  (선형 vs 비선형 압축 비교)")

fig = plt.figure(figsize=(15, 4))
for i in range(6):
    a = fig.add_subplot(3, 12, i + 1); a.imshow(Xs[i].reshape(8, 8), cmap="gray"); a.axis("off"); a.set_title("원본" if i == 0 else "", fontsize=8)
    a = fig.add_subplot(3, 12, 12 + i + 1); a.imshow(rec[i].reshape(8, 8), cmap="gray"); a.axis("off"); a.set_title("AE 복원" if i == 0 else "", fontsize=8)
    a = fig.add_subplot(3, 12, 24 + i + 1); a.imshow(rec_pca[i].reshape(8, 8), cmap="gray"); a.axis("off"); a.set_title("PCA 복원" if i == 0 else "", fontsize=8)
a = fig.add_subplot(1, 2, 2)
Z = encode(ae, Xs); sc = a.scatter(Z[:, 0], Z[:, 1], c=y, cmap="tab10", s=5); fig.colorbar(sc, ax=a)
a.set_title("오토인코더 2D 잠재 공간 (색=숫자)")
save(fig, "dl06_autoencoder.png")

# ---------------- 2) Mini DDPM ----------------
T = 40
betas = np.linspace(1e-4, 0.25, T)
alphas = 1 - betas; abar = np.cumprod(alphas)
data, _ = make_moons(4000, noise=0.05, random_state=0); data = (data - data.mean(0)) / data.std(0)

def q_sample(x0, t, eps):                           # forward: x_t = √ᾱ_t x0 + √(1-ᾱ_t) ε  (한 번에 점프 가능)
    return np.sqrt(abar[t])[:, None] * x0 + np.sqrt(1 - abar[t])[:, None] * eps

def t_embed(t):                                     # 시간 정보를 사인/코사인으로
    f = np.array([1, 2, 4, 8])
    return np.c_[np.sin(t[:, None] / T * np.pi * f), np.cos(t[:, None] / T * np.pi * f)]

# 학습 데이터: (x_t, t) → ε 예측  (DDPM 손실 ||ε - ε_θ(x_t,t)||²)
N = 60000
x0 = data[rng.integers(0, len(data), N)]; t = rng.integers(0, T, N); eps = rng.normal(size=(N, 2))
net = MLPRegressor(hidden_layer_sizes=(256, 256, 256), activation="relu", max_iter=60, batch_size=512,
                   learning_rate_init=2e-3, random_state=0, early_stopping=False)
net.fit(np.c_[q_sample(x0, t, eps), t_embed(t)], eps)
print(f"\n[Diffusion] 노이즈 예측 학습 최종 손실 = {net.loss_:.4f} (예측 안 하면 1.0)")

# 역과정 샘플링 (DDPM)
x = rng.normal(size=(2000, 2)); snaps = {T: x.copy()}
for ti in reversed(range(T)):
    tt = np.full(len(x), ti)
    eps_hat = net.predict(np.c_[x, t_embed(tt)])
    x = (x - betas[ti] / np.sqrt(1 - abar[ti]) * eps_hat) / np.sqrt(alphas[ti])
    if ti > 0: x += np.sqrt(betas[ti]) * rng.normal(size=x.shape)
    if ti in (30, 20, 10, 0): snaps[ti] = x.copy()

fig, ax = plt.subplots(2, 5, figsize=(16, 6.5))
for a, tt in zip(ax[0], [0, 5, 15, 25, 39]):
    a.scatter(*q_sample(data[:1500], np.full(1500, tt), rng.normal(size=(1500, 2))).T, s=2); a.set_title(f"Forward: t={tt}"); a.set_xlim(-3, 3); a.set_ylim(-3, 3)
for a, tt in zip(ax[1], [T, 30, 20, 10, 0]):
    a.scatter(*snaps[tt].T, s=2, c="C3"); a.set_title(f"Reverse(생성): t={tt}"); a.set_xlim(-3, 3); a.set_ylim(-3, 3)
save(fig, "dl06_diffusion.png")

# 생성 품질 간단 평가: 생성 샘플 → 가장 가까운 실제 데이터까지 평균 거리
d_gen = np.sqrt(((snaps[0][:500, None] - data[None]) ** 2).sum(-1)).min(1).mean()
d_noise = np.sqrt(((rng.normal(size=(500, 2))[:, None] - data[None]) ** 2).sum(-1)).min(1).mean()
print(f"[Diffusion] 생성 샘플→실제 데이터 최근접 평균거리 = {d_gen:.3f}  (순수 가우시안 노이즈는 {d_noise:.3f})")
