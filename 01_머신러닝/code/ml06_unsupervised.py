"""
[실습 06] 비지도 학습: K-Means 직접 구현, 엘보우/실루엣, DBSCAN, PCA, t-SNE
"""
import numpy as np
from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.ensemble import IsolationForest
from _plot_setup import plt, save

# 1) K-Means 밑바닥 구현
def kmeans_numpy(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for it in range(iters):
        labels = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)   # 할당
        newC = np.array([X[labels == j].mean(0) for j in range(k)])            # 갱신
        if np.allclose(newC, C): break
        C = newC
    inertia = ((X - C[labels]) ** 2).sum()
    return labels, C, inertia, it + 1

Xb, yb = make_blobs(600, centers=4, cluster_std=1.0, random_state=3)
lab, C, inertia, n_it = kmeans_numpy(Xb, 4)
sk = KMeans(4, n_init=10, random_state=0).fit(Xb)
print(f"[직접 구현 K-Means] {n_it}회 반복 후 수렴, inertia={inertia:.1f}")
print(f"[sklearn KMeans]   inertia={sk.inertia_:.1f},  두 결과 일치도(ARI)={adjusted_rand_score(lab, sk.labels_):.3f}")

# 2) 엘보우 + 실루엣으로 K 고르기
ks = range(2, 10); inert, sil = [], []
for k in ks:
    m = KMeans(k, n_init=10, random_state=0).fit(Xb)
    inert.append(m.inertia_); sil.append(silhouette_score(Xb, m.labels_))
print("K별 실루엣:", {k: round(s, 3) for k, s in zip(ks, sil)}, "→ 최적 K =", list(ks)[int(np.argmax(sil))])

fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
ax[0].scatter(Xb[:, 0], Xb[:, 1], c=lab, s=8, cmap="tab10"); ax[0].scatter(C[:, 0], C[:, 1], c="k", marker="X", s=150)
ax[0].set_title("직접 구현한 K-Means (K=4)")
ax[1].plot(ks, inert, "o-"); ax[1].set_title("엘보우 방법 (Inertia)"); ax[1].set_xlabel("K")
ax[2].plot(ks, sil, "s-", c="g"); ax[2].set_title("실루엣 점수"); ax[2].set_xlabel("K")
save(fig, "ml06_kmeans_elbow.png")

# 3) 초승달 데이터: K-Means vs 계층적 vs GMM vs DBSCAN
Xm, ym = make_moons(500, noise=0.06, random_state=0)
Xm_s = StandardScaler().fit_transform(Xm)
algos = {"K-Means": KMeans(2, n_init=10, random_state=0), "Agglomerative(single)": AgglomerativeClustering(2, linkage="single"),
         "GMM": GaussianMixture(2, random_state=0), "DBSCAN(eps=0.3)": DBSCAN(eps=0.3, min_samples=5)}
fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
print("\n[초승달 데이터 군집 성능 ARI (1이면 완벽)]")
for ax, (name, a) in zip(axes, algos.items()):
    l = a.fit_predict(Xm_s)
    ari = adjusted_rand_score(ym, l); print(f"{name:22s}: ARI={ari:.3f}")
    ax.scatter(Xm[:, 0], Xm[:, 1], c=l, s=8, cmap="viridis"); ax.set_title(f"{name}\nARI={ari:.2f}", fontsize=9)
save(fig, "ml06_moons_clustering.png")

# 4) PCA & t-SNE (손글씨 숫자 8x8 = 64차원)
X, y = load_digits(return_X_y=True)
Xs = StandardScaler().fit_transform(X)
pca = PCA().fit(Xs)
cum = np.cumsum(pca.explained_variance_ratio_)
print(f"\n[PCA] 64차원 중 분산 90% 설명에 필요한 주성분 수: {np.argmax(cum >= 0.9) + 1}개, 95%: {np.argmax(cum >= 0.95) + 1}개")
P2 = PCA(2).fit_transform(Xs)
T2 = TSNE(2, perplexity=30, random_state=0, init="pca").fit_transform(Xs)
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(np.arange(1, 65), cum); ax[0].axhline(0.9, ls=":", c="r"); ax[0].set_title("PCA 누적 설명 분산"); ax[0].set_xlabel("주성분 수")
for a, Z, t in [(ax[1], P2, "PCA 2D"), (ax[2], T2, "t-SNE 2D")]:
    sc = a.scatter(Z[:, 0], Z[:, 1], c=y, s=5, cmap="tab10"); a.set_title(t)
fig.colorbar(sc, ax=ax[2], label="숫자")
save(fig, "ml06_pca_tsne.png")

# 5) 이상 탐지
rng = np.random.default_rng(0)
Xa = np.r_[rng.normal(0, 1, (300, 2)), rng.uniform(-6, 6, (15, 2))]
pred = IsolationForest(contamination=0.05, random_state=0).fit_predict(Xa)
print(f"\n[IsolationForest] 이상치로 판정: {np.sum(pred == -1)}개 (실제 주입한 이상치 15개)")
