"""
[실습 03] 합성곱/풀링을 NumPy로 직접 구현해서 CNN의 동작 원리 확인
- conv2d(stride, padding) 구현 → 출력 크기 공식 검증
- 엣지 검출 필터로 특징 맵 시각화
- 파라미터 수 계산: MLP vs CNN
- "합성곱 특징 + 풀링" 이 원본 픽셀보다 위치 변화에 강하다는 것 실험
"""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from _plot_setup import plt, save

def conv2d(x, k, stride=1, pad=0):
    """x: (H,W), k: (K,K). 딥러닝 프레임워크의 conv는 사실 cross-correlation."""
    x = np.pad(x, pad)
    K = k.shape[0]
    H = (x.shape[0] - K) // stride + 1; W = (x.shape[1] - K) // stride + 1
    out = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(x[i*stride:i*stride+K, j*stride:j*stride+K] * k)
    return out

def maxpool(x, s=2):
    H, W = x.shape[0] // s, x.shape[1] // s
    return x[:H*s, :W*s].reshape(H, s, W, s).max(axis=(1, 3))

out_size = lambda N, K, S, P: (N + 2 * P - K) // S + 1

# 1) 출력 크기 공식 검증
x = np.random.default_rng(0).normal(size=(32, 32))
print("[출력 크기 공식 (N+2P-K)/S+1 검증]")
for K, S, P in [(3, 1, 0), (3, 1, 1), (5, 2, 2), (7, 2, 3)]:
    print(f"  N=32, K={K}, S={S}, P={P} → 공식 {out_size(32, K, S, P)},  실제 {conv2d(x, np.ones((K, K)), S, P).shape[0]}")

# 2) 엣지 검출 필터
digits = load_digits()
img = digits.images[0]
filters = {"수직 엣지 (Sobel-x)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
           "수평 엣지 (Sobel-y)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
           "블러 (평균)": np.ones((3, 3)) / 9,
           "샤픈": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])}
fig, ax = plt.subplots(1, 6, figsize=(16, 2.9))
ax[0].imshow(img, cmap="gray"); ax[0].set_title("원본 8x8")
for a, (name, k) in zip(ax[1:5], filters.items()):
    a.imshow(conv2d(img, k, pad=1), cmap="RdBu"); a.set_title(name, fontsize=9)
ax[5].imshow(maxpool(np.abs(conv2d(img, filters["수직 엣지 (Sobel-x)"], pad=1))), cmap="gray"); ax[5].set_title("수직엣지 → MaxPool 2x2", fontsize=9)
for a in ax: a.axis("off")
save(fig, "dl03_filters.png")

# 3) 파라미터 수 비교 (224x224x3 입력)
mlp = 224 * 224 * 3 * 1000 + 1000
conv = 3 * 3 * 3 * 64 + 64
print(f"\n[파라미터 수] 224x224x3 → 은닉 1000 FC층: {mlp:,}개  vs  3x3 conv 64채널: {conv:,}개 (가중치 공유)")
def resnet_block_params(c): return 2 * (3 * 3 * c * c) + 2 * (2 * c)   # conv 2개 + BN 2개
print(f"ResNet basic block(64ch) 파라미터: {resnet_block_params(64):,}개")

# 4) 수용 영역(receptive field) 계산
rf, jump = 1, 1
print("\n[수용 영역] 3x3 conv(s=1) 와 2x2 pool(s=2)을 번갈아 쌓을 때")
for i, (k, s, name) in enumerate([(3, 1, "conv"), (3, 1, "conv"), (2, 2, "pool"), (3, 1, "conv"), (3, 1, "conv"), (2, 2, "pool"), (3, 1, "conv")]):
    rf += (k - 1) * jump; jump *= s
    print(f"  {i+1}층 {name}: 수용 영역 {rf}x{rf}")

# 5) 이동(shift)에 대한 강건성: 원본 픽셀 vs conv+pool 특징
X, y = digits.images, digits.target
rng = np.random.default_rng(0)
bank = [filters["수직 엣지 (Sobel-x)"], filters["수평 엣지 (Sobel-y)"], -filters["수직 엣지 (Sobel-x)"], -filters["수평 엣지 (Sobel-y)"]] + \
       [rng.normal(size=(3, 3)) for _ in range(12)]          # 고정 필터 16개 (학습 X)
def conv_features(im):
    return np.concatenate([maxpool(np.maximum(0, conv2d(im, k, pad=1)), 4).ravel() for k in bank])   # conv→ReLU→pool
def shift(im, dx): return np.roll(im, dx, axis=1)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
raw = LogisticRegression(max_iter=3000).fit(Xtr.reshape(len(Xtr), -1), ytr)
F = lambda arr: np.array([conv_features(i) for i in arr])
cnnf = LogisticRegression(max_iter=3000).fit(F(Xtr), ytr)
print("\n[테스트 이미지를 오른쪽으로 dx 픽셀 이동했을 때 정확도]")
print("  dx | 원본 픽셀 | conv+ReLU+pool 특징")
for dx in (0, 1, 2):
    Xs = np.array([shift(i, dx) for i in Xte])
    print(f"  {dx}  |  {raw.score(Xs.reshape(len(Xs), -1), yte):.3f}    |  {cnnf.score(F(Xs), yte):.3f}")
print("→ 풀링 덕분에 작은 위치 변화에 덜 민감 (CNN의 위치 불변성)")
