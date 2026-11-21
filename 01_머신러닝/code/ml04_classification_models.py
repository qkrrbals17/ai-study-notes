"""
[실습 04] 분류 모델 5종 비교 + 결정 경계 시각화 + 지니 불순도 직접 계산
- 유방암 진단 데이터(569명, 30개 특성)로 5-Fold CV 비교
- make_moons 2D 데이터로 모델별 결정 경계 그리기
"""
import numpy as np, time
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from _plot_setup import plt, save

def models():
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "SVM (RBF)":          make_pipeline(StandardScaler(), SVC(C=1, gamma="scale")),
        "KNN (k=5)":          make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
        "DecisionTree (d=4)": DecisionTreeClassifier(max_depth=4, random_state=0),
        "GaussianNB":         GaussianNB(),
    }

X, y = load_breast_cancer(return_X_y=True)
cv = StratifiedKFold(5, shuffle=True, random_state=0)
print(f"데이터: {X.shape}, 양성(1) 비율={y.mean():.2f}\n")
print(f"{'모델':20s} | acc    | f1     | roc_auc | 학습시간")
for name, m in models().items():
    t = time.time()
    r = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
    print(f"{name:20s} | {r['test_accuracy'].mean():.4f} | {r['test_f1'].mean():.4f} | "
          f"{r['test_roc_auc'].mean():.4f}  | {time.time()-t:.2f}s")

# 스케일링 유무가 KNN/SVM에 미치는 영향
print("\n[스케일링 효과]")
for name, raw, scaled in [("KNN", KNeighborsClassifier(5), make_pipeline(StandardScaler(), KNeighborsClassifier(5))),
                          ("SVM", SVC(), make_pipeline(StandardScaler(), SVC()))]:
    a = cross_validate(raw, X, y, cv=cv)["test_score"].mean()
    b = cross_validate(scaled, X, y, cv=cv)["test_score"].mean()
    print(f"{name}: 스케일링 X = {a:.4f}  →  스케일링 O = {b:.4f}")

# 결정 경계
Xm, ym = make_moons(400, noise=0.25, random_state=0)
xx, yy = np.meshgrid(np.linspace(-1.6, 2.6, 300), np.linspace(-1.2, 1.7, 300))
fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
for ax, (name, m) in zip(axes, models().items()):
    m.fit(Xm, ym)
    Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
    ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, s=8, cmap="coolwarm", edgecolor="k", linewidth=0.2)
    ax.set_title(f"{name}\ntrain acc={m.score(Xm, ym):.2f}", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
save(fig, "ml04_decision_boundaries.png")

# SVM 하이퍼파라미터 C, gamma 영향
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for i, C in enumerate([0.1, 100]):
    for j, g in enumerate([0.1, 1, 30]):
        m = SVC(C=C, gamma=g).fit(Xm, ym)
        Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax = axes[i, j]; ax.contourf(xx, yy, Z, alpha=0.3, cmap="coolwarm")
        ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, s=6, cmap="coolwarm")
        ax.set_title(f"C={C}, gamma={g}  (SV {len(m.support_)}개)", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
save(fig, "ml04_svm_C_gamma.png")

# 지니 불순도 손계산 vs sklearn
def gini(labels):
    p = np.bincount(labels) / len(labels)
    return 1 - np.sum(p ** 2)
tree = DecisionTreeClassifier(max_depth=1).fit(X, y)
f, thr = tree.tree_.feature[0], tree.tree_.threshold[0]
left, right = y[X[:, f] <= thr], y[X[:, f] > thr]
w_gini = len(left) / len(y) * gini(left) + len(right) / len(y) * gini(right)
print(f"\n[지니 불순도] 루트={gini(y):.4f}, 분할 후 가중 평균={w_gini:.4f}, 정보이득={gini(y)-w_gini:.4f}")
print("sklearn이 계산한 루트 불순도:", round(tree.tree_.impurity[0], 4))
print("\n깊이 2 결정트리 규칙:")
names = load_breast_cancer().feature_names
print(export_text(DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y), feature_names=list(names)))
