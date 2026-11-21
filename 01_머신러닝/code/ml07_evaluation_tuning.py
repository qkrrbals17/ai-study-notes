"""
[실습 07] 불균형 데이터에서의 평가 지표, 임계값 조정, 교차검증 전략, 하이퍼파라미터 튜닝
- 사기 탐지처럼 양성이 3%인 데이터 생성
"""
import numpy as np
from scipy.stats import loguniform, randint
from sklearn.datasets import make_classification
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold, TimeSeriesSplit,
                                     GridSearchCV, RandomizedSearchCV, learning_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (confusion_matrix, classification_report, precision_recall_curve,
                             roc_curve, roc_auc_score, average_precision_score, f1_score, accuracy_score)
from sklearn.calibration import calibration_curve
from _plot_setup import plt, save

X, y = make_classification(n_samples=20000, n_features=15, n_informative=8, weights=[0.97],
                           flip_y=0.01, random_state=1)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
print(f"양성 비율: {y.mean():.3%}")

# 1) Accuracy의 함정
dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
print(f"[모두 음성이라고 예측하는 바보 모델] accuracy={accuracy_score(y_te, dummy.predict(X_te)):.4f}, "
      f"F1={f1_score(y_te, dummy.predict(X_te)):.4f}  ← 정확도만 보면 속는다")

clf = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)
print("\n[임계값 0.5] 혼동 행렬 [[TN FP] [FN TP]]:\n", confusion_matrix(y_te, pred))
print(classification_report(y_te, pred, digits=3))
print(f"ROC-AUC={roc_auc_score(y_te, proba):.4f}   PR-AUC(AP)={average_precision_score(y_te, proba):.4f}  (기준선 PR-AUC = 양성비율 {y_te.mean():.3f})")

# 2) 임계값 조정
prec, rec, thr = precision_recall_curve(y_te, proba)
f1s = 2 * prec * rec / (prec + rec + 1e-12)
best = np.argmax(f1s[:-1])
print(f"\nF1 최대 임계값 = {thr[best]:.3f} → precision={prec[best]:.3f}, recall={rec[best]:.3f}, F1={f1s[best]:.3f}")
i90 = np.where(rec[:-1] >= 0.90)[0][-1]
print(f"재현율 90% 이상 보장 임계값 = {thr[i90]:.3f} → precision={prec[i90]:.3f}  (놓치면 안 되는 문제에 사용)")

fpr, tpr, _ = roc_curve(y_te, proba)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set_title(f"ROC 곡선 (AUC={roc_auc_score(y_te, proba):.3f})")
ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR(Recall)")
ax[1].plot(rec, prec); ax[1].axhline(y_te.mean(), ls=":", c="k", label="무작위 기준선")
ax[1].scatter(rec[best], prec[best], c="r", zorder=3, label=f"F1 최대 (thr={thr[best]:.2f})")
ax[1].set_title(f"PR 곡선 (AP={average_precision_score(y_te, proba):.3f})"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend(fontsize=8)
ax[2].plot(thr, prec[:-1], label="precision"); ax[2].plot(thr, rec[:-1], label="recall"); ax[2].plot(thr, f1s[:-1], label="F1")
ax[2].set_xlabel("임계값"); ax[2].set_title("임계값에 따른 트레이드오프"); ax[2].legend()
save(fig, "ml07_roc_pr_threshold.png")

# 3) class_weight 효과
for cw in (None, "balanced"):
    m = LogisticRegression(max_iter=2000, class_weight=cw).fit(X_tr, y_tr)
    p = m.predict(X_te)
    print(f"LogReg class_weight={str(cw):8s} → recall={np.mean(p[y_te==1]==1):.3f}, F1={f1_score(y_te, p):.3f}")

# 4) CV 전략별 fold 구성 확인
yy = np.array([0]*8 + [1]*4)
print("\n[CV 분할 비교] 12개 샘플, 양성 4개")
for name, cv in [("KFold(셔플X)", KFold(3)), ("StratifiedKFold", StratifiedKFold(3)), ("TimeSeriesSplit", TimeSeriesSplit(3))]:
    print(f"{name:16s}:", [f"val={v.tolist()} 양성{yy[v].sum()}개" for _, v in cv.split(np.zeros(12), yy)])

# 5) Grid vs Random Search
Xs, ys = X_tr[:4000], y_tr[:4000]
cv = StratifiedKFold(3, shuffle=True, random_state=0)
grid = GridSearchCV(HistGradientBoostingClassifier(random_state=0),
                    {"learning_rate": [0.03, 0.1, 0.3], "max_leaf_nodes": [15, 31, 63], "l2_regularization": [0, 1.0]},
                    scoring="average_precision", cv=cv, n_jobs=-1).fit(Xs, ys)
rand = RandomizedSearchCV(HistGradientBoostingClassifier(random_state=0),
                          {"learning_rate": loguniform(0.01, 0.5), "max_leaf_nodes": randint(8, 128),
                           "l2_regularization": loguniform(1e-3, 10), "min_samples_leaf": randint(5, 100)},
                          n_iter=18, scoring="average_precision", cv=cv, random_state=0, n_jobs=-1).fit(Xs, ys)
print(f"\nGridSearch  (18조합): best AP={grid.best_score_:.4f}  {grid.best_params_}")
print(f"RandomSearch(18회)  : best AP={rand.best_score_:.4f}  { {k: (round(float(v), 3) if isinstance(v, float) else int(v)) for k, v in rand.best_params_.items()} }")

# 6) 학습 곡선 + 확률 보정
sizes, tr_s, va_s = learning_curve(HistGradientBoostingClassifier(random_state=0), X_tr, y_tr, cv=3,
                                   train_sizes=np.linspace(0.1, 1, 6), scoring="average_precision", n_jobs=-1)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(sizes, tr_s.mean(1), "o-", label="train"); ax[0].plot(sizes, va_s.mean(1), "s-", label="validation")
ax[0].set_xlabel("훈련 샘플 수"); ax[0].set_ylabel("AP"); ax[0].set_title("학습 곡선"); ax[0].legend()
for name, p in [("HistGB", proba), ("LogReg", LogisticRegression(max_iter=2000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1])]:
    fr, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax[1].plot(mp, fr, "o-", label=name)
ax[1].plot([0, 1], [0, 1], "k:"); ax[1].set_title("확률 보정 곡선 (대각선=완벽)"); ax[1].set_xlabel("예측 확률"); ax[1].set_ylabel("실제 양성 비율"); ax[1].legend()
save(fig, "ml07_learning_calibration.png")
