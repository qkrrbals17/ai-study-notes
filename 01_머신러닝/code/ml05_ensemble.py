"""
[실습 05] 단일 트리 vs 배깅 vs 랜덤포레스트 vs 부스팅 비교
- n_estimators 증가에 따른 성능 변화, OOB 점수, 특성 중요도 vs Permutation Importance
- (XGBoost/LightGBM이 설치돼 있으면 함께 비교, 없으면 sklearn HistGradientBoosting 사용)
"""
import numpy as np, time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from _plot_setup import plt, save

X, y = make_classification(n_samples=4000, n_features=20, n_informative=6, n_redundant=4,
                           flip_y=0.05, class_sep=0.8, random_state=7)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

models = {
    "DecisionTree":     DecisionTreeClassifier(random_state=0),
    "Bagging(트리100)":  BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, n_jobs=-1, random_state=0),
    "RandomForest":     RandomForestClassifier(n_estimators=300, oob_score=True, n_jobs=-1, random_state=0),
    "AdaBoost":         AdaBoostClassifier(n_estimators=200, random_state=0),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3, random_state=0),
    "HistGB(LightGBM식)": HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                                        early_stopping=True, random_state=0),
}
try:
    import lightgbm as lgb
    models["LightGBM"] = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, verbose=-1)
except ImportError:
    print("(lightgbm 미설치 → HistGradientBoosting으로 대체 비교)")
try:
    import xgboost as xgb
    models["XGBoost"] = xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=4)
except ImportError:
    print("(xgboost 미설치 → 건너뜀)")

print(f"\n{'모델':20s} | train acc | test acc | 시간")
for name, m in models.items():
    t = time.time(); m.fit(X_tr, y_tr)
    print(f"{name:20s} | {m.score(X_tr, y_tr):.4f}    | {m.score(X_te, y_te):.4f}   | {time.time()-t:.2f}s")
print(f"\nRandomForest OOB 점수 = {models['RandomForest'].oob_score_:.4f} (별도 검증셋 없이 추정한 성능)")
print("→ 단일 트리는 train=1.0(과적합), 앙상블은 test 성능이 크게 오름")

# 트리 개수에 따른 성능
ns = [1, 5, 10, 25, 50, 100, 200, 400]
rf_s = [RandomForestClassifier(n, n_jobs=-1, random_state=0).fit(X_tr, y_tr).score(X_te, y_te) for n in ns]
gb = GradientBoostingClassifier(n_estimators=400, learning_rate=0.1, max_depth=3, random_state=0).fit(X_tr, y_tr)
gb_s = [np.mean(p == y_te) for p in gb.staged_predict(X_te)]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(ns, rf_s, "o-", label="RandomForest"); ax[0].plot(range(1, 401), gb_s, label="GradientBoosting (staged)")
ax[0].set_xscale("log"); ax[0].set_xlabel("트리 개수"); ax[0].set_ylabel("test accuracy"); ax[0].legend(); ax[0].set_title("트리 수에 따른 성능")

# 특성 중요도 비교
rf = models["RandomForest"]
pi = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0, n_jobs=-1)
order = np.argsort(pi.importances_mean)[::-1][:10]
w = 0.4; xs = np.arange(len(order))
ax[1].bar(xs - w/2, rf.feature_importances_[order], w, label="Impurity 기반")
ax[1].bar(xs + w/2, pi.importances_mean[order], w, label="Permutation")
ax[1].set_xticks(xs); ax[1].set_xticklabels([f"f{i}" for i in order]); ax[1].legend(); ax[1].set_title("특성 중요도 Top10")
save(fig, "ml05_ensemble.png")

# 보팅 & 스태킹
base = [("lr", LogisticRegression(max_iter=1000)), ("rf", RandomForestClassifier(200, n_jobs=-1, random_state=0)),
        ("hgb", HistGradientBoostingClassifier(random_state=0))]
print("\n[보팅/스태킹 5-Fold CV accuracy]")
for name, m in [("LogReg 단독", base[0][1]), ("RF 단독", base[1][1]), ("HGB 단독", base[2][1]),
                ("Soft Voting", VotingClassifier(base, voting="soft")),
                ("Stacking", StackingClassifier(base, final_estimator=LogisticRegression(), cv=5))]:
    print(f"{name:12s}: {cross_val_score(m, X, y, cv=5, n_jobs=-1).mean():.4f}")
