"""
[실습 08] 엔드투엔드 미니 프로젝트: 당뇨 진행도 '고위험군' 예측
문제정의 → 분할 → EDA → 베이스라인 → 모델 비교 → 튜닝 → 오류분석 → 최종평가 → 저장/로드 → 예측 API 함수
"""
import json, os, time, joblib
import numpy as np, pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report
from scipy.stats import loguniform, randint

SEED = 42
np.random.seed(SEED)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts"); os.makedirs(OUT, exist_ok=True)

# 1. 문제 정의: 1년 후 진행도(target)가 상위 30%인 '고위험' 환자를 조기 선별 (이진 분류, 지표 ROC-AUC)
data = load_diabetes(as_frame=True, scaled=False)
df = data.frame.copy()
df["high_risk"] = (df.target >= df.target.quantile(0.7)).astype(int)
df = df.drop(columns="target")

# 2. 테스트셋을 가장 먼저 분리
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.high_risk, random_state=SEED)
X, y = train_df.drop(columns="high_risk"), train_df.high_risk
X_test, y_test = test_df.drop(columns="high_risk"), test_df.high_risk
print(f"train={len(train_df)}, test={len(test_df)}, 고위험 비율={y.mean():.2f}")

# 3. EDA (train에서만!)
corr = train_df.corr()["high_risk"].drop("high_risk").sort_values(ascending=False)
print("\n[EDA] 타깃과의 상관계수 상위:\n", corr.round(3).head(5).to_string())

# 4. 베이스라인 → 후보 모델 비교
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cands = {
    "Dummy(베이스라인)": DummyClassifier(strategy="prior"),
    "LogisticRegression": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
    "RandomForest": RandomForestClassifier(300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
    "HistGB": HistGradientBoostingClassifier(random_state=SEED),
}
log = []
print("\n[모델 비교: 5-Fold ROC-AUC]")
for name, m in cands.items():
    s = cross_val_score(m, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    log.append({"model": name, "auc_mean": round(s.mean(), 4), "auc_std": round(s.std(), 4)})
    print(f"  {name:20s} {s.mean():.4f} ± {s.std():.4f}")

# 5. 상위 모델 튜닝 (작은 데이터라 로지스틱이 강함 → C 튜닝)
search = RandomizedSearchCV(
    Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, solver="saga"))]),
    {"clf__C": loguniform(1e-3, 1e2), "clf__l1_ratio": [0.0, 0.5, 1.0]},   # l1_ratio 0=L2, 1=L1
    n_iter=40, cv=cv, scoring="roc_auc", random_state=SEED, n_jobs=-1).fit(X, y)
print(f"\n[튜닝] best CV AUC={search.best_score_:.4f}, params={ {k: round(float(v), 4) for k, v in search.best_params_.items()} }")
best = search.best_estimator_
log.append({"model": "LogReg(tuned)", "auc_mean": round(search.best_score_, 4), **{k: str(v) for k, v in search.best_params_.items()}})

# 6. 오류 분석 (OOF 예측으로 train 내부에서)
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(best, X, y, cv=cv, method="predict_proba")[:, 1]
err = train_df.assign(proba=oof, wrong=((oof >= 0.5) != y))
print("\n[오류 분석] 틀린 샘플 vs 맞춘 샘플의 평균 bmi / s5:")
print(err.groupby("wrong")[["bmi", "s5", "proba"]].mean().round(3).to_string())

# 7. 최종 평가 (테스트셋은 여기서 단 한 번)
best.fit(X, y)
p_test = best.predict_proba(X_test)[:, 1]
print(f"\n[최종 테스트] ROC-AUC = {roc_auc_score(y_test, p_test):.4f}")
print(classification_report(y_test, (p_test >= 0.5).astype(int), digits=3))

coef = pd.Series(best.named_steps["clf"].coef_[0], index=X.columns).sort_values(key=abs, ascending=False)
print("해석 - 영향이 큰 특성:", coef.round(3).head(5).to_dict())

# 8. 저장 & 로드 & 예측 함수 (API 서빙의 핵심)
joblib.dump(best, os.path.join(OUT, "diabetes_risk_model.joblib"))
json.dump({"experiments": log, "test_auc": round(roc_auc_score(y_test, p_test), 4),
           "features": list(X.columns), "seed": SEED}, open(os.path.join(OUT, "experiment_log.json"), "w"),
          ensure_ascii=False, indent=2)
loaded = joblib.load(os.path.join(OUT, "diabetes_risk_model.joblib"))

def predict_api(patient: dict) -> dict:
    """FastAPI 등에서 그대로 감쌀 수 있는 예측 함수"""
    row = pd.DataFrame([patient])[X.columns]
    p = float(loaded.predict_proba(row)[0, 1])
    return {"high_risk_probability": round(p, 3), "label": "고위험" if p >= 0.5 else "일반"}

sample = X_test.iloc[0].to_dict()
print("\n[예측 API 테스트] 입력:", {k: round(v, 2) for k, v in list(sample.items())[:4]}, "...")
print("출력:", predict_api(sample), "| 실제:", "고위험" if y_test.iloc[0] else "일반")
print("\n저장된 파일:", os.listdir(OUT))
