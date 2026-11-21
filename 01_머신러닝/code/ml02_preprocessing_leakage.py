"""
[실습 02] 전처리 파이프라인 + 데이터 누수(leakage) 재현 실험
1) 결측치/범주형이 섞인 가상 고객 데이터 → ColumnTransformer 파이프라인
2) 누수 실험: '정답과 무관한' 랜덤 데이터인데도, 특성 선택을 CV 밖에서 하면 정확도가 높게 나옴
"""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score, StratifiedKFold

rng = np.random.default_rng(0)
n = 1000
df = pd.DataFrame({
    "age": rng.integers(18, 70, n).astype(float),
    "income": rng.lognormal(8, 0.6, n),
    "visits": rng.poisson(5, n).astype(float),
    "region": rng.choice(["서울", "부산", "대구", "광주"], n),
    "plan": rng.choice(["basic", "pro", "premium"], n, p=[.6, .3, .1]),
})
logit = -3 + 0.04 * df.age + 0.25 * df.visits + (df.plan == "premium") * 1.5
df["churn"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
for col in ["age", "income"]:                       # 결측치 인위적으로 만들기
    df.loc[rng.random(n) < 0.1, col] = np.nan
df.loc[rng.random(n) < 0.05, "region"] = np.nan

print("== 데이터 요약 ==")
print(df.head().to_string())
print("\n결측치 수:\n", df.isna().sum().to_string())
print("\n타깃 비율:", df.churn.value_counts(normalize=True).round(3).to_dict())

# 1) 파이프라인 (전처리를 CV의 각 fold 안에서 fit → 누수 없음)
num_cols, cat_cols = ["age", "income", "visits"], ["region", "plan"]
num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                     ("oh", OneHotEncoder(handle_unknown="ignore"))])
pre = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
model = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=1000))])

cv = StratifiedKFold(5, shuffle=True, random_state=0)
X, y = df.drop(columns="churn"), df.churn
auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"\n[파이프라인] ROC-AUC = {auc.mean():.3f} ± {auc.std():.3f}")

model.fit(X, y)
feat_names = model.named_steps["pre"].get_feature_names_out()
coefs = pd.Series(model.named_steps["clf"].coef_[0], index=feat_names).sort_values()
print("\n로지스틱 회귀 계수 (스케일된 특성 기준):\n", coefs.round(3).to_string())

# 2) 데이터 누수 실험: 완전 랜덤 데이터 (진짜 정확도는 50%여야 정상)
Xr = rng.normal(size=(100, 5000))
yr = rng.integers(0, 2, 100)
# ❌ 잘못된 방법: 전체 데이터로 특성 선택 후 CV
Xsel = SelectKBest(f_classif, k=20).fit_transform(Xr, yr)
bad = cross_val_score(LogisticRegression(max_iter=1000), Xsel, yr, cv=cv).mean()
# ✅ 올바른 방법: 특성 선택을 파이프라인 안에 넣음
good_pipe = Pipeline([("sel", SelectKBest(f_classif, k=20)), ("clf", LogisticRegression(max_iter=1000))])
good = cross_val_score(good_pipe, Xr, yr, cv=cv).mean()
print(f"\n[누수 실험 - 랜덤 데이터]  잘못된 CV 정확도 = {bad:.3f}   올바른 CV 정확도 = {good:.3f}")
print("→ 랜덤 데이터인데 잘못된 방법은 높은 점수가 나옴 = 데이터 누수!")
