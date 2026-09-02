import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------
# 1. Load binary classification dataset
# ------------------------------------------------------------------
X, y = load_breast_cancer(return_X_y=True)

# ------------------------------------------------------------------
# 2. Choose 5 random samples from each class
# ------------------------------------------------------------------
rng = np.random.default_rng()

classes = np.unique(y)

selected_idx = []
for cls in classes:
    cls_idx = np.where(y == cls)[0]
    selected_idx.extend(
        rng.choice(cls_idx, size=20, replace=False)
    )

selected_idx = np.array(selected_idx)

X_small = X[selected_idx]
y_small = y[selected_idx]

print(f"Selected dataset size: {len(y_small)}")
print("Class counts:",
      {c: np.sum(y_small == c) for c in np.unique(y_small)})

# ------------------------------------------------------------------
# Common CV setup
# ------------------------------------------------------------------
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=123
)

gbm = GradientBoostingClassifier(random_state=123)

# ------------------------------------------------------------------
# 3A. Scale using mean/std from WHOLE dataset (leakage)
# ------------------------------------------------------------------
scaler_global = StandardScaler()
X_global_scaled = scaler_global.fit_transform(X_small)

scores_A = cross_val_score(
    gbm,
    X_global_scaled,
    y_small,
    cv=cv,
    scoring="accuracy"
)

error_A = 1 - scores_A.mean()

# ------------------------------------------------------------------
# 3B. Scale within each training fold only
# ------------------------------------------------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("gbm", GradientBoostingClassifier())
])

scores_B = cross_val_score(
    pipeline,
    X_small,
    y_small,
    cv=cv,
    scoring="accuracy"
)

error_B = 1 - scores_B.mean()

# ------------------------------------------------------------------
# 4. Print results
# ------------------------------------------------------------------
print("\nResults")
print("-" * 40)
print(f"Case A (global scaling):  CV error = {error_A:.4f}")
print(f"Case B (fold scaling):    CV error = {error_B:.4f}")

print("\nFold accuracies:")
print("A:", np.round(scores_A, 4))
print("B:", np.round(scores_B, 4))
