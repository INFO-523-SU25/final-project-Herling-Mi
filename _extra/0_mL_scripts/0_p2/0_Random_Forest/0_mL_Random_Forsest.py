"""
Random Forest Classification with LOOCV Hyperparameter Sweep
Numeric-only features, scatter plot of hyperparameters, and learning curve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, ParameterGrid, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------------
# Load CSV
# -------------------------------
df = pd.read_csv(r"N_audio_features_parallel_output_encoded.csv")  # Replace with your CSV path

# Keep only numeric columns
df_numeric = df.select_dtypes(include=[np.number])

# Define features and target
X = df_numeric.drop("genre_enc", axis=1)
y = df_numeric["genre_enc"]

# -------------------------------
# Standardize features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Define hyperparameter grid
# -------------------------------
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}

grid = list(ParameterGrid(param_grid))

# -------------------------------
# LOOCV hyperparameter sweep
# -------------------------------
loo = LeaveOneOut()
hyper_results = []

for params in grid:
    y_true_all = []
    y_pred_all = []
    for train_index, test_index in loo.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        rf = RandomForestClassifier(**params, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        y_true_all.append(y_test.values[0])
        y_pred_all.append(y_pred[0])

    # Metrics
    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)

    hyper_results.append({
        **params,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })

# Convert results to DataFrame
results_df = pd.DataFrame(hyper_results)

# Find best hyperparameters by accuracy
best_idx = results_df["accuracy"].idxmax()
best_params = results_df.loc[best_idx]
print("Best Hyperparameters:")
print(best_params)

# -------------------------------
# Scatter plot: n_estimators vs accuracy
# -------------------------------
plt.figure(figsize=(8,6))
plt.scatter(results_df["n_estimators"], results_df["accuracy"], c='blue', s=50)
plt.title("Random Forest Hyperparameter Sweep: n_estimators vs Accuracy")
plt.xlabel("n_estimators")
plt.ylabel("LOOCV Accuracy")
plt.grid(True)
plt.show()

# -------------------------------
# Learning curve
# -------------------------------
rf_best = RandomForestClassifier(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    random_state=42
)

train_sizes, train_scores, test_scores = learning_curve(
    rf_best, X_scaled, y, cv=loo, scoring="accuracy", train_sizes=np.linspace(0.1, 1.0, 5)
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8,6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label="Cross-validation score")
plt.title("Random Forest Learning Curve")
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.legend(loc="best")
plt.grid(True)
plt.show()
