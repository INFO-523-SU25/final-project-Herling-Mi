import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut, ParameterGrid, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv(r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\0_Final_Proj_N\0_Random_Forest\N_audio_features_parallel_output_encoded.csv")
target_column = 'genre_enc'

# -----------------------------
# Keep only numeric columns
# -----------------------------
df_numeric = df.select_dtypes(include=np.number)
X = df_numeric.drop(target_column, axis=1)
y = df_numeric[target_column]

# -----------------------------
# Hyperparameter grid
# -----------------------------
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = list(ParameterGrid(param_grid))
results = []

# -----------------------------
# LOOCV hyperparameter sweep
# -----------------------------
print(f"Starting LOOCV over {len(grid)} hyperparameter combinations on {X.shape[0]} samples...")
for idx, params in enumerate(grid, 1):
    loo = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(**params, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        y_true_all.append(y_test.values[0])
        y_pred_all.append(y_pred[0])

    # Compute metrics for this hyperparameter combination
    acc = accuracy_score(y_true_all, y_pred_all)
    prec = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    rec = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
    f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)

    results.append({**params, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1})

    print(f"[{idx}/{len(grid)}] Params: {params} -> "
          f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

# -----------------------------
# Convert results to DataFrame
# -----------------------------
results_df = pd.DataFrame(results)

# Find best hyperparameters based on F1 score
best_idx = results_df['f1_score'].idxmax()
best_params = results_df.loc[best_idx]
print("\nBest hyperparameters based on F1 score:")
print(best_params)

# -----------------------------
# Scatter plot of hyperparameter sweep
# -----------------------------
plt.figure(figsize=(10,6))
plt.scatter(range(len(results_df)), results_df['f1_score'], c='blue', label='F1 Score')
plt.xlabel('Hyperparameter Combination Index')
plt.ylabel('F1 Score')
plt.title('Random Forest Hyperparameter Sweep F1 Scores')
plt.grid(True)
plt.legend()
plt.show()

# -----------------------------
# Learning curve with best hyperparameters
# -----------------------------
clf_best = RandomForestClassifier(
    n_estimators=int(best_params['n_estimators']),
    max_depth=None if pd.isna(best_params['max_depth']) else int(best_params['max_depth']),
    min_samples_split=int(best_params['min_samples_split']),
    min_samples_leaf=int(best_params['min_samples_leaf']),
    random_state=42
)

train_sizes, train_scores, test_scores = learning_curve(
    estimator=clf_best,
    X=X,
    y=y,
    cv=5,                 # use 5-fold instead of LOOCV
    train_sizes=np.linspace(0.1, 1.0, 5),
    scoring='accuracy',
    n_jobs=1              # avoid TerminatedWorkerError
)

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Accuracy')
plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation Accuracy')  # Corrected legend
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.title('Learning Curve for Best Random Forest')
plt.legend()
plt.grid(True)
plt.show()
