import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split, GridSearchCV, validation_curve,
    learning_curve, LeaveOneOut
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    precision_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# 1. Load dataset
file_path = r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\0_Final_Proj_N\0_knn\N_audio_features_parallel_output_encoded.csv"
df = pd.read_csv(file_path)

# 2. Define numeric features only – excluding original string categoricals
numeric_features = [
    'fundamental_freq', 'freq_e_1', 'freq_e_2', 'freq_e_3',
    'duration', 'zero_crossing_rate', 'mfcc_mean', 'mfcc_std',
    'tempo', 'rms_energy',
    'artist_enc', 'song_name_enc', 'genre_enc', 'key_enc'
]
X = df[numeric_features]
y = df['genre_enc']

# 3. Preprocessing via ColumnTransformer: Variance threshold + PCA
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('vt', VarianceThreshold(threshold=0.0)),
        ('pca', PCA(n_components=0.95))  # keep components that explain 95% of variance
    ]), numeric_features)
], remainder='drop')  # drop any columns not in numeric_features

# 4. Full pipeline: preprocessing, scaling, and KNN classifier
pipeline = Pipeline([
    ('prep', preprocessor),
    ('scale', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Hyperparameter sweep via GridSearchCV
param_grid = {
    'knn__n_neighbors': list(range(1, 21, 2)),
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': ['auto', 'kd_tree', 'ball_tree'],
    'knn__p': [1, 2]
}
grid = GridSearchCV(
    pipeline, param_grid, cv=5,
    scoring='accuracy', return_train_score=True, error_score='raise'
)
grid.fit(X_train, y_train)

# 7. Save results and display best performance
results_df = pd.DataFrame(grid.cv_results_)
csv_out = "knn_hyperparam_sweep.csv"
results_df.to_csv(csv_out, index=False)
print(f"Sweep results saved to {os.path.abspath(csv_out)}\n")

best = grid.best_estimator_
print("Best Hyperparameters:", grid.best_params_)

y_pred = best.predict(X_test)
print(f"Test Precision (weighted): {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Test F1 Score (weighted): {f1_score(y_test, y_pred, average='weighted'):.4f}\n")

print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix - Test Set")
plt.show()

# 8. Validation Curve: Accuracy vs n_neighbors
param_range = list(range(1, 21, 2))
train_scores, test_scores = validation_curve(
    pipeline, X_train, y_train,
    param_name='knn__n_neighbors', param_range=param_range,
    cv=5, scoring='accuracy'
)

plt.figure()
plt.plot(param_range, np.mean(train_scores, axis=1), label='Training Score')
plt.plot(param_range, np.mean(test_scores, axis=1), label='Cross-validation Score')
plt.title('Validation Curve for K (n_neighbors)')
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# 9. Learning Curve: Accuracy vs Training Examples
train_sizes, train_scores_lc, test_scores_lc = learning_curve(
    pipeline, X_train, y_train, cv=5,
    scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5)
)
train_mean = np.mean(train_scores_lc, axis=1)
test_mean = np.mean(test_scores_lc, axis=1)
train_std = np.std(train_scores_lc, axis=1)
test_std = np.std(test_scores_lc, axis=1)

plt.figure()
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='r')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='g')
plt.plot(train_sizes, train_mean, 'o-', color='r', label='Training Score')
plt.plot(train_sizes, test_mean, 'o-', color='g', label='Cross-val Score')
plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# 10. Final Evaluation with LOOCV
loo = LeaveOneOut()
loo_scores = []
for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    best.fit(X_tr, y_tr)
    loo_scores.append(best.score(X_te, y_te))

print(f"LOOCV Accuracy: {np.mean(loo_scores):.4f}")
