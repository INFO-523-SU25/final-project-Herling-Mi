import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

# =============================
# 1. Load dataset
# =============================
file_path = r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\0_Final_Proj_N\0_knn\N_audio_features_parallel_output_encoded.csv"
df = pd.read_csv(file_path)

# =============================
# 2. Define features and target
# =============================
numeric_features = [
    'fundamental_freq', 'freq_e_1', 'freq_e_2', 'freq_e_3',
    'duration', 'zero_crossing_rate', 'mfcc_mean', 'mfcc_std',
    'tempo', 'rms_energy',
    'artist_enc', 'song_name_enc', 'genre_enc', 'key_enc'
]
X = df[numeric_features]
y = df['genre_enc']

# =============================
# 3. Preprocessing pipeline
# =============================
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('vt', VarianceThreshold(threshold=0.0)),
        ('pca', PCA(n_components=0.95))
    ]), numeric_features)
], remainder='drop')

pipeline = Pipeline([
    ('prep', preprocessor),
    ('scale', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# =============================
# 4. Hyperparameter sweep
# =============================
param_grid = {
    'knn__n_neighbors': list(range(1, 21, 2)),
    'knn__weights': ['uniform', 'distance'],
    'knn__algorithm': ['auto', 'kd_tree', 'ball_tree'],
    'knn__p': [1, 2]
}

loo = LeaveOneOut()
results = []

for n_neighbors in param_grid['knn__n_neighbors']:
    for weights in param_grid['knn__weights']:
        for algorithm in param_grid['knn__algorithm']:
            for p in param_grid['knn__p']:
                pipeline.set_params(knn__n_neighbors=n_neighbors,
                                    knn__weights=weights,
                                    knn__algorithm=algorithm,
                                    knn__p=p)
                loo_scores = []
                for train_idx, test_idx in loo.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    pipeline.fit(X_train, y_train)
                    y_pred = pipeline.predict(X_test)
                    loo_scores.append(f1_score(y_test, y_pred, average='weighted'))
                results.append({
                    'n_neighbors': n_neighbors,
                    'weights': weights,
                    'algorithm': algorithm,
                    'p': p,
                    'mean_f1_score': np.mean(loo_scores)
                })

results_df = pd.DataFrame(results)
results_df.to_csv("knn_loocv_results.csv", index=False)
print(f"LOOCV results saved to {os.path.abspath('knn_loocv_results.csv')}")

# =============================
# 5. Hyperparameter F1 score plot
# =============================
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(
    results_df['n_neighbors'],
    results_df['mean_f1_score'],
    c=results_df['mean_f1_score'],
    cmap='viridis',
    s=100,
    edgecolor='k'
)
ax.set_xlabel('Number of Neighbors (k)')
ax.set_ylabel('Mean F1 Score (LOOCV)')
ax.set_title('KNN Hyperparameter Tuning')
fig.colorbar(scatter, label='Mean F1 Score')
plt.grid(True)
plt.show()

# =============================
# 6. Learning curve
# =============================
# Use only the best hyperparameters for learning curve to reduce memory usage
best_row = results_df.iloc[results_df['mean_f1_score'].idxmax()]
pipeline.set_params(
    knn__n_neighbors=best_row['n_neighbors'],
    knn__weights=best_row['weights'],
    knn__algorithm=best_row['algorithm'],
    knn__p=best_row['p']
)

train_sizes, train_scores, test_scores = learning_curve(
    pipeline,
    X, y,
    cv=5,
    scoring='f1_weighted',
    train_sizes=np.linspace(0.1, 1.0, 5),  # reduce sizes for memory
    n_jobs=1  # sequential to avoid memory crashes
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
plt.plot(train_sizes, test_mean, 'o-', color='green', label='Cross-validation score')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='green')
plt.xlabel('Training Set Size')
plt.ylabel('F1 Score')
plt.title('Learning Curve: KNN Classifier (Best Hyperparameters)')
plt.legend(loc='best')
plt.grid(True)
plt.show()
