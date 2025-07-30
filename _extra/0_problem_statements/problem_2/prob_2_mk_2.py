from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

easy_text = """
Easy Path: Basic Collaborative Filtering + Simple Metadata Features

1. Data Input & Preprocessing:
- Load user interaction data with track metadata including artist, genre, and basic audio features (Mean, Tempo, Danceability).
- Encode categorical metadata (artist, genre) using one-hot or label encoding.
- Normalize numerical features (Mean, Tempo, Danceability) using Min-Max scaling or StandardScaler.
- Prepare user-item interaction matrix (implicit feedback or ratings).

2. Feature Engineering:
- Construct feature vectors combining user preferences and track metadata features.
- Optionally incorporate vocal and non-vocal audio statistics separately.
- Aggregate user listening history to represent preferences.

3. Model Selection & Training:
- Split dataset using Train-Test Split.
- Train simple recommendation models such as K-Nearest Neighbors or Logistic Regression to predict user preferences.
- Use collaborative filtering based on nearest neighbors.

4. Model Evaluation:
- Evaluate using Accuracy, Precision@K, and Recall@K.
- Analyze recommendation relevance by comparing recommended tracks against user history.
- Perform offline validation on held-out test set.

Handling Vocal/Audio Tracks:
- If vocal and audio tracks are available, include vocal track features as primary inputs.
- Optionally append non-vocal track statistics for enhanced feature vectors.
- Process vocal and audio features separately before integration.

ML Techniques Used:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Train-Test Split
- Accuracy, Precision, Recall metrics
- One-hot Encoding / Label Encoding
- Feature Normalization (Min-Max Scaling, StandardScaler)
"""

medium_text = """
Medium Path: Content-Based Filtering with Advanced Audio Features and Clustering

1. Data Input & Preprocessing:
- Load track metadata including full audio features (Mean, Variance, Skewness, Kurtosis, RMS Energy, Zero Crossing Rate, Tempo, Loudness) separately for vocal and non-vocal tracks.
- Normalize all numerical features using StandardScaler or RobustScaler.
- Encode categorical features (Genre, Artist) using embeddings or one-hot encoding.
- Apply PCA to reduce dimensionality of high-dimensional audio features.

2. Feature Engineering:
- Combine vocal and non-vocal track features into a joint feature vector.
- Cluster tracks and users using K-Means or Hierarchical Clustering on reduced features.
- Represent each user by cluster memberships or aggregated cluster centroids.

3. Model Selection & Training:
- Use Stratified K-Fold Cross-Validation for train-test splits.
- Train Random Forest classifiers or Gradient Boosting Machines on clustered feature sets.
- Integrate user preferences and cluster information for personalized recommendations.

4. Model Evaluation:
- Evaluate using Recall@K, F1-score, and cluster validity metrics.
- Conduct ablation studies comparing vocal-only, non-vocal-only, and combined feature sets.
- Analyze recommendation diversity and cluster cohesion.

Handling Vocal/Audio Tracks:
- Extract and preprocess vocal and audio features separately.
- Use separate pipelines for vocal and non-vocal features before merging.
- Experiment with feature importance to determine contributions from each track type.

ML Techniques Used:
- Random Forest
- Gradient Boosting Machines (GBM)
- K-Means Clustering
- Hierarchical Clustering
- PCA (Principal Component Analysis)
- Stratified K-Fold Cross-Validation
- Recall, F1-score
- One-hot Encoding / Embeddings
- Feature Normalization (StandardScaler, RobustScaler)
"""

hard_text = """
Hard Path: Hybrid Deep Learning with Sequential Patterns and Metadata Fusion

1. Data Input & Preprocessing:
- Load raw audio data with separated vocal and non-vocal tracks.
- Extract Mel-Spectrograms, STFT, and sequence mining features (e.g., Generalized Sequential Pattern (GSP)) from user-item interaction logs.
- Normalize features separately for vocal and non-vocal tracks.
- Encode categorical metadata (Artist, Genre, Key) as learned embeddings.

2. Feature Engineering:
- Develop multi-input feature sets combining spectrogram embeddings, sequential pattern features, and metadata embeddings.
- Use autoencoders or deep feature extractors for dimensionality reduction.
- Fuse vocal and non-vocal track features via learned representation layers.

3. Model Selection & Training:
- Use Nested Cross-Validation or Bayesian Optimization for hyperparameter tuning.
- Build hybrid deep learning architectures with CNNs or Transformers on spectrogram inputs and RNNs on sequential data.
- Combine with ensemble methods (XGBoost, stacking) on metadata and sequence features.

4. Model Evaluation:
- Evaluate with advanced metrics: Mean Average Precision (MAP), personalized recommendation accuracy, and user satisfaction metrics.
- Perform ablation studies on vocal vs non-vocal input contributions.
- Analyze error patterns per language, genre, and user segment.

Handling Vocal/Audio Tracks:
- Design separate model branches for vocal and non-vocal audio inputs.
- Augment vocal and audio tracks independently during training.
- Integrate multi-modal inputs with metadata embeddings for end-to-end training.

ML Techniques Used:
- Deep Learning (CNN, RNN, Transformers)
- Autoencoders
- Sequence Mining (GSP)
- XGBoost, Stacking
- Nested Cross-Validation
- Bayesian Optimization
- Mean Average Precision (MAP)
- Embeddings for categorical data
- Feature Normalization
"""

def generate_pdf(filename, title, content):
    doc = SimpleDocTemplate(filename, pagesize=LETTER,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    styleH = styles['Heading1']
    styleN = styles['Normal']

    story = []
    story.append(Paragraph(title, styleH))
    story.append(Spacer(1, 12))

    for paragraph in content.strip().split('\n\n'):
        story.append(Paragraph(paragraph.replace('\n', '<br />'), styleN))
        story.append(Spacer(1, 12))

    doc.build(story)
    print(f"PDF created: {filename}")

if __name__ == "__main__":
    generate_pdf("Easy_Music_Recommendation_Detailed.pdf", "Easy Path Pipeline", easy_text)
    generate_pdf("Medium_Music_Recommendation_Detailed.pdf", "Medium Path Pipeline", medium_text)
    generate_pdf("Hard_Music_Recommendation_Detailed.pdf", "Hard Path Pipeline", hard_text)
