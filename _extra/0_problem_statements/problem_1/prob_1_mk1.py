from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

# Pipeline texts (your original texts)
easy_text = """
Easy Path: Statistical Features from Vocal & Audio Tracks + Traditional ML

Pipeline Outline

1. Data Input & Preprocessing:
   - Load audio files with separated vocal and non-vocal (instrumental/audio) tracks.
   - Extract or load precomputed statistical features (Mean, Variance, Skewness, Kurtosis, RMS Energy, Zero Crossing Rate, Tempo, Loudness) separately for vocal and audio tracks.
   - Normalize numerical features (e.g., Min-Max or Standard Scaling).
   - Encode categorical metadata (if using, e.g., Genre, Artist).

2. Feature Engineering:
   - Keep vocal track statistics as primary features.
   - Optionally append audio track statistics as additional features.
   - Combine all features into one feature vector per sample.

3. Model Selection & Training:
   - Split dataset into train/test sets (Stratified K-Fold recommended).
   - Train classical classifiers (Logistic Regression, Random Forest, or SVM) on combined feature vectors.

4. Model Evaluation:
   - Evaluate with Accuracy, Precision, Recall, F1-score.
   - Compare performance with vocal-only features vs vocal+audio features.
   - Use confusion matrix to inspect language-specific errors.

Handling Vocal/Audio Tracks:
- Load and process vocal and audio tracks separately at every stage.
- Use vocal track features as main input; add audio features optionally.
"""

medium_text = """
Medium Path: Time-Frequency Features from Vocal & Audio Tracks + Classical ML (e.g., GMMs)

Pipeline Outline

1. Data Input & Preprocessing:
   - Load separate vocal and audio waveforms for each sample.
   - Extract time-frequency features such as Mel-Frequency Cepstral Coefficients (MFCCs), spectral centroid, bandwidth, or STFT.
   - Normalize or standardize features per track.

2. Feature Engineering & Integration:
   - Aggregate or summarize time-frequency features across frames (e.g., means, variances).
   - Combine vocal and audio track features into a single feature vector.

3. Model Selection & Training:
   - Use classical models suited for time-series or distributions, e.g., Gaussian Mixture Models (GMM), Hidden Markov Models (HMM), or classical classifiers like Random Forest and SVM on summarized features.
   - Perform hyperparameter tuning via grid/random search.

4. Validation:
   - Use stratified K-Fold cross-validation to ensure balanced representation across languages.
   - Optionally use likelihood-based scoring for GMMs or standard classification metrics for others.

5. Evaluation:
   - Evaluate with Accuracy, Precision, Recall, F1-score.
   - Use confusion matrices to identify common misclassifications.
   - Perform ablation comparing vocal-only features, audio-only, and combined features.

Handling Vocal/Audio Tracks:
- Process vocal and audio tracks separately up to feature extraction.
- Merge features carefully to leverage complementary information.
"""

ml_medium = """
ML Techniques Used:
- Gaussian Mixture Models (GMM)
- Hidden Markov Models (HMM) (optional)
- Random Forest
- Support Vector Machines (SVM)
- Stratified K-Fold Cross-Validation
- Accuracy, Precision, Recall, F1-score metrics
- Confusion Matrix analysis
- Feature Normalization (StandardScaler, Min-Max Scaling)
- Hyperparameter tuning via grid or random search
"""

hard_text = """
Hard Path: End-to-End Multi-Input Deep Learning with Vocal & Audio + Sequential Models

Pipeline Outline

1. Data Input & Preprocessing:
   - Load raw audio for vocal and non-vocal (instrumental/audio) tracks separately.
   - Segment into fixed-length frames if necessary.
   - Extract time-frequency features (Mel-Spectrogram or raw waveform) for both vocal and audio tracks.
   - Normalize features independently.

2. Metadata Preparation:
   - Encode metadata (Artist, Tempo, Genre) as embeddings or one-hot vectors.

3. Model Architecture:
   - Dual-input deep model:
     - Branch 1: CNN or Transformer layers on vocal track features.
     - Branch 2: CNN or Transformer layers on audio track features.
     - Branch 3: Metadata input processed via dense layers.
   - Concatenate outputs of three branches.
   - Fully connected layers for final classification.

4. Training Setup:
   - Use advanced augmentation on vocal/audio tracks separately.
   - Use Nested Cross-Validation or Bayesian Optimization for hyperparameter tuning.
   - Apply dropout, batch normalization, and early stopping.

5. Evaluation:
   - Calculate accuracy, macro-averaged F1-score, confusion matrix.
   - Conduct ablation to evaluate vocal-only, audio-only, and combined model contributions.
   - Perform error analysis per language and per input type.

Handling Vocal/Audio Tracks:
- Use separate model branches for vocal and audio inputs.
- Normalize and augment vocal and audio inputs independently.
- Integrate metadata as auxiliary input.
"""

# ML Techniques lists for each path
ml_easy = """
ML Techniques Used:
- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)
- Stratified K-Fold Cross-Validation
- Accuracy, Precision, Recall, F1-score metrics
- Confusion Matrix analysis
- Feature Normalization (Min-Max, Standard Scaling)
- Encoding categorical metadata (One-hot, Label Encoding)
"""

ml_medium = """
ML Techniques Used:
- Convolutional Neural Networks (CNN)
- K-Fold Cross-Validation
- Early Stopping
- Accuracy, Macro F1-score
- Ablation Studies
- Feature Normalization (StandardScaler, RobustScaler)
- Data Augmentation (time-shifting, pitch-shifting, noise addition)
"""

ml_hard = """
ML Techniques Used:
- Deep Learning Architectures (CNN, Transformer, RNN)
- Nested Cross-Validation
- Bayesian Optimization
- Dropout, Batch Normalization, Early Stopping
- Accuracy, Macro-averaged F1-score, Confusion Matrix
- Ablation Studies
- Embeddings for metadata
- Data Augmentation on multi-input branches
"""

def generate_pdf(filename, title, content, ml_techniques):
    doc = SimpleDocTemplate(filename, pagesize=LETTER,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    styleN = styles['Normal']
    styleH = styles['Heading1']

    story = []

    # Title
    story.append(Paragraph(title, styleH))
    story.append(Spacer(1, 12))

    # Pipeline content paragraphs
    for paragraph in content.strip().split('\n\n'):
        paragraph = paragraph.strip()
        if paragraph:
            story.append(Paragraph(paragraph.replace('\n', '<br />'), styleN))
            story.append(Spacer(1, 12))

    # Add ML techniques section at the end
    story.append(Spacer(1, 24))
    story.append(Paragraph("Machine Learning Techniques Summary", styleH))
    story.append(Spacer(1, 12))
    for paragraph in ml_techniques.strip().split('\n\n'):
        paragraph = paragraph.strip()
        if paragraph:
            story.append(Paragraph(paragraph.replace('\n', '<br />'), styleN))
            story.append(Spacer(1, 12))

    doc.build(story)
    print(f"Created PDF: {filename}")

if __name__ == "__main__":
    generate_pdf("Easy_Path_Pipeline_ReportLab.pdf", "Easy Path Pipeline", easy_text, ml_easy)
    generate_pdf("Medium_Path_Pipeline_ReportLab.pdf", "Medium Path Pipeline", medium_text, ml_medium)
    generate_pdf("Hard_Path_Pipeline_ReportLab.pdf", "Hard Path Pipeline", hard_text, ml_hard)
