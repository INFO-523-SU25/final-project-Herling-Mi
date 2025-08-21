import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import f1_score, precision_score, accuracy_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ==========================
# File paths and columns
# ==========================
CSV_PATH = r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\0_Final_Proj_N\0_col_Mel_Tran\output_sorted_with_viridis.csv"
IMG_COLUMN = "color_png"
TARGET_COLUMN = "genre_enc"

# ==========================
# Load dataframe
# ==========================
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df['genre_enc'] = df['genre_enc'].astype(str)

# ==========================
# Image settings
# ==========================
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16

# ==========================
# Train-test split
# ==========================
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df[TARGET_COLUMN],
    random_state=42
)

# ==========================
# Image data generators
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df,
    x_col=IMG_COLUMN,
    y_col=TARGET_COLUMN,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_gen = val_datagen.flow_from_dataframe(
    val_df,
    x_col=IMG_COLUMN,
    y_col=TARGET_COLUMN,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================
# Hyperparameter sweep
# ==========================
param_grid = {
    "conv_filters": [[32, 64], [64, 128], [32, 64, 128]],
    "kernel_size": [(3,3), (5,5)],
    "dropout": [0.2, 0.3, 0.5],
    "learning_rate": [1e-2, 1e-3, 1e-4]
}

results = []

for params in ParameterGrid(param_grid):
    print(f"Training with params: {params}")

    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(params['conv_filters'][0], params['kernel_size'], activation='relu'),
        MaxPooling2D(),
        Conv2D(params['conv_filters'][1], params['kernel_size'], activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dropout(params['dropout']),
        Dense(128, activation='relu'),
        Dense(len(train_gen.class_indices), activation='softmax')
    ])

    optimizer = Adam(learning_rate=params['learning_rate'])
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=30,
        callbacks=[early_stop],
        verbose=2
    )

    # ==========================
    # Compute sklearn metrics
    # ==========================
    val_gen.reset()
    y_true = val_gen.labels
    y_pred_probs = model.predict(val_gen, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    val_acc = accuracy_score(y_true, y_pred)
    val_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    val_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    results.append({
        **params,
        'val_acc': val_acc,
        'val_precision': val_precision,
        'val_f1': val_f1,
        'conv_filters_str': str(params['conv_filters'])
    })

# ==========================
# Results dataframe & plot
# ==========================
results_df = pd.DataFrame(results)
plt.figure(figsize=(10,6))
results_df['dropout'] = results_df['dropout'].astype(float)

scatter = plt.scatter(
    x=results_df['conv_filters_str'],
    y=results_df['val_acc'],
    c=results_df['dropout'],
    cmap='viridis',
    s=150
)

plt.colorbar(scatter, label='Dropout Rate')
plt.title("Hyperparameter Sweep Accuracy (Viridis)")
plt.xlabel("Conv Filters")
plt.ylabel("Validation Accuracy")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ==========================
# Train best model
# ==========================
best_params = results_df.loc[results_df['val_f1'].idxmax()].to_dict()  # choose based on F1
print(f"Best params: {best_params}")

best_model = Sequential([
    Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    Conv2D(best_params['conv_filters'][0], best_params['kernel_size'], activation='relu'),
    MaxPooling2D(),
    Conv2D(best_params['conv_filters'][1], best_params['kernel_size'], activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(best_params['dropout']),
    Dense(128, activation='relu'),
    Dense(len(train_gen.class_indices), activation='softmax')
])

optimizer = Adam(learning_rate=best_params['learning_rate'])
best_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

history = best_model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[early_stop],
    verbose=1
)

# ==========================
# Compute final metrics for best model
# ==========================
val_gen.reset()
y_true = val_gen.labels
y_pred_probs = best_model.predict(val_gen, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

final_acc = accuracy_score(y_true, y_pred)
final_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
final_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"Final Best Model Metrics:\nAccuracy: {final_acc:.4f}, Precision: {final_precision:.4f}, F1: {final_f1:.4f}")

# ==========================
# Learning curves
# ==========================
plt.figure(figsize=(10,6))
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Learning Curve of Best Model")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
