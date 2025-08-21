import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score
import tkinter as tk
from tkinter import ttk
from itertools import product

# ---------- CONFIG ----------
CSV_PATH = r"output_sorted_with_grey.csv""
IMG_BASE_PATH = r"C:\Users\Mr_Green\OneDrive\Desktop\0_Sum_25\INFO_523\0_Final_Proj_N\wav_files"
IMG_COLUMN = "grey_png"
TARGET_COLUMN = "genre_enc"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
# Hyperparameters to sweep
EPOCHS_LIST = [10, 15, 30]
NUM_CONV_LIST = [2, 3]
PATIENCE_LIST = [2, 5]

# ---------- LOAD CSV ----------
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
df.columns = df.columns.str.strip()
cols_to_ignore = ['artist', 'song_name', 'genre']
df = df.drop(columns=[col for col in cols_to_ignore if col in df.columns], errors='ignore')

if IMG_COLUMN not in df.columns:
    possible_img_cols = [col for col in df.columns if col.lower().endswith('.png')]
    if possible_img_cols:
        IMG_COLUMN = possible_img_cols[0]
        print(f"Using inferred image column: {IMG_COLUMN}")
    else:
        raise KeyError(f"Image column '{IMG_COLUMN}' not found. Available columns: {df.columns.tolist()}")

# ---------- UI SETUP ----------
root = tk.Tk()
root.title("Training Progress")
root.geometry("400x100")
progress_label = tk.Label(root, text="Starting training...")
progress_label.pack(pady=10)
progress_bar = ttk.Progressbar(root, orient="horizontal", length=350, mode="determinate")
progress_bar.pack(pady=10)
root.update_idletasks()

# ---------- DATA PREPARATION ----------
images, labels = [], []
total_files = len(df)
progress_bar['maximum'] = total_files

for idx, row in df.iterrows():
    img_name = str(row[IMG_COLUMN]).strip()
    img_path = os.path.join(IMG_BASE_PATH, img_name)
    try:
        img = load_img(img_path, color_mode="grayscale", target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(row[TARGET_COLUMN])
    except Exception as e:
        print(f"Error loading {img_path}: {e}")

    progress_bar['value'] = idx + 1
    progress_label.config(text=f"Processing {idx+1}/{total_files}: {img_name}")
    root.update_idletasks()

X = np.array(images)
y = to_categorical(np.array(labels))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- HYPERPARAMETER SWEEP ----------
results = []

for epochs, num_conv, patience in product(EPOCHS_LIST, NUM_CONV_LIST, PATIENCE_LIST):
    progress_label.config(text=f"Training: conv={num_conv}, epochs={epochs}, patience={patience}")
    root.update_idletasks()

    model = Sequential()
    model.add(Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)))  # Input layer

    # Add convolutional layers with Batch Normalization
    for i in range(num_conv):
        filters = 32 * (2 ** i)
        model.add(Conv2D(filters, (3,3), activation='relu'))
        model.add(BatchNormalization())  # <-- Batch Norm added
        model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())  # <-- Batch Norm after dense
    model.add(Dropout(0.5))
    model.add(Dense(y.shape[1], activation='softmax', kernel_regularizer=l2(0.001)))

    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[early_stop]
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)

    results.append({
        'conv_layers': num_conv,
        'epochs': epochs,
        'patience': patience,
        'accuracy': acc,
        'f1_score': f1,
        'precision': prec,
        'history': history
    })

# ---------- PLOTS ----------
plt.figure(figsize=(10,6))
for r in results:
    plt.scatter(r['conv_layers'], r['accuracy'], s=r['epochs']*10, label=f"patience={r['patience']}")
plt.xlabel("Num Conv Layers")
plt.ylabel("Accuracy")
plt.title("Hyperparameter Sweep: Accuracy vs Conv Layers")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
history = results[-1]['history']
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Learning Curve (last run)")
plt.legend()
plt.show()

# ---------- DISPLAY METRICS ----------
for r in results:
    print(f"Conv Layers={r['conv_layers']}, Epochs={r['epochs']}, Patience={r['patience']}")
    print(f"Accuracy={r['accuracy']:.4f}, F1={r['f1_score']:.4f}, Precision={r['precision']:.4f}")
    print("-"*50)

root.destroy()
