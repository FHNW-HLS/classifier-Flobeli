# ============================================================
# Edge-only Binary Classifier (1-Kanal) + Threshold-Suche
# ============================================================

import os
import random
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
)
from sklearn.utils import class_weight

# -------------------- Configuration --------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 20
NUM_FOLDS = 5

DATA_PATHS = {
    "open": "class_open",
    "closed": "class_closed",
}

OUT_DIR = "training_plots"
MODEL_PATH = "best_edge_classifier_model.h5"
os.makedirs(OUT_DIR, exist_ok=True)

# -------------------- Data Loading --------------------
def load_edge_images(folder, label, img_size):
    images, labels = [], []

    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        path = os.path.join(folder, fn)
        with Image.open(path) as img:
            img = img.resize((img_size, img_size), resample=Image.NEAREST)
            arr = np.array(img)

            # RGB / RGBA -> robuster Edge-Kanal
            if arr.ndim == 3:
                arr = arr[..., :3].max(axis=2)

            images.append(arr.astype(np.float32))
            labels.append(label)

    return images, labels


# -------------------- Model --------------------
def build_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


# -------------------- Threshold Search --------------------
def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_f1 = 0.5, -1.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


# -------------------- Load Dataset --------------------
X_open, y_open = load_edge_images(DATA_PATHS["open"], 1, IMG_SIZE)
X_closed, y_closed = load_edge_images(DATA_PATHS["closed"], 0, IMG_SIZE)

X = np.array(X_open + X_closed, dtype=np.float32) / 255.0
y = np.array(y_open + y_closed, dtype=np.int32)

X = X[..., np.newaxis]

print("X shape:", X.shape)
print("Samples:", len(X), "Open:", np.sum(y == 1), "Closed:", np.sum(y == 0))


# -------------------- Cross Validation --------------------
skf = StratifiedKFold(
    n_splits=NUM_FOLDS,
    shuffle=True,
    random_state=SEED,
)

best_auc = -1.0
all_val_probs = []
all_val_true = []

fold = 1
for train_idx, val_idx in skf.split(X, y):
    print(f"\n--- Fold {fold}/{NUM_FOLDS} ---")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = {i: cw[i] for i in range(len(cw))}

    model = build_model(X.shape[1:])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5),
    ]

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Validation probabilities
    y_prob = model.predict(X_val, batch_size=BATCH_SIZE).ravel()

    # ---- Threshold search (FOLD-LOKAL)
    best_t, best_f1 = find_best_threshold(y_val, y_prob)
    y_pred = (y_prob >= best_t).astype(int)

    print(f"Best threshold: {best_t:.3f} | F1: {best_f1:.4f}")

    all_val_probs.extend(y_prob)
    all_val_true.extend(y_val)

    auc = tf.keras.metrics.AUC()(y_val, y_prob).numpy()

    if auc > best_auc:
        best_auc = auc
        model.save(MODEL_PATH)
        print("Best model saved.")

    K.clear_session()
    fold += 1


# -------------------- GLOBAL Threshold --------------------
all_val_probs = np.array(all_val_probs)
all_val_true = np.array(all_val_true)

global_t, global_f1 = find_best_threshold(all_val_true, all_val_probs)
final_preds = (all_val_probs >= global_t).astype(int)

cm = confusion_matrix(all_val_true, final_preds)
cr = classification_report(
    all_val_true,
    final_preds,
    target_names=["Closed", "Open"],
)

print("\n=== Overall (Global Threshold) ===")
print("Threshold:", global_t)
print(cm)
print(cr)
print("Overall F1:", global_f1)

np.save(os.path.join(OUT_DIR, "overall_confusion_matrix.npy"), cm)
with open(os.path.join(OUT_DIR, "overall_classification_report.txt"), "w") as f:
    f.write(cr)

print("\nTraining finished.")
print("Best model:", MODEL_PATH)
