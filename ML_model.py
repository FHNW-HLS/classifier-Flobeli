# ============================================================
# Edge-only Binary Classifier (Schwarz-Weiß, 1-Kanal)
# Komplett neu geschrieben
# ============================================================

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
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
    """
    Lädt SCHWARZ-WEISS Edge-Bilder.
    Falls Datei als RGB gespeichert ist, wird exakt EIN Kanal genommen.
    Keine zusätzliche Grayscale-Konvertierung.
    """
    images, labels = [], []

    for fn in sorted(os.listdir(folder)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue

        path = os.path.join(folder, fn)
        with Image.open(path) as img:
            img = img.resize((img_size, img_size))
            arr = np.array(img)

            # RGB/RGBA -> 1 Kanal
            if arr.ndim == 3:
                arr = arr[:, :, 0]

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

        layers.Flatten(),
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
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


# -------------------- Load Dataset --------------------
X_open, y_open = load_edge_images(DATA_PATHS["open"], 1, IMG_SIZE)
X_closed, y_closed = load_edge_images(DATA_PATHS["closed"], 0, IMG_SIZE)

X = np.array(X_open + X_closed, dtype=np.float32) / 255.0
y = np.array(y_open + y_closed, dtype=np.int32)

# (N,H,W) -> (N,H,W,1)
X = X[..., np.newaxis]

print("X shape:", X.shape)
print("Samples:", len(X), "Open:", np.sum(y == 1), "Closed:", np.sum(y == 0))

# -------------------- K-Fold Training --------------------
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

best_auc = -1.0
overall_y_true, overall_y_pred = [], []

fold = 1
for train_idx, val_idx in kf.split(X):
    print(f"\n--- Fold {fold}/{NUM_FOLDS} ---")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Class weights
    cw = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = {i: cw[i] for i in range(len(cw))}

    # Augmentation (safe for edges)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    model = build_model(input_shape=X.shape[1:])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6),
    ]

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=SEED),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Validation
    y_prob = model.predict(X_val, batch_size=BATCH_SIZE).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    overall_y_true.extend(y_val)
    overall_y_pred.extend(y_pred)

    auc = tf.keras.metrics.AUC()(y_val, y_prob).numpy()
    f1 = f1_score(y_val, y_pred, zero_division=0)

    print(f"AUC: {auc:.4f} | F1: {f1:.4f}")

    # Save best model
    if auc > best_auc:
        best_auc = auc
        model.save(MODEL_PATH)
        print("Best model saved.")

        np.save(os.path.join(OUT_DIR, "best_confusion_matrix.npy"),
                confusion_matrix(y_val, y_pred))
        with open(os.path.join(OUT_DIR, "best_classification_report.txt"), "w") as f:
            f.write(classification_report(y_val, y_pred,
                                          target_names=["Closed", "Open"]))

    K.clear_session()
    fold += 1

# -------------------- Overall Results --------------------
overall_y_true = np.array(overall_y_true)
overall_y_pred = np.array(overall_y_pred)

cm = confusion_matrix(overall_y_true, overall_y_pred)
cr = classification_report(overall_y_true, overall_y_pred,
                           target_names=["Closed", "Open"])
f1 = f1_score(overall_y_true, overall_y_pred, zero_division=0)

print("\n=== Overall ===")
print(cm)
print(cr)
print("Overall F1:", f1)

np.save(os.path.join(OUT_DIR, "overall_confusion_matrix.npy"), cm)
with open(os.path.join(OUT_DIR, "overall_classification_report.txt"), "w") as f:
    f.write(cr)

print("\nTraining finished.")
print("Best model:", MODEL_PATH)
