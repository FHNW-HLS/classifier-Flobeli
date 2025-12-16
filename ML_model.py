import os
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# ---------------- Configuration ----------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 64                 # You can increase to 128/224 for better features (more RAM/time)
EPOCHS = 20                   # increased epochs (early stopping will usually stop earlier)
BATCH_SIZE = 32
NUM_FOLDS = 5

DATA_PATHS = {
    'open': 'class_open',
    'closed': 'class_closed'
}
MODEL_SAVE_PATH = 'best_edge_classifier_model.h5'
PLOTS_DIR = 'training_plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------- Utilities ----------------
def load_images_from_folder(folder, label, img_size):
    images, labels = [], []
    if not os.path.isdir(folder):
        print(f"⚠️ Folder not found: {folder}")
        return images, labels

    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder, filename)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, (img_size, img_size))

                # --- EDGE DETECTION ---
                img = cv2.GaussianBlur(img, (3, 3), 0)
                edges = cv2.Canny(img, threshold1=50, threshold2=150)

                images.append(edges.astype(np.float32))
                labels.append(label)

            except Exception as e:
                print(f"⚠️ Failed to load {img_path}: {e}")

    return images, labels


def build_model(input_shape):
    # Small but effective CNN with BatchNorm and Dropout
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),

        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
        ]
    )
    return model

def plot_and_save_history(history, fold, outdir=PLOTS_DIR):
    # Save multiple plots: accuracy, loss, auc, precision, recall
    hist = history.history
    epochs_range = range(1, len(hist.get('loss', [])) + 1)

    # 1) Accuracy
    plt.figure()
    plt.plot(epochs_range, hist.get('accuracy', []), label='train_acc')
    plt.plot(epochs_range, hist.get('val_accuracy', []), label='val_acc')
    plt.title(f'Fold {fold} - Accuracy')
    plt.legend()
    path = os.path.join(outdir, f'fold_{fold}_accuracy.png'); plt.savefig(path); plt.close()
    print(f"Saved {path}")

    # 2) Loss
    plt.figure()
    plt.plot(epochs_range, hist.get('loss', []), label='train_loss')
    plt.plot(epochs_range, hist.get('val_loss', []), label='val_loss')
    plt.title(f'Fold {fold} - Loss')
    plt.legend()
    path = os.path.join(outdir, f'fold_{fold}_loss.png'); plt.savefig(path); plt.close()
    print(f"Saved {path}")

    # 3) AUC
    if 'auc' in hist:
        plt.figure()
        plt.plot(epochs_range, hist.get('auc', []), label='train_auc')
        plt.plot(epochs_range, hist.get('val_auc', []), label='val_auc')
        plt.title(f'Fold {fold} - AUC')
        plt.legend()
        path = os.path.join(outdir, f'fold_{fold}_auc.png'); plt.savefig(path); plt.close()
        print(f"Saved {path}")

    # 4) Precision
    if 'precision' in hist:
        plt.figure()
        plt.plot(epochs_range, hist.get('precision', []), label='train_precision')
        plt.plot(epochs_range, hist.get('val_precision', []), label='val_precision')
        plt.title(f'Fold {fold} - Precision')
        plt.legend()
        path = os.path.join(outdir, f'fold_{fold}_precision.png'); plt.savefig(path); plt.close()
        print(f"Saved {path}")

    # 5) Recall
    if 'recall' in hist:
        plt.figure()
        plt.plot(epochs_range, hist.get('recall', []), label='train_recall')
        plt.plot(epochs_range, hist.get('val_recall', []), label='val_recall')
        plt.title(f'Fold {fold} - Recall')
        plt.legend()
        path = os.path.join(outdir, f'fold_{fold}_recall.png'); plt.savefig(path); plt.close()
        print(f"Saved {path}")

# ---------------- Load dataset ----------------
images_open, labels_open = load_images_from_folder(DATA_PATHS['open'], label=1, img_size=IMG_SIZE)
images_closed, labels_closed = load_images_from_folder(DATA_PATHS['closed'], label=0, img_size=IMG_SIZE)

if len(images_open) + len(images_closed) == 0:
    raise SystemExit("No images found in specified data folders. Please check DATA_PATHS and files.")

X = np.array(images_open + images_closed, dtype=np.float32) / 255.0
y = np.array(labels_open + labels_closed, dtype=np.int32)

# Add channel dimension for grayscale
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print(f"Total samples: {len(X)} (open={sum(y==1)}, closed={sum(y==0)})")

# ---------------- K-Fold Cross-Validation ----------------
kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
fold = 1
accuracies = []
aucs = []
precisions = []
recalls = []
f1s = []

best_val_auc = -1.0
best_model_path = MODEL_SAVE_PATH

# For overall metrics across folds
overall_y_true = []
overall_y_pred = []

for train_idx, val_idx in kf.split(X):
    print(f"\n--- Fold {fold}/{NUM_FOLDS} ---")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Class weights to handle imbalance
    try:
        cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = {i: cw_val for i, cw_val in enumerate(cw)}
        print(f"Class weights: {class_weights}")
    except Exception as e:
        class_weights = None
        print(f"Could not compute class weights ({e}), proceeding without them.")

    # Data augmentation generator
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.08,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train, seed=SEED)

    # Build model
    model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 1))

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    # Fit using augmented generator
    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=SEED),
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )

    # Predict on validation set
    y_val_pred_prob = model.predict(X_val, batch_size= BATCH_SIZE, verbose=0).ravel()
    y_val_pred = (y_val_pred_prob >= 0.5).astype(int)

    # Store overall predictions
    overall_y_true.extend(list(y_val))
    overall_y_pred.extend(list(y_val_pred))

    # Compute metrics for this fold
    fold_acc = np.mean(y_val_pred == y_val)
    fold_auc = tf.keras.metrics.AUC()(y_val, y_val_pred_prob).numpy()
    fold_precision = tf.keras.metrics.Precision()(y_val, y_val_pred).numpy()
    fold_recall = tf.keras.metrics.Recall()(y_val, y_val_pred).numpy()
    fold_f1 = f1_score(y_val, y_val_pred, zero_division=0)

    accuracies.append(fold_acc)
    aucs.append(fold_auc)
    precisions.append(fold_precision)
    recalls.append(fold_recall)
    f1s.append(fold_f1)

    print(f"Fold {fold} results:")
    print(f"  Accuracy : {fold_acc:.4f}")
    print(f"  AUC      : {fold_auc:.4f}")
    print(f"  Precision: {fold_precision:.4f}")
    print(f"  Recall   : {fold_recall:.4f}")
    print(f"  F1       : {fold_f1:.4f}")

    # Save model if this fold has the best validation AUC so far
    if fold_auc > best_val_auc:
        best_val_auc = fold_auc
        model.save(best_model_path)
        print(f"✅ New best model (by val AUC) saved to: {best_model_path}")

        # Save confusion matrix and classification report for best fold
        cm = confusion_matrix(y_val, y_val_pred)
        cr = classification_report(y_val, y_val_pred, target_names=['Closed', 'Open'])
        best_cm_path = os.path.join(PLOTS_DIR, f'best_fold_confusion_matrix.npy')
        np.save(best_cm_path, cm)
        with open(os.path.join(PLOTS_DIR, 'best_fold_classification_report.txt'), 'w') as f:
            f.write(cr)
        print(f"Saved best-fold confusion matrix -> {best_cm_path}")
        print("Saved best-fold classification report ->", os.path.join(PLOTS_DIR, 'best_fold_classification_report.txt'))

    # Save per-fold confusion matrix and classification report
    cm_fold = confusion_matrix(y_val, y_val_pred)
    with open(os.path.join(PLOTS_DIR, f'fold_{fold}_classification_report.txt'), 'w') as f:
        f.write(classification_report(y_val, y_val_pred, target_names=['Closed', 'Open']))
    np.save(os.path.join(PLOTS_DIR, f'fold_{fold}_confusion_matrix.npy'), cm_fold)
    print(f"Saved fold-{fold} classification report and confusion matrix.")

    # Plot history
    plot_and_save_history(history, fold)

    # Clean up for next fold
    K.clear_session()
    fold += 1

# ---------------- Final summary ----------------
print("\n=== Cross-Validation Summary ===")
for i, (a, au, p, r, f1s_i) in enumerate(zip(accuracies, aucs, precisions, recalls, f1s), start=1):
    print(f"Fold {i}: Acc={a:.4f}, AUC={au:.4f}, Precision={p:.4f}, Recall={r:.4f}, F1={f1s_i:.4f}")

print("\nAverages:")
print(f"  Accuracy mean : {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"  AUC mean      : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
print(f"  Precision mean: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"  Recall mean   : {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"  F1 mean       : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

# ---------------- Overall confusion & classification report ----------------
overall_y_true = np.array(overall_y_true)
overall_y_pred = np.array(overall_y_pred)
print("\n=== Overall Results (all folds combined) ===")
overall_cm = confusion_matrix(overall_y_true, overall_y_pred)
overall_cr = classification_report(overall_y_true, overall_y_pred, target_names=['Closed', 'Open'])
overall_f1 = f1_score(overall_y_true, overall_y_pred, zero_division=0)

print("Confusion Matrix:\n", overall_cm)
print("\nClassification Report:\n", overall_cr)
print(f"\nOverall F1: {overall_f1:.4f}")

# Save overall results to disk
np.save(os.path.join(PLOTS_DIR, 'overall_confusion_matrix.npy'), overall_cm)
with open(os.path.join(PLOTS_DIR, 'overall_classification_report.txt'), 'w') as f:
    f.write(overall_cr)
print(f"\nSaved overall confusion matrix and classification report in: {PLOTS_DIR}")

print(f"\nBest model (by val AUC) saved at: {best_model_path}")
print("Done.")