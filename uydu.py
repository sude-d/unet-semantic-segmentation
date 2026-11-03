import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

keras = tf.keras
layers = keras.layers

def load_dataset(root, img_size=(128, 128)):
    images, masks = [], []

    for tile in sorted(os.listdir(root)):
        img_dir = os.path.join(root, tile, "images")
        mask_dir = os.path.join(root, tile, "masks")

        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            continue

        for f in sorted(os.listdir(img_dir)):
            if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(img_dir, f)
            mask_path = os.path.join(mask_dir, os.path.splitext(f)[0] + ".png")

            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)
            img = img.astype("float32") / 255.0

            mask = cv2.resize(
                mask,
                img_size,
                interpolation=cv2.INTER_NEAREST
            )

            mask = (mask > 127).astype("float32")
            mask = np.expand_dims(mask, axis=-1)

            images.append(img)
            masks.append(mask)

    return (
        np.array(images, dtype="float32"),
        np.array(masks, dtype="float32")
    )

X, y = load_dataset(
    "Semantic segmentation dataset",
    img_size=(128, 128)
)

print(f"{len(X)} images and {len(y)} masks loaded.")

if len(X) == 0:
    raise ValueError("No images found. Check your dataset directory and file names.")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training images:", X_train.shape)
print("Validation images:", X_val.shape)
print("Training masks:", y_train.shape)
print("Validation masks:", y_val.shape)

def unet_model(input_size=(128, 128, 3)):
    inputs = keras.Input(shape=input_size)

    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(16, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, 3, activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D()(c3)

    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(p3)
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(c4)
    p4 = layers.MaxPooling2D()(c4)

    c5 = layers.Conv2D(256, 3, activation="relu", padding="same")(p4)
    c5 = layers.Conv2D(256, 3, activation="relu", padding="same")(c5)

    u6 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, 3, activation="relu", padding="same")(u6)
    c6 = layers.Conv2D(128, 3, activation="relu", padding="same")(c6)

    u7 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, 3, activation="relu", padding="same")(u7)
    c7 = layers.Conv2D(64, 3, activation="relu", padding="same")(c7)

    u8 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(u8)
    c8 = layers.Conv2D(32, 3, activation="relu", padding="same")(c8)

    u9 = layers.Conv2DTranspose(16, 2, strides=2, padding="same")(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(16, 3, activation="relu", padding="same")(u9)
    c9 = layers.Conv2D(16, 3, activation="relu", padding="same")(c9)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c9)

    return keras.Model(inputs=inputs, outputs=outputs)

model = unet_model()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "unet_model.keras",
        monitor="val_loss",
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=16,
    epochs=10,
    callbacks=callbacks
)

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("U-Net Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()

def show_prediction(idx=0):
    if idx < 0 or idx >= len(X_val):
        raise IndexError("Invalid validation image index.")

    img = X_val[idx]
    mask_true = y_val[idx].squeeze()

    pred_raw = model.predict(img[None, ...], verbose=0)[0].squeeze()
    mask_pred = (pred_raw > 0.5).astype("float32")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_true, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(mask_pred, cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

show_prediction(0)
