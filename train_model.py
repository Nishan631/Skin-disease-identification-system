import math
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(BASE_DIR, "Skin_Diseases", "train")
TEST_DIR = os.path.join(BASE_DIR, "Skin_Diseases", "test")
MODEL_OUT = os.path.join(BASE_DIR, "Flask", "Skin_Diseases.h5")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20
CLASS_NAMES = ["Acne", "Melanoma", "Psoriasis", "Rosacea", "Vitiligo"]


def build_model():
    model = models.Sequential(
        [
            layers.Rescaling(1.0 / 255, input_shape=(*IMG_SIZE, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(5, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        height_shift_range=0.2,
        width_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
    )
    test_datagen = ImageDataGenerator()

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="categorical",
    )
    test_gen = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        classes=CLASS_NAMES,
        class_mode="categorical",
        shuffle=False,
    )
    if train_gen.samples == 0:
        raise ValueError(f"No training images found in: {TRAIN_DIR}")
    if test_gen.samples == 0:
        raise ValueError(f"No test images found in: {TEST_DIR}")

    model = build_model()
    steps_per_epoch = max(1, math.ceil(train_gen.samples / BATCH_SIZE))
    validation_steps = max(1, math.ceil(test_gen.samples / BATCH_SIZE))

    model.fit(
        train_gen,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_gen,
        validation_steps=validation_steps,
    )

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"Saved model to: {MODEL_OUT}")


if __name__ == "__main__":
    tf.random.set_seed(42)
    main()
