import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Configuration
DATASET_DIR = 'datasets'
IMG_HEIGHT = 180
IMG_WIDTH = 180
BATCH_SIZE = 32
EPOCHS = 50

def train_model():
    print("Loading datasets...")
    
    # Load training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    # Load validation data
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)

    class_names = train_ds.class_names
    print(f"Found classes: {class_names}")

    # Configure dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Data augmentation
    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",
                          input_shape=(IMG_HEIGHT,
                                       IMG_WIDTH,
                                       3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )

    # Build the model
    num_classes = len(class_names)

    # Build the model using Transfer Learning (MobileNetV2)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False

    model = Sequential([
      data_augmentation,
      layers.Rescaling(1./127.5, offset=-1), # MobileNetV2 expects [-1, 1]
      base_model,
      layers.GlobalAveragePooling2D(),
      layers.Dropout(0.2),
      layers.Dense(len(class_names))
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    # Train the model
    print("Starting training...")
    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=EPOCHS
    )

    # Save the model
    print("Saving model to flower_model.keras...")
    model.save('flower_model.keras')
    
    # Save class names to a file for app.py to read
    with open('class_names.txt', 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
            
    print("Training complete!")

if __name__ == "__main__":
    train_model()
