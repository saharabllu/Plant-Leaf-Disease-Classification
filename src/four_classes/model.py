"""CNN Architecture for 4-Class Plant Classification."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Flatten, Dense, Dropout
)

def build_cnn_4_classes() -> Sequential:
    """Builds the 4-layer CNN model specifically for the 4 plant classes."""
    NUM_CLASSES = 4
    
    model = Sequential([
        # Layer 1
        Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(150, 150, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Layer 2
        Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Layer 3
        Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Layer 4
        Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Classification Phase
        Flatten(),
        Dense(512, kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        # Output Layer (4 Classes)
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model
