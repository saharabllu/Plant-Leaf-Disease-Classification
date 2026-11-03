"""CNN Architecture for 27-Class Disease Classification."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Flatten, Dense, Dropout
)

def build_cnn_27_classes() -> Sequential:
    """Builds the 5-layer CNN model specifically for the 27 disease classes."""
    NUM_CLASSES = 27
    
    model = Sequential([
        # Layer 1: Low-level features (edges, colors)
        Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=(150, 150, 3)),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Layer 2: Combinations of edges
        Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Layer 3: Textures and patterns
        Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Layer 4: Complex patterns (lesions, necrosis)
        Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Layer 5: High-level representation for highly similar classes (e.g., Tomato/Potato)
        Conv2D(512, (3, 3), padding='same', kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D((2, 2)),

        # Classification Phase
        Flatten(),
        Dense(512, kernel_initializer='he_normal'),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.5),

        # Output Layer (27 Classes)
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model
