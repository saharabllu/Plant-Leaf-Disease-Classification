"""CNN model architecture for Plant Leaf Disease Classification."""

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, Activation,
    MaxPooling2D, Flatten, Dense, Dropout
)

def build_cnn(num_classes: int) -> Sequential:
    """Builds and returns a custom CNN model.
    
    Args:
        num_classes (int): Number of output classes (4 for plant types, 27 for diseases).
    """
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

        # Layer 5: High-level representation for highly similar classes
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

        # Output Layer
        Dense(num_classes, activation='softmax')
    ])
    
    return model
