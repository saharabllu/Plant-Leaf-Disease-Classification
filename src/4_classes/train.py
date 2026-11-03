"""Training loop for the 4-Class Plant Classification model."""

import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_4_classes(model, train_gen, val_gen, save_path: str, epochs: int = 50):
    """Compiles and trains the 4-class CNN model."""

    os.makedirs(os.path.dirname(save_path), exist_ok=True
    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=1
    )
    model_checkpoint = ModelCheckpoint(
        filepath=save_path, monitor='val_accuracy', save_best_only=True, verbose=1
    )
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # Training
    history = model.fit(
        train_gen,
        epochs=epochs,
        batch_size=32,
        validation_data=val_gen,
        callbacks=[early_stop, model_checkpoint],
        verbose=1
    )
    return history
