"""Preprocessing for 4-Class Plant Classification (Kentang, Lemon, Teh, Tomat)."""

import os
import shutil
import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def split_dataset_4_classes(raw_path: str, split_path: str, seed: int = 42):
    """Splits dataset into train, val, test for the 4 main plant classes."""
    os.makedirs(split_path, exist_ok=True)
    plants = ["Kentang", "Lemon", "Teh", "Tomat"]

    for plant in plants:
        plant_src = os.path.join(raw_path, plant)
        if not os.path.exists(plant_src):
            continue

        plant_output = os.path.join(split_path, f"_tmp_{plant}")
        
        # Split per plant folder (80% train, 10% val, 10% test)
        splitfolders.ratio(plant_src, output=plant_output, seed=seed, ratio=(0.8, 0.1, 0.1), move=False)

        for subset in ["train", "val", "test"]:
            src_subset = os.path.join(plant_output, subset)
            dst_subset = os.path.join(split_path, subset, plant)
            
            if os.path.exists(src_subset):
                os.makedirs(dst_subset, exist_ok=True)
                for disease_folder in os.listdir(src_subset):
                    s = os.path.join(src_subset, disease_folder)
                    d = os.path.join(dst_subset, disease_folder)
                    if os.path.isdir(s):
                        if os.path.exists(d):
                            shutil.rmtree(d)
                        shutil.copytree(s, d)

        shutil.rmtree(plant_output)

def create_generators_4_classes(split_path: str, img_size=(150, 150), batch_size=32):
    """Creates data generators for the 4-class experiment (no brightness/channel shift)."""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(split_path, "train"), target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(split_path, "val"), target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(split_path, "test"), target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False
    )

    return train_gen, val_gen, test_gen
