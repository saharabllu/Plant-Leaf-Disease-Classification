"""Preprocessing for 27-Class Disease Classification."""

import os
import shutil
import splitfolders
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def split_dataset_27_classes(raw_path: str, split_path: str, seed: int = 42):
    """Splits dataset into train, val, test and flattens the 27 disease classes."""
    if os.path.exists(split_path):
        shutil.rmtree(split_path)
    os.makedirs(split_path, exist_ok=True)

    plants = ["Kentang", "Lemon", "Teh", "Tomat"]
    
    for plant in plants:
        plant_src = os.path.join(raw_path, plant)
        if not os.path.exists(plant_src):
            continue

        plant_tmp = os.path.join(split_path, "_tmp")
        if os.path.exists(plant_tmp): 
            shutil.rmtree(plant_tmp)

        # Split per plant folder (ratio 80% train, 10% val, 10% test)
        splitfolders.ratio(plant_src, output=plant_tmp, seed=seed, ratio=(0.8, 0.1, 0.1), move=False)

        for subset in ["train", "val", "test"]:
            src_subset = os.path.join(plant_tmp, subset)
            dst_subset = os.path.join(split_path, subset)
            os.makedirs(dst_subset, exist_ok=True)

            for disease in os.listdir(src_subset):
                new_class_name = f"{plant}_{disease}"
                s = os.path.join(src_subset, disease)
                d = os.path.join(dst_subset, new_class_name)
                if os.path.isdir(s):
                    shutil.copytree(s, d)

        shutil.rmtree(plant_tmp)


def create_generators_27_classes(split_path: str, img_size=(150, 150), batch_size=32):
    """Creates data generators for the 27-class experiment (with brightness & channel shift)."""
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2],
        channel_shift_range=0.1
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
