import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
import pickle

# Parameters
dataset_path =r'\Users\abina\OneDrive\Desktop\hand_symbol\Dataset'  # Update to your dataset path
img_size = 128
batch_size = 32
epochs = 10
model_save_path = 'mobilenetv2_gesture_model.keras'
label_encoder_save_path = 'label_encoder.pkl'

# Step 1: Create ImageDataGenerator for data augmentation and splitting
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Reserve 20% for validation
)

# Step 2: Create data generators for training and validation
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Use 80% for training
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Use 20% for validation
)

# Save class labels (for decoding predictions later)
class_indices = train_generator.class_indices
label_encoder = {v: k for k, v in class_indices.items()}  # Reverse mapping

with open(label_encoder_save_path, 'wb') as le_file:
    pickle.dump(label_encoder, le_file)

# Step 3: Build the MobileNetV2 model
base_model = MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base model layers

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(len(class_indices), activation='softmax')  # Output layer for classes
])

# Step 4: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model with generators
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs
)

# Step 6: Save the trained model
model.save(model_save_path)
