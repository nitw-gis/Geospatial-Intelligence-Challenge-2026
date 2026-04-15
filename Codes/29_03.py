# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:15:44 2026

@author: lab
"""

# Image processing libraries
import rasterio
from rasterio.plot import show
from matplotlib import pyplot as plt

import os
import numpy as np
from rasterio.windows import Window

img_processed = []
mask_processed = []
image_dir=r"D:\Hackathon_1\29_03\filtered_image_tiles"
mask_dir=r"D:\Hackathon_1\29_03\filtered_mask_tiles"

# Get the list of images
image_files = sorted(os.listdir(image_dir))

for img_name in image_files:
    # 1. Extract the number (e.g., 'image11.tif' -> '11')
    # We remove 'image' and '.tif' to get just the ID
    file_id = img_name.replace('image_tile_', '').replace('.tif', '')

    # 2. Build the expected mask name
    expected_mask_name = f"mask_tile_{file_id}.tif"
    mask_path = os.path.join(mask_dir, expected_mask_name)
    img_path = os.path.join(image_dir, img_name)

    # 3. Check if that specific mask actually exists
    if os.path.exists(mask_path):
        with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
            win = Window(0, 0, 512, 512)

            # Read and Prepare Image
            img_data = src_img.read([1, 2, 3], window=win)
            img_data = np.transpose(img_data, (1, 2, 0))

            # Read and Prepare Mask
            mask_data = src_mask.read(1, window=win)

            # Check if mask contains any roof classes (1, 2, 3, or 4)
            unique_mask_values = np.unique(mask_data)
            if any(val in [1, 2, 3, 4] for val in unique_mask_values):
                # Final shapes
                mask_data = np.nan_to_num(mask_data, nan=0)
                mask_data[mask_data == -9999] = 0
                img_processed.append(img_data)
                mask_processed.append(np.expand_dims(mask_data, axis=-1).astype(np.uint8))
            else:
                print(f"Skipping {img_name}: Mask contains no roof classes (1-4).")
    else:
        print(f"⚠️ Warning: No mask found for {img_name} (Expected {expected_mask_name})")

# Convert to final arrays
X_train = np.array(img_processed)
Y_train = np.array(mask_processed)

print(f"✅ Success! Loaded {len(X_train)} matching pairs with roof types.")

# Please review the output from the cell above. If the `.tif` files are present, ensure their names and extensions are correct. If the directories appear empty or don't exist, please double-check the path to your shared Google Drive folder and the permissions.

import numpy as np

# Flatten the Y_train array to count unique pixel values
flattened_masks = Y_train.flatten()

# Get unique values and their counts
unique_values, counts = np.unique(flattened_masks, return_counts=True)

print("Class Distribution in Y_train:")
for val, count in zip(unique_values, counts):
    print(f"Class {val}: {count} pixels")

# Calculate percentages for better understanding
total_pixels = len(flattened_masks)
print("\nClass Distribution (Percentage):")
for val, count in zip(unique_values, counts):
    percentage = (count / total_pixels) * 100
    print(f"Class {val}: {percentage:.2f}%")


#Please execute the cell above and carefully examine the output to find the exact path to your 'Hackathon' folder. Shared folders often appear as shortcuts directly under `MyDrive` or sometimes have slightly different naming conventions. Once you identify the correct path (e.g., `/content/drive/MyDrive/Hackathon` or something similar), you will need to update the `image_directory` and `mask_directory` variables in cell `48ca10c3` with the correct path.

# Calculate class weights
# The 'counts' array is available from the previous execution of cell f559a4cc
# It contains the pixel counts for each class: [Class 0, Class 1, Class 2, Class 3, Class 4]

total_pixels = np.sum(counts)
num_classes = len(counts)

# Calculate inverse frequency weights
# A common approach is to use total_pixels / (num_classes * class_count)
class_weights = total_pixels / (num_classes * counts)

# Normalize weights so they sum to 1 or scale them as needed (optional, but good practice)
# For sparse_categorical_crossentropy, Keras usually handles the scaling internally if you pass them directly.
# However, it's good to ensure their relative proportions are correct.
# Let's just use the inverse frequency directly for now.

class_weights = class_weights / np.mean(class_weights) # Normalize to average weight of 1

print("Calculated Class Weights:", class_weights)


NUM_CLASSES = 5   # Example: background + 3 roof types
import tensorflow as tf
import random

import numpy as np

IMG_HEIGHT=512
IMG_WIDTH=512
IMG_CHANNELS=3

def multi_dice_loss(y_true, y_pred):
    # y_true shape: (Batch, 512, 512, 1)
    # y_pred shape: (Batch, 512, 512, 5)

    # 1. Convert integer labels to one-hot encoding
    # We squeeze the last dim (1) so it becomes (Batch, 512, 512) before one-hotting to (Batch, 512, 512, 5)
    y_true_oh = tf.one_hot(tf.cast(tf.squeeze(y_true, -1), tf.int32), depth=NUM_CLASSES)

    smooth = 1e-6

    # 2. Calculate intersection and union over spatial dimensions (Height and Width)
    # axis=[1, 2] means we sum up the pixels for each class separately
    intersection = tf.reduce_sum(y_true_oh * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true_oh, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])

    # 3. Compute Dice score per class, then average across classes and batch
    dice = (2. * intersection + smooth) / (union + smooth)

    # We subtract from 1 because we want to MINIMIZE this value
    return 1 - tf.reduce_mean(dice)

def total_loss(y_true, y_pred):
    # Apply class weights to sparse categorical crossentropy
    # Ensure y_true is int32 and y_pred is float32
    y_true_int = tf.cast(y_true, tf.int32)
    y_pred_float = tf.cast(y_pred, tf.float32)

    # Calculate sparse categorical crossentropy. This will have shape (batch_size, IMG_HEIGHT, IMG_WIDTH)
    scce_loss = tf.keras.losses.sparse_categorical_crossentropy(
        tf.squeeze(y_true_int, axis=-1), y_pred_float, from_logits=False # from_logits=False because activation is softmax
    )

    # Get a flattened version of y_true to map weights
    y_true_flat = tf.reshape(y_true_int, [-1])
    # Map each true class label to its corresponding weight
    sample_weights = tf.gather(class_weights, y_true_flat)
    # Reshape sample_weights to match scce_loss dimensions
    sample_weights_reshaped = tf.reshape(sample_weights, tf.shape(scce_loss))

    # Apply sample weights
    weighted_scce = tf.reduce_mean(scce_loss * tf.cast(sample_weights_reshaped, tf.float32))

    # Combined they help the model see "shapes" (Dice) and "pixels" (SCCE)
    return weighted_scce + multi_dice_loss(y_true, y_pred)
opt=tf.keras.optimizers.Adam()

#
inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

#Contraction path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.Dropout(0.1)(c1)
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.Dropout(0.1)(c2)
c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.Dropout(0.2)(c3)
c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.Dropout(0.2)(c4)
c4 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.Dropout(0.3)(c5)
c5 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

#Expansive path
u6 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
# same - adds padding in input feature map whereas 'valid' doesn't add padding but reduces computational efficiency (used when input is less than output feature map)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.Dropout(0.2)(c6)
c6 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

u7 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.Dropout(0.2)(c7)
c7 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

u8 = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.Dropout(0.1)(c8)
c8 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

u9 = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.Dropout(0.1)(c9)
c9 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

outputs = tf.keras.layers.Conv2D(NUM_CLASSES, (1, 1), activation='softmax', name='roof_output')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

# Explicitly build the model before compiling
model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

model.compile(optimizer=opt, loss={'roof_output':total_loss}, metrics={'roof_output':'sparse_categorical_accuracy'})
model.summary()

# Model check points
checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_hack.keras', verbose = 1, save_best_only=True)

callbacks=[
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    tf.keras.callbacks.EarlyStopping(patience=4,monitor='val_loss'),
    checkpointer,
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=8,min_lr=1e-6,verbose=1)]

# Stack the list of images and masks into single numpy arrays
x_train=[]
y_train=[]
x_train = np.stack(img_processed, axis=0)
y_train = np.stack(mask_processed, axis=0)

# Set validation_split to a value greater than 0 (e.g., 0.1 or 0.2)
results = model.fit(x_train, y_train, validation_split=0.3, batch_size=4, epochs=20, callbacks=callbacks)

#The output of the cell above will show the pixel count and percentage for each class (0, 1, 2, 3, 4) in your training masks. If class 0 is significantly higher, we'll need to consider techniques to address this imbalance, such as applying class weights to your loss function.

#Now that we have the class weights, we need to modify the `total_loss` function to apply them. This involves using `tf.nn.weighted_cross_entropy_with_logits` or manually applying weights to `sparse_categorical_crossentropy`.

for i in range(len(img_processed)):
    # Get raw predictions for the current image
    preds = model.predict(np.expand_dims(img_processed[i], axis=0))

    # Get the predicted class for each pixel (argmax across classes)
    predicted_mask_indices = np.argmax(preds, axis=-1) # Shape will be (1, 512, 512)

    # Squeeze to remove batch dimension for plotting
    predicted_mask = np.squeeze(predicted_mask_indices)
    true_mask = np.squeeze(mask_processed[i])

    plt.figure(figsize=(15, 10)) # Adjust figure size for a 2x2 grid
    plt.suptitle(f"Image {i+1} Performance Comparison", fontsize=16)

    # 1. Original Image
    plt.subplot(2, 2, 1)
    plt.title("Input Image")
    plt.imshow(img_processed[i])
    plt.axis('off')

    # 2. Ground Truth Mask
    plt.subplot(2, 2, 2)
    plt.title("True Mask")
    plt.imshow(true_mask, cmap='viridis', vmin=0, vmax=NUM_CLASSES-1)
    plt.colorbar(ticks=range(NUM_CLASSES), fraction=0.046, pad=0.04)
    plt.axis('off')
    print(f"Unique values in True Mask for Image {i+1}: {np.unique(true_mask, return_counts=True)}")

    # 3. Predicted Mask (Argmax)
    plt.subplot(2, 2, 3)
    plt.title('Predicted Mask (Argmax)')
    plt.imshow(predicted_mask, cmap='viridis', vmin=0, vmax=NUM_CLASSES-1)
    plt.colorbar(ticks=range(NUM_CLASSES), fraction=0.046, pad=0.04)
    plt.axis('off')
    print(f"Unique values in Predicted Mask for Image {i+1}: {np.unique(predicted_mask, return_counts=True)}")

    # 4. Error Map
    plt.subplot(2, 2, 4)
    plt.title("Error Map (Yellow=Wrong)")
    error_map = np.abs(true_mask - predicted_mask)
    plt.imshow(error_map, cmap='inferno')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle
    plt.show()
