# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:15:44 2026

@author: lab
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.windows import Window
from matplotlib import pyplot as plt
import tensorflow as tf
import datetime

#z=0

# 1. Map your specific folder pairs
# This ensures Image Folder 1 only looks for masks in Mask Folder 1
NUM_CLASSES = 5
dataset_pairs = [
    {
        'img_dir': r"D:\Hackathon_1\train_images",
        'mask_dir': r"D:\Hackathon_1\train_masks"
    },
    {
        'img_dir': r"D:\Hackathon_1\29_03\filtered_image_tiles",
        'mask_dir': r"D:\Hackathon_1\29_03\filtered_mask_tiles"
    }
]

img_processed = []
mask_processed = []

# 2. Outer Loop: Iterate through each pair of folders
for pair in dataset_pairs:
    current_img_dir = pair['img_dir']
    current_mask_dir = pair['mask_dir']
    
    print(f"--- Processing folder: {current_img_dir} ---")
    
    # Get all .tif files in the current image directory
    image_files = sorted(glob.glob(os.path.join(current_img_dir, "*.tif")))

    # 3. Inner Loop: Process each image in the current folder
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        
        # Robust ID extraction: image_tile_11.tif -> 11
        # We split by 'tile_' and take the part before '.tif'
        try:
            file_id = img_name.split('tile_')[-1].split('.')[0]
        except IndexError:
            print(f"Skipping {img_name}: Filename format doesn't match 'tile_ID.tif'")
            continue

        # Build the expected mask path using the CURRENT pair's mask directory
        expected_mask_name = f"mask_tile_{file_id}.tif"
        mask_path = os.path.join(current_mask_dir, expected_mask_name)

        # 4. Check if the specific mask exists in the corresponding folder
        if os.path.exists(mask_path):
            with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
                win = Window(0, 0, 512, 512)

                # Read Image (Bands 1, 2, 3)
                img_data = src_img.read([1, 2, 3], window=win)
                img_data = np.transpose(img_data, (1, 2, 0))

                # Read Mask
                mask_data = src_mask.read(1, window=win)

                # Filter: Check if mask contains any roof classes (1, 2, 3, or 4)
                if np.any(np.isin(mask_data, [1, 2, 3, 4])):
                    # Clean up invalid values
                    mask_data = np.nan_to_num(mask_data, nan=0)
                    mask_data[mask_data == -9999] = 0
                    
                    # CPU OPTIMIZATION: Store as uint8 to save RAM
                    # Since your U-Net has a normalization layer, keep data as 0-255
                    img_processed.append(img_data.astype(np.uint8))
                    mask_processed.append(np.expand_dims(mask_data, axis=-1).astype(np.uint8))
                else:
                    # Optional: print skipped images if you need to debug
                    #z+=1
                    #print(f"Skipping {img_name}: No roof classes found.")
                    pass
        else:
            print(f"⚠️ Warning: No mask found for {img_name} in {current_mask_dir}")

# 5. Convert to final NumPy arrays
X_train = np.array(img_processed)
Y_train = np.array(mask_processed)

print("-" * 30)
print(f"✅ Success! Total matching pairs loaded: {len(X_train)}")
print(f"X_train shape: {X_train.shape} (Data type: {X_train.dtype})")
print(f"Y_train shape: {Y_train.shape} (Data type: {Y_train.dtype})")
#print(z)

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

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_hack.keras', verbose = 1, save_best_only=True)

callbacks=[
    tf.keras.callbacks.TensorBoard(log_dir='logs'),
    tf.keras.callbacks.EarlyStopping(patience=4,monitor='val_loss'),
    checkpointer,
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',patience=4,min_lr=1e-6,verbose=1)]

import tensorflow as tf
from tensorflow.keras.models import load_model

# 1. Map your custom loss functions
custom_objects = {
    'total_loss': total_loss,
    'multi_dice_loss': multi_dice_loss
}

# 2. Add safe_mode=False to allow the Lambda layer to load
model = load_model(
    'roof_unet_model.keras', 
    custom_objects=custom_objects, 
    safe_mode=False  # This fixes the error
)

print("✅ Model loaded successfully including the Lambda layer!")

results = model.fit(X_train, Y_train, validation_split=0.3, batch_size=32, epochs=50, callbacks=callbacks)

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

# print(model.optimizer.learning_rate.numpy())
model.save('final_train.keras')
