# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:40:14 2026

@author: lab
"""

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
        'img_dir': r"D:\Hackathon_1\test_images",
    },
    ]

img_processed = []
mask_processed = []

# 2. Outer Loop: Iterate through each pair of folders
for pair in dataset_pairs:
    current_img_dir = pair['img_dir']
    
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
        with rasterio.open(img_path) as src_img:
            win = Window(0, 0, 512, 512)
            # Read Image (Bands 1, 2, 3)
            img_data = src_img.read([1, 2, 3], window=win)
            img_data = np.transpose(img_data, (1, 2, 0))


            # CPU OPTIMIZATION: Store as uint8 to save RAM
            # Since your U-Net has a normalization layer, keep data as 0-255
            img_processed.append(img_data.astype(np.uint8))
            
# 5. Convert to final NumPy arrays
X_test = np.array(img_processed)

print("-" * 30)
print(f"✅ Success! Total test images loaded: {len(X_test)}")
print(f"X_test shape: {X_test.shape}")

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

# 6. Load the model (weights are already inside!)
# We still need custom_objects because the model architecture 
# refers to 'total_loss' even if we aren't using it for training.
custom_objects = {
    'total_loss': total_loss,
    'multi_dice_loss': multi_dice_loss
}

model = tf.keras.models.load_model(
    'model_for_hack.keras', 
    custom_objects=custom_objects, 
    safe_mode=False
)

# 7. Run Predictions and Plot
for i in range(len(X_test)):
    # Prepare image: normalize and add batch dimension
    img_to_predict = X_test[i].astype('float32')
    img_batch = np.expand_dims(img_to_predict, axis=0)

    # Predict
    preds = model.predict(img_batch)
    predicted_mask = np.argmax(preds[0], axis=-1)

    # Plotting (Only 2 subplots since we don't have "True Masks" for test images)
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Test Image {i+1} Prediction", fontsize=14)

    # Original Image
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(X_test[i])
    plt.axis('off')

    # Predicted Mask
    plt.subplot(1, 2, 2)
    plt.title("Model Prediction")
    plt.imshow(predicted_mask, cmap='viridis', vmin=0, vmax=NUM_CLASSES-1)
    plt.colorbar(ticks=range(NUM_CLASSES), label="Class ID")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    # OPTIONAL: Save the mask as a .tif for your hackathon submission
    # from PIL import Image
    # mask_output = Image.fromarray(predicted_mask.astype(np.uint8))
    # mask_output.save(f"prediction_tile_{i}.tif")
