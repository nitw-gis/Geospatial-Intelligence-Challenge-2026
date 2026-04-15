# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 20:42:51 2026

@author: lab
"""

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
from tf.keras.optimizers import Adam
opt=Adam(learning_rate = 0.0001)

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

model.save("roof_unet_model.keras")
