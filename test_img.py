# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 17:50:46 2026

@author: lab
"""

import os
import shutil

# Image processing libraries
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np

image_path = r"D:\Hackathon_1\kartapur_proj.tif"
tile_size = 512
output_image_path = r"D:\Hackathon_1\test_images"
os.makedirs(output_image_path, exist_ok=True)

# ----------------------------
# OPEN IMAGE
# ----------------------------
with rasterio.open(image_path) as src:

    width = src.width
    height = src.height

    tile_id = 0

    # Loop over tiles
    for row in range(0, height, tile_size):
        for col in range(0, width, tile_size):

            window = Window(col, row, tile_size, tile_size)

            # Skip incomplete edge tiles (optional)
            if (row + tile_size > height) or (col + tile_size > width):
                continue

            transform = src.window_transform(window)

            # ----------------------------
            # READ IMAGE TILE
            # ----------------------------
            img_tile = src.read(window=window)

            profile = src.profile.copy()
            profile.update({
                "height": tile_size,
                "width": tile_size,
                "transform": transform,
                "compress": "lzw"
            })

            image_tile_path = os.path.join(
                output_image_path,
                f"image_tile_{tile_id}.tif"
            )

            with rasterio.open(image_tile_path, "w", **profile) as dst:
                dst.write(img_tile)

            # ----------------------------

            tile_id += 1

print(" All tiles generated successfully.")