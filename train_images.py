# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 18:11:11 2026

@author: lab
"""

import os
import shutil
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np

image_path = r"D:\Hackathon_1\train\Timmowal_Village.tif"
vector_path = r"D:\Hackathon_1\train\Timmowal_Built_up.geojson"
attribute_field = "Roof_type"
tile_size = 512
output_image_path = r"D:\Hackathon_1\train_images"
output_mask_path = r"D:\Hackathon_1\train_masks"
os.makedirs(output_image_path, exist_ok=True)
os.makedirs(output_mask_path, exist_ok=True)

# READ VECTOR
gdf = gpd.read_file(vector_path)

# READ VECTOR
gdf = gpd.read_file(vector_path)

# ----------------------------
# OPEN IMAGE
# ----------------------------
with rasterio.open(image_path) as src:

    # Reproject vector to match raster CRS
    gdf = gdf.to_crs(src.crs)

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
            profile.update(
                {
                    "height": tile_size,
                    "width": tile_size,
                    "transform": transform,
                    "compress": "lzw",
                    "tiled": True,
                    "blockxsize": tile_size,  # Explicitly set block size
                    "blockysize": tile_size,  # Explicitly set block size
                }
            )

            image_tile_path = os.path.join(
                output_image_path, f"image_tile_{tile_id}.tif"
            )

            with rasterio.open(image_tile_path, "w", **profile) as dst:
                dst.write(img_tile)

            # ----------------------------
            # CREATE MASK TILE
            # ----------------------------

            # Get tile bounds
            bounds = rasterio.windows.bounds(window, src.transform)

            # Clip vector to tile extent (faster processing)
            tile_gdf = gdf.cx[bounds[0] : bounds[2], bounds[1] : bounds[3]]

            shapes = [
                (geom, value)
                for geom, value in zip(tile_gdf.geometry, tile_gdf[attribute_field])
            ]

            mask = rasterize(
                shapes=shapes,
                out_shape=(tile_size, tile_size),
                transform=transform,
                fill=0,
                dtype="uint8",
            )

            mask_profile = profile.copy()
            mask_profile.update({"count": 1, "dtype": "uint8", "compress": "lzw", "tiled": True, "blockxsize": tile_size, "blockysize": tile_size})

            mask_tile_path = os.path.join(
                output_mask_path, f"mask_tile_{tile_id}.tif"
            )

            with rasterio.open(mask_tile_path, "w", **mask_profile) as dst:
                dst.write(mask, 1)

            tile_id += 1

print(" All tiles generated successfully.")