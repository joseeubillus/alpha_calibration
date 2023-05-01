import os
import numpy as np
import cv2

# path to segmented image
path = 'C:/Users/josee/Box/2022-2023 GRA/Permedia_Ubillus/test_3d/test_ml_ij.tif'

# read image
img = cv2.imread(path, -1)

# Reshape to a 300 x 300 array
img = np.resize(img, (300, 300))

# Repeat 10 time img for 3D array
vol_3d = np.repeat(img[:, :, np.newaxis], 10, axis=2)

# Generate raw file from 1D and 2D array

# Flip both arrays upside down and rotate 90 degrees
model_1d = np.flipud(np.rot90(img,1))
model_3d = np.flipud(np.rot90(vol_3d,1))

# Flatten 2D and 3D arrays
flattened_1d = model_1d.flatten('F')
flattened_3d = model_3d.flatten('F')

# Create txt file for 1D array
filename = 'model.raw'
with open(filename, 'w') as f:
    f.write('# .raw version 0.700\n')
    f.write('origin: 0 0 0\n')
    f.write('cellsize: 0.002 0.02 0.002\n')
    f.write('rotation: 0\n')
    f.write('data: ascii\n')
    for item in flattened_1d:
        f.write("%s\n" % item)

# Create txt file for 3D array
# 1D array
