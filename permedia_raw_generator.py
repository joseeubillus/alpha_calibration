import os
import numpy as np
import cv2

# path to segmented image
path = 'C:/Users/ubillusj/Box/2022-2023 GRA/Permedia_Ubillus/out_phase_folder/op_new.tiff'

# read image
img = cv2.imread(path,-1)

# Reshape to a 300 x 300 array
img_reshape = cv2.resize(img, (300, 300))

# Repeat 10 time img for 3D array
vol_3d = np.repeat(img_reshape[:, :, np.newaxis], 10, axis=2)

# Generate raw file from 1D and 2D array

# Flip both arrays upside down and rotate 90 degrees
model_1d = np.flipud(np.rot90(img_reshape,1))
model_3d = np.flipud(np.rot90(vol_3d,1))

# Shape of both arrays
dim1, dim2 = model_1d.shape
dim3, dim4, dim5 = model_3d.shape

# Flatten 2D and 3D arrays
flattened_1d = model_1d.flatten('F')
flattened_3d = model_3d.flatten('F')

# Create txt file for 1D array
filename = 'C:/Users/ubillusj/Box/2022-2023 GRA/Permedia_Ubillus/out_phase_folder/model_1D.raw'
with open(filename, 'w') as f:
    f.write('# .raw version 0.700\n')
    f.write('extents: ' + str(dim1) + ' ' + str(1) + ' ' + str(dim2) + '\n')
    f.write('origin: 0 0 0\n')
    f.write('cellsize: 0.002 0.02 0.002\n')
    f.write('rotation: 0\n')
    f.write('data: ascii\n')
    for item in flattened_1d:
        f.write("%s\n" % item)

# Create txt file for 3D array
filename = 'C:/Users/ubillusj/Box/2022-2023 GRA/Permedia_Ubillus/out_phase_folder/model_3D.raw'
with open(filename, 'w') as f:
    f.write('# .raw version 0.700\n')
    f.write('extents: ' + str(dim3) + ' ' + str(dim5) + ' ' + str(dim4) + '\n')
    f.write('origin: 0 0 0\n')
    f.write('cellsize: 0.002 0.002 0.002\n')
    f.write('rotation: 0\n')
    f.write('data: ascii\n')
    for item in flattened_3d:
        f.write("%s\n" % item)
