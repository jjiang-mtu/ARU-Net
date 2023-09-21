import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])
import shutil
from time import time

import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
import scipy.ndimage as ndimage

import parameter as para

if os.path.exists(para.training_set_path):
    shutil.rmtree(para.training_set_path)

new_ct_path = os.path.join(para.training_set_path, 'ct')
new_seg_dir = os.path.join(para.training_set_path, 'seg')

os.mkdir(para.training_set_path)
os.mkdir(new_ct_path)
os.mkdir(new_seg_dir)

start = time()
for file in tqdm(os.listdir(para.train_ct_path)):

    # Reading CTA image and mask into memory
    ct = sitk.ReadImage(os.path.join(para.train_ct_path, file), sitk.sitkUInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(para.train_seg_path, file.replace('volume', 'segmentation')), sitk.sitkInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    # Fusion of labels from the mask
    seg_array[seg_array > 0] = 1

    # Truncate grayscale values outside of the threshold
    ct_array[ct_array > para.upper] = para.upper
    ct_array[ct_array < para.lower] = para.lower

    # Downsampling the CT data on the transects and resampling was performed to adjust the spacing of the z-axis to 1 mm for all data
    #ct_array = ndimage.zoom(ct_array, (ct.GetSpacing()[-1] / para.slice_thickness, para.down_scale, para.down_scale), order=3)
    #seg_array = ndimage.zoom(seg_array, (ct.GetSpacing()[-1] / para.slice_thickness, 1, 1), order=0)

    # Find the slice at the beginning and end of IA and expand outward respectively
    z = np.any(seg_array, axis=(1, 2))
    start_slice, end_slice = np.where(z)[0][[0, -1]]

    # Expand slice in each direction
    start_slice = max(0, start_slice - para.expand_slice)
    end_slice = min(seg_array.shape[0] - 1, end_slice + para.expand_slice)

    # If the number of slice after cropping is less than the size of the patch, discard the data
    if end_slice - start_slice + 1 < para.size:
        print('!!!!!!!!!!!!!!!!')
        print(file, 'Too few slices', ct_array.shape[0])
        print('!!!!!!!!!!!!!!!!')
        continue

    ct_array = ct_array[start_slice:end_slice + 1, :, :]
    seg_array = seg_array[start_slice:end_slice + 1, :, :]

    # Save preprocessing data as nii.gz file
    new_ct = sitk.GetImageFromArray(ct_array)

    new_ct.SetDirection(ct.GetDirection())
    new_ct.SetOrigin(ct.GetOrigin())
    new_ct.SetSpacing((ct.GetSpacing()[0] * int(1 / para.down_scale), ct.GetSpacing()[1] * int(1 / para.down_scale), ct.GetSpacing()[2] * int(1 / para.down_scale)))

    new_seg = sitk.GetImageFromArray(seg_array)

    new_seg.SetDirection(ct.GetDirection())
    new_seg.SetOrigin(ct.GetOrigin())
    new_seg.SetSpacing((ct.GetSpacing()[0]* int(1 / para.down_scale), ct.GetSpacing()[1]* int(1 / para.down_scale), ct.GetSpacing()[2] * int(1 / para.down_scale)))

    sitk.WriteImage(new_ct, os.path.join(new_ct_path, file))
    sitk.WriteImage(new_seg, os.path.join(new_seg_dir, file.replace('volume', 'segmentation'))) 
