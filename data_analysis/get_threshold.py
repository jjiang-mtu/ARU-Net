"""
Selection of a suitable truncation threshold
"""

import os

from tqdm import tqdm
import SimpleITK as sitk

import sys
sys.path.append(os.path.split(sys.path[0])[0])

import parameter as para

num_point = 0.0
num_inlier = 0.0

for file in tqdm(os.listdir(para.train_ct_path)):

    ct = sitk.ReadImage(os.path.join(para.train_ct_path, file), sitk.sitkUInt16)
    ct_array = sitk.GetArrayFromImage(ct)

    seg = sitk.ReadImage(os.path.join(para.train_seg_path, file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)

    IA_roi = ct_array[seg_array > 0]
    inliers = ((IA_roi < para.upper) * (IA_roi > para.lower)).astype(int).sum()

    print('{:.4}%'.format(inliers / IA_roi.shape[0] * 100))
    print('------------')

    num_point += IA_roi.shape[0]
    num_inlier += inliers

print(num_inlier / num_point)

# Maximum and minimum thresholds (6.5e4, 1.5e4) for IA CTA images
# Training set: 0.9999318043754671
# Testing set: 0.999951487261941



