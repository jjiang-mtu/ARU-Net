"""
Optimization of Post-Processing for 3D Fully Connected Conditional Random Fields(CRF)
"""

import os
import sys
sys.path.append(os.path.split(sys.path[0])[0])

import collections

import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

import skimage.measure as measure
import skimage.morphology as morphology
from utilities.calculate_metrics import Metirc

import parameter as para

file_name = []  # File name

# Definition of evaluation indicators
IA_score = collections.OrderedDict()
IA_score['dice'] = []
IA_score['jacard'] = []
IA_score['voe'] = []
IA_score['fnr'] = []
IA_score['fpr'] = []
IA_score['assd'] = []
IA_score['rmsd'] = []
IA_score['msd'] = []

# Define two variables in order to calculate global dice
dice_intersection = 0.0  
dice_union = 0.0

for file_index, file in enumerate(os.listdir(os.path.join(para.test_set_path, 'ct'))):

    print('file index:', file_index, file)
    
    file_name.append(file)

    ct = sitk.ReadImage(os.path.join(os.path.join(para.test_set_path, 'ct'), file), sitk.sitkInt16)
    ct_array = sitk.GetArrayFromImage(ct)    
    ct_array = ct_array.astype(np.float32)    

    pred = sitk.ReadImage(os.path.join(para.pred_path, file.replace('volume', 'pred')), sitk.sitkUInt8)
    pred_array = sitk.GetArrayFromImage(pred)

    seg = sitk.ReadImage(os.path.join(os.path.join(para.test_set_path, 'seg'), file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 0] = 1

    new_ct_array = ct_array
    new_pred_array = pred_array

    # Defining Conditional Random Fields(CRF)
    print('Begin CRF post-processing')
    n_labels = 2
    d = dcrf.DenseCRF(np.prod(new_ct_array.shape), n_labels)

    # Obtaining the unary potential
    unary = np.zeros_like(new_pred_array, dtype=np.float32)
    unary[new_pred_array == 0] = 0.1
    unary[new_pred_array == 1] = 0.9

    U = np.stack((1 - unary, unary), axis=0)
    d.setUnaryEnergy(unary_from_softmax(U))

    # Obtaining the pairwise potential 
    # This creates the color-independent features and then add them to the CRF
    feats = create_pairwise_gaussian(sdims=(para.s1, para.s1, para.s1), shape=new_ct_array.shape)
    d.addPairwiseEnergy(feats, compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    feats = create_pairwise_bilateral(sdims=(para.s2, para.s2, para.s2), schan=(para.s3,), img=new_ct_array)
    d.addPairwiseEnergy(feats, compat=10, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Draw inferences
    Q, tmp1, tmp2 = d.startInference()
    for i in tqdm(range(para.max_iter)):
        # print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)

    # Getting predictive labeling results
    MAP = np.argmax(np.array(Q), axis=0).reshape(new_pred_array.shape)
    IA_seg = np.zeros_like(seg_array, dtype=np.uint8)
    IA_seg= MAP.astype(np.uint8)
    IA_seg = measure.label(IA_seg, connectivity=1)
    props = measure.regionprops(IA_seg)
        
    max_area = 0
    max_index = 0
    for index, prop in enumerate(props, start=1):
        if prop.area > max_area:
            max_area = prop.area
            max_index = index
    
    IA_seg[IA_seg != max_index] = 0
    IA_seg[IA_seg == max_index] = 1
    
    IA_seg = IA_seg.astype(np.bool_)
    morphology.remove_small_holes(IA_seg, para.maximum_hole, connectivity=2, in_place=True)

    IA_seg = IA_seg.astype(np.uint8)      
            
    # Calculation of segmentation evaluation indicators
    IA_metric = Metirc(seg_array, IA_seg, ct.GetSpacing())

    IA_score['dice'].append(IA_metric.get_dice_coefficient()[0])
    IA_score['jacard'].append(IA_metric.get_jaccard_index())
    IA_score['voe'].append(IA_metric.get_VOE())
    IA_score['fnr'].append(IA_metric.get_FNR())
    IA_score['fpr'].append(IA_metric.get_FPR())
    IA_score['assd'].append(IA_metric.get_ASSD())
    IA_score['rmsd'].append(IA_metric.get_RMSD())
    IA_score['msd'].append(IA_metric.get_MSD())

    dice_intersection += IA_metric.get_dice_coefficient()[1]
    dice_union += IA_metric.get_dice_coefficient()[2]

    # Saving CRF post-processing results as nii data
    pred_seg = sitk.GetImageFromArray(IA_seg)
    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(para.crf_path, file.replace('volume', 'crf')))

    print('dice:', IA_score['dice'][-1])
    print('--------------------------------------------------------------')


# Write evaluation indicators to excel file
IA_data = pd.DataFrame(IA_score, index=file_name)

IA_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(IA_data.columns))
IA_statistics.loc['mean'] = IA_data.mean()
IA_statistics.loc['std'] = IA_data.std()
IA_statistics.loc['min'] = IA_data.min()
IA_statistics.loc['max'] = IA_data.max()

writer = pd.ExcelWriter('./result-CRF.xlsx')
IA_data.to_excel(writer, 'IA')
IA_statistics.to_excel(writer, 'IA_statistics')
writer.save()

# Print global dice
print('dice global:', dice_intersection / dice_union)
