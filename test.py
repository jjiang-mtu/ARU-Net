"""
Test Script
"""

import os
import copy
import collections
from time import time

import torch
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import SimpleITK as sitk
import skimage.measure as measure
import skimage.morphology as morphology

from net.ARUNet import ARUNet
from utilities.calculate_metrics import Metirc

import parameter as para

os.environ['CUDA_VISIBLE_DEVICES'] = para.gpu

# Define two variables to calculate global dice
dice_intersection = 0.0  
dice_union = 0.0

file_name = []  # File name
time_pre_case = []  # Running time

# Defining evaluation indicators
IA_score = collections.OrderedDict()
IA_score['dice'] = []
IA_score['jacard'] = []
IA_score['voe'] = []
IA_score['fnr'] = []
IA_score['fpr'] = []
IA_score['assd'] = []
IA_score['rmsd'] = []
IA_score['msd'] = []

# Define the network and load the parameters
net = torch.nn.DataParallel(ARUNet(training=False)).cuda()
net.load_state_dict(torch.load(para.module_path))
net.eval()

def Sobel_CT(image):
    
    ct_array = sitk.GetArrayFromImage(image)
    ct_array = ct_array.astype(np.float32)
    
    new_ct = sitk.GetImageFromArray(ct_array)
    new_ct.SetDirection(image.GetDirection())
    new_ct.SetOrigin(image.GetOrigin())
    new_ct.SetSpacing(image.GetSpacing())  
    normalizeFilter = sitk.NormalizeImageFilter()
    new_ct = normalizeFilter.Execute(new_ct)  # Set mean and std deviation
    
    image_float = sitk.Cast(new_ct, sitk.sitkFloat32)
    sobel_op = sitk.SobelEdgeDetectionImageFilter()
    sobel_sitk = sobel_op.Execute(image_float)
    sobel_sitk = sitk.Cast(sobel_sitk, sitk.sitkInt16)
    resacleFilter = sitk.RescaleIntensityImageFilter()
    resacleFilter.SetOutputMaximum(para.lower) #Set maximum value
    resacleFilter.SetOutputMinimum(para.lower/2) # Set minimum value
    sobel_sitk = resacleFilter.Execute(sobel_sitk)  
    
    return sobel_sitk
           
for file_index, file in enumerate(os.listdir(os.path.join(para.test_set_path, 'ct'))):

    start = time()

    file_name.append(file)    

    # Read CTA images into memory
    ct = sitk.ReadImage(os.path.join(os.path.join(para.test_set_path, 'ct'), file), sitk.sitkInt16)    
    ct_array = sitk.GetArrayFromImage(ct)
    ct_array = ct_array.astype(np.float32)  

    ct_Sobel = Sobel_CT(ct)            
    ct_Sobel = sitk.GetArrayFromImage(ct_Sobel)
    ct_Sobel = ct_Sobel.astype(np.float32)   
   
    ct_array = ct_array + ct_Sobel
       
    origin_shape = ct_array.shape
    
    # Use padding for CTA data with too little slice
    too_small = False    
    if ct_array.shape[0] < para.size:
        depth = ct_array.shape[0]
        temp = np.ones((para.size, int(256 * para.down_scale), int(256 * para.down_scale))) * para.lower
        temp[0: depth] = ct_array
        ct_array = temp 
        too_small = True
    
    # Sampling prediction based on sliding windows
    start_slice = 0
    end_slice = start_slice + para.size - 1
    count = np.zeros((ct_array.shape[0], 256, 256), dtype=np.int16)
    probability_map = np.zeros((ct_array.shape[0], 256, 256), dtype=np.float32)

    with torch.no_grad():
        while end_slice < ct_array.shape[0]:

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)

            outputs = net(ct_tensor)

            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            del outputs      
            
            start_slice += para.stride
            end_slice = start_slice + para.size - 1
    
        if end_slice != ct_array.shape[0] - 1:
            end_slice = ct_array.shape[0] - 1
            start_slice = end_slice - para.size + 1

            ct_tensor = torch.FloatTensor(ct_array[start_slice: end_slice + 1]).cuda()
            ct_tensor = ct_tensor.unsqueeze(dim=0).unsqueeze(dim=0)
            outputs = net(ct_tensor)

            count[start_slice: end_slice + 1] += 1
            probability_map[start_slice: end_slice + 1] += np.squeeze(outputs.cpu().detach().numpy())

            del outputs
            
        pred_seg = np.zeros_like(probability_map)
        #print(probability_map.shape)   
        pred_seg[probability_map >= (para.threshold * count)] = 1

        if too_small:
            temp = np.zeros((depth, 256, 256), dtype=np.float32)
            temp += pred_seg[0: depth]
            pred_seg = temp
    
    # Reading masks into memory
    seg = sitk.ReadImage(os.path.join(os.path.join(para.test_set_path, 'seg'), file.replace('volume', 'segmentation')), sitk.sitkUInt8)
    seg_array = sitk.GetArrayFromImage(seg)
    seg_array[seg_array > 0] = 1

    # 3D connected component optimization (3DCCO)
    # Maximum connectivity domain extraction for IA, removal of small regions, and filling of internal voids
    pred_seg = pred_seg.astype(np.uint8)
    IA_seg = copy.deepcopy(pred_seg)
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

    # Save the predicted results as nii data
    pred_seg = sitk.GetImageFromArray(IA_seg)

    pred_seg.SetDirection(ct.GetDirection())
    pred_seg.SetOrigin(ct.GetOrigin())
    pred_seg.SetSpacing(ct.GetSpacing())

    sitk.WriteImage(pred_seg, os.path.join(para.pred_path, file.replace('volume', 'pred')))

    speed = time() - start
    time_pre_case.append(speed)
    
    print(file, 'This case use {:.3f} s'.format(speed))  
    print('Dice: {:.4f}'.format(IA_metric.get_dice_coefficient()[0]))
    print('-----------------------')

# Write evaluation indicators to excel file
IA_data = pd.DataFrame(IA_score, index=file_name)
IA_data['time'] = time_pre_case

IA_statistics = pd.DataFrame(index=['mean', 'std', 'min', 'max'], columns=list(IA_data.columns))
IA_statistics.loc['mean'] = IA_data.mean()
IA_statistics.loc['std'] = IA_data.std()
IA_statistics.loc['min'] = IA_data.min()
IA_statistics.loc['max'] = IA_data.max()

writer = pd.ExcelWriter('./result-pred.xlsx')
IA_data.to_excel(writer, 'IA')
IA_statistics.to_excel(writer, 'IA_statistics')
writer.save()

# Print global dice
print('Global dice :', dice_intersection / dice_union)
