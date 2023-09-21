"""
Parameters
"""

## Path-related parameters

train_ct_path = './dataset/IAData/train/ct/'  # Path to the input image of the original training set

train_seg_path = './dataset/IAData/train/seg/'  # Path to the labeled image of the original training set

test_ct_path = './dataset/IAData/test/ct/'  # Path to the input image of the original test set

test_seg_path = './dataset/IAData/test/seg/'  # Path to the labeled image of the original test set

training_set_path = './train/'  # Path of the preprocessed training set

test_set_path = './test/'  # Path of the preprocessed test set

pred_path = './res-pred'  # Path to save network prediction results

crf_path = './res-crf'  # Path to save CRF optimization results

module_path = './model/Bestnet512-0.163-0.161.pth'  #  Load test model


## Parameters related to training data

size = 48  # Using 48 consecutive slices as input patch to the network

down_scale = 1  # Transect downsampling factor

expand_slice = 20  # Cropping each image patch containing 20 slices above and below the aneurysm as a training sample

slice_thickness = 1  # Normalize the spacing of the image data on the z-axis to 1mm

upper, lower = 6.5e4, 1.5e4  #  Maximum and minimum grayscale truncation thresholds for CTA images


## Network structure-related parameters

drop_rate = 0.3  # Probability of random dropout


## Network training-related parameters

gpu = '0'  # Serial number of the graphics card used

Epoch = 600 

learning_rate = 1e-4   

learning_rate_decay = [30, 60, 90, 120, 200, 400, 500]

alpha = 0.33  # Depth supervision attenuation factor

batch_size = 3

num_workers = 3

pin_memory = True

cudnn_benchmark = True


## Parameters related to model testing

threshold = 0.5  # Threshold 

stride = 12  # Slide sampling step

maximum_hole = 5e4  # Maximum hole size


## Parameters related to CRF post-processing optimization

z_expand, x_expand, y_expand = 10, 30, 30  # Number of extensions in three directions based on predictions

max_iter = 10  # Number of CRF iterations

s1, s2, s3 = 1, 10, 10  # CRF Gaussian kernel parameters


