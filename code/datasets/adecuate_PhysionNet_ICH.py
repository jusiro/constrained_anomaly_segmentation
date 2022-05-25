import nibabel as nib
import os
import numpy as np
import random
import argparse
import cv2
from scipy import ndimage
from skimage import measure
from matplotlib import pyplot as plt
import SimpleITK as sitk


def volume_registration(fixed_image, moving_image, mask=None):

    fixed_image = sitk.GetImageFromArray(fixed_image)
    moving_image = sitk.GetImageFromArray(moving_image)
    # Initial transformation
    '''
    transform_to_displacment_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacment_field_filter.SetReferenceImage(fixed_image)
    initial_transform = sitk.DisplacementFieldTransform(
        transform_to_displacment_field_filter.Execute(sitk.Transform(2, sitk.sitkIdentity)))
    '''
    initial_transform = sitk.CenteredTransformInitializer(fixed_image,
                                                          moving_image,
                                                          sitk.Euler3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.GEOMETRY)

    registration_method = sitk.ImageRegistrationMethod()
    # Similarity metric settings.
    #registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricAsCorrelation()
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Don't optimize in-place, we would possibly like to run this cell multiple times.
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    # Apply transformation
    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32),
                                                  sitk.Cast(moving_image, sitk.sitkFloat32))

    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())
    out = sitk.GetArrayFromImage(moving_resampled)

    if mask is None:
        return out
    else:  # Apply transformation to gt mask
        mask_mov = sitk.GetImageFromArray(mask)
        moving_resampled = sitk.Resample(mask_mov, fixed_image, final_transform, sitk.sitkNearestNeighbor,
                                         0, mask_mov.GetPixelID())
        out_mask = sitk.GetArrayFromImage(moving_resampled)

        return out, out_mask


def preprocess_vol_ICH(vol, mask):
    w_level = 40
    w_width = 120

    # Intensity normalization

    vol = (vol - (w_level - (w_width / 2))) * (255 / (w_width))
    vol[vol < 0] = 0
    vol[vol > 255] = 255

    # Get tissue mask

    tissue_mask = np.ones(vol.shape)
    tissue_mask[vol == 0] = 0
    tissue_mask[vol == 255] = 0
    tissue_mask = ndimage.binary_opening(tissue_mask, structure=np.ones((10, 10, 1))).astype(tissue_mask.dtype)
    tissue_mask = ndimage.binary_erosion(tissue_mask, structure=np.ones((5, 5, 1))).astype(tissue_mask.dtype)

    # Keep larger objetc in CT
    for iSlice in np.arange(0, vol.shape[-1]):
        tissue_mask_i = tissue_mask[:, :, iSlice]

        if np.max(tissue_mask_i) > 0:

            labels = measure.label(tissue_mask_i)
            props = measure.regionprops(labels)

            areas = [i_prop.area for i_prop in props]
            labels_given = [i_prop.label for i_prop in props]
            idx_areas = np.argsort(areas)

            if np.mean(tissue_mask_i[labels == (labels_given[idx_areas[-1]])]) != 0:
                label = labels_given[idx_areas[-1]]
            else:
                label = labels_given[idx_areas[-2]]

            tissue_mask_i = labels == (label)
            tissue_mask[:, :, iSlice] = tissue_mask_i

    vol = vol * tissue_mask
    mask = mask * tissue_mask

    return vol, mask

np.random.seed(42)
random.seed(42)

dir_dataset = '../data/PhysioNet-ICH/'
dir_out = '../data/PhysioNetICH_5slices_registered_new/'
scan = 'CT'
nSlices = 5
partitions = ['train', 'test']
Ncases = np.array([50, 25])

if not os.path.isdir(dir_out):
    os.mkdir(dir_out)
if not os.path.isdir(dir_out + '/' + scan + '/'):
    os.mkdir(dir_out + '/' + scan + '/')

cases = os.listdir(dir_dataset + 'ct_scans/')
cases = [dir_dataset + 'ct_scans/' + iCase for iCase in cases if iCase != '.DS_Store']

random.shuffle(cases)

for iPartition in np.arange(0, len(partitions)):

    if not os.path.isdir(dir_out + '/' + scan + '/' + partitions[iPartition]):
        os.mkdir(dir_out + '/' + scan + '/' + partitions[iPartition])
    if not os.path.isdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/benign'):
        os.mkdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/benign')
    if not os.path.isdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/malign'):
        os.mkdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/malign')
    if not os.path.isdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/ground_truth'):
        os.mkdir(dir_out + '/' + scan + '/' + partitions[iPartition] + '/ground_truth')

    cases_partition = cases[np.sum(Ncases[:iPartition]):np.sum(Ncases[:iPartition+1])]

# Load volume reference
vol_ref = nib.load('../data/PhysioNet-ICH/ct_scans/071.nii')
vol_ref = (vol_ref.get_fdata())
mask_ref = nib.load('../data/PhysioNet-ICH/masks/071.nii')
mask_ref = (mask_ref.get_fdata())
mask_ref[mask_ref > 0] = 255
vol_ref, mask_ref = preprocess_vol_ICH(vol_ref, mask_ref)

c = 0
for iCase in cases:
    c += 1

    img_path = iCase
    mask_path = iCase.replace('ct_scans', 'masks')

    # Load volume and mask
    img = nib.load(img_path)
    img = (img.get_fdata())[:, :, :]
    mask = nib.load(mask_path)
    mask = (mask.get_fdata())
    mask[mask > 0] = 255

    print(str(c) + '/' + str(len(cases)) + ' || ' + 'slices: ' + str(img.shape[-1]))

    # Preprocess
    img, mask = preprocess_vol_ICH(img, mask)

    img = img[:, :, round(img.shape[-1] / 2) - nSlices+5:round(img.shape[-1] / 2) + nSlices+5]
    mask = mask[:, :, round(mask.shape[-1] / 2) - nSlices+5:round(mask.shape[-1] / 2) + nSlices+5]

    if np.max(mask) == 255:
        part = 'test'
    else:
        part = 'train'

    mask = mask.astype(np.uint8)

    for iSlice in np.arange(0, nSlices*2):
        filename = iCase.split('/')[-1].split('.')[0] + '_' + str(iSlice) + '.jpg'

        i_image = img[:, :, iSlice]
        i_mask = mask[:, :, iSlice]

        if np.any(i_mask == 255):
            label = 'malign'
            cv2.imwrite(dir_out + '/' + scan + '/' + part + '/ground_truth/' + filename, i_mask)
        else:
            label = 'benign'

        cv2.imwrite(dir_out + '/' + scan + '/' + part + '/' + label + '/' + filename, i_image)
