import nibabel as nib
import os
import numpy as np
import random
import argparse
import cv2

np.random.seed(42)
random.seed(42)


def adecuate_BRATS(args):

    dir_dataset = args.dir_dataset
    dir_out = args.dir_out
    scan = args.scan
    nSlices = args.nSlices

    partitions = ['train', 'val', 'test']
    Ncases = np.array([271, 32, 32])

    if not os.path.isdir(dir_out):
        os.mkdir(dir_out)
    if not os.path.isdir(dir_out + '/' + scan + '/'):
        os.mkdir(dir_out + '/' + scan + '/')

    cases_LGG = os.listdir(dir_dataset + 'LGG/')
    cases_LGG = [dir_dataset + 'LGG/' + iCase for iCase in cases_LGG if iCase != '.DS_Store']

    cases_HGG = os.listdir(dir_dataset + 'HGG/')
    cases_HGG = [dir_dataset + 'HGG/' + iCase for iCase in cases_HGG if iCase != '.DS_Store']

    cases = cases_LGG + cases_HGG

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

        c = 0
        for iCase in cases_partition:
            c += 1
            print(str(c) + '/' + str(len(cases_partition)))

            img_path = iCase + '/' + iCase.split('/')[-1] + '_' + scan + '.nii.gz'
            mask_path = iCase + '/' + iCase.split('/')[-1] + '_seg.nii.gz'

            img = nib.load(img_path)
            img = (img.get_fdata())[:, :, :]
            img = (img/img.max())*255
            img = img.astype(np.uint8)

            mask = nib.load(mask_path)
            mask = (mask.get_fdata())
            mask[mask > 0] = 255
            mask = mask.astype(np.uint8)

            for iSlice in np.arange(round(img.shape[-1]/2) - nSlices, round(img.shape[-1]/2) + nSlices):
                filename = iCase.split('/')[-1] + '_' + str(iSlice) + '.jpg'

                i_image = img[:, :, iSlice]
                i_mask = mask[:, :, iSlice]

                if np.any(i_mask == 255):
                    label = 'malign'
                    cv2.imwrite(dir_out + '/' + scan + '/' + partitions[iPartition] + '/ground_truth/' + filename, i_mask)
                else:
                    label = 'benign'

                cv2.imwrite(dir_out + '/' + scan + '/' + partitions[iPartition] + '/' + label + '/' + filename, i_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_dataset", default='../data/MICCAI_BraTS_2019_Data_Training/', type=str)
    parser.add_argument("--dir_out", default='../data/BRATS_10slices/', type=str)
    parser.add_argument("--scan", default='flair', type=str)
    parser.add_argument("--nSlices", default=5, type=int)

    args = parser.parse_args()
    adecuate_BRATS(args)


