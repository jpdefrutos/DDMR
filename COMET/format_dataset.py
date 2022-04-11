import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import h5py
import nibabel as nib
from nilearn.image import resample_img
import re
import numpy as np
from scipy.ndimage import zoom
from skimage.measure import regionprops
from tqdm import tqdm
from argparse import ArgumentParser
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure

import pandas as pd

from DeepDeformationMapRegistration.utils import constants as C
from DeepDeformationMapRegistration.utils.misc import segmentation_cardinal_to_ohe, segmentation_ohe_to_cardinal

SEGMENTATION_NR2LBL_LUT = {0: 'background',
                           1: 'parenchyma',
                           2: 'vessel'}
SEGMENTATION_LBL2NR_LUT = {v: k for k, v in SEGMENTATION_NR2LBL_LUT.items()}

IMG_DIRECTORY = '/mnt/EncryptedData1/Users/javier/ext_datasets/COMET_dataset/OSLO_COMET_CT/Volumes'
SEG_DIRECTORY = '/mnt/EncryptedData1/Users/javier/ext_datasets/COMET_dataset/OSLO_COMET_CT/Segmentations' # '/home/jpdefrutos/workspace/LiverSegmentation_UNet3D/data/prediction'

IMG_NAME_PATTERN = '(.*).nii.gz'
SEG_NAME_PATTERN = '(.*).nii.gz'

OUT_DIRECTORY = '/mnt/EncryptedData1/Users/javier/ext_datasets/COMET_dataset/OSLO_COMET_CT/Formatted_128x128x128'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--crop', action='store_true')  # If present, args.crop = True, else args.crop = False
    parser.add_argument('--offset', type=int, default=C.MAX_AUG_DISP_ISOT + 10, help='Crop offset in mm')
    parser.add_argument('--dilate-segmentations', type=bool, default=False)
    args = parser.parse_args()

    img_list = [os.path.join(IMG_DIRECTORY, f) for f in os.listdir(IMG_DIRECTORY) if f.endswith('.nii.gz')]
    img_list.sort()

    seg_list = [os.path.join(SEG_DIRECTORY, f) for f in os.listdir(SEG_DIRECTORY) if f.endswith('.nii.gz')]
    seg_list.sort()

    zoom_file = pd.DataFrame(columns=['scale_i', 'scale_j', 'scale_k'])
    os.makedirs(OUT_DIRECTORY, exist_ok=True)
    binary_ball = generate_binary_structure(3, 1)
    for seg_file in tqdm(seg_list):
        img_name = re.match(SEG_NAME_PATTERN, os.path.split(seg_file)[-1])[1]
        img_file = os.path.join(IMG_DIRECTORY, img_name + '.nii.gz')

        img = resample_img(nib.load(img_file), np.eye(3))
        seg = resample_img(nib.load(seg_file), np.eye(3), interpolation='nearest')

        img = np.asarray(img.dataobj)
        seg = np.asarray(seg.dataobj)

        segs_are_ohe = bool(len(seg.shape) > 3 and seg.shape[3] > 1)
        if args.crop:
            parenchyma = regionprops(seg[..., 0])[0]
            bbox = np.asarray(parenchyma.bbox) + [*[-args.offset]*3, *[args.offset]*3]
            # check that the new bbox is within the image limits!
            bbox[:3] = np.maximum(bbox[:3], [0, 0, 0])
            bbox[3:] = np.minimum(bbox[3:], img.shape)
            img = img[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
            seg = seg[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5], ...]
        # Resize to 128x128x128
        isot_shape = img.shape

        zoom_factors = (np.asarray([128]*3) / np.asarray(img.shape)).tolist()

        img = zoom(img, zoom_factors, order=3)
        if args.dilate_segmentations:
            seg = binary_dilation(seg, binary_ball, iterations=1)
        seg = zoom(seg, zoom_factors + [1]*(len(seg.shape) - len(img.shape)), order=0)
        zoom_file = zoom_file.append({'scale_i': zoom_factors[0],
                                      'scale_j': zoom_factors[1],
                                      'scale_k': zoom_factors[2]}, ignore_index=True)

        # seg -> cardinal
        # seg_expanded -> OHE
        if segs_are_ohe:
            seg_expanded = seg.copy()
            seg = segmentation_ohe_to_cardinal(seg)  # Ordinal encoded. argmax returns the first ocurrence of the maximum. Hence the previoous multiplication operation
        else:
            seg_expanded = segmentation_cardinal_to_ohe(seg)

        h5_file = h5py.File(os.path.join(OUT_DIRECTORY, img_name + '.h5'), 'w')

        h5_file.create_dataset('image', data=img[..., np.newaxis], dtype=np.float32)
        h5_file.create_dataset('segmentation', data=seg.astype(np.uint8), dtype=np.uint8)
        h5_file.create_dataset('segmentation_expanded', data=seg_expanded.astype(np.uint8), dtype=np.uint8)
        h5_file.create_dataset('segmentation_labels', data=np.unique(seg)[1:])  # Remove the 0 (background label)
        h5_file.create_dataset('isotropic_shape', data=isot_shape)

        print('{}: Segmentation labels {}'.format(img_name, np.unique(seg)[1:]))
        h5_file.close()

    zoom_file.to_csv(os.path.join(OUT_DIRECTORY, 'zoom_factors.csv'))
    print("Average")
    print(zoom_file.mean().to_list())

    print("Standard deviation")
    print(zoom_file.std().to_list())

    print("Average + STD")
    print((zoom_file.mean() + zoom_file.std()).to_list())



