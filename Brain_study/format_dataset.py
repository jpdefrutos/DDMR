import h5py
import nibabel as nib
from nilearn.image import resample_img
import os
import re
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

SEGMENTATION_NR2LBL_LUT = {0: 'background',
                           2: 'parietal-right-gm',
                           3: 'lateral-ventricle-left',
                           4: 'occipital-right-gm',
                           6: 'parietal-left-gm',
                           8: 'occipital-left-gm',
                           9: 'lateral-ventricle-right',
                           11: 'globus-pallidus-right',
                           12: 'globus-pallidus-left',
                           14: 'putamen-left',
                           16: 'putamen-right',
                           20: 'brain-stem',
                           23: 'subthalamic-nucleus-right',
                           29: 'fornix-left',
                           33: 'subthalamic-nucleus-left',
                           39: 'caudate-left',
                           53: 'caudate-right',
                           67: 'cerebellum-left',
                           76: 'cerebellum-right',
                           102: 'thalamus-left',
                           203: 'thalamus-right',
                           210: 'frontal-left-gm',
                           211: 'frontal-right-gm',
                           218: 'temporal-left-gm',
                           219: 'temporal-right-gm',
                           232: '3rd-ventricle',
                           233: '4th-ventricle',
                           254: 'fornix-right',
                           255: 'csf'}
SEGMENTATION_LBL2NR_LUT = {v: k for k, v in SEGMENTATION_NR2LBL_LUT.items()}

IMG_DIRECTORY = '/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1'
SEG_DIRECTORY = '/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1/anatomical_masks'

IMG_NAME_PATTERN = '(.*).nii.gz'
SEG_NAME_PATTERN = '(.*)_lobes.nii.gz'

OUT_DIRECTORY = '/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1/test'

if __name__ == '__main__':
    img_list = [os.path.join(IMG_DIRECTORY, f) for f in os.listdir(IMG_DIRECTORY) if f.endswith('.nii.gz')]
    img_list.sort()

    seg_list = [os.path.join(SEG_DIRECTORY, f) for f in os.listdir(SEG_DIRECTORY) if f.endswith('.nii.gz')]
    seg_list.sort()

    os.makedirs(OUT_DIRECTORY, exist_ok=True)
    for seg_file in tqdm(seg_list):
        img_name = re.match(SEG_NAME_PATTERN, os.path.split(seg_file)[-1])[1]
        img_file = os.path.join(IMG_DIRECTORY, img_name + '.nii.gz')

        img = resample_img(nib.load(img_file), np.eye(3))
        seg = resample_img(nib.load(seg_file), np.eye(3), interpolation='nearest')

        isot_shape = img.shape

        # Resize to 128x128x128
        img = np.asarray(img.dataobj)
        img = zoom(img, np.asarray([128]*3) / np.asarray(isot_shape), order=3)

        seg = np.asarray(seg.dataobj)
        seg = zoom(seg, np.asarray([128]*3) / np.asarray(isot_shape), order=0)

        unique_lbls = np.unique(seg)[1:]     # Omit background
        seg_expanded = np.tile(np.zeros_like(seg)[..., np.newaxis], (1, 1, 1, len(unique_lbls)))
        for ch, lbl in enumerate(unique_lbls):
            seg_expanded[seg == lbl, ch] = 1

        h5_file = h5py.File(os.path.join(OUT_DIRECTORY, img_name + '.h5'), 'w')

        h5_file.create_dataset('image', data=img[..., np.newaxis], dtype=np.float32)
        h5_file.create_dataset('segmentation', data=seg[..., np.newaxis].astype(np.uint8), dtype=np.uint8)
        h5_file.create_dataset('segmentation_expanded', data=seg_expanded.astype(np.uint8), dtype=np.uint8)
        h5_file.create_dataset('segmentation_labels', data=unique_lbls)
        h5_file.create_dataset('isotropic_shape', data=isot_shape)

        h5_file.close()





