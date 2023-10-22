import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import h5py
import numpy as np
import zipfile
import re
import dicom2nifti as d2n
import nibabel as nib
from ddmr.utils.nifti_utils import save_nifti
from ddmr.utils.misc import try_mkdir
from tqdm import tqdm
import shutil


IRCA_PATH = '/mnt/EncryptedData1/Users/javier/vessel_registration/3Dirca/'

SEGMENTATIONS = ('venoussystem', 'venacava', 'portalvein', 'liver', 'livertumor')

SEGS_VESSELS = ('venoussystem', 'venacava', 'portalvein')
SEGS_PARENCH = ('liver',)
SEGS_TUMOR = ('livertumor', 'tumor', 'liverkyst', 'livercyst')

SEGMENTATIONS = (SEGS_PARENCH + SEGS_VESSELS + SEGS_TUMOR)

DEST_FOLDER = os.path.join(IRCA_PATH, 'nifti3')

ZIP_EXT = '.zip'
NIFTI_EXT = '.nii.gz'
H5_EXT = '.hd5'

PATIENT_DICOM = 'PATIENT_DICOM'
TEMP = 'temp'
SEGS_DICOM = 'MASKS_DICOM'

CONVERTED_FILE = 'none' + NIFTI_EXT
VOL_FILE = 'volume-{:04d}' + NIFTI_EXT
SEG_FILE = 'segmentation-{:04d}' + NIFTI_EXT

SEG_TO_CT_ORIENTATION_MAT = np.eye(4)
SEG_TO_CT_ORIENTATION_MAT[0] = -1


def merge_segmentations(file_list):
    nib_file = nib.concat_images(file_list)
    np_file = np.asarray(nib_file.dataobj)
    np_file = np.sign(np.sum(np_file, -1)) * np.max(np_file)
    return nib.Nifti1Image(np_file, nib_file.affine)


if __name__ == '__main__':
    # 1. List of folders
    folder_list = [os.path.join(IRCA_PATH, d) for d in os.listdir(IRCA_PATH) if d.lower().startswith('3dircadb1.')]
    folder_list.sort()

    try_mkdir(DEST_FOLDER)
    folder_iter = tqdm(folder_list)
    for pat_dir in folder_iter:
        pat_dir = folder_list[13]
        i = int(pat_dir.split('.')[-1])
        # 2. Unzip PATIENT_DICOM.zip
        temp_folder = os.path.join(pat_dir, TEMP)
        folder_iter.set_description('Volume DICOM: Unzipping PATIENT_DICOM.zip')
        zipfile.ZipFile(os.path.join(pat_dir, PATIENT_DICOM + ZIP_EXT)).extractall(temp_folder)

        folder_iter.set_description('Volume DICOM: Converting DICOM to Nifti')
        d2n.convert_directory(os.path.join(temp_folder, PATIENT_DICOM), os.path.join(temp_folder, PATIENT_DICOM))
        os.rename(os.path.join(temp_folder, PATIENT_DICOM, CONVERTED_FILE), os.path.join(DEST_FOLDER, VOL_FILE.format(i)))

        folder_iter.set_description('Volume DICOM: CT stored in: ' + os.path.join(DEST_FOLDER, VOL_FILE.format(i)))
        # os.rename also moves the file to the destination path. So the original one ceases to exist

        # 3. Unzip MASKS_DICOM.zip
        folder_iter.set_description('Segmentations DICOM: Unzipping MASKS_DICOM.zip')
        zipfile.ZipFile(os.path.join(pat_dir, SEGS_DICOM + ZIP_EXT)).extractall(temp_folder)
        seg_nib = list()
        seg_ves = list()
        seg_par = list()
        seg_tumor = list()
        seg_dirs = list()
        for root, dir_list, file_list in os.walk(os.path.join(temp_folder, SEGS_DICOM)):
            for fold in dir_list:
                if fold.startswith(SEGMENTATIONS):
                    # if 'liverkyst' in fold:
                    #     continue
                    # else:
                    seg_dirs.append(fold)

        seg_dirs.sort()
        for fold in seg_dirs:
            folder_iter.set_description('Segmentations DICOM: Converting ' + fold)
            d2n.convert_directory(os.path.join(temp_folder, SEGS_DICOM, fold),
                                  os.path.join(temp_folder, SEGS_DICOM))
            os.rename(os.path.join(temp_folder, SEGS_DICOM, CONVERTED_FILE),
                      os.path.join(temp_folder, SEGS_DICOM, fold + '_nifti_' + NIFTI_EXT))
            if fold.startswith(SEGS_VESSELS):
                seg_ves.append(os.path.join(temp_folder, SEGS_DICOM, fold + '_nifti_' + NIFTI_EXT))
            elif fold.startswith(SEGS_TUMOR):
                seg_tumor.append(os.path.join(temp_folder, SEGS_DICOM, fold + '_nifti_' + NIFTI_EXT))
            elif fold.startswith(SEGS_PARENCH):
                seg_par.append(os.path.join(temp_folder, SEGS_DICOM, fold + '_nifti_' + NIFTI_EXT))
            else:
                continue

        folder_iter.set_description('Segmentations DICOM: Concatenating segmentations')
        # Merge the vessel segmentations

        segs_to_merge = tuple()
        if len(seg_par) > 1:
            segs_to_merge += (merge_segmentations(seg_par),)
        else:
            segs_to_merge += tuple(seg_par)  # seg_par is a list

        if len(seg_ves) > 1:
            segs_to_merge += (merge_segmentations(seg_ves),)
        else:
            segs_to_merge += tuple(seg_ves)  # seg_ves is a list

        if len(seg_tumor) > 1:
            segs_to_merge += (merge_segmentations(seg_tumor),)
        else:
            segs_to_merge += tuple(seg_tumor)  # seg_tumor is a list

        # # Merge the tumors segmentations
        # if len(seg_tumor):
        #     segs_to_merge.append(merge_segmentations(seg_tumor))
        # else:
        #     print('No tumors found in ' + pat_dir)


        # Merge with the parenchyma and save
        folder_iter.set_description('Segmentations DICOM: Saving segmentations')
        if len(segs_to_merge) > 1:
            nib.save(nib.concat_images(segs_to_merge, check_affines=True), os.path.join(DEST_FOLDER, SEG_FILE.format(i)))
        else:
            nib.save(segs_to_merge[0], os.path.join(DEST_FOLDER, SEG_FILE.format(i)))

        folder_iter.set_description('Segmentations DICOM: Segmentation stored in ' + os.path.join(DEST_FOLDER, SEG_FILE.format(i)))

        shutil.rmtree(temp_folder)
        folder_iter.set_description('Temporal file deleted')
        # 4. Load DICOM and transform to nifti
        # 5. Store as nifty in hd5
