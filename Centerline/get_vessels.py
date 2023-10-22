from ddmr.utils.nifti_utils import save_nifti
from tqdm import tqdm
import os
import h5py
import ddmr.utils.constants as C

DATASET_LOCATION = '/mnt/EncryptedData1/Users/javier/vessel_registration/3Dirca/dataset/EVAL'
DATASET_NAMES = ['Affine', 'None', 'Translation']
DATASET_FILENAME = 'volume'


if __name__ == '__main__':
    for dataset_name in DATASET_NAMES:
        dataset_loc = os.path.join(DATASET_LOCATION, dataset_name)
        dataset_files = os.listdir(dataset_loc)
        dataset_files.sort()
        dataset_files = [os.path.join(dataset_loc, f) for f in dataset_files if DATASET_FILENAME in f]

        iterator = tqdm(dataset_files)
        for fn in iterator:
            f = os.path.split(fn)[-1].split('.hd5')[0]
            vol_file = h5py.File(fn, 'r')
            fix_vessels = vol_file[C.H5_FIX_VESSELS_MASK][..., 0]
            mov_vessels = vol_file[C.H5_MOV_VESSELS_MASK][..., 0]

            dst_folder = os.path.join(os.getcwd(), 'VESSELS', dataset_name)
            os.makedirs(dst_folder, exist_ok=True)
            save_nifti(fix_vessels, os.path.join(dst_folder, f+'_fix.nii.gz'))
            save_nifti(mov_vessels, os.path.join(dst_folder, f+'_mov.nii.gz'))
            vol_file.close()
