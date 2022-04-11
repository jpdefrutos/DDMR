import os
import nibabel as nb
import numpy as np
import zipfile


TEMP_UNZIP_PATH = '/mnt/EncryptedData1/Users/javier/ext_datasets/LITS17/temp'
NII_EXTENSION = '.nii'


def save_nifti(data, save_path, header=None, verbose=True):
    if header is None:
        data_nifti = nb.Nifti1Image(data, affine=np.eye(4))
    else:
        data_nifti = nb.Nifti1Image(data, affine=None, header=header)

    data_nifti.header.get_xyzt_units()
    try:
        nb.save(data_nifti, save_path)  # Save as NiBabel file
        if verbose:
            print('Saved {}'.format(save_path))
    except ValueError:
        print('Could not save {}'.format(save_path))


def unzip_nii_file(file_path):
    file_dir, file_name = os.path.split(file_path)
    file_name = file_name.split('.zip')[0]

    dest_path = os.path.join(TEMP_UNZIP_PATH, file_name)
    zipfile.ZipFile(file_path).extractall(TEMP_UNZIP_PATH)

    if not os.path.exists(dest_path):
        print('ERR: File {} not unzip-ed!'.format(file_path))
        dest_path = None
    return dest_path


def delete_temp(file_path, verbose=False):
    assert NII_EXTENSION in file_path
    os.remove(file_path)
    if verbose:
        print('Deleted file: ', file_path)
