import os
import h5py
import numpy as np
from tqdm import tqdm
import ddmr.utils.constants as C


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
LITS_NONE = '/mnt/EncryptedData1/Users/javier/vessel_registration/LiTS/None'
LITS_TRANS = '/mnt/EncryptedData1/Users/javier/vessel_registration/LiTS/Translation'
LITS_AFFINE = '/mnt/EncryptedData1/Users/javier/vessel_registration/LiTS/Affine'


IMG_SHAPE = (64, 64, 64, 1)
for dataset in [LITS_NONE, LITS_AFFINE, LITS_TRANS]:
    dataset_files = [os.path.join(dataset, d) for d in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, d))]
    f_iter = tqdm(dataset_files)
    f_iter.set_description('Analyzing ' + dataset)
    inv_shape_count = 0
    inv_type_count = 0
    for i, d in enumerate(f_iter):
        f = h5py.File(d, 'r')
        if f[C.H5_FIX_IMG][:].shape != IMG_SHAPE:
            print(d + ' Invalid FIX IMG. Shape: ' + str(f[C.H5_FIX_IMG][:].shape))
            inv_shape_count += 1
        if f[C.H5_MOV_IMG][:].shape != IMG_SHAPE:
            print(d + ' Invalid MOV IMG. Shape: ' + str(f[C.H5_MOV_IMG][:].shape))
            inv_shape_count += 1
        if f[C.H5_FIX_PARENCHYMA_MASK][:].shape != IMG_SHAPE:
            print(d + ' Invalid FIX PARENCHYMA. Shape: ' + str(f[C.H5_FIX_PARENCHYMA_MASK][:].shape))
            inv_shape_count += 1
        if f[C.H5_MOV_PARENCHYMA_MASK][:].shape != IMG_SHAPE:
            print(d + ' Invalid MOV PARENCHYMA. Shape: ' + str(f[C.H5_MOV_PARENCHYMA_MASK][:].shape))
            inv_shape_count += 1
        if f[C.H5_FIX_TUMORS_MASK][:].shape != IMG_SHAPE:
            print(d + ' Invalid FIX TUMORS. Shape: ' + str(f[C.H5_FIX_TUMORS_MASK][:].shape))
            inv_shape_count += 1
        if f[C.H5_MOV_TUMORS_MASK][:].shape != IMG_SHAPE:
            print(d + ' Invalid MOV TUMORS. Shape: ' + str(f[C.H5_MOV_TUMORS_MASK][:].shape))
            inv_shape_count += 1

        if f[C.H5_FIX_IMG][:].dtype != np.float32:
            print(d + ' Invalid FIX IMG. Type: ' + str(f[C.H5_FIX_IMG][:].dtype))
            inv_type_count += 1
        if f[C.H5_MOV_IMG][:].dtype != np.float32:
            print(d + ' Invalid MOV IMG. Type: ' + str(f[C.H5_MOV_IMG][:].dtype))
            inv_type_count += 1
        if f[C.H5_FIX_PARENCHYMA_MASK][:].dtype != np.float32:
            print(d + ' Invalid FIX PARENCHYMA. Type: ' + str(f[C.H5_FIX_PARENCHYMA_MASK][:].dtype))
            inv_type_count += 1
        if f[C.H5_MOV_PARENCHYMA_MASK][:].dtype != np.float32:
            print(d + ' Invalid MOV PARENCHYMA. Type: ' + str(f[C.H5_MOV_PARENCHYMA_MASK][:].dtype))
            inv_type_count += 1
        if f[C.H5_FIX_TUMORS_MASK][:].dtype != np.float32:
            print(d + ' Invalid FIX TUMORS. Type: ' + str(f[C.H5_FIX_TUMORS_MASK][:].dtype))
            inv_type_count += 1
        if f[C.H5_MOV_TUMORS_MASK][:].dtype != np.float32:
            print(d + ' Invalid MOV TUMORS. Type: ' + str(f[C.H5_MOV_TUMORS_MASK][:].dtype))
            inv_type_count += 1

    print('\n\n>>>>SUMMARY ' + dataset)
    print('\t\tInvalid shape: ' + str(inv_shape_count) + '\n\t\tInvalid type: ' + str(inv_type_count))
