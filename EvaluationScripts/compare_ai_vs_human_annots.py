import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import nibabel as nib
import re
import medpy.metric as medpy_metrics

import pandas as pd

import numpy as np

from tqdm import tqdm

import multiprocessing as mp
IMG_DIRECTORY = '/mnt/EncryptedData1/Laparoscopy/OSLO_COMET_dataset/OSLO_COMET_CT/Volumes_nii/test_set'
VES_SEG_DIRECTORY = '/mnt/EncryptedData1/Laparoscopy/OSLO_COMET_dataset/OSLO_COMET_CT/Vessels'
PAR_SEG_DIRECTORY = '/mnt/EncryptedData1/Laparoscopy/OSLO_COMET_dataset/OSLO_COMET_CT/Parenchyma'

LM_SEG_DIRECTORY = '/mnt/EncryptedData1/Laparoscopy/OSLO_COMET_dataset/OSLO_COMET_CT/LiverMask/test_set'

nii_FILENAME_PATTERN = '(.*)_CT.nii'
LV_VESSEL_FILENAME_PATTERN = '(.*)-vessels.nii'
LV_PARENCHYMA_FILENAME_PATTERN = '(.*)-livermask.nii'

OUT_DIRECTORY = '/mnt/EncryptedData1/Users/javier/ext_datasets/COMET_ai_vs_human'


def process_group(file_list):
    # print('Got: ' + ','.join(file_list))
    img_file, par_file, ves_file, lm_par_file, lm_ves_file = file_list
    img_num = int(re.match(nii_FILENAME_PATTERN, os.path.split(img_file)[-1])[1])

    human_par = nib.load(par_file)
    header = human_par.header
    human_par = np.asarray(human_par.dataobj)
    human_ves = np.asarray(nib.load(ves_file).dataobj)

    lm_par = np.asarray(nib.load(lm_par_file).dataobj)
    lm_ves = np.asarray(nib.load(lm_ves_file).dataobj)
    lm_ves[lm_ves > 0] = 1

    dsc_par = medpy_metrics.dc(human_par, lm_par)
    dsc_ves = medpy_metrics.dc(human_ves, lm_ves)

    hd_par = medpy_metrics.hd(human_par, lm_par, voxelspacing=header['pixdim'][1:4])
    hd_ves = medpy_metrics.hd(human_ves, lm_ves, voxelspacing=header['pixdim'][1:4])

    hd95_par = medpy_metrics.hd95(human_par, lm_par, voxelspacing=header['pixdim'][1:4])
    hd95_ves = medpy_metrics.hd95(human_ves, lm_ves, voxelspacing=header['pixdim'][1:4])

    return img_num, dsc_par, dsc_ves, hd_par, hd_ves, hd95_par, hd95_ves


if __name__ == '__main__':
    img_list = [os.path.join(IMG_DIRECTORY, f) for f in os.listdir(IMG_DIRECTORY) if f.endswith('.nii')]
    img_list.sort()

    ves_seg_list = [os.path.join(VES_SEG_DIRECTORY, os.path.split(f)[-1]+'.gz') for f in img_list]
    ves_seg_list.sort()

    par_seg_list = [os.path.join(PAR_SEG_DIRECTORY, os.path.split(f)[-1]+'.gz') for f in img_list]
    par_seg_list.sort()

    lm_ves_seg_list = [os.path.join(LM_SEG_DIRECTORY, f) for f in os.listdir(LM_SEG_DIRECTORY) if f.endswith('-vessels.nii')]
    lm_ves_seg_list.sort()

    lm_par_seg_list = [os.path.join(LM_SEG_DIRECTORY, f) for f in os.listdir(LM_SEG_DIRECTORY) if f.endswith('-livermask.nii')]
    lm_par_seg_list.sort()

    os.makedirs(OUT_DIRECTORY, exist_ok=True)

    zip_lists = zip(img_list, par_seg_list, ves_seg_list, lm_par_seg_list, lm_ves_seg_list)
    mets = ['DSC', 'DSC', 'HD', 'HD', 'H95', 'H95']
    segs = ['Parenchyma', 'Vessels', 'Parenchyma', 'Vessels', 'Parenchyma', 'Vessels']
    cols = list(zip(*[mets, segs]))
    idx = pd.MultiIndex.from_tuples(cols, names=['Metrics', 'Labels'])
    df = pd.DataFrame(index=idx)

    print('\nLaunching processes...')
    with mp.Pool(11, maxtasksperchild=1) as p:
        results = p.map_async(process_group, zip_lists)
        for v in results.get():
            df[v[0]] = v[1:]

    # for i, (img_file, par_file, ves_file, lm_par_file, lm_ves_file) in tqdm(enumerate(zip_lists), total=len(img_list)):
    #     img_name = re.match(nii_FILENAME_PATTERN, os.path.split(img_file)[-1])[1]
    #
    #     img = nib.load(img_file)
    #     human_par = nib.load(par_file)
    #     header = human_par.header
    #     human_par = np.asarray(human_par.dataobj)
    #     human_ves = np.asarray(nib.load(ves_file).dataobj)
    #
    #     lm_par = np.asarray(nib.load(lm_par_file).dataobj)
    #     lm_ves = np.asarray(nib.load(lm_ves_file).dataobj)
    #     lm_ves[lm_ves > 0] = 1
    #
    #     dsc_par = medpy_metrics.dc(human_par, lm_par)
    #     dsc_ves = medpy_metrics.dc(human_ves, lm_ves)
    #
    #     hd_par = medpy_metrics.hd(human_par, lm_par, voxelspacing=header['pixdim'][1:4])
    #     hd_ves = medpy_metrics.hd(human_ves, lm_ves, voxelspacing=header['pixdim'][1:4])
    #
    #     hd95_par = medpy_metrics.hd95(human_par, lm_par, voxelspacing=header['pixdim'][1:4])
    #     hd95_ves = medpy_metrics.hd95(human_ves, lm_ves, voxelspacing=header['pixdim'][1:4])
    #
    #     df[i] = [dsc_par, dsc_ves, hd_par, hd_ves, hd95_par, hd95_ves]
    print('\nResults...')
    print(df)

    print('\nSummary...')
    print(df.describe())

    df.to_csv('comparison.csv', sep=';')
