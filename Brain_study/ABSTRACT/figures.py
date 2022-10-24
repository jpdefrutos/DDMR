import os
import argparse
import re
import warnings

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

segm_cm = cm.get_cmap('Dark2', 256)
segm_cm = segm_cm(np.linspace(0, 1, 28))
segm_cm[0, :] = np.asarray([0, 0, 0, 0])
segm_cm = ListedColormap(segm_cm)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dir', type=str, help='Directories where the models are stored', default=None)
    parser.add_argument('-o', '--output', type=str, help='Output directory', default=os.getcwd())
    parser.add_argument('--overwrite', type=bool, default=True)
    args = parser.parse_args()
    assert args.dir is not None, "No directories provided. Stopping"

    list_fix_img = list()
    list_mov_img = list()
    list_fix_seg = list()
    list_mov_seg = list()
    list_pred_img = list()
    list_pred_seg = list()
    print('Fetching data...')
    for r, d, f in os.walk(args.dir):
        if os.path.split(r)[1] == 'Evaluation_paper':
            for name in f:
                if re.search('^050', name) and name.endswith('nii.gz'):
                    if re.search('fix_img', name) and name.endswith('nii.gz'):
                        list_fix_img.append(os.path.join(r, name))
                    elif re.search('mov_img', name):
                        list_mov_img.append(os.path.join(r, name))
                    elif re.search('fix_seg', name):
                        list_fix_seg.append(os.path.join(r, name))
                    elif re.search('mov_seg', name):
                        list_mov_seg.append(os.path.join(r, name))
                    elif re.search('pred_img', name):
                        list_pred_img.append(os.path.join(r, name))
                    elif re.search('pred_seg', name):
                        list_pred_seg.append(os.path.join(r, name))

    # Figure: all coronal views
    # Fix img | Mov img
    # BASELINE 1 | BASELINE 2 | SEGGUIDED
    # UW 1 | UW 2 | UW 3
    list_fix_img.sort()
    list_fix_seg.sort()
    list_mov_img.sort()
    list_mov_seg.sort()
    list_pred_img.sort()
    list_pred_seg.sort()
    print('Making Test_data.png...')
    selected_slice = 30
    fix_img = np.asarray(nib.load(list_fix_img[0]).dataobj)[..., selected_slice, 0]
    mov_img = np.asarray(nib.load(list_mov_img[0]).dataobj)[..., selected_slice, 0]
    fix_seg = np.asarray(nib.load(list_fix_seg[0]).dataobj)[..., selected_slice, 0]
    mov_seg = np.asarray(nib.load(list_mov_seg[0]).dataobj)[..., selected_slice, 0]

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(9, 3), dpi=200)

    for i, (img, title) in enumerate(zip([(fix_img, fix_seg), (mov_img, mov_seg)],
                                         [('Fixed image', 'Fixed Segms.'), ('Moving image', 'Moving Segms.')])):

        ax[i].imshow(img[0], origin='lower', cmap='Greys_r')
        ax[i+2].imshow(img[0], origin='lower', cmap='Greys_r')
        ax[i+2].imshow(img[1], origin='lower', cmap=segm_cm, alpha=0.6)

        ax[i].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
        ax[i+2].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)

        ax[i].set_xlabel(title[0], fontsize=16)
        ax[i+2].set_xlabel(title[1], fontsize=16)

    plt.tight_layout()
    if not args.overwrite and os.path.exists(os.path.join(args.output, 'Test_data.png')):
        warnings.warn('File Test_data.png already exists. Skipping')
    else:
        plt.savefig(os.path.join(args.output, 'Test_data.png'), format='png')
    plt.close()

    print('Making Pred_data.png...')
    fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(9, 3), dpi=200)

    for i, (pred_img_path, pred_seg_path) in enumerate(zip(list_pred_img, list_pred_seg)):
        img = np.asarray(nib.load(pred_img_path).dataobj)[..., selected_slice, 0]
        seg = np.asarray(nib.load(pred_seg_path).dataobj)[..., selected_slice, 0]

        ax[0, i].imshow(img, origin='lower', cmap='Greys_r')
        ax[1, i].imshow(img, origin='lower', cmap='Greys_r')
        ax[1, i].imshow(seg, origin='lower', cmap=segm_cm, alpha=0.6)

        ax[0, i].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
        ax[1, i].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)

        model = re.search('((UW|SEGGUIDED|BASELINE).*)_{2,}MET', pred_img_path).group(1).rstrip('_')
        model = model.replace('_Lsim', ' ')
        model = model.replace('_Lseg', ' ')
        model = model.replace('_L', ' ')
        model = model.replace('_', ' ')
        model = model.upper()
        model = ' '.join(model.split())

        ax[1, i].set_xlabel(model, fontsize=9)
    plt.tight_layout()
    if not args.overwrite and os.path.exists(os.path.join(args.output, 'Pred_data.png')):
        warnings.warn('File Pred_data.png already exists. Skipping')
    else:
        plt.savefig(os.path.join(args.output, 'Pred_data.png'), format='png')
    plt.close()

    print('Making Pred_data_large.png...')
    fig, ax = plt.subplots(nrows=2, ncols=8, figsize=(9, 3), dpi=200)
    list_pred_img = [list_mov_img[0]] + list_pred_img
    list_pred_img = [list_fix_img[0]] + list_pred_img
    list_pred_seg = [list_mov_seg[0]] + list_pred_seg
    list_pred_seg = [list_fix_seg[0]] + list_pred_seg

    for i, (pred_img_path, pred_seg_path) in enumerate(zip(list_pred_img, list_pred_seg)):
        img = np.asarray(nib.load(pred_img_path).dataobj)[..., selected_slice, 0]
        seg = np.asarray(nib.load(pred_seg_path).dataobj)[..., selected_slice, 0]

        ax[0, i].imshow(img, origin='lower', cmap='Greys_r')
        ax[1, i].imshow(img, origin='lower', cmap='Greys_r')
        ax[1, i].imshow(seg, origin='lower', cmap=segm_cm, alpha=0.6)

        ax[0, i].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
        ax[1, i].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)

        if i > 1:
            model = re.search('((UW|SEGGUIDED|BASELINE).*)_{2,}MET', pred_img_path).group(1).rstrip('_')
            model = model.replace('_Lsim', ' ')
            model = model.replace('_Lseg', ' ')
            model = model.replace('_L', ' ')
            model = model.replace('_', ' ')
            model = model.upper()
            model = ' '.join(model.split())
        elif i == 0:
            model = 'Moving image'
        else:
            model = 'Fixed image'

        ax[1, i].set_xlabel(model, fontsize=7)
    plt.tight_layout()
    if not args.overwrite and os.path.exists(os.path.join(args.output, 'Pred_data_large.png')):
        warnings.warn('File Pred_data.png already exists. Skipping')
    else:
        plt.savefig(os.path.join(args.output, 'Pred_data_large.png'), format='png')
    plt.close()

    print('...done!')
