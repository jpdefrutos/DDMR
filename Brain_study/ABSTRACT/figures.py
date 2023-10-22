import os
import argparse
import re
import warnings

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, to_rgba, CSS4_COLORS
import tikzplotlib

from ddmr.utils.misc import segmentation_ohe_to_cardinal

# segm_cm = np.asarray([to_rgba(CSS4_COLORS[c], 1) for c in CSS4_COLORS.keys()])
# # segm_cm.sort()
# segm_cm = segm_cm[np.linspace(0, len(segm_cm), 4, endpoint=False).astype(int), ...]
segm_cm = cm.get_cmap('jet').reversed()
segm_cm = segm_cm(np.linspace(0, 1, 30))
segm_cm[0, :] = np.asarray([0, 0, 0, 0])
segm_cm = ListedColormap(segm_cm)

DICT_MODEL_NAMES = {'BASELINE': 'BL',
                    'SEGGUIDED': 'SG',
                    'UW': 'UW'}

DICT_METRICS_NAMES = {'NCC': 'N',
                      'SSIM': 'S',
                      'DICE': 'D',
                      'DICE_MACRO': 'D',
                      'HD': 'H', }


def get_model_name(in_path: str):
    model = re.search('((UW|SEGGUIDED|BASELINE).*)_\d+-\d+', in_path)
    if model:
        model = model.group(1).rstrip('_')
        model = model.replace('_Lsim', '')
        model = model.replace('_Lseg', '')
        model = model.replace('_L', '')
        model = model.replace('_', ' ')
        model = model.upper()
        elements = model.split()
        model = elements[0]
        metrics = list()
        model = DICT_MODEL_NAMES[model]
        for m in elements[1:]:
            if m != 'MACRO':
                metrics.append(DICT_METRICS_NAMES[m])

        return '{}-{}'.format(model, ''.join(metrics))
    else:
        try:
            model = re.search('(SyNCC|SyN)', in_path).group(1)
        except AttributeError:
            raise ValueError('Unknown folder name/model: '+ in_path)
        return model


def load_segmentation(file_path) -> np.ndarray:
    segm = np.asarray(nib.load(file_path).dataobj)
    if segm.shape[-1] > 1:
        segm = segmentation_ohe_to_cardinal(segm)
    return segm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dir', type=str, help='Directories where the models are stored', default=None)
    parser.add_argument('-o', '--output', type=str, help='Output directory', default=os.getcwd())
    parser.add_argument('--overwrite', type=bool, default=True)
    parser.add_argument('--fileno', type=int, default=2)
    parser.add_argument('--tikz', type=bool, default=False)
    args = parser.parse_args()
    assert args.dir is not None, "No directories provided. Stopping"
    os.makedirs(args.output, exist_ok=True)
    list_fix_img = list()
    list_mov_img = list()
    list_fix_seg = list()
    list_mov_seg = list()
    list_pred_img = list()
    list_pred_seg = list()
    print('Fetching data...')
    init_lvl = args.dir.count(os.sep)
    for r, d, f in os.walk(args.dir):
        current_lvl = r.count(os.sep) - init_lvl
        if current_lvl < 3:
            for name in f:
                if re.search('^{:03d}'.format(args.fileno), name) and name.endswith('nii.gz'):
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
    selected_slice = 64
    fix_img = np.asarray(nib.load(list_fix_img[0]).dataobj)[selected_slice, ..., 0].T
    mov_img = np.asarray(nib.load(list_mov_img[0]).dataobj)[selected_slice, ..., 0].T
    fix_seg = load_segmentation(list_fix_seg[0])[selected_slice, ..., 0].T
    mov_seg = load_segmentation(list_mov_seg[0])[selected_slice, ..., 0].T

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(9, 3), dpi=200)

    for i, (img, title) in enumerate(zip([(fix_img, fix_seg), (mov_img, mov_seg)],
                                         [('Fixed image', 'Fixed segms.'), ('Moving image', 'Moving segms.')])):

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
        if args.tikz:
            tikzplotlib.save(os.path.join(args.output, 'Test_data.tex'))
    plt.close()

    print('Making Pred_data.png...')
    fig, ax = plt.subplots(nrows=2, ncols=len(list_pred_img), figsize=(9, 3), dpi=200)

    for i, (pred_img_path, pred_seg_path) in enumerate(zip(list_pred_img, list_pred_seg)):
        img = np.asarray(nib.load(pred_img_path).dataobj)[selected_slice, ..., 0].T
        seg = load_segmentation(pred_seg_path)[selected_slice, ..., 0].T

        ax[0, i].imshow(img, origin='lower', cmap='Greys_r')
        ax[1, i].imshow(img, origin='lower', cmap='Greys_r')
        ax[1, i].imshow(seg, origin='lower', cmap=segm_cm, alpha=0.6)

        ax[0, i].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
        ax[1, i].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)

        model = get_model_name(pred_img_path)

        ax[1, i].set_xlabel(model, fontsize=9)
    plt.tight_layout()
    if not args.overwrite and os.path.exists(os.path.join(args.output, 'Pred_data.png')):
        warnings.warn('File Pred_data.png already exists. Skipping')
    else:
        plt.savefig(os.path.join(args.output, 'Pred_data.png'), format='png')
        if args.tikz:
            tikzplotlib.save(os.path.join(args.output, 'Pred_data.tex'))
    plt.close()

    print('Making Pred_data_large.png...')
    fig, ax = plt.subplots(nrows=2, ncols=len(list_pred_img) + 2, figsize=(9, 3), dpi=200)
    list_pred_img = [list_mov_img[0]] + list_pred_img
    list_pred_img = [list_fix_img[0]] + list_pred_img
    list_pred_seg = [list_mov_seg[0]] + list_pred_seg
    list_pred_seg = [list_fix_seg[0]] + list_pred_seg

    for i, (pred_img_path, pred_seg_path) in enumerate(zip(list_pred_img, list_pred_seg)):
        img = np.asarray(nib.load(pred_img_path).dataobj)[selected_slice, ..., 0].T
        seg = load_segmentation(pred_seg_path)[selected_slice, ..., 0].T

        ax[0, i].imshow(img, origin='lower', cmap='Greys_r')
        ax[1, i].imshow(img, origin='lower', cmap='Greys_r')
        ax[1, i].imshow(seg, origin='lower', cmap=segm_cm, alpha=0.6)

        ax[0, i].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
        ax[1, i].tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)

        if i > 1:
            model = get_model_name(pred_img_path)
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
        if args.tikz:
            tikzplotlib.save(os.path.join(args.output, 'Pred_data_large.png'))
    plt.close()

    print('...done!')
