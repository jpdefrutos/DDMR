import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

from Brain_study.Train_UncertaintyWeighted import launch_train
import argparse

TRAIN_DATASET = '/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1/training'

err = list()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Location of the training data', default=TRAIN_DATASET)
    parser.add_argument('--validation', type=str, help='Location of the validation data', default=None)
    parser.add_argument('--similarity', nargs='+', type=str, help='Similarity loss function: mse, ncc, ssim', default=[])
    parser.add_argument('--segmentation', nargs='+', type=str, help='Segmentation loss function: hd, dice', default=[])
    parser.add_argument('--output', type=str, help='Output directory', default=TRAIN_DATASET)
    parser.add_argument('--gpu', type=str, help='GPU number', default='0')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--rw', type=float, help='Regularization weigh', default=5e-3)

    args = parser.parse_args()

    output_folder = os.path.join(args.output, 'UW_Lsim_{}__Lseg_{}__MET_mse_ncc_ssim'.format('__'.join(args.similarity),
                                                                                             '__'.join(args.segmentation)))
    print('TRAIN ' + args.dataset)
    launch_train(dataset_folder=args.dataset,
                 validation_folder=args.validation,
                 output_folder=output_folder,
                 gpu_num=args.gpu,
                 prior_reg_w=args.rw,
                 lr=args.lr,
                 simil=args.similarity,
                 segm=args.segmentation)


