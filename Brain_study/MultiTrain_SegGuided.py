import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

from Brain_study.Train_SegmentationGuided import launch_train
import argparse

TRAIN_DATASET = '/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1/training'

err = list()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Location of the training data', default=TRAIN_DATASET)
    parser.add_argument('--validation', type=str, help='Location of the validation data', default=None)
    parser.add_argument('--similarity', type=str, help='Similarity loss function: mse, ncc, ssim')
    parser.add_argument('--segmentation', type=str, help='Segmentation loss function: hd, dice')
    parser.add_argument('--output', type=str, help='Output directory', default=TRAIN_DATASET)
    parser.add_argument('--gpu', type=str, help='GPU number', default='0')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--rw', type=float, help='Regularization weigh', default=2e-2)

    args = parser.parse_args()

    print('TRAIN ' + args.dataset)
    launch_train(dataset_folder=args.dataset,
                 validation_folder=args.validation,
                 output_folder=os.path.join(args.output, 'SEGGUIDED_Lsim_{}__Lseg_{}__MET_mse_ncc_ssim'.format(args.similarity, args.segmentation)),
                 gpu_num=args.gpu,
                 lr=args.lr,
                 rw=args.rw,
                 simil=args.similarity,
                 segm=args.segmentation)
