import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import argparse
from datetime import datetime

TRAIN_DATASET = '/mnt/EncryptedData1/Users/javier/ext_datasets/COMET_dataset/OSLO_COMET_CT/Formatted_128x128x128/train'

err = list()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to the model file', required=True)
    parser.add_argument('--dataset', type=str, help='Location of the training data', default=TRAIN_DATASET)
    parser.add_argument('--validation', type=str, help='Location of the validation data', default=None)
    parser.add_argument('--similarity', nargs='+', help='Similarity metric: mse, ncc, ssim', default=['ncc'])
    parser.add_argument('--segmentation', nargs='+', help='Segmentation loss function: hd, dice', default=['dice'])
    parser.add_argument('--output', type=str, help='Output directory', default=TRAIN_DATASET)
    parser.add_argument('--gpu', type=str, help='GPU number', default='0')
    parser.add_argument('--lr', type=float, help='Learning rate', default=1e-4)
    parser.add_argument('--rw', type=float, help='Regularization weigh', default=5e-3)
    parser.add_argument('--epochs', type=int, help='Max number of epochs', default=1000)
    parser.add_argument('--name', type=str, default='COMET')
    parser.add_argument('--uw', type=bool, default=False)
    parser.add_argument('--freeze', nargs='+', help='What layers to freeze: INPUT, OUTPUT, ENCODER, DECODER, TOP, BOTTOM', default=None)
    parser.add_argument('--epochs', default=1000)
    parser.add_argument('--batch', default=16)
    args = parser.parse_args()

    print('TRAIN ' + args.dataset)

    if args.uw:
        from COMET.COMET_train_UW import launch_train
        simil = args.similarity
        segm = args.segmentation
        output_folder = os.path.join(args.output, '{}_Lsim_{}__Lseg_{}'.format(args.name, '_'.join(simil), '_'.join(segm)))
    else:
        from COMET.COMET_train import launch_train
        simil = args.similarity[0]
        segm = args.segmentation[0]
        output_folder = os.path.join(args.output, '{}_Lsim_{}__Lseg_{}'.format(args.name, simil, segm))
    output_folder = output_folder + '_' + datetime.now().strftime("%H%M%S-%d%m%Y")

    if args.freeze is not None:
        assert all(s in ['INPUT', 'OUTPUT', 'ENCODER', 'DECODER', 'TOP', 'BOTTOM'] for s in args.freeze),\
            'Invalid option for "freeze". Expected one or several of: INPUT, OUTPUT, ENCODER, DECODER, TOP, BOTTOM'
        args.freeze = list(set(args.freeze))

    launch_train(dataset_folder=args.dataset,
                 validation_folder=args.validation,
                 output_folder=output_folder,
                 gpu_num=args.gpu,
                 lr=args.lr,
                 rw=args.rw,
                 simil=simil,
                 segm=segm,
                 max_epochs=args.epochs,
                 model_file=args.model,
                 freeze_layers=args.freeze)
