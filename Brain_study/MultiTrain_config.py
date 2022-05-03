import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import argparse
from configparser import ConfigParser
from datetime import datetime

import DeepDeformationMapRegistration.utils.constants as C

TRAIN_DATASET = '/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1/training'

err = list()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ini', help='Configuration file')
    args = parser.parse_args()

    configFile = ConfigParser()
    configFile.read(args.ini)
    print('Loaded configuration file: ' + args.ini)
    print({section: dict(configFile[section]) for section in configFile.sections()})
    print('\n\n')

    trainConfig = configFile['TRAIN']
    lossesConfig = configFile['LOSSES']
    datasetConfig = configFile['DATASETS']
    othersConfig = configFile['OTHERS']
    augmentationConfig = configFile['AUGMENTATION']

    simil = lossesConfig['similarity'].split(',')
    segm = lossesConfig['segmentation'].split(',')
    if trainConfig['name'].lower() == 'uw':
        from Brain_study.Train_UncertaintyWeighted import launch_train
        loss_config = {'simil': simil, 'segm': segm}
    elif trainConfig['name'].lower() == 'segguided':
        from Brain_study.Train_SegmentationGuided import launch_train
        loss_config = {'simil': simil[0], 'segm': segm[0]}
    else:
        from Brain_study.Train_Baseline import launch_train
        loss_config = {'simil': simil[0]}

    output_folder = os.path.join(othersConfig['outputFolder'],
                                 '{}_Lsim_{}__Lseg_{}'.format(trainConfig['name'], '_'.join(simil), '_'.join(segm)))
    output_folder = output_folder + '_' + datetime.now().strftime("%H%M%S-%d%m%Y")

    print('TRAIN ' + datasetConfig['train'])

    if augmentationConfig:
        C.GAMMA_AUGMENTATION = augmentationConfig['gamma'].lower() == 'true'
        C.BRIGHTNESS_AUGMENTATION = augmentationConfig['brightness'].lower() == 'true'

    try:
        unet = [int(x) for x in trainConfig['unet'].split(',')]
    except KeyError as e:
        unet = [16, 32, 64, 128, 256]

    try:
        head = [int(x) for x in trainConfig['head'].split(',')]
    except KeyError as e:
        head = [16, 16]

    launch_train(dataset_folder=datasetConfig['train'],
                 validation_folder=datasetConfig['validation'],
                 output_folder=output_folder,
                 gpu_num=eval(trainConfig['gpu']),
                 lr=eval(trainConfig['learningRate']),
                 rw=eval(trainConfig['regularizationWeight']),
                 acc_gradients=eval(trainConfig['accumulativeGradients']),
                 batch_size=eval(trainConfig['batchSize']),
                 max_epochs=eval(trainConfig['epochs']),
                 image_size=eval(trainConfig['imageSize']),
                 early_stop_patience=eval(trainConfig['earlyStopPatience']),
                 unet=unet,
                 head=head,
                 **loss_config)
