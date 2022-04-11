import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)  # PYTHON > 3.3 does not allow relative referencing

import argparse
from configparser import ConfigParser
from shutil import copy2
import os
from datetime import datetime

TRAIN_DATASET = '/mnt/EncryptedData1/Users/javier/ext_datasets/COMET_dataset/OSLO_COMET_CT/Formatted_128x128x128/train'

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

    print('TRAIN MODEL IN' + trainConfig['model'])

    simil = lossesConfig['similarity'].split(',')
    segm = lossesConfig['segmentation'].split(',')
    if trainConfig['name'].lower() == 'uw':
        from COMET.COMET_train_UW import launch_train
        output_folder = os.path.join(othersConfig['outputFolder'], '{}_Lsim_{}__Lseg_{}'.format(trainConfig['name'], '_'.join(simil), '_'.join(segm)))
    else:
        from COMET.COMET_train import launch_train
        simil = simil[0]
        segm = segm[0]
        output_folder = os.path.join(othersConfig['outputFolder'], '{}_Lsim_{}__Lseg_{}'.format(trainConfig['name'], simil, segm))
    output_folder = output_folder + '_' + datetime.now().strftime("%H%M%S-%d%m%Y")

    try:
        froozen_layers = eval(trainConfig['freeze'])
    except NameError as err:
        froozen_layers = [trainConfig['freeze'].upper()]
    if froozen_layers is not None:
        assert all(s in ['INPUT', 'OUTPUT', 'ENCODER', 'DECODER', 'TOP', 'BOTTOM'] for s in froozen_layers),\
            'Invalid option for "freeze". Expected one or several of: INPUT, OUTPUT, ENCODER, DECODER, TOP, BOTTOM'
        froozen_layers = list(set(froozen_layers))  # Unique elements

    # copy the configuration file to the destionation folder
    os.makedirs(output_folder, exist_ok=True)
    copy2(args.ini, os.path.join(output_folder, os.path.split(args.ini)[-1]))

    launch_train(dataset_folder=datasetConfig['train'],
                 validation_folder=datasetConfig['validation'],
                 output_folder=output_folder,
                 gpu_num=eval(trainConfig['gpu']),
                 lr=eval(trainConfig['learningRate']),
                 rw=eval(trainConfig['regularizationWeight']),
                 simil=simil,
                 segm=segm,
                 max_epochs=eval(trainConfig['epochs']),
                 model_file=trainConfig['model'],
                 freeze_layers=froozen_layers,
                 acc_gradients=eval(trainConfig['accumulativeGradients']),
                 batch_size=eval(trainConfig['batchSize']))