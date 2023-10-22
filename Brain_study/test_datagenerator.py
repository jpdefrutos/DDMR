from Brain_study.data_generator import BatchGenerator

import ddmr.utils.constants as C
from tqdm import tqdm

from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import os
import voxelmorph as vxm
from ddmr.utils.acummulated_optimizer import AdamAccumulated
from ddmr.losses import NCC, StructuralSimilarity, StructuralSimilarity_simplified


def named_logs(model, logs, validation=False):
    result = {'size': C.BATCH_SIZE} # https://gist.github.com/erenon/91f526302cd8e9d21b73f24c0f9c4bb8#gistcomment-3041181
    for l in zip(model.metrics_names, logs):
        k = ('val_' if validation else '') + l[0]
        result[k] = l[1]
    return result

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

    C.BATCH_SIZE = 12
    C.TRAINING_DATASET = '/mnt/EncryptedData1/Users/javier/ext_datasets/IXI_dataset/T1/training'
    output_folder = "/mnt/EncryptedData1/Users/javier/train_output/Brain_study/ERASE"

    data_generator = BatchGenerator(C.TRAINING_DATASET, C.BATCH_SIZE, True, C.TRAINING_PERC, True, ['none'])
    train_generator = data_generator.get_train_generator()
    val_generator = data_generator.get_validation_generator()

    e_iter = tqdm(range(100))
    t_iter = tqdm(train_generator)
    v_iter = tqdm(val_generator)

    e_iter.set_description('Epoch')
    t_iter.set_description('Train')
    v_iter.set_description('Val')
    #
    # for s in e_iter:
    #     for b in t_iter:
    #         continue
    #
    #     for b in v_iter:
    #         continue

    # Build model
    enc_features = [16, 32, 32, 32]  # const.ENCODER_FILTERS
    dec_features = [32, 32, 32, 32, 32, 16, 16]  # const.ENCODER_FILTERS[::-1]
    nb_features = [enc_features, dec_features]
    network = vxm.networks.VxmDense(inshape=(64, 64, 64),
                                    nb_unet_features=nb_features,
                                    int_steps=0)

    d = os.path.join(os.getcwd(), 'tensorboard_test')
    os.makedirs(d, exist_ok=True)
    callback_tensorboard = TensorBoard(log_dir=d,
                                       batch_size=C.BATCH_SIZE, write_images=False, histogram_freq=0, update_freq='epoch',
                                       write_graph=True,
                                       write_grads=True)

    losses = {'transformer': StructuralSimilarity_simplified(patch_size=2, dim=3, dynamic_range=1.).loss,
              'flow': vxm.losses.Grad('l2').loss}
    metrics = {'transformer': [StructuralSimilarity_simplified(patch_size=2, dim=3, dynamic_range=1.).metric,
                               tf.keras.losses.MSE],
               # 'flow': vxm.losses.Grad('l2').loss
               }
    loss_weights = {'transformer': 1.,
                    'flow': 5e-3}

    optimizer = AdamAccumulated(C.ACCUM_GRADIENT_STEP, C.LEARNING_RATE)

    network.compile(optimizer=optimizer,
                    loss=losses,
                    loss_weights=loss_weights,
                    metrics=metrics)
    callback_tensorboard.set_model(network)
    dummy = lambda x: named_logs(network, [x, 0, x, 0, 0])

    callback_tensorboard.on_train_begin()
    for s in e_iter:
        callback_tensorboard.on_epoch_begin(s)

        for n in range(100):
            callback_tensorboard.on_train_batch_begin(n)
            input('Press enter')
            callback_tensorboard.on_train_batch_end(n, dummy(n))

        callback_tensorboard.on_epoch_end(s, dummy)
    callback_tensorboard.on_train_end()
