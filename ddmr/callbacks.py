import tensorflow as tf
import tensorflow.keras.backend as K
import logging
import numpy as np


class RollingAverageWeighting(tf.keras.callbacks.Callback):
    def __init__(self, weights: list, loss_names: list, ref_loss: str, epoch_update):
        super(RollingAverageWeighting, self).__init__()
        assert len(weights) == len(loss_names)
        self.weights = weights
        self.loss_weights = dict()
        for name, w in zip(loss_names, weights):
            self.loss_weights[name] = w
        self.epoch_update = epoch_update - 1  # Epoch is zero based
        self.rolling_avg = dict()
        self.ref_loss = ref_loss
        loss_names.append(ref_loss)
        for name in loss_names:
            self.rolling_avg[name] = 0

    def on_epoch_end(self, epoch, logs=None):
        # Get the average loss for each loss function
        if epoch > self.epoch_update:
            # Updated loss weights
            for i, name in enumerate(self.rolling_avg.keys()):
                # avg[n] = avg[n-1] + 1/n * (new_val - avg[n-1]), where n is the size of the rolling avg
                self.rolling_avg[name] += (1 / self.epoch_update) * (logs.get(name) - self.rolling_avg[name])
        else:
            for i, name in enumerate(self.rolling_avg.keys()):
                self.rolling_avg[name] += logs.get(name)
                if epoch == self.epoch_update:  # Time to start updating the weights!
                    self.rolling_avg[name] /= self.epoch_update

        if not epoch % self.epoch_update:
            self.update_weights()

    def update_weights(self):
        new_weights = list()
        for name in self.loss_weights.keys():
            K.set_value(self.loss_weights[name], self.rolling_avg[self.ref_loss] / self.rolling_avg[name])
            new_weights.append(self.rolling_avg[self.ref_loss] / self.rolling_avg[name])

        out_str = ''
        for name, val in zip(self.loss_weights.keys(), new_weights):
            out_str += '{}: {:7.2f}\t'.format(name, val)
        print('WEIGHTS UPDATE: ' + out_str)


class UncertaintyWeightingRollingAverageCallback(tf.keras.callbacks.Callback):
    def __init__(self, method, epoch_update):
        super(UncertaintyWeightingRollingAverageCallback, self).__init__()
        self.method = method
        self.epoch_update = epoch_update

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.epoch_update:
            self.method()
            print('Calling method: '+self.method.__name__)

