import numpy as np
import ddmr.utils.constants as C

class SummaryDictionary:
    def __init__(self, model, batch_size, accumulative_gradients_step=None):
        self.train_names = model.metrics_names
        self.val_names = ['val_'+n for n in self.train_names]
        self.batch_size = batch_size
        self.acc_grad_step = accumulative_gradients_step
        self._reset()

    def _reset(self):
        self.summary_dict = {'size': self.batch_size}
        if self.acc_grad_step is not None:
            self.summary_dict = {'accumulative_grad_step': self.acc_grad_step}
        for k in self.train_names + self.val_names:
            self.summary_dict[k] = list()

    def on_train_batch_end(self, values):
        for k, v in zip(self.train_names, values):
            self.summary_dict[k].append(v)

    def on_validation_batch_end(self, values):
        for k, v in zip(self.val_names, values):
            self.summary_dict[k].append(v)

    def on_epoch_end(self):
        for k, v in self.summary_dict.items():
            self.summary_dict[k] = np.asarray(v).mean()

        ret_val = self.summary_dict.copy()
        self._reset()
        return ret_val


def named_logs(model, logs, validation=False):
    result = {'size': C.BATCH_SIZE} # https://gist.github.com/erenon/91f526302cd8e9d21b73f24c0f9c4bb8#gistcomment-3041181
    for l in zip(model.metrics_names, logs):
        k = ('val_' if validation else '') + l[0]
        result[k] = l[1]
    return result
