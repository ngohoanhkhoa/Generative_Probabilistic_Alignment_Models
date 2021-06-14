from torchtext import data
import pandas as pd
import os
import logging as logging

SOS = '_SOS'
EOS = '_EOS' # also function as PAD
UNK = '_UNK'
PAD = '_PAD'

def singleton(cls):
    instances = {}
    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance()

@singleton
class Logger(object):
    """Logs to stdout and to file simultaneously."""
    def __init__(self):
        pass

    def setup(self, log_file=None, timestamp=True):
        _format = '%(message)s'
        if timestamp:
            _format = '%(asctime)s ' + _format

        self.formatter = logging.Formatter(_format)
        self._logger = logging.getLogger('alignment')
        self._logger.setLevel(logging.DEBUG)
        self._ch = logging.StreamHandler()
        self._ch.setFormatter(self.formatter)
        self._logger.addHandler(self._ch)

        if log_file:
            self._fh = logging.FileHandler(log_file, mode='w')
            self._fh.setFormatter(self.formatter)
            self._logger.addHandler(self._fh)

    def get(self):
        return self._logger

def prepare_data_torchtext(dataset_csv):
    # NB: using batch_first=True
    SRC = data.Field(batch_first=True, lower=True, include_lengths=True,
                     unk_token=None, pad_token=None, init_token=None,
                     eos_token=None)
    # QUESTION: maybe don't use SOS on target
    TRG = data.Field(batch_first=True, lower=True, include_lengths=True,
                     unk_token=None, pad_token=None, init_token=None,
                     eos_token=None)

    data_fields = [('Source', SRC), ('Target', TRG)]
    traindev_dataset = data.TabularDataset(path=dataset_csv, format='csv',
                                           fields=data_fields, skip_header=True)
    # build vocabs
    SRC.build_vocab(traindev_dataset)
    TRG.build_vocab(traindev_dataset)
    PAD_INDEX = TRG.vocab.stoi[PAD]

    return traindev_dataset, SRC, TRG, PAD_INDEX

def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.Source, batch.Target, pad_idx)
