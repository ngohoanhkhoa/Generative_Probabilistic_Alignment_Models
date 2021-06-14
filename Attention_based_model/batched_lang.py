#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import logging
import torch
from torchtext import data, datasets
import pandas as pd
import os


UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
# UNK_NUM = 0
# PAD_NUM = 1
# SOS_NUM = 2
# EOS_NUM = 3

USE_CUDA = torch.cuda.is_available()

class Batch:
    """Object for holding a batch of data with mask during training.
    Input is a batch from a torch text iterator.
    """

    def __init__(self, src, trg, pad_index=0):

        src, src_lengths = src

        self.src = src
        self.src_lengths = src_lengths
        self.src_mask = (src != pad_index).unsqueeze(-2)
        self.nseqs = src.size(0)

        trg, trg_lengths = trg
        # NOTE: commenting that out
        # self.trg = trg[:, :-1]  # QUESTION: why cut EOS? Because it's never fed in teacher forcing? (careful with that maybe!)
        self.trg = trg
        self.trg_lengths = trg_lengths
        self.trg_y = trg[:, 1:]  # I understand better why we cut SOS here for eval though
        self.trg_mask = (self.trg_y != pad_index)
        self.ntokens = (self.trg_y != pad_index).data.sum().item()

        if USE_CUDA:  # could probably write this with .to(device) instead
            self.src = self.src.cuda()
            self.src_mask = self.src_mask.cuda()

            if trg is not None:
                self.trg = self.trg.cuda()
                self.trg_y = self.trg_y.cuda()
                self.trg_mask = self.trg_mask.cuda()

def rebatch(pad_idx, batch):
    """Wrap torchtext batch into our own Batch class for pre-processing"""
    return Batch(batch.Source, batch.Target, pad_idx)


# Using torchtext
# ---------------

def prepare_data_torchtext(data_train_file, data_test_file):
    # build a torchtext TabularDataset from csv file, using Fields
    # NB: using batch_first=True
    SRC = data.Field(batch_first=True, lower=False, include_lengths=True,
                     unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=None,
                     eos_token=EOS_TOKEN)
    # QUESTION: maybe don't use SOS on target
    TRG = data.Field(batch_first=True, lower=False, include_lengths=True,
                     unk_token=UNK_TOKEN, pad_token=PAD_TOKEN, init_token=SOS_TOKEN,
                     eos_token=EOS_TOKEN)

    data_fields = [('Source', SRC), ('Target', TRG)]
    train_dataset = data.TabularDataset(path=data_train_file, format='csv',
                                           fields=data_fields, skip_header=True)
    # build vocabs
    SRC.build_vocab(train_dataset)
    TRG.build_vocab(train_dataset)
    PAD_INDEX = TRG.vocab.stoi[PAD_TOKEN]
    
    test_dataset = data.TabularDataset(path=data_test_file, format='csv',
                                           fields=data_fields, skip_header=True)
    

    return train_dataset, test_dataset, SRC, TRG, PAD_INDEX



def sentence_from_tensor(FIELD, tensor):
    local = list(tensor.view(-1).cpu().numpy())
    words = [FIELD.vocab.itos[idx] for idx in local]
    return words

def log_data_info(train_data, src_field, trg_field):


    output = "\n- Train size (number of sentence pairs):"
    output += str(len(train_data))

    output += "\n- First training example:\n"
    output += "src: " + " ".join(vars(train_data[0])['Source']) + '\n'
    output += "trg: " + " ".join(vars(train_data[0])['Target'])

    output += "\n- Most common words (src):\n"
    output += "\n".join(["%10s %10d" % x for x in src_field.vocab.freqs.most_common(10)])
    output += "\n- Most common words/characters (trg):\n"
    output += "\n".join(["%10s %10d" % x for x in trg_field.vocab.freqs.most_common(10)])

    output += "\n- First 10 words (src):\n"
    output += "\n".join('%02d %s' % (i, t) for i, t in enumerate(src_field.vocab.itos[:10]))
    output += "\n- First 10 words (trg):\n"
    output += "\n".join('%02d %s' % (i, t) for i, t in enumerate(trg_field.vocab.itos[:10]))

    output += f"\n- Number of Source words (types): {len(src_field.vocab)}"
    output += f"\n- Number of Target words (types): {len(trg_field.vocab)} \n"

    logging.info(output)


