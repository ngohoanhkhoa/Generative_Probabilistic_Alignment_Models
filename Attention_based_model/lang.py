#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
import logging


# SOS_token = 0
# EOS_token = 1
# PAD_token = 2

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_NUM = 0
PAD_NUM = 1
SOS_NUM = 2
EOS_NUM = 3


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_NUM: SOS_TOKEN, EOS_NUM: EOS_TOKEN, PAD_NUM: PAD_TOKEN, UNK_NUM: UNK_TOKEN}
        self.n_words = 4  # Count predefined tokens

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def readLangs(source, target):
    logging.info("Reading lines...")

    # Read the file and split into lines
    # PG: could use readlines() rather probably
    source_lines = open(source, encoding='utf-8').read().strip().split('\n')
    target_lines = open(target, encoding='utf-8').read().strip().split('\n')

    # Pair lines
    pairs = list(zip(source_lines, target_lines)) # /!\ tuples instead of lists

    # Make Lang instances
    input_lang = Lang(source)
    output_lang = Lang(target)

    return input_lang, output_lang, pairs

def prepare_data(source, target):
    input_lang, output_lang, pairs = readLangs(source, target)
    logging.info("Read %s sentence pairs" % len(pairs))
    logging.info("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    logging.info("Counted words:")
    logging.info("vocab size for {}: {}".format(input_lang.name, input_lang.n_words))
    logging.info("vocab size for {}: {}".format(output_lang.name, output_lang.n_words))
    return input_lang, output_lang, pairs


