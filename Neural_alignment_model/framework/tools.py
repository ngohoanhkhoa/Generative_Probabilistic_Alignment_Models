# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:14:16 2017

@author: ngoho
"""
from __future__ import print_function
import logging as logging
import random
import json
import gzip
import os
import io
import tempfile
import datetime
import numpy as np
import re

try:
    import cPickle as pkl
except:
    import pickle as pkl
from collections import OrderedDict

import framework.shuffle as shuffle


#==============================================================================
# Extra vocabulary symbols 
#==============================================================================
SOS = '_SOS'
EOS = '_EOS' # also function as PAD
UNK = '_UNK'
NULL = '_NULL'

extra_tokens = [SOS, EOS, UNK, NULL]

start_token = extra_tokens.index(SOS)
end_token = extra_tokens.index(EOS)
unk_token = extra_tokens.index(UNK)
null_token = extra_tokens.index(NULL)


#==============================================================================
# Extra character vocabulary symbols 
#==============================================================================
SOW = '_SOW'
EOW = '_EOW' # also function as PAD
UNK = '_UNK'
NULL = '_NULL'

extra_character_tokens = [SOW, EOW, UNK, NULL]

start_character_token = extra_character_tokens.index(SOW)
end_character_token = extra_character_tokens.index(EOW)
unk_character_token = extra_character_tokens.index(UNK)
null_character_token = extra_character_tokens.index(NULL)

def get_time_now():
    return datetime.datetime.now().strftime('%y.%m.%d %H:%M:%S')

def unicode_to_utf8(d):
    return dict((key.encode("UTF-8"), value) for (key,value) in d.items())

def load_dict(filename):
    try:
        with io.open(filename, 'rb') as f:
            return unicode_to_utf8(json.load(f))
    except:
        with io.open(filename, 'rb') as f:
            return pkl.load(f)
        
def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return io.open(filename, mode, encoding='UTF-8')
    
def invert_dictionary(d):
    return OrderedDict([(v,k) for k,v in d.items()])
    
def idx_to_sent(ivocab, idxs, join=True):
    sent = []
    for widx in idxs:
        if widx == 0:
            break
        sent.append(ivocab.get(widx, UNK))
    if join:
        return " ".join(sent)
    else:
        return sent

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

class BiTextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, 
                 target,
                 source_dict, 
                 target_dict,
                 batch_size=1,
                 maxlen=None,
                 minlen=None,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=False,
                 maxibatch_size=100):
        
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = io.open(source, 'r', encoding='utf-8')
            self.target = io.open(target, 'r', encoding='utf-8')
            
        self.source_dict = load_dict(source_dict)
        self.target_dict = load_dict(target_dict)
        
        self.all_source_dict = self.source_dict
        self.all_target_dict = self.target_dict
        
        self.inverse_source_dict = invert_dictionary(self.source_dict)
        self.inverse_target_dict = invert_dictionary(self.target_dict)
        
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)
            

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            # sort by target buffer
            if self.sort_by_length:
                tlen = np.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict
                      else unk_token for w in ss]
                      
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source 
                          else unk_token for w in ss]
                      
                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict 
                      else unk_token for w in tt]
                      
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target 
                          else unk_token for w in tt]

                if self.maxlen:
                    if len(ss) > self.maxlen or len(tt) > self.maxlen:
                        continue
                if self.minlen:
                    if len(ss) < self.minlen or len(tt) < self.minlen:
                        continue
                if self.skip_empty and (not ss or not tt):
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target
    next = __next__
    
    
class BiTextWordCharacterIterator:
    """Simple Bitext iterator."""
    def __init__(self,
                 source,
                 target,
                 source_dict,
                 target_dict,
                 character_source_dict,
                 character_target_dict,
                 batch_size=1,
                 maxlen=None,
                 minlen=None,
                 maxlen_source_character=None,
                 maxlen_target_character=None,
                 n_words_source=-1,
                 n_words_target=-1,
                 n_characters_source=-1,
                 n_characters_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=False,
                 maxibatch_size=100,
                 use_source_character=False,
                 use_source_sub_vocabulary=False,
                 source_sub_vocabulary_size=-1):
        
        self.source_directory = source
        
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = io.open(source, 'r', encoding='utf-8')
            self.target = io.open(target, 'r', encoding='utf-8')
            
        self.source_dict = load_dict(source_dict)
        self.target_dict = load_dict(target_dict)
        
        self.inverse_source_dict = invert_dictionary(self.source_dict)
        self.inverse_target_dict = invert_dictionary(self.target_dict)
        
        self.source_sub_vocabulary_size = source_sub_vocabulary_size
        self.use_source_sub_vocabulary = use_source_sub_vocabulary
        
        
        if self.use_source_sub_vocabulary:
            self.sub_source_dict = self.build_sub_source_vocabulary()

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        
        self.character_source_dict = load_dict(character_source_dict)
        self.character_target_dict = load_dict(character_target_dict)
        
        #self.inverse_character_source_dict = invert_dictionary(self.character_source_dict)
        #self.inverse_character_target_dict = invert_dictionary(self.character_target_dict)
        
        self.maxlen_source_character = maxlen_source_character
        self.maxlen_target_character = maxlen_target_character
        
        self.n_characters_source=n_characters_source
        self.n_characters_target=n_characters_target
        
        self.use_source_character=use_source_character
        
        self.end_of_data = False
        

    def build_sub_source_vocabulary(self):
        source = source = io.open(self.source_directory, 'r', encoding='utf-8')
        word_freqs_known = OrderedDict()
        word_freqs_unknown = OrderedDict()
        
        for line in source:
            words_in = line.strip().split(' ')
            for w in words_in:
                if w in source:
                    word_freqs_known[w] = self.source_dict[w]
                else:
                    if w not in word_freqs_unknown:
                        word_freqs_unknown[w] = 0
                    word_freqs_unknown[w] += 1
                    
        for w, idx_w  in self.source_dict.items():
            if ( len(word_freqs_known.keys()) + len(word_freqs_unknown.keys()) ) >= self.source_sub_vocabulary_size: break
        
            if w not in word_freqs_known:
                word_freqs_known[w] = self.source_dict[w]
                        
        words_known = list(word_freqs_known.keys())
        freqs_known = list(word_freqs_known.values())
        sorted_idx_known = np.argsort(freqs_known)
        sorted_words_known = [words_known[ii] for ii in sorted_idx_known]
        
        words_unknown = list(word_freqs_unknown.keys())
        freqs_unknown = list(word_freqs_unknown.values())
        sorted_idx_unknown = np.argsort(freqs_unknown)
        sorted_words_unknown = [words_unknown[ii] for ii in sorted_idx_unknown[::-1]]
    
        worddict = OrderedDict()
        worddict[SOS] = 0
        worddict[EOS] = 1
        worddict[UNK] = 2
        worddict[NULL] = 3
        for ii, ww in enumerate(sorted_words_known):
            worddict[ww] = ii
        len_word_known_dict = len(worddict.keys())
        for ii, ww in enumerate(sorted_words_unknown):
            worddict[ww] = len_word_known_dict+ii
    
        with open('%s_sub.pkl'%self.source_directory, 'wb') as f:
            pkl.dump(worddict, f, protocol=2)
    
        print('Create sub vocabulary')
        
        return worddict

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)
            

    def __next__(self):
        
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        source_s=[]
        
        target_character = []
        source_character = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            # sort by target buffer
            if self.sort_by_length:
                tlen = np.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        
        try:

            # actual work here
            while True:
                
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                    tt = self.target_buffer.pop()
                    
                    ss_character = ss
                    tt_character = tt
                    
                except IndexError:
                    break
                
                # ------------------------------------------------------------- 
                tt = [self.target_dict[w] if w in self.target_dict 
                          else unk_token for w in tt]
                          
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target 
                          else unk_token for w in tt]
                    
                tt_character = [[ self.character_target_dict[c] if c in self.character_target_dict 
                else unk_character_token for c in w] for w in tt_character]
                if self.n_characters_target > 0:
                    tt_character = [[ c if c < self.n_characters_target 
                                        else unk_character_token for c in w] for w in tt_character]
                                        
                if self.maxlen_target_character:
                    tt_character = [w[:self.maxlen_target_character] for w in tt_character]

                # -------------------------------------------------------------
                
                if self.use_source_character:
                        ss_character = [[ self.character_source_dict[c] if c in self.character_source_dict 
                        else unk_character_token for c in w] for w in ss_character]
                        if self.n_characters_source > 0:
                            ss_character = [[ c if c < self.n_characters_source 
                                                else unk_character_token for c in w] for w in ss_character]
                                                
                        if self.maxlen_source_character:
                            ss_character = [w[:self.maxlen_source_character] for w in ss_character]
                            
                # -------------------------------------------------------------
                
                if self.use_source_sub_vocabulary:
                    ss_s = [self.sub_source_dict[w] if w in self.sub_source_dict
                          else unk_token for w in ss]
                
                    if self.source_sub_vocabulary_size > 0:
                        ss_s = [w if w < self.source_sub_vocabulary_size 
                              else unk_token for w in ss_s]
                    source_s.append(ss_s)
                    
                
                ss_w = [self.source_dict[w] if w in self.source_dict
                          else unk_token for w in ss]
                
                if self.n_words_source > 0:
                    ss_w = [w if w < self.n_words_source 
                         else unk_token for w in ss_w]

                if self.maxlen:
                    if len(ss_w) > self.maxlen or len(tt) > self.maxlen:
                        continue
                    
                if self.minlen:
                    if len(ss) < self.minlen or len(tt) < self.minlen:
                        continue
                    
                if self.skip_empty and (not ss or not tt):
                    continue

                source.append(ss_w)
                target.append(tt)
                
                target_character.append(tt_character)
                source_character.append(ss_character)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if self.use_source_character:
            if len(source) == 0 or len(target) == 0:
                source, target, target_character, source_character = self.next()
                
            return source, target, target_character, source_character
        
        if self.use_source_sub_vocabulary:
            # all sentence pairs in maxibatch filtered out because of length
            if len(source) == 0 or len(target) == 0:
                source, target, target_character, source_s = self.next()
            return source, target, target_character, source_s
        
        if not self.use_source_character and not self.use_source_sub_vocabulary:
            if len(source) == 0 or len(target) == 0:
                    source, target, target_character = self.next()
            return source, target, target_character
        
    next = __next__
    
    
class BiTextBPEIterator:
    def __init__(self,
                 source, 
                 target,
                 batch_size=1,
                 maxlen=None,
                 minlen=None,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=False,
                 maxibatch_size=100):
        
        if shuffle_each_epoch:
            self.source_orig = source
            self.target_orig = target
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source = io.open(source, 'r', encoding='utf-8')
            self.target = io.open(target, 'r', encoding='utf-8')
        
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source
        self.n_words_target = n_words_target

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source, self.target = shuffle.main([self.source_orig, self.target_orig], temporary=True)
        else:
            self.source.seek(0)
            self.target.seek(0)
            

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())

            # sort by target buffer
            if self.sort_by_length:
                tlen = np.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]
                _tbuf = [self.target_buffer[i] for i in tidx]

                self.source_buffer = _sbuf
                self.target_buffer = _tbuf

            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()

        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                
                # Need to check when shuffle_each_epoch
                ss[0] = re.sub('[^0-9]', '', ss[0])
                ss[-1] = re.sub('[^0-9]', '', ss[-1])
                ss_ = []
                for s in ss:
                    if s != '':
                        ss_.append(int(s))
                ss = ss_
                
                tt = self.target_buffer.pop()
                tt[0] = re.sub('[^0-9]', '', tt[0])
                tt[-1] = re.sub('[^0-9]', '', tt[-1])
                tt_ = []
                for t in tt:
                    if t != '':
                        tt_.append(int(t))
                tt = tt_
                
                if self.minlen:
                    if len(ss) < self.minlen or len(tt) < self.minlen:
                        continue
                if self.skip_empty and (not ss or not tt):
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()

        return source, target
    next = __next__
    
class MonoTextBPEIterator:
    def __init__(self,
                 source,
                 batch_size=1,
                 maxlen=None,
                 minlen=None,
                 n_words_source=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=False,
                 maxibatch_size=100):
        
        
        self.source = io.open(source, 'r', encoding='utf-8')
        
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.skip_empty = skip_empty

        self.n_words_source = n_words_source

        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length

        self.source_buffer = []
        self.target_buffer = []
        self.k = batch_size * maxibatch_size
        self.end_of_data = False

    def __iter__(self):
        return self

    def __len__(self):
        return sum([1 for _ in self])
    
    def reset(self):
        if self.shuffle:
            self.source.seek(0)
        else:
            self.source.seek(0)
            

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []

        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip().split())

            # sort by target buffer
            if self.sort_by_length:
                tlen = np.array([len(s) for s in self.source_buffer])
                tidx = tlen.argsort()

                _sbuf = [self.source_buffer[i] for i in tidx]

                self.source_buffer = _sbuf

            else:
                self.source_buffer.reverse()

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                
                # Need to check when shuffle_each_epoch
                ss[0] = re.sub('[^0-9]', '', ss[0])
                ss[-1] = re.sub('[^0-9]', '', ss[-1])
                ss_ = []
                for s in ss:
                    if s != '':
                        ss_.append(int(s))
                ss = ss_
                
                if self.minlen:
                    if len(ss) < self.minlen:
                        continue
                if self.skip_empty and (not ss):
                    continue

                source.append(ss)

                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(source) == 0:
            source = self.next()

        return source
    next = __next__