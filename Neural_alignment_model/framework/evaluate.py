# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 12:00:42 2017

@author: ngoho
"""
import tensorflow as tf
import os
# Remember to change
from alignment import Alignment

# Avoid thread explosion
#os.environ["OMP_NUM_THREADS"] = "8"
#os.environ["MKL_NUM_THREADS"] = "8"

# For GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = ""

tf.app.flags.DEFINE_string('note', '', 'Note about the training') #********

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# MODEL PARAMETERS
tf.app.flags.DEFINE_string('model', 'evaluate_IBM1', 'Model for training') #********
tf.app.flags.DEFINE_string('model_parameter_loaded_from_checkpoint', None, 'Load Model parameters from file, if None get the last checkpoint')

# ----------------- IBM1
tf.app.flags.DEFINE_integer('ibm1_freq',1, 'Train IBM1 in ibm1_freq epoches, no training IBM1 (= 0), train more # time (IBM1_parameter_loaded_from_file is not None, ibm1_freq=-#)')
tf.app.flags.DEFINE_integer('pretrain_freq',1, 'Pre-train model in pretrain_freq eppches')
tf.app.flags.DEFINE_string('IBM1_parameter_loaded_from_file', "/vol/work2/2017-NeuralAlignments/exp-ngoho/models-cpu/Version3/en-fr_ibm1_10i_1/IBM1_params.en-fr_ibm1_10i_1.10000.5i.pkl" , 'Load IBM1 parameters (Lexicon table) from file')
tf.app.flags.DEFINE_boolean('IBM1_parameter_export_to_file', False, 'Export IBM1 parameters (Lexicon table) to file')

# ----------------- Transition
tf.app.flags.DEFINE_float('jump_distance_update_freq', 1, 'Jump distance update rate')

tf.app.flags.DEFINE_float('p0', 0.01, 'P0')
tf.app.flags.DEFINE_integer('max_distance', 5, 'Max distance')

tf.app.flags.DEFINE_string('transition_parameter_loaded_from_file', None , 'Load transition parameters from file')
tf.app.flags.DEFINE_string('p0_parameter_loaded_from_file', None , 'Load p0 parameter from file')
tf.app.flags.DEFINE_string('initialize_jump_set', 'heuristic', 'Initialize Jump distance set (heuristic, uniform, random)')
tf.app.flags.DEFINE_string('initial_transition_pi', 'heuristic', 'Set value for initial transition pi (heuristic, uniform, random)')
tf.app.flags.DEFINE_float('transition_heuristic_prob', 0.9, 'Set value for initial transition at Jump Width = 1 (heuristic)')

# ----------------- Extending
tf.app.flags.DEFINE_integer('negative_source_vocab_num', 0, 'Number of negative class for output softmax (Complementary Sum Sampling)')
tf.app.flags.DEFINE_integer('target_window_size', 2, 'Size of window for target sentence - context')
tf.app.flags.DEFINE_integer('source_window_size', 0, 'Size of window for source sentence - history')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# FOLDER
tf.app.flags.DEFINE_string('model_name', 'evaluate', 'File name used for model') #********
tf.app.flags.DEFINE_string('model_dir', '/vol/work2/2017-NeuralAlignments/exp-ngoho/models', 'Path to save model')
tf.app.flags.DEFINE_string('log_name', 'log', 'File name used for model log')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# TRAINING PARAMETERS
tf.app.flags.DEFINE_integer('source_vocabulary_size', 10000, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('target_vocabulary_size', 10000, 'Target vocabulary size')
tf.app.flags.DEFINE_integer('max_seq_length', 20, 'Maximum sequence length')
tf.app.flags.DEFINE_integer('min_seq_length', 1, 'Maximum sequence length')

tf.app.flags.DEFINE_boolean('shuffle_each_epoch', False, 'Shuffle training dataset for each epoch')
tf.app.flags.DEFINE_boolean('sort_by_length', False, 'Sort pre-fetched minibatches by their target sequence lengths')

tf.app.flags.DEFINE_integer('max_epochs', 10000, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('max_load_batches', 100, 'Maximum # of batches to load at one time')
tf.app.flags.DEFINE_integer('batch_size', 5, 'Batch size')
tf.app.flags.DEFINE_integer('display_freq', 1, 'Display training status every this iteration')
tf.app.flags.DEFINE_integer('save_freq', 20, 'Save model checkpoint every this iteration')
tf.app.flags.DEFINE_integer('valid_freq', 1, 'Evaluate model every this iteration: valid_data needed')

# ----------------- Optimizer
tf.app.flags.DEFINE_string('optimizer', 'adam', 'Optimizer for training: (adadelta, adam, rmsprop)')
tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
tf.app.flags.DEFINE_float('max_gradient_norm', 1.0, 'Clip gradients to this norm')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# NEURAL NETWORK PARAMETERS
tf.app.flags.DEFINE_integer('hidden_units', 64, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('embedding_size', 64, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_float('keep_prob', 1.0, 'Embedding dimensions of encoder and decoder inputs')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# DATA

tf.app.flags.DEFINE_string('source_vocabulary', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/training/europarl-v7.en-fr.cln.low.en.pkl', 'Path to source vocabulary')
tf.app.flags.DEFINE_string('target_vocabulary', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/training/europarl-v7.en-fr.cln.low.fr.pkl', 'Path to target vocabulary')
tf.app.flags.DEFINE_string('source_train_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/training/europarl-v7.en-fr.cln.low.en.lenSent50', 'Path to source training data')
tf.app.flags.DEFINE_string('target_train_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/training/europarl-v7.en-fr.cln.low.fr.lenSent50', 'Path to target training data')
tf.app.flags.DEFINE_string('source_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/testing/testing.low.en', 'Path to source validation data')
tf.app.flags.DEFINE_string('target_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/testing/testing.low.fr', 'Path to target validation data')
tf.app.flags.DEFINE_string('reference_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/testing/testing.en-fr.align', 'Path to alignment validation data')

tf.app.flags.DEFINE_integer('evaluate_alignment_start_from', 1, 'For evaluation, alignment starts from (0 or 1)')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# RESEARCH
tf.app.flags.DEFINE_integer('export_valid_alignment_freq', 1, 'Export validation alignment result')
tf.app.flags.DEFINE_boolean('get_distance_file', True, 'Research: Export distance parameters to file')
tf.app.flags.DEFINE_boolean('get_p0_file', True, 'Research: Export p0 parameters to file')
tf.app.flags.DEFINE_boolean('get_transition_file', False, 'Research: Export transition matrix to file')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# RUNTIME PARAMETERS
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')
tf.app.flags.DEFINE_string('mode', 'train', 'Mode (train, evaluate)')

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

FLAGS = tf.app.flags.FLAGS

def main(_):
    
    alignment = Alignment(FLAGS)
    alignment.run()
    
if __name__ == '__main__':
    tf.app.run()

#==============================================================================
# 
#==============================================================================

# 100K EN-FR

#tf.app.flags.DEFINE_string('source_vocabulary', '/vol/work2/2017-NeuralAlignments/exp-ngoho/data/europarl-v7.en-fr.cln.low.100000.fr.pkl', 'Path to source vocabulary')
#tf.app.flags.DEFINE_string('target_vocabulary', '/vol/work2/2017-NeuralAlignments/exp-ngoho/data/europarl-v7.en-fr.cln.low.100000.en.pkl', 'Path to target vocabulary')
#tf.app.flags.DEFINE_string('source_train_data', '/vol/work2/2017-NeuralAlignments/exp-ngoho/data/europarl-v7.en-fr.cln.low.100000.fr.lenSent50', 'Path to source training data')
#tf.app.flags.DEFINE_string('target_train_data', '/vol/work2/2017-NeuralAlignments/exp-ngoho/data/europarl-v7.en-fr.cln.low.100000.en.lenSent50', 'Path to target training data')
#tf.app.flags.DEFINE_string('source_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/testing/testing.low.fr', 'Path to source validation data')
#tf.app.flags.DEFINE_string('target_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/testing/testing.low.en', 'Path to target validation data')
#tf.app.flags.DEFINE_string('reference_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/testing/testing.fr-en.align', 'Path to alignment validation data')

# EN-FR

#tf.app.flags.DEFINE_string('source_vocabulary', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/training/europarl-v7.en-fr.cln.low.en.pkl', 'Path to source vocabulary')
#tf.app.flags.DEFINE_string('target_vocabulary', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/training/europarl-v7.en-fr.cln.low.fr.pkl', 'Path to target vocabulary')
#tf.app.flags.DEFINE_string('source_train_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/training/europarl-v7.en-fr.cln.low.en.lenSent50', 'Path to source training data')
#tf.app.flags.DEFINE_string('target_train_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/training/europarl-v7.en-fr.cln.low.fr.lenSent50', 'Path to target training data')
#tf.app.flags.DEFINE_string('source_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/testing/testing.low.en', 'Path to source validation data')
#tf.app.flags.DEFINE_string('target_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/testing/testing.low.fr', 'Path to target validation data')
#tf.app.flags.DEFINE_string('reference_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-fr/formatted/testing/testing.en-fr.align', 'Path to alignment validation data')

# EN-RO

#tf.app.flags.DEFINE_string('source_vocabulary', '/vol/work2/2017-NeuralAlignments/data/en-ro/formatted/training/train.merg.en-ro.cln.ro.utf8.low.pkl', 'Path to source vocabulary')
#tf.app.flags.DEFINE_string('target_vocabulary', '/vol/work2/2017-NeuralAlignments/data/en-ro/formatted/training/train.merg.en-ro.cln.en.utf8.low.pkl', 'Path to target vocabulary')
#tf.app.flags.DEFINE_string('source_train_data', '/vol/work2/2017-NeuralAlignments/data/en-ro/formatted/training/train.merg.en-ro.cln.ro.utf8.low.lenSent50', 'Path to source training data')
#tf.app.flags.DEFINE_string('target_train_data', '/vol/work2/2017-NeuralAlignments/data/en-ro/formatted/training/train.merg.en-ro.cln.en.utf8.low.lenSent50', 'Path to target training data')
#tf.app.flags.DEFINE_string('source_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-ro/formatted/testing/corp.test.ro-en.cln.ro.low', 'Path to source validation data')
#tf.app.flags.DEFINE_string('target_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-ro/formatted/testing/corp.test.ro-en.cln.en.low', 'Path to target validation data')
#tf.app.flags.DEFINE_string('reference_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-ro/formatted/testing/test.ro-en.ali.startFrom1', 'Path to alignment validation data')

# EN-DE

#tf.app.flags.DEFINE_string('source_vocabulary', '/vol/work2/2017-NeuralAlignments/data/en-de/formatted/training/corp.train.de-en.de.low.pkl', 'Path to source vocabulary')
#tf.app.flags.DEFINE_string('target_vocabulary', '/vol/work2/2017-NeuralAlignments/data/en-de/formatted/training/corp.train.de-en.en.low.pkl', 'Path to target vocabulary')
#tf.app.flags.DEFINE_string('source_train_data', '/vol/work2/2017-NeuralAlignments/data/en-de/formatted/training/corp.train.de-en.low.cln.de.final.lenSent50', 'Path to source training data')
#tf.app.flags.DEFINE_string('target_train_data', '/vol/work2/2017-NeuralAlignments/data/en-de/formatted/training/corp.train.de-en.low.cln.en.final.lenSent50', 'Path to target training data')
#tf.app.flags.DEFINE_string('source_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-de/formatted/testing/corp.test.de-en.de.low.ngoho', 'Path to source validation data')
#tf.app.flags.DEFINE_string('target_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-de/formatted/testing/corp.test.de-en.en.low.ngoho', 'Path to target validation data')
#tf.app.flags.DEFINE_string('reference_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-de/formatted/testing/alignmentDeEn.fixed.ali.startFrom1.de-en.ngoho', 'Path to alignment validation data')

# EN-CZ

#tf.app.flags.DEFINE_string('source_vocabulary', '/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/training/training.en-cz.en.tok.low.pkl', 'Path to source vocabulary')
#tf.app.flags.DEFINE_string('target_vocabulary', '/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/training/training.en-cz.cz.tok.low.pkl', 'Path to target vocabulary')
#tf.app.flags.DEFINE_string('source_train_data', '/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/training/training.en-cz.en.tok.low.lenSent50', 'Path to source training data')
#tf.app.flags.DEFINE_string('target_train_data', '/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/training/training.en-cz.cz.tok.low.lenSent50', 'Path to target training data')
#tf.app.flags.DEFINE_string('source_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/testing/testing.en-cz.en.low', 'Path to source validation data')
#tf.app.flags.DEFINE_string('target_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/testing/testing.en-cz.cz.low', 'Path to target validation data')
#tf.app.flags.DEFINE_string('reference_valid_data', '/vol/work2/2017-NeuralAlignments/data/en-cz/formatted/testing/testing.en-cz.alignment.fixed', 'Path to alignment validation data')