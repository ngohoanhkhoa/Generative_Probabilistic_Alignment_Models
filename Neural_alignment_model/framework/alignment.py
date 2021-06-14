#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:06:34 2017

@author: ngohoanhkhoa
"""

import os, datetime
import importlib

import tensorflow as tf

import framework.tools as tools


# =============================================================================
# 
# =============================================================================

class Alignment(object):
    
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        
        # Create folder to store model files
        self.model_path = os.path.join(self.FLAGS.model_dir, self.FLAGS.data+'_'+self.FLAGS.model)
        if self.FLAGS.model_name is not None:
            self.model_path = os.path.join(self.FLAGS.model_dir, self.FLAGS.data+'_'+self.FLAGS.model+'_'+self.FLAGS.model_name)

        if not os.path.exists(self.model_path): os.makedirs(self.model_path)
        
        self.log_fname = os.path.join(self.model_path, 
                                 'log' + '.' + self.FLAGS.model + '.' + \
                                 '{}'.format(datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')))
        
        # Start logging module (both to terminal and to file)
        tools.Logger.setup(log_file=self.log_fname, timestamp=False)
        self.log = tools.Logger.get()
        
        self.model = None

    def run(self):
        # Initiate TF session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=self.FLAGS.allow_soft_placement,
                                              log_device_placement=self.FLAGS.log_device_placement,
                                              intra_op_parallelism_threads=self.FLAGS.intra_op_parallelism_threads,
                                              inter_op_parallelism_threads=self.FLAGS.inter_op_parallelism_threads,
                                              gpu_options=tf.GPUOptions(allow_growth=False))) as sess:
            
            self.print_training_info()
            
            # Load parallel data to train
            self.log.info('LOAD TRAINING DATA')
            if 'character' in self.FLAGS.model:
                train_set = tools.BiTextWordCharacterIterator(source=self.FLAGS.source_train_data,
                                                             target=self.FLAGS.target_train_data,
                                                             source_dict=self.FLAGS.source_vocabulary,
                                                             target_dict=self.FLAGS.target_vocabulary,
                                                             n_words_source=self.FLAGS.source_vocabulary_size,
                                                             n_words_target=self.FLAGS.target_vocabulary_size,
                                                             batch_size=self.FLAGS.batch_size,
                                                             maxlen=self.FLAGS.max_seq_length,
                                                             minlen=self.FLAGS.min_seq_length,
                                                             shuffle_each_epoch=self.FLAGS.shuffle_each_epoch,
                                                             sort_by_length=self.FLAGS.sort_by_length,
                                                             maxibatch_size=self.FLAGS.max_load_batches,
                                                             skip_empty=True,
                                                             
                                                             character_source_dict=self.FLAGS.source_character_vocabulary,
                                                             character_target_dict=self.FLAGS.target_character_vocabulary,
                                                             n_characters_source=self.FLAGS.character_source_vocabulary_size,
                                                             n_characters_target=self.FLAGS.character_target_vocabulary_size,
                                                             maxlen_source_character=self.FLAGS.max_word_source_length,
                                                             maxlen_target_character=self.FLAGS.max_word_target_length,
                                                             use_source_character=self.FLAGS.use_source_character,
                                                             
                                                             use_source_sub_vocabulary=False)
            elif 'bpe' in self.FLAGS.model:
                train_set = tools.BiTextBPEIterator(source=self.FLAGS.source_train_data,
                                                            target=self.FLAGS.target_train_data,
                                                            n_words_source=self.FLAGS.source_vocabulary_size,
                                                            n_words_target=self.FLAGS.target_vocabulary_size,
                                                            batch_size=self.FLAGS.batch_size,
                                                            minlen=self.FLAGS.min_seq_length,
                                                            shuffle_each_epoch=self.FLAGS.shuffle_each_epoch,
                                                            sort_by_length=self.FLAGS.sort_by_length,
                                                            maxibatch_size=self.FLAGS.max_load_batches,
                                                            skip_empty=True)
            else: 
                train_set = tools.BiTextIterator(source=self.FLAGS.source_train_data,
                                                            target=self.FLAGS.target_train_data,
                                                            source_dict=self.FLAGS.source_vocabulary,
                                                            target_dict=self.FLAGS.target_vocabulary,
                                                            n_words_source=self.FLAGS.source_vocabulary_size,
                                                            n_words_target=self.FLAGS.target_vocabulary_size,
                                                            batch_size=self.FLAGS.batch_size,
                                                            maxlen=self.FLAGS.max_seq_length,
                                                            minlen=self.FLAGS.min_seq_length,
                                                            shuffle_each_epoch=self.FLAGS.shuffle_each_epoch,
                                                            sort_by_length=self.FLAGS.sort_by_length,
                                                            maxibatch_size=self.FLAGS.max_load_batches,
                                                            skip_empty=True)
                
            self.log.info('LOAD VALIDATION DATA')
            
            if 'character' in self.FLAGS.model:
                if 'source_sub' in self.FLAGS.model:
                    valid_set = tools.BiTextWordCharacterIterator(source=self.FLAGS.source_valid_data,
                                                             target=self.FLAGS.target_valid_data,
                                                             source_dict=self.FLAGS.source_vocabulary,
                                                             target_dict=self.FLAGS.target_vocabulary,
                                                             batch_size=1,
                                                             n_words_source=self.FLAGS.source_vocabulary_size,
                                                             n_words_target=self.FLAGS.target_vocabulary_size,
                                                             
                                                             character_source_dict=self.FLAGS.source_character_vocabulary,
                                                             character_target_dict=self.FLAGS.target_character_vocabulary,
                                                             n_characters_source=self.FLAGS.character_source_vocabulary_size,
                                                             n_characters_target=self.FLAGS.character_target_vocabulary_size,
                                                             maxlen_source_character=self.FLAGS.max_word_source_length,
                                                             maxlen_target_character=self.FLAGS.max_word_target_length,
                                                             
                                                             use_source_sub_vocabulary=True,
                                                             source_sub_vocabulary_size=self.FLAGS.source_sub_vocabulary_size)
                    
                else:
                    valid_set = tools.BiTextWordCharacterIterator(source=self.FLAGS.source_valid_data,
                                                                         target=self.FLAGS.target_valid_data,
                                                                         source_dict=self.FLAGS.source_vocabulary,
                                                                         target_dict=self.FLAGS.target_vocabulary,
                                                                         batch_size=1,
                                                                         n_words_source=self.FLAGS.source_vocabulary_size,
                                                                         n_words_target=self.FLAGS.target_vocabulary_size,
                                                                         
                                                                         character_source_dict=self.FLAGS.source_character_vocabulary,
                                                                         character_target_dict=self.FLAGS.target_character_vocabulary,
                                                                         n_characters_source=self.FLAGS.character_source_vocabulary_size,
                                                                         n_characters_target=self.FLAGS.character_target_vocabulary_size,
                                                                         maxlen_source_character=self.FLAGS.max_word_source_length,
                                                                         maxlen_target_character=self.FLAGS.max_word_target_length,
                                                                         
                                                                         use_source_character=self.FLAGS.use_source_character)
            elif 'bpe' in self.FLAGS.model:
                valid_set = tools.BiTextBPEIterator(source=self.FLAGS.source_valid_data,
                                                                target=self.FLAGS.target_valid_data,
                                                                batch_size=1,
                                                                n_words_source=self.FLAGS.source_vocabulary_size,
                                                                n_words_target=self.FLAGS.target_vocabulary_size)
            else:
                valid_set = tools.BiTextIterator(source=self.FLAGS.source_valid_data,
                                                                target=self.FLAGS.target_valid_data,
                                                                source_dict=self.FLAGS.source_vocabulary,
                                                                target_dict=self.FLAGS.target_vocabulary,
                                                                batch_size=1,
                                                                n_words_source=self.FLAGS.source_vocabulary_size,
                                                                n_words_target=self.FLAGS.target_vocabulary_size)
            
            
            # Create a new model or reload existing checkpoint
            self.log.info('BUILD MODEL: %s', self.FLAGS.model)
                
            self.model = importlib.import_module('model.%s' % self.FLAGS.model).Model(self.FLAGS, sess, self.log)
            self.model.print_model_info()
            self.log.info(' ')
            
            self.model.train_set = train_set
            self.model.valid_set = valid_set
            
            if self.FLAGS.model_parameter_loaded_from_checkpoint != None:
                self.log.info('Reloading model parameters from checkpoint: %s', self.FLAGS.model_parameter_loaded_from_checkpoint)
                self.model.restore(self.FLAGS.model_parameter_loaded_from_checkpoint)
            else:
                ckpt = tf.train.get_checkpoint_state(self.model_path)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    self.log.info('Reloading model parameters from the most recent checkpoint: %s', ckpt.model_checkpoint_path)
                    self.model.restore(ckpt.model_checkpoint_path)
                else:
                    if not os.path.exists(self.model_path):
                        os.makedirs(self.model_path)
                    self.log.info('Creating new model parameters in %s', self.model_path)
                    sess.run(tf.global_variables_initializer())

            self.log.info('-----------------------------------------------------------------')
            self.log.info('-----------------------------------------------------------------')
            self.log.info('-----------------------------------------------------------------')
            
            # Train for all epochs
            self.model.train()
            
            # Save last model
            self.log.info('SAVE LAST MODEL in %s', self.model.checkpoint_path)
            self.model.save(self.model.checkpoint_path, global_step=self.model.global_step)
            
        self.log.info('TRAINING IS TERMINATED')
        
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
    
    def print_training_info(self):
        self.log.info('-----------------------------------------------------------------')
        self.log.info('-----------------------------------------------------------------')
        self.log.info('-----------------------------------------------------------------')
        self.log.info('-----------------------------------------------------------------')
        self.log.info('ALIGNMENT MODEL: %s', self.FLAGS.model)
        self.log.info('-----------------------------------------------------------------')
        self.log.info('Note: %s', self.FLAGS.note)
        self.log.info('-----------------------------------------------------------------')
        self.log.info('DATA:')
        self.log.info('Source training data: %s', self.FLAGS.source_train_data)
        self.log.info('Target training data: %s', self.FLAGS.target_train_data)
        self.log.info('Source validation data: %s', self.FLAGS.source_valid_data)
        self.log.info('Target validation data: %s', self.FLAGS.target_valid_data)
        self.log.info('Reference validation data: %s', self.FLAGS.reference_valid_data)
        
        self.log.info('-----------------------------------------------------------------')
        self.log.info('FOLDER:')
        self.log.info('Model path: %s', self.model_path)
        self.log.info('Log path: %s', self.log_fname)
        self.log.info(' ')
        
        self.log.info('-----------------------------------------------------------------')
        self.log.info('RUNTIME PARAMETERS:')
        self.log.info('Allow device soft placement: %s', self.FLAGS.allow_soft_placement)
        self.log.info('Log placement of ops on devices: %s', self.FLAGS.log_device_placement)
        self.log.info('Use # cpu intra_op_parallelism_threads: %s', self.FLAGS.intra_op_parallelism_threads)
        self.log.info('Use # cpu inter_op_parallelism_threads: %s', self.FLAGS.inter_op_parallelism_threads)
        self.log.info(' ')
        
        self.log.info('-----------------------------------------------------------------')
        self.log.info('TRAINING PARAMETERS:')
        self.log.info('Shuffle training dataset for each epoch: %s', self.FLAGS.shuffle_each_epoch)
        self.log.info('Sort pre-fetched minibatches by their target sequence lengths: %s', self.FLAGS.sort_by_length)
        self.log.info(' ')
        self.log.info('Maximum # of training epochs: %s', self.FLAGS.max_epochs)
        self.log.info('Maximum # of batches to load at one time: %s', self.FLAGS.max_load_batches)
        self.log.info('Batch size: %s', self.FLAGS.batch_size)
        self.log.info('Save model checkpoint every: %s', self.FLAGS.save_freq)
        self.log.info('Display training status every: %s', self.FLAGS.display_freq)
        self.log.info('Evaluate model every: %s', self.FLAGS.valid_freq)
        self.log.info(' ')
        self.log.info('Optimizer for training: %s', self.FLAGS.optimizer)
        self.log.info('Learning rate: %s', self.FLAGS.learning_rate)
        self.log.info('Keep probability: %s', self.FLAGS.keep_prob)
        self.log.info('Clip gradients to this norm: %s', self.FLAGS.max_gradient_norm)
        self.log.info(' ')
        self.log.info('Maximum sequence length: %s', self.FLAGS.max_seq_length) 
        self.log.info('Minimum sequence length: %s', self.FLAGS.min_seq_length)
        self.log.info(' ')
        if 'variational' not in self.FLAGS.model:
            self.log.info('Update emission fixed parameter after: %s epoches/batches', self.FLAGS.emission_update_freq)
            self.log.info('Update trainsition fixed parameter after: %s epoches/batches', self.FLAGS.jump_width_update_freq)
            self.log.info(' ')
        self.log.info('Max jump width: %d', self.FLAGS.max_jump_width)
        self.log.info(' ')
        self.log.info('-----------------------------------------------------------------')
        self.log.info('-----------------------------------------------------------------')
        