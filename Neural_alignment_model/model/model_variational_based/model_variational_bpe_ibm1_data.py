#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, time, math
import numpy as np
import tensorflow as tf
import framework.tools as tools

from model.model_variational_based.model_variational_bpe_ibm1 import Model

class Model(Model):

    def __init__(self, FLAGS, session, log):
        
        super(Model, self).__init__(FLAGS, session, log)
        
    def build_update(self):
        self.annealing_KL_divergence = tf.Variable(0., trainable=False, name='annealing_KL_divergence' ,
                                                   dtype=self.tf_float_dtype)
        
        reconstruction_expectation = tf.reduce_mean(tf.reduce_sum(self.reconstruction_expectation_log, axis=1))
        
        alignment_expectation = tf.reduce_mean(self.alignment_expectation)
        
        KL_divergence = tf.reduce_mean(tf.reduce_sum(self.KL_divergence, axis=1))
        
        self.KL_divergence_eval = KL_divergence
        
        self.cost_reconstruction_expectation =  (-reconstruction_expectation)
        self.cost_alignment_expectation =  (-alignment_expectation)
        self.cost_KL_divergence = KL_divergence * self.annealing_KL_divergence
        
        self.cost = (self.FLAGS.alpha_reconstruction_expectation * self.cost_reconstruction_expectation) \
        + (self.FLAGS.alpha_alignment_expectation * self.cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.cost_KL_divergence)
        
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        gradients = tf.gradients(self.cost, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = self.opt.apply_gradients(zip(clip_gradients,trainable_params),
                                                         global_step=self.global_step)
        
        gradients_reconstruction = tf.gradients(self.cost_reconstruction_expectation, trainable_params)
        clip_gradients_reconstruction, _ = tf.clip_by_global_norm(gradients_reconstruction, self.max_gradient_norm)
        self.updates_reconstruction = self.opt.apply_gradients(zip(clip_gradients_reconstruction,trainable_params),
                                                         global_step=self.global_step)
        
    def train(self):
        # Training loop
        self.log.info('TRAINING %s', tools.get_time_now())
        
        mono_train_set = tools.MonoTextBPEIterator(source=self.FLAGS.target_train_data_mono,
                                                 n_words_source=self.FLAGS.source_vocabulary_size,
                                                 batch_size=self.FLAGS.batch_size,
                                                 minlen=self.FLAGS.min_seq_length,
                                                 shuffle_each_epoch=self.FLAGS.shuffle_each_epoch,
                                                 sort_by_length=self.FLAGS.sort_by_length,
                                                 maxibatch_size=self.FLAGS.max_load_batches,
                                                 skip_empty=True)
        
        self.evaluate()
        for epoch_idx in range(self.FLAGS.max_epochs):
            self.log.info('------------------')
            if self.global_epoch_step.eval() >= self.FLAGS.max_epochs:
                self.log.info('Training is complete. Reach the max epoch number: %d ', self.global_epoch_step.eval())
                break
   
            for train_seq_mono, train_seq in zip(mono_train_set, self.train_set):
                start_time = time.time()
                batch_mono = self.prepare_batch_mono(train_seq_mono)
                update_info_mono = self.train_batch_reconstruction(*batch_mono)
                
                batch = self.prepare_batch(*train_seq)
                update_info = self.train_batch(*batch)
                
                self.print_log(update_info, update_info_mono, start_time)
                
                self.evaluate()
                self.save_model()

            #------------------------------------------------------------------
            #------------------------------------------------------------------
            # Increase the epoch index of the model
            self.log.info('Epoch %d done at %s', 
                          self.global_epoch_step.eval(), 
                          tools.get_time_now())
            self.global_epoch_step_op.eval()
            
    def print_log(self, update_info, update_info_mono, start_time):
        cost = update_info[0]
        cost_reconstruction = update_info[1]
        cost_alignment = update_info[2]
        cost_KL = update_info[3]
        
        if self.global_step.eval() % self.FLAGS.display_freq == 0:
            self.log.info('Epoch %d , Step %d , Cost: %.5f (T: %.5f) in %ds at %s', 
                          self.global_epoch_step.eval(),
                          self.global_step.eval(),
                          update_info_mono,update_info_mono,
                          time.time() - start_time,
                          tools.get_time_now())
            
            self.log.info('Epoch %d , Step %d , Cost: %.5f (T: %.5f, A: %.5f, KL: %.5f) in %ds at %s', 
                          self.global_epoch_step.eval(),
                          self.global_step.eval(),
                          cost,cost_reconstruction,cost_alignment, cost_KL,
                          time.time() - start_time,
                          tools.get_time_now())
            
            self.start_time = time.time()
            
    def train_batch_reconstruction(self,
                  targets,
                  targets_null,
                  target_lengths):
        
        input_feed = {}
        input_feed[self.keep_prob.name] = self.FLAGS.keep_prob
        input_feed[self.targets.name] = targets
        input_feed[self.target_lengths.name] = target_lengths
        
        input_feed[self.targets_null.name] = targets_null
        
        
        output_feed = [self.updates_reconstruction,
                       self.cost_reconstruction_expectation]
                       
        outputs = self.sess.run(output_feed, input_feed)
        
        return outputs[1]
        
    
    def prepare_batch_mono(self,seqs_y):
        y_lengths = np.array([len(s) for s in seqs_y])
        y_lengths_max = np.max(y_lengths)
        
        self.batch_size = len(seqs_y)
        self.update_freq = len(seqs_y)
                     
        y_null = [self.FLAGS.target_vocabulary_size - 1]
        
        word_y = np.ones((self.batch_size,
                     y_lengths_max),
                    dtype=self.int_dtype) * tools.end_token
        
        word_y_lengths = np.ones((self.batch_size),
                                    dtype=self.int_dtype)
                
            
        for idx, s_y in enumerate(seqs_y):
            word_y[idx, :y_lengths[idx]] = s_y
            word_y_lengths[idx] = y_lengths[idx]

        return word_y, y_null, word_y_lengths