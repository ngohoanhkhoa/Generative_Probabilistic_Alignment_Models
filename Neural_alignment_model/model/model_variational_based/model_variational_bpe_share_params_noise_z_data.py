#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, time, math
import numpy as np
import tensorflow as tf
import framework.tools as tools

from tensorflow.python import pywrap_tensorflow

from model.model_variational_based.model_variational_bpe_share_params_noise_z import Model

class Model(Model):

    def __init__(self, FLAGS, session, log):
        
        self.global_step_reconstruction_y = tf.Variable(0, trainable=False, name='global_step_reconstruction_y')
        self.global_step_reconstruction_x = tf.Variable(0, trainable=False, name='global_step_reconstruction_x')
        
        super(Model, self).__init__(FLAGS, session, log)
    
# =============================================================================
# 
# =============================================================================
    def build_update(self):
        self.annealing_KL_divergence = tf.Variable(0., trainable=False, name='annealing_KL_divergence' ,
                                                   dtype=self.tf_float_dtype)
        
        # Cost for e -> y -> e, f
        e_y_reconstruction_expectation = tf.reduce_mean(self.e_y_reconstruction_expectation)
        f_y_alignment_expectation = tf.reduce_mean(self.f_y_alignment_expectation)
        y_KL_divergence = tf.reduce_mean(tf.reduce_sum(self.y_KL_divergence, axis=1))
        
        self.e_y_cost_reconstruction_expectation =  (-e_y_reconstruction_expectation)
        self.f_y_cost_alignment_expectation =  (-f_y_alignment_expectation)
        self.y_cost_KL_divergence = y_KL_divergence * self.annealing_KL_divergence
        self.y_cost_KL_divergence_eval = y_KL_divergence
        
        self.y_cost = (self.FLAGS.alpha_reconstruction_expectation * self.e_y_cost_reconstruction_expectation) \
        + (self.FLAGS.alpha_alignment_expectation * self.f_y_cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.y_cost_KL_divergence)
                
        
        # Cost for f -> x -> f, e
        f_x_reconstruction_expectation = tf.reduce_mean(self.f_x_reconstruction_expectation)
        e_x_alignment_expectation = tf.reduce_mean(self.e_x_alignment_expectation)
        x_KL_divergence = tf.reduce_mean(tf.reduce_sum(self.x_KL_divergence, axis=1))
        
        self.f_x_cost_reconstruction_expectation =  (-f_x_reconstruction_expectation)
        self.e_x_cost_alignment_expectation =  (-e_x_alignment_expectation)
        self.x_cost_KL_divergence = x_KL_divergence * self.annealing_KL_divergence
        self.x_cost_KL_divergence_eval = x_KL_divergence
               
        self.x_cost = (self.FLAGS.alpha_reconstruction_expectation * self.f_x_cost_reconstruction_expectation) \
        + (self.FLAGS.alpha_alignment_expectation * self.e_x_cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.x_cost_KL_divergence)
        
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        gradients = tf.gradients(self.y_cost + self.x_cost, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = self.opt.apply_gradients(zip(clip_gradients,trainable_params),
                                                         global_step=self.global_step)
        
        
        y_gradients_reconstruction = tf.gradients(self.e_y_cost_reconstruction_expectation, trainable_params)
        y_clip_gradients_reconstruction, _ = tf.clip_by_global_norm(y_gradients_reconstruction, self.max_gradient_norm)
        self.updates_reconstruction_y = self.opt.apply_gradients(zip(y_clip_gradients_reconstruction,trainable_params),
                                                         global_step=self.global_step_reconstruction_y)
        
        x_gradients_reconstruction = tf.gradients(self.f_x_cost_reconstruction_expectation, trainable_params)
        x_clip_gradients_reconstruction, _ = tf.clip_by_global_norm(x_gradients_reconstruction, self.max_gradient_norm)
        self.updates_reconstruction_x = self.opt.apply_gradients(zip(x_clip_gradients_reconstruction,trainable_params),
                                                         global_step=self.global_step_reconstruction_x)


    def update_with_pretrained_params(self):
        if self.FLAGS.model_parameter_loaded_from_checkpoint_IBM1 is not None:
            self.pretrained_param = pywrap_tensorflow.NewCheckpointReader(self.FLAGS.model_parameter_loaded_from_checkpoint_IBM1)
            var_to_shape_map = self.pretrained_param.get_variable_to_shape_map()

            assign_ops = []
            trainable_params = tf.trainable_variables()
            for v in trainable_params:
                for key in var_to_shape_map:
                    if v.name[:-2] == key:
                        self.log.info(key)
                        assign_ops.append(tf.assign(v,self.pretrained_param.get_tensor(str(v.name)[:-2])))
            self.sess.run(tf.group(*assign_ops))

            
    def train(self):
        
        self.update_with_pretrained_params()
        
        mono_train_set_y = tools.MonoTextBPEIterator(source=self.FLAGS.target_train_data_mono,
                                         n_words_source=self.FLAGS.target_vocabulary_size,
                                         batch_size=self.FLAGS.batch_size_mono,
                                         minlen=self.FLAGS.min_seq_length,
                                         shuffle_each_epoch=self.FLAGS.shuffle_each_epoch,
                                         sort_by_length=self.FLAGS.sort_by_length,
                                         maxibatch_size=self.FLAGS.max_load_batches,
                                         skip_empty=True)
        
        mono_train_set_x = tools.MonoTextBPEIterator(source=self.FLAGS.source_train_data_mono,
                                                 n_words_source=self.FLAGS.source_vocabulary_size,
                                                 batch_size=self.FLAGS.batch_size_mono,
                                                 minlen=self.FLAGS.min_seq_length,
                                                 shuffle_each_epoch=self.FLAGS.shuffle_each_epoch,
                                                 sort_by_length=self.FLAGS.sort_by_length,
                                                 maxibatch_size=self.FLAGS.max_load_batches,
                                                 skip_empty=True)
            
        # Training loop
        self.log.info('TRAINING %s', tools.get_time_now())
        
        self.evaluate()
        for epoch_idx in range(self.FLAGS.max_epochs):
            self.log.info('------------------')
            if self.global_epoch_step.eval() >= self.FLAGS.max_epochs:
                self.log.info('Training is complete. Reach the max epoch number: %d ', self.global_epoch_step.eval())
                break
            
            for train_seq_mono_y, train_seq_mono_x, train_seq in zip(mono_train_set_y, mono_train_set_x, self.train_set):
                start_time = time.time()
                update_info_mono_y = 0
                update_info_mono_x = 0
                update_info = [0,0,0,0,0,0,0,0]
                if self.FLAGS.target_train_data_mono_use != 0:
                    batch_mono_y = self.prepare_batch_mono(train_seq_mono_y)
                    update_info_mono_y = self.train_batch_reconstruction_y(*batch_mono_y)
                
                if self.FLAGS.source_train_data_mono_use != 0:
                    batch_mono_x = self.prepare_batch_mono(train_seq_mono_x)
                    update_info_mono_x = self.train_batch_reconstruction_x(*batch_mono_x)
                
                if self.FLAGS.train_data_bi_use != 0:
                    batch = self.prepare_batch(*train_seq)
                    update_info = self.train_batch(*batch)
                
                self.print_log(update_info, [update_info_mono_y, update_info_mono_x], start_time)
                
                self.evaluate()
                self.save_model()
                
            

            #------------------------------------------------------------------
            #------------------------------------------------------------------
            # Increase the epoch index of the model
            self.log.info('Epoch %d done at %s', 
                          self.global_epoch_step.eval(), 
                          tools.get_time_now())
            self.global_epoch_step_op.eval()
            
    def save_model(self):
        # Save the model checkpoint
        global_step = self.global_step.eval()
        if self.FLAGS.target_train_data_mono_use != 0:
            global_step = self.global_step_reconstruction_y.eval()
                
        if self.FLAGS.source_train_data_mono_use != 0:
            global_step = self.global_step_reconstruction_x.eval()
                    
        if self.FLAGS.train_data_bi_use != 0:
            global_step = self.global_step.eval()
                    
        if self.FLAGS.save_freq != 0 and global_step % self.FLAGS.save_freq == 0:
            self.log.info('SAVE MODEL: %s' , self.checkpoint_path)
            self.save(self.checkpoint_path, global_step=self.global_step)
            
    def evaluate(self):
        # Save the model checkpoint
        global_step = self.global_step.eval()
        if self.FLAGS.target_train_data_mono_use != 0:
            global_step = self.global_step_reconstruction_y.eval()
                
        if self.FLAGS.source_train_data_mono_use != 0:
            global_step = self.global_step_reconstruction_x.eval()
                    
        if self.FLAGS.train_data_bi_use != 0:
            global_step = self.global_step.eval()
            
        if self.FLAGS.valid_freq!= 0 and global_step % self.FLAGS.valid_freq == 0:
            self.log.info('------------------')
            self.evaluate_alignment()
            self.evaluate_reconstruction()
            self.evaluate_kl()
            self.log.info('------------------')
            
    def print_log(self, update_info, update_info_mono, start_time):
        y_cost = update_info[0]
        e_y_cost_reconstruction = update_info[1]
        f_y_cost_alignment = update_info[2]
        y_cost_KL = update_info[3]
        
        x_cost = update_info[4]
        f_x_cost_reconstruction = update_info[5]
        e_x_cost_alignment = update_info[6]
        x_cost_KL = update_info[7]
        
        update_info_mono_y = update_info_mono[0]
        update_info_mono_x = update_info_mono[1]
        
        global_step = self.global_step.eval()
        if self.FLAGS.target_train_data_mono_use != 0:
            global_step = self.global_step_reconstruction_y.eval()
                
        if self.FLAGS.source_train_data_mono_use != 0:
            global_step = self.global_step_reconstruction_x.eval()
                    
        if self.FLAGS.train_data_bi_use != 0:
            global_step = self.global_step.eval()
        
        if global_step % self.FLAGS.display_freq == 0:
            
            self.log.info('Epoch %d , Step %d , Cost e -> y: %.5f (R: %.5f, A: %.5f, KL: %.5f, R1: %.5f) in %ds at %s', 
                          self.global_epoch_step.eval(),
                          global_step,
                          y_cost,e_y_cost_reconstruction,f_y_cost_alignment, y_cost_KL, update_info_mono_y,
                          time.time() - self.start_time,
                          tools.get_time_now())
            
            self.log.info('Epoch %d , Step %d , Cost f -> x: %.5f (R: %.5f, A: %.5f, KL: %.5f, R1: %.5f) in %ds at %s', 
                          self.global_epoch_step.eval(),
                          global_step,
                          x_cost,f_x_cost_reconstruction,e_x_cost_alignment, x_cost_KL, update_info_mono_x,
                          time.time() - self.start_time,
                          tools.get_time_now())
            
            self.start_time = time.time()
            
    def train_batch_reconstruction_y(self,
                                      targets,
                                      targets_null,
                                      target_lengths,
                                      targets_noise,
                                      target_noise_lengths,
                                      target_mask_noise):
        
        input_feed = {}
        input_feed[self.keep_prob.name] = self.FLAGS.keep_prob
        input_feed[self.targets.name] = targets
        input_feed[self.target_lengths.name] = target_lengths
        
        input_feed[self.targets_null.name] = targets_null
        
        input_feed[self.targets_noise.name] = targets_noise
        input_feed[self.target_noise_lengths.name] = target_noise_lengths
        
        output_feed = [self.updates_reconstruction_y,
                       self.e_y_cost_reconstruction_expectation]
                       
        outputs = self.sess.run(output_feed, input_feed)
        
        return outputs[1]
    
    def train_batch_reconstruction_x(self,
                                      sources,
                                      sources_null,
                                      source_lengths,
                                      sources_noise,
                                      source_noise_lengths,
                                      source_mask_noise,):
        
        input_feed = {}
        input_feed[self.keep_prob.name] = self.FLAGS.keep_prob
        input_feed[self.sources.name] = sources
        input_feed[self.source_lengths.name] = source_lengths
        
        input_feed[self.sources_null.name] = sources_null
        
        input_feed[self.sources_noise.name] = sources_noise
        input_feed[self.source_noise_lengths.name] = source_noise_lengths
        
        output_feed = [self.updates_reconstruction_x,
                       self.f_x_cost_reconstruction_expectation]
                       
        outputs = self.sess.run(output_feed, input_feed)
        
        return outputs[1]

    def prepare_batch_mono(self,seqs_y):
        y_lengths = np.array([len(s) for s in seqs_y])
        y_lengths_max = np.max(y_lengths)
        
        self.batch_size_mono = len(seqs_y)
                     
        y_null = [self.FLAGS.target_vocabulary_size - 1]
        
        word_y = np.ones((self.batch_size_mono,
                     y_lengths_max),
                    dtype=self.int_dtype) * tools.end_token
        
        word_y_lengths = np.ones((self.batch_size_mono),
                                    dtype=self.int_dtype)
                
            
        for idx, s_y in enumerate(seqs_y):
            word_y[idx, :y_lengths[idx]] = s_y
            word_y_lengths[idx] = y_lengths[idx]
            
        word_y_noise, word_y_noise_lengths, mask_ys = self.add_noise(word_y, seqs_y, self.batch_size_mono)

        return word_y, y_null, word_y_lengths, word_y_noise, word_y_noise_lengths, mask_ys
    
    def train_batch(self,
             sources,
             sources_null,
             source_lengths,
             targets,
             targets_null,
             target_lengths,
             sources_noise,
             source_noise_lengths,
             source_mask_noise,
             targets_noise,
             target_noise_lengths,
             target_mask_noise):
        
        if self.global_step.eval() % self.FLAGS.alpha_KL_divergence_freq == 0:
            if self.annealing_KL_divergence.eval() <= 1.:
                self.sess.run(tf.assign(self.annealing_KL_divergence,
                                        self.annealing_KL_divergence +
                                        tf.cast(10e-3, dtype=self.tf_float_dtype)))
            else:
                self.sess.run(tf.assign(self.annealing_KL_divergence,
                                        tf.cast(1., dtype=self.tf_float_dtype)))
        
        input_feed = {}
        input_feed[self.keep_prob.name] = self.FLAGS.keep_prob
        input_feed[self.sources.name] = sources
        input_feed[self.targets.name] = targets
        input_feed[self.source_lengths.name] = source_lengths
        input_feed[self.target_lengths.name] = target_lengths
        
        input_feed[self.targets_null.name] = targets_null
        input_feed[self.sources_null.name] = sources_null
        
        input_feed[self.sources_noise.name] = sources
        input_feed[self.targets_noise.name] = targets
        
        input_feed[self.source_noise_lengths.name] = source_lengths
        input_feed[self.target_noise_lengths.name] = target_lengths
        
        output_feed = [self.updates,
                       self.y_cost,
                       self.e_y_cost_reconstruction_expectation,
                       self.f_y_cost_alignment_expectation,
                       self.y_cost_KL_divergence,
                       self.x_cost,
                       self.f_x_cost_reconstruction_expectation,
                       self.e_x_cost_alignment_expectation,
                       self.x_cost_KL_divergence
                       ]
        
        outputs = self.sess.run(output_feed, input_feed)
        
        return outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6], outputs[7], outputs[8]