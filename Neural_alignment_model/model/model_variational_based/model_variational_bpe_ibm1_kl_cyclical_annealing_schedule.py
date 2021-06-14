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
        
    def build_model(self):
        
        self.get_reference_BPE_idx()
        
        self.build_optimizer()
        self.build_initializer()
        self.build_word_model_placeholders()
        self.initialize_transition_parameter_nn()
        
        self.build_variational_encoder()
        self.build_reconstruction()
        self.build_alignment()
        
        self.build_update()
        
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
        
    def evaluate(self):
        if self.FLAGS.valid_freq!= 0 and self.global_step.eval() % self.FLAGS.valid_freq == 0:
            self.log.info('------------------')
            self.evaluate_alignment()
            self.evaluate_reconstruction()
            self.evaluate_kl()
            self.log.info('------------------')
            
    def evaluate_kl(self):
        # Execute a validation step
        if self.FLAGS.valid_freq!= 0 and self.global_step.eval() % self.FLAGS.valid_freq == 0:
            
            alignment_cost_batch = []
            reconstruction_cost_batch = []
            kl_cost_batch = []
            
            for valid_seq in self.valid_set:
                batch = self.prepare_batch(*valid_seq)
                eval_info = self.get_kl(*batch)
                
                alignment_cost_batch.append(eval_info[0])
                reconstruction_cost_batch.append(eval_info[1])
                kl_cost_batch.append(eval_info[2])
                    
            alignment_cost = np.mean(alignment_cost_batch)
            reconstruction_cost = np.mean(reconstruction_cost_batch)
            kl_cost = np.mean(kl_cost_batch)
            
            self.log.info('VALIDATION: Epoch %d , Step %d , Alignment: %.5f, Reconstruction: %.5f, KL: %5f',
                          self.global_epoch_step.eval(),
                          self.global_step.eval() , 
                          alignment_cost, 
                          reconstruction_cost,kl_cost)
            
    def get_kl(self,
             sources,
             source_lengths,
             targets,
             targets_null,
             target_lengths):
        
        target_length = target_lengths[0]
        source_length = source_lengths[0]
        source = sources[0][:source_length]
        
        input_feed = {}
        input_feed[self.keep_prob.name] = 1.0
        input_feed[self.sources.name] = sources
        input_feed[self.targets.name] = targets
        input_feed[self.source_lengths.name] = source_lengths
        input_feed[self.target_lengths.name] = target_lengths
        
        input_feed[self.targets_null.name] = targets_null
        
        output_feed = [self.cost_alignment_expectation,
                       self.cost_reconstruction_expectation,
                       self.KL_divergence_eval]
        
        #----------------------------------------------------------------------
        outputs = self.sess.run(output_feed, input_feed)
        
        return outputs[0], outputs[1], outputs[2]
    
    def train_batch(self,
                  sources,
                  source_lengths,
                  targets,
                  targets_null,
                  target_lengths):
        
        if self.global_step.eval() % self.FLAGS.alpha_KL_divergence_freq == 0.:
            if self.annealing_KL_divergence.eval() < 1.:
                self.sess.run(tf.assign(self.annealing_KL_divergence,
                                        self.annealing_KL_divergence +
                                        tf.cast(10e-3, dtype=self.tf_float_dtype)))
            else:
                self.sess.run(tf.assign(self.annealing_KL_divergence,
                                        tf.cast(1., dtype=self.tf_float_dtype)))
                
        if self.global_step.eval() % self.FLAGS.alpha_KL_divergence_freq_0 == 0.:
            self.sess.run(tf.assign(self.annealing_KL_divergence,
                                        tf.cast(0., dtype=self.tf_float_dtype)))
            
        print(self.annealing_KL_divergence.eval())
        
        input_feed = {}
        input_feed[self.keep_prob.name] = self.FLAGS.keep_prob
        input_feed[self.sources.name] = sources
        input_feed[self.targets.name] = targets
        input_feed[self.source_lengths.name] = source_lengths
        input_feed[self.target_lengths.name] = target_lengths
        
        input_feed[self.targets_null.name] = targets_null
        
        output_feed = [self.updates,
                       self.cost,
                       self.cost_reconstruction_expectation,
                       self.cost_alignment_expectation,
                       self.cost_KL_divergence
                       ]
                       
        outputs = self.sess.run(output_feed, input_feed)
        
        return outputs[1], outputs[2], outputs[3], outputs[4]
    