#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, time, math
import numpy as np
import tensorflow as tf
import framework.tools as tools
from tensorflow.python import pywrap_tensorflow


from model.model_variational_based.model_variational_bpe_share_params import Model

class Model(Model):

    def __init__(self, FLAGS, session, log):

        super(Model, self).__init__(FLAGS, session, log)
        
    def build_model(self):
        
        self.best_AER_f_e = 100.
        self.best_AER_e_f = 100.
        
        
        self.get_reference_BPE_idx()
        
        self.build_optimizer()
        self.build_initializer()
        self.build_word_model_placeholders()
        self.initialize_transition_parameter_nn()
        
        self.build_variational_encoder()
        self.build_decoder()
        self.build_reconstruction()
        self.build_alignment()
        
        self.build_update()
        
    def build_update(self):
        self.annealing_KL_divergence = tf.Variable(0., trainable=False, name='annealing_KL_divergence' ,
                                                   dtype=self.tf_float_dtype)
        
        # Cost for e -> y -> e, f
        e_y_reconstruction_expectation = tf.reduce_mean(tf.reduce_sum(self.e_y_reconstruction_expectation_log, axis=1))
        f_y_alignment_expectation = tf.reduce_mean(self.f_y_alignment_expectation)
        y_KL_divergence = tf.reduce_mean(tf.reduce_sum(self.y_KL_divergence, axis=1))
        
        self.e_y_cost_reconstruction_expectation =  (-e_y_reconstruction_expectation)
        self.f_y_cost_alignment_expectation =  (-f_y_alignment_expectation)
        self.y_cost_KL_divergence = y_KL_divergence * self.annealing_KL_divergence
        
        self.y_cost = (self.FLAGS.alpha_reconstruction_expectation * self.e_y_cost_reconstruction_expectation) \
        + (self.FLAGS.alpha_alignment_expectation * self.f_y_cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.y_cost_KL_divergence)
        
        self.e_y_cost_reconstruction_expectation_eval =  self.e_y_cost_reconstruction_expectation
        self.f_y_cost_alignment_expectation_eval =  - tf.reduce_mean(self.f_y_alignment_expectation_eval)
        self.y_cost_KL_divergence_eval = y_KL_divergence
        
        # Cost for f -> x -> f, e
        f_x_reconstruction_expectation = tf.reduce_mean(tf.reduce_sum(self.f_x_reconstruction_expectation_log, axis=1))
        e_x_alignment_expectation = tf.reduce_mean(self.e_x_alignment_expectation)
        x_KL_divergence = tf.reduce_mean(tf.reduce_sum(self.x_KL_divergence, axis=1))
        
        self.f_x_cost_reconstruction_expectation =  (-f_x_reconstruction_expectation)
        self.e_x_cost_alignment_expectation =  (-e_x_alignment_expectation)
        self.x_cost_KL_divergence = x_KL_divergence * self.annealing_KL_divergence
    
        
        self.x_cost = (self.FLAGS.alpha_reconstruction_expectation * self.f_x_cost_reconstruction_expectation) \
        + (self.FLAGS.alpha_alignment_expectation * self.e_x_cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.x_cost_KL_divergence)
        
        self.f_x_cost_reconstruction_expectation_eval = self.f_x_cost_reconstruction_expectation
        self.e_x_cost_alignment_expectation_eval = - tf.reduce_mean(self.e_x_alignment_expectation_eval)
        self.x_cost_KL_divergence_eval = x_KL_divergence
        
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        gradients = tf.gradients(self.y_cost + self.x_cost, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = self.opt.apply_gradients(zip(clip_gradients,trainable_params),
                                                         global_step=self.global_step)
        
    def build_word_model_placeholders(self):
        
        self.sources = tf.placeholder(dtype=self.tf_int_dtype,
                                      shape=(None,None),
                                      name='sources')
        
        self.targets = tf.placeholder(dtype=self.tf_int_dtype,
                                      shape=(None,None),
                                      name='targets')
                                      
        self.source_lengths = tf.placeholder(dtype=self.tf_int_dtype, 
                                             shape=(None),
                                             name='source_lengths')
                                             
        self.target_lengths = tf.placeholder(dtype=self.tf_int_dtype,
                                             shape=(None),
                                             name='target_lengths')
                                      
        self.targets_null = tf.placeholder(dtype=self.tf_int_dtype,
                                          shape=(1),
                                          name='targets_null')
        
        self.sources_null = tf.placeholder(dtype=self.tf_int_dtype,
                                          shape=(1),
                                          name='sources_null')
        
        self.alignment_matrix_target_source = tf.placeholder(dtype=self.tf_float_dtype,
                                          shape=(None, None,None),
                                          name='alignment_matrix_target_source')
        
        self.alignment_matrix_source_target = tf.placeholder(dtype=self.tf_float_dtype,
                                          shape=(None, None,None),
                                          name='alignment_matrix_source_target')
        
    
    def build_alignment(self):
        with tf.variable_scope('alignment', reuse=tf.AUTO_REUSE):
        
            self.f_y_emission = self.get_emission(self.f_y_prob, self.f_y_prob_null, self.sources)
            self.f_y_alignment, self.f_y_alignment_expectation_eval = self.get_alignment(input_prob_softmax= self.f_y_prob, 
                                                    input_prob_softmax_null= self.f_y_prob_null,
                                                    input_sentences= self.targets,
                                                    output_sentences= self.sources,
                                                    input_lengths= self.target_lengths,
                                                    output_lengths= self.source_lengths)
            
            self.f_y_alignment_expectation = self.get_alignment_expectation(input_prob_softmax= self.f_y_hidden_prob, 
                                                                            input_prob_softmax_null= self.f_y_hidden_prob_null,
                                                                            input_sentences= self.targets,
                                                                            output_sentences= self.sources,
                                                                            input_lengths= self.target_lengths,
                                                                            output_lengths= self.source_lengths,
                                                                            mask_supervised=self.alignment_matrix_source_target)
            
            
            self.e_x_emission = self.get_emission(self.e_x_prob, self.e_x_prob_null, self.targets)
            self.e_x_alignment, self.e_x_alignment_expectation_eval = self.get_alignment(input_prob_softmax= self.e_x_prob, 
                                                                            input_prob_softmax_null= self.e_x_prob_null,
                                                                            input_sentences= self.sources,
                                                                            output_sentences= self.targets,
                                                                            input_lengths= self.source_lengths,
                                                                            output_lengths= self.target_lengths)
            
            self.e_x_alignment_expectation = self.get_alignment_expectation(input_prob_softmax= self.e_x_hidden_prob, 
                                                                            input_prob_softmax_null= self.e_x_hidden_prob_null,
                                                                            input_sentences= self.sources,
                                                                            output_sentences= self.targets,
                                                                            input_lengths= self.source_lengths,
                                                                            output_lengths= self.target_lengths,
                                                                            mask_supervised=self.alignment_matrix_target_source)
        
    def forward_backward_batch(self, emission_prob_expectation, transition_prob_expectation):
        
        def forward_function(last_forward, yi):
            tmp = tf.multiply(tf.matmul(last_forward, transition_prob_expectation), yi)
            return tmp / tf.reduce_sum(tmp, axis=2, keep_dims=True)
        
        def backward_function(last_backward, yi):
            # Combine transition matrix with observations
            combined = tf.multiply(tf.expand_dims(transition_prob_expectation, 1), tf.expand_dims(yi, 2))
            tmp = tf.reduce_sum(tf.multiply(combined, tf.expand_dims(last_backward, 2)), axis=3)
            return tmp / tf.reduce_sum(tmp, axis=2, keep_dims=True)

        emission_prob_expectation_shape = tf.shape(emission_prob_expectation)
        # Shape:
        batch_size_ = emission_prob_expectation_shape[0]
        target_size_ = emission_prob_expectation_shape[1]
        source_size_ = emission_prob_expectation_shape[2]
        
        # obs_seq from self.emission_prob_expectation:
        #(batch_size_,target_size_,source_size_) 
        # > (source_size_,batch_size_,target_size_,target_size_)
        obs_seq = tf.transpose(tf.tile(tf.expand_dims(emission_prob_expectation,axis=1), 
                                    [1,target_size_,1,1 ]),(3,0,1,2))
        
        # Calculate FORWARD
        # forward: (source_size_,batch_size_,target_size_,target_size_)
        forward = tf.scan(forward_function, obs_seq)
        # forward: (source_size_,batch_size_,1,target_size_)
        forward = tf.gather(forward,[0], axis=2)
        
        # Calculate BACKWARD
        # final_: (1,batch_size_,target_size_,target_size_)
        final_transition = tf.ones((1,batch_size_,target_size_,target_size_), 
                                   dtype=self.tf_float_dtype)
        
        # obs_seq_b: (0:source_size_+1,batch_size_,target_size_,target_size_)
        obs_seq_b = tf.concat([obs_seq,final_transition], axis=0)
        # backward: 
        # (0:source_size_+1,batch_size_,target_size_,target_size_) 
        # > (source_size_+1:0,batch_size_,target_size_,target_size_)
        backward = tf.scan(backward_function,tf.reverse(obs_seq_b, [0]))
        # backward:
        # (source_size_+1:0,batch_size_,target_size_,target_size_)
        # > (0:source_size_+1,batch_size_,target_size_,target_size_)
        # > (1:source_size_+1,batch_size_,target_size_,target_size_)
        # > (source_size_,batch_size_,1,target_size_)
        backward = tf.gather(tf.gather(tf.reverse(backward, [0]), tf.range(1, source_size_+1), axis=0),[0], axis=2)

        # Calculate EMISSION POSTERIOR
        # emission_posterior: (source_size_,batch_size_,1,target_size_)
        emission_posterior = tf.multiply(forward, backward)
        emission_posterior_sum = tf.reduce_sum(emission_posterior, axis=3, keep_dims=True)
            
        emission_posterior_sum_zero_replaced_by_one = tf.where(tf.is_inf(tf.log(emission_posterior_sum)),
                                                               tf.ones_like(emission_posterior_sum), emission_posterior_sum)
        emission_posterior = emission_posterior / emission_posterior_sum_zero_replaced_by_one
        
        # emission_posterior: 
        # (source_size_,batch_size_,1,target_size_)
        # > (source_size_,batch_size_,target_size_)
        # > (batch_size_,target_size_,source_size_)
        emission_posterior = tf.transpose(tf.squeeze(emission_posterior, axis=2), (1,2,0))
        
        return emission_posterior
    
    
    def get_alignment_expectation(self,
                                  input_prob_softmax, 
                                  input_prob_softmax_null, 
                                  input_sentences, 
                                  output_sentences,
                                  input_lengths,
                                  output_lengths,
                                  mask_supervised):
        
        emission = self.get_emission(input_prob_softmax, input_prob_softmax_null, output_sentences)

                
        input_mask = tf.cast(tf.sequence_mask(input_lengths,tf.shape(input_sentences)[1]), self.tf_float_dtype)
        input_mask = tf.concat([input_mask, input_mask], axis=1)
        transition = tf.div(input_mask, tf.cast(tf.expand_dims(input_lengths, -1), self.tf_float_dtype))
        transition = tf.expand_dims(transition, 1)
        transition = tf.tile(transition, [1, tf.shape(output_sentences)[1], 1])
        
        emission = emission * mask_supervised
        alignment_expectation_prob = tf.matmul(transition, emission)
        
        alignment_expectation_log = tf.log(alignment_expectation_prob)
        
        alignment_expectation = alignment_expectation_log * tf.eye(num_rows=tf.shape(alignment_expectation_log)[1], 
                                                                   num_columns=tf.shape(alignment_expectation_log)[2], 
                                                                   batch_shape=[tf.shape(alignment_expectation_log)[0]],
                                                                   dtype= self.tf_float_dtype)
                
        alignment_expectation = tf.reduce_sum(alignment_expectation, axis=-1)
        
        output_mask = tf.cast(tf.sequence_mask(output_lengths,tf.shape(output_sentences)[1]), self.tf_float_dtype)
        alignment_expectation = output_mask * alignment_expectation
        
        alignment_expectation = tf.reduce_sum(alignment_expectation, axis=-1)
        
        return alignment_expectation
            
    def get_alignment(self,
                      input_prob_softmax, 
                      input_prob_softmax_null, 
                      input_sentences, 
                      output_sentences,
                      input_lengths,
                      output_lengths):
        
        emission = self.get_emission(input_prob_softmax, input_prob_softmax_null, output_sentences)

        input_mask = tf.cast(tf.sequence_mask(input_lengths,tf.shape(input_sentences)[1]), self.tf_float_dtype)
        input_mask = tf.concat([input_mask, input_mask], axis=1)
        transition = tf.div(input_mask, tf.cast(tf.expand_dims(input_lengths, -1), self.tf_float_dtype))
        transition = tf.expand_dims(transition, 1)
        transition = tf.tile(transition, [1, tf.shape(output_sentences)[1], 1])
        
        alignment_expectation_prob_only = tf.multiply(tf.transpose(transition, (0,2,1)), emission)
        
        alignment_expectation_prob = tf.matmul(transition, emission)
        
        alignment_expectation_log = tf.log(alignment_expectation_prob)
        
        alignment_expectation = alignment_expectation_log * tf.eye(num_rows=tf.shape(alignment_expectation_log)[1], 
                                                                   num_columns=tf.shape(alignment_expectation_log)[2], 
                                                                   batch_shape=[tf.shape(alignment_expectation_log)[0]],
                                                                   dtype= self.tf_float_dtype)
                
        alignment_expectation = tf.reduce_sum(alignment_expectation, axis=-1)
        
        output_mask = tf.cast(tf.sequence_mask(output_lengths,tf.shape(output_sentences)[1]), self.tf_float_dtype)
        alignment_expectation = output_mask * alignment_expectation
        
        alignment_expectation = tf.reduce_sum(alignment_expectation, axis=-1)
        
        return alignment_expectation_prob_only, alignment_expectation
    
    
    def get_alignment_supervised_from_opposite_direction(self, alignment_matrix):
        alignment_matrix_non_null = alignment_matrix[:, :int(np.shape(alignment_matrix)[1]/2), :]
        alignment_shape = np.shape(alignment_matrix_non_null)
        
        alignment_matrix_non_null_output = np.ones(alignment_shape, dtype=self.float_dtype)
        for idx_s in range(alignment_shape[0]):
            for idx_obs in range(alignment_shape[1]):
                is_higher = False
                for idx_state in range(alignment_shape[2]):
                    if alignment_matrix_non_null[idx_s, idx_obs, idx_state] >= self.FLAGS.supervised_mask_threshold:
                        is_higher = True
                        break
                if is_higher:
                    for idx_state in range(alignment_shape[2]):
                        if alignment_matrix_non_null[idx_s, idx_obs, idx_state] >= self.FLAGS.supervised_mask_threshold:
                            alignment_matrix_non_null_output[idx_s, idx_obs, idx_state] = 1.
                        else:
                            alignment_matrix_non_null_output[idx_s, idx_obs, idx_state] = 0.
                            
        alignment_matrix_non_null_transposed_output = np.transpose(alignment_matrix_non_null_output, axes=[0,2,1])
        alignment_matrix_output = np.ones((alignment_shape[0], alignment_shape[2]*2, alignment_shape[1]),dtype=self.float_dtype)
        alignment_matrix_output[:, :alignment_shape[2], :] = alignment_matrix_non_null_transposed_output
        
        return alignment_matrix_output


    def train_batch(self,
                  sources,
                  sources_null,
                  source_lengths,
                  targets,
                  targets_null,
                  target_lengths):
        
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
        
        output_feed = [self.f_y_alignment,
                       self.e_x_alignment]
                       
        outputs = self.sess.run(output_feed, input_feed)

        
        f_y_alignment = np.array(outputs[0])
        e_x_alignment = np.array(outputs[1])
        
        e_x_alignment_mask_from_f_y = self.get_alignment_supervised_from_opposite_direction(f_y_alignment)
        f_y_alignment_mask_from_e_x = self.get_alignment_supervised_from_opposite_direction(e_x_alignment)
                        
        
        input_feed[self.alignment_matrix_target_source.name] = e_x_alignment_mask_from_f_y 
        input_feed[self.alignment_matrix_source_target.name] = f_y_alignment_mask_from_e_x

        
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
    
    def get_kl(self,
             sources,
             sources_null,
             source_lengths,
             targets,
             targets_null,
             target_lengths):
        
        input_feed = {}
        input_feed[self.keep_prob.name] = 1.0
        input_feed[self.sources.name] = sources
        input_feed[self.targets.name] = targets
        input_feed[self.source_lengths.name] = source_lengths
        input_feed[self.target_lengths.name] = target_lengths
        
        input_feed[self.targets_null.name] = targets_null
        input_feed[self.sources_null.name] = sources_null
        
        output_feed = [self.f_y_cost_alignment_expectation_eval,
                       self.e_y_cost_reconstruction_expectation_eval,
                       self.y_cost_KL_divergence_eval,
                       self.e_x_cost_alignment_expectation_eval,
                       self.f_x_cost_reconstruction_expectation_eval,
                       self.x_cost_KL_divergence_eval]
        
        #----------------------------------------------------------------------
        outputs = self.sess.run(output_feed, input_feed)
        
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]