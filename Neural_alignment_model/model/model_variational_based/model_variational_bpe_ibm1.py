#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, time, math
import numpy as np
import tensorflow as tf
import framework.tools as tools

from model.model_variational_based.model_variational_bpe import Model

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
        
# =============================================================================
# 
# =============================================================================
        
    def get_transition_evaluation_nn(self, sentence_length):
        transition_matrix = np.ones((2*sentence_length, 2*sentence_length),dtype=self.float_dtype)
    
        transition_matrix_unit_null = np.zeros((sentence_length, sentence_length), dtype=self.float_dtype)
        np.fill_diagonal(transition_matrix_unit_null, self.float_dtype(1.))
            
        transition_matrix[:sentence_length, sentence_length: ] = transition_matrix_unit_null 
        transition_matrix[sentence_length:] = transition_matrix[:sentence_length]
          
        initial_transition = np.zeros((2*sentence_length), dtype=self.float_dtype)
        initial_transition[:sentence_length] = self.float_dtype(1.)
        initial_transition = initial_transition/ np.sum(initial_transition)

        return initial_transition, transition_matrix

# =============================================================================
# 
# =============================================================================

    def build_alignment(self):
        
        self.emission = self.get_emission(self.y_hidden, self.y_hidden_null)
        
        emission = self.get_emission(self.y_hidden_variable, self.y_hidden_variable_null)
                
        target_mask = tf.cast(tf.sequence_mask(self.target_lengths,tf.shape(self.targets)[1]), self.tf_float_dtype)
        target_mask = tf.concat([target_mask, target_mask], axis=1)
        transition = tf.div(target_mask, tf.cast(tf.expand_dims(self.target_lengths, -1), self.tf_float_dtype))
        transition = tf.expand_dims(transition, 1)
        transition = tf.tile(transition, [1, tf.shape(self.sources)[1], 1])
        
        alignment_expectation_prob = tf.matmul(transition, emission)
        
        alignment_expectation_log = tf.log(alignment_expectation_prob)
        
        alignment_expectation = alignment_expectation_log * tf.eye(num_rows=tf.shape(alignment_expectation_log)[1], 
                                                                   num_columns=tf.shape(alignment_expectation_log)[2], 
                                                                   batch_shape=[tf.shape(alignment_expectation_log)[0]],
                                                                   dtype= self.tf_float_dtype)
                
        alignment_expectation = tf.reduce_sum(alignment_expectation, axis=-1)
        
        source_mask = tf.cast(tf.sequence_mask(self.source_lengths,tf.shape(self.sources)[1]), self.tf_float_dtype)
        alignment_expectation = source_mask * alignment_expectation
        
        self.alignment_expectation = tf.reduce_sum(alignment_expectation, axis=-1)
        
    def eval(self,
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
        
        output_feed = [self.emission]
        #----------------------------------------------------------------------
        outputs = self.sess.run(output_feed, input_feed)
        
        emission_final = np.array(outputs[0][0])
        
        initial_transition, transition = self.get_transition_evaluation_nn(target_length)
        
        
        state_seq, likelihood_seq = self.viterbi(np.array(source),
                                        target_length*2,
                                        initial_transition,
                                        transition, 
                                        emission_final)
        
        return [state_seq], [likelihood_seq], [emission_final]