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
        
    def get_emission(self, y_hidden_variable, y_hidden_variable_null):
        with tf.variable_scope('emission', reuse=tf.AUTO_REUSE):
            
            nnLayer_emission = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                        activation=tf.nn.tanh,
                                        use_bias=True,
                                        name='nnLayerEmission')
                                               
            target_state = nnLayer_emission(y_hidden_variable)
            target_state_null = nnLayer_emission(y_hidden_variable_null)
            
#            target_state = tf.nn.dropout(x=target_state,
#                                           keep_prob=self.keep_prob,
#                                           name='nnDropoutNonNull')
#                                           
#            target_state_null = tf.nn.dropout(x=target_state_null,
#                                           keep_prob=self.keep_prob,
#                                           name='nnDropoutNull')
            
            nnLayerSourceVocabulary = tf.layers.Dense(units=self.FLAGS.source_vocabulary_size,
                                                      name='nnLayerSourceVocabulary')
            
            emission_value = nnLayerSourceVocabulary(target_state)
            target_state_null = nnLayerSourceVocabulary(target_state_null)
            
            emission_prob_softmax = tf.nn.softmax(emission_value)
            self.target_state_null = tf.nn.softmax(target_state_null)
            
            input_scan = (emission_prob_softmax, self.sources)
            
            def get_source(value):
                target_state = value[0]
                sources = value[1]
                
                target_state = tf.gather(target_state, sources, axis=-1)
                target_state_null = tf.gather(self.target_state_null, sources, axis=-1)
                
                target_state_null = tf.tile(target_state_null, [tf.shape(target_state)[0], 1])
                
                emission_prob = tf.concat([target_state, target_state_null], axis=0)
                
                return emission_prob, 0
        
            emission_prob, _ = tf.map_fn(get_source,input_scan, dtype=(self.tf_float_dtype,
                                                                           tf.int32))
            
        return emission_prob
    
    def get_target(self, target_state):
        with tf.variable_scope('reconstruction', reuse=tf.AUTO_REUSE):
            target_value = tf.layers.Dense(units=self.FLAGS.target_vocabulary_size,
                                           use_bias=True,
                                           name='nnLayerTargetVocabulary')(target_state)
            
            target_value = tf.nn.dropout(x=target_value,
                                           keep_prob=self.keep_prob,
                                           name='nnDropoutNonNull')
            
            target_prob_softmax = tf.nn.softmax(target_value)
            
            target_predicted = tf.argmax(target_prob_softmax,axis=-1)
            
            input_scan = (target_prob_softmax, self.targets )
            def get_target_(value):
                target_prob = value[0]
                targets = value[1]
                
                target_prob = tf.gather(target_prob, targets, axis=-1)
                target_prob = target_prob * tf.eye(tf.shape(target_prob)[0], tf.shape(target_prob)[1], dtype= self.tf_float_dtype)
                target_prob = tf.reduce_sum(target_prob, axis=-1)
                
                return target_prob, 0
            
            target_prob, _ = tf.map_fn(get_target_,input_scan, dtype=(self.tf_float_dtype,
                                                                           tf.int32))
            
            
            target_mask = tf.sequence_mask(lengths=self.target_lengths, maxlen=tf.shape(target_prob)[1],
                                           dtype=self.tf_float_dtype, name='targetMask')
            
            reconstruction_expectation = target_prob * target_mask
            reconstruction_expectation_log = tf.log(target_prob) * target_mask
        
        return reconstruction_expectation, reconstruction_expectation_log, target_predicted