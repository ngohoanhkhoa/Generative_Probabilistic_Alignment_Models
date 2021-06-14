#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import time
import framework.tools as tools

from model.model_hmm_based.model_hmm import Model

# =============================================================================
# Target Word
# Embedding: Feed forward
# Emission: Feed forward
# Transition: Discrete
# =============================================================================

class Model(Model):

    def __init__(self, FLAGS, session, log):

        super(Model, self).__init__(FLAGS, session, log)
        
    def build_model(self):
        self.initialize_transition_parameter_discrete()
        
        self.build_optimizer()
        self.build_initializer()
        
        self.build_word_model_placeholders()
        self.build_model_transition_discrete_placeholders()
        
        self.build_emission()
        self.build_forward_backward_discrete()
        
        self.build_update_emission()

# =============================================================================
#             
# =============================================================================
        
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
        
    def build_model_transition_discrete_placeholders(self):
        # Size not fixed for training
        self.transition = tf.placeholder(dtype=self.tf_float_dtype,
                                         shape=(None,
                                                None,
                                                None),
                                         name='transition')
                                      
        self.initial_transition = tf.placeholder(dtype=self.tf_float_dtype,
                                                 shape=(None,
                                                        None), 
                                                 name='initial_transition')
                                                 
    def assign_variable_training_transition(self):
        self.update_jump_width_probability_set()
    
# =============================================================================
# 
# =============================================================================
                                                 
    def get_emission(self):
        # NN parameters Null
        target_embedding= tf.get_variable(name='targetEmbedding',
                                                shape=[self.FLAGS.target_vocabulary_size,
                                                       self.FLAGS.embedding_size],
                                                initializer=self.initializer,
                                                dtype=self.tf_float_dtype)
                                                
        target_embedded_null = tf.nn.embedding_lookup(params=target_embedding,
                                                      ids=self.targets_null)
        
        target_embedded_word = tf.nn.embedding_lookup(params=target_embedding,
                                                      ids=self.targets)
        
        nnLayer1 = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                    activation=tf.nn.tanh,
                                    use_bias=True,
                                    name='nnLayer1')
        
        target_state_null = nnLayer1(target_embedded_null)
        target_state = nnLayer1(target_embedded_word)
        
        nnLayer2 = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                    activation=tf.nn.tanh,
                                    use_bias=True,
                                    name='nnLayer2')
        
        target_state_null = nnLayer2(target_state_null)
        target_state = nnLayer2(target_state)
        
        #----------------------------------------------------------------------

        target_state = tf.nn.dropout(x=target_state,
                                       keep_prob=self.keep_prob,
                                       name='nnDropoutNonNull')
                                       
        target_state_null = tf.nn.dropout(x=target_state_null,
                                       keep_prob=self.keep_prob,
                                       name='nnDropoutNull')
        
        nnLayerVocabulary = tf.layers.Dense(units=self.FLAGS.source_vocabulary_size,
                                 name='nnLayerVocabulary')
        
        target_state = nnLayerVocabulary(target_state)
        target_state_null = nnLayerVocabulary(target_state_null)
        
        target_state = tf.nn.softmax(target_state)
        self.target_state_null = tf.nn.softmax(target_state_null)
        
        input_scan = (target_state, self.sources)
        
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
            
# =============================================================================
# 
# =============================================================================
        
    def prepare_batch(self, seqs_x, seqs_y):
        x_lengths = np.array([len(s) for s in seqs_x])
        y_lengths = np.array([len(s) for s in seqs_y])
        x_lengths_max = np.max(x_lengths)
        y_lengths_max = np.max(y_lengths)
        
        self.batch_size = len(seqs_x)
        self.update_freq = len(seqs_x)
                     
        y_null = [tools.null_token]
        
        word_x = np.ones((self.batch_size,
                     x_lengths_max),
                    dtype=self.int_dtype) * tools.end_token
                
        word_x_lengths = np.ones((self.batch_size),
                                    dtype=self.int_dtype)
        
        word_y = np.ones((self.batch_size,
                     y_lengths_max),
                    dtype=self.int_dtype) * tools.end_token
        
        word_y_lengths = np.ones((self.batch_size),
                                    dtype=self.int_dtype)
                
        for idx, s_x in enumerate(seqs_x):
            word_x[idx, :x_lengths[idx]] = s_x
            word_x_lengths[idx] = x_lengths[idx]
            
        for idx, s_y in enumerate(seqs_y):
            word_y[idx, :y_lengths[idx]] = s_y
            word_y_lengths[idx] = y_lengths[idx]

        return word_x, word_x_lengths, word_y, y_null, word_y_lengths
    
    def train_batch(self,
                  sources,
                  source_lengths,
                  targets,
                  targets_null,
                  target_lengths):
                        
        initial_transitions, transitions \
            = self.prepare_transition_matrix_discrete(target_lengths, np.shape(targets)[1])
        
        input_feed = {}
        input_feed[self.keep_prob.name] = self.FLAGS.keep_prob
        input_feed[self.sources.name] = sources
        input_feed[self.targets.name] = targets
        input_feed[self.source_lengths.name] = source_lengths
        input_feed[self.target_lengths.name] = target_lengths
        
        input_feed[self.targets_null.name] = targets_null
        
        input_feed[self.transition.name] = transitions
        input_feed[self.initial_transition.name] = initial_transitions
        
        output_feed = [self.updates_emission,
                       self.cost_emission,
                       self.transition_posteriors_expectation
                       ]
                       
        outputs = self.sess.run(output_feed, input_feed)
        
        for transition, target_length in zip(outputs[2], target_lengths):
            self.get_transition_posterior_for_each_batch(transition, target_length, np.shape(targets)[1])

        cost = outputs[1]
        
        return cost
    
    
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
        
        output_feed = [self.emission_prob]
        #----------------------------------------------------------------------
        outputs = self.sess.run(output_feed, input_feed)
        
        emission_final = np.array(outputs[0][0])
        
        initial_transition, transition = self.get_transition_evaluation_discrete(target_length)

        state_seq, likelihood_seq = self.viterbi(np.array(source),
                                        target_length*2, 
                                        initial_transition, 
                                        transition, 
                                        emission_final)
        
        return [state_seq], [likelihood_seq], [emission_final]
                    
#==============================================================================
# 
#==============================================================================
    
    def print_model_info(self):
        self.log.info('Source word vocabulary size: %s', self.FLAGS.source_vocabulary_size)
        self.log.info('Target word vocabulary size: %s', self.FLAGS.target_vocabulary_size)
        self.log.info('Word hidden units: %s', self.FLAGS.hidden_units)
        self.log.info('Word embedding size: %s', self.FLAGS.embedding_size)
