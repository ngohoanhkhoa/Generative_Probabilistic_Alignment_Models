#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import io, time
import framework.tools as tools

# =============================================================================
# Target BPE
# Embedding: Feed forward
# Emission: Bi-LSTM
# Transition: Discrete
# =============================================================================

#==============================================================================
# Extra vocabulary BPE symbols 
unk_token_bpe = 0
start_token_bpe = 1
end_token_bpe = 2
#==============================================================================

from model.model_hmm_based.model_hmm_transition_discrete_emission_bpe import Model

class Model(Model):

    def __init__(self, FLAGS, session, log):

        super(Model, self).__init__(FLAGS, session, log)
        
    def build_model(self):
        self.get_reference_BPE_idx()
        
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
        
        nnLayer = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                    activation=tf.nn.tanh,
                                    use_bias=True,
                                    name='nnLayer')
                                                 
        target_state_null = nnLayer(target_embedded_null)

        lstm_cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.FLAGS.hidden_units)
        lstm_cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.FLAGS.hidden_units)
        
                                                                       
        with tf.variable_scope('lstmTarget'):

                (output_fw, output_bw),\
                (output_state_fw, output_state_bw) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                cell_bw=lstm_cell_bw, 
                                                inputs=target_embedded_word,
                                                sequence_length=self.target_lengths,
                                                dtype=self.tf_float_dtype)
                                                
                target_state = tf.concat([output_fw, output_bw], axis=2)
                
                nnLayer_lstm = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                    activation=None,
                                    use_bias=False,
                                    name='nnLayerLSTM')

        target_state = nnLayer_lstm(target_state)
        #----------------------------------------------------------------------
        target_state = nnLayer(target_state)

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
            
#==============================================================================
# 
#==============================================================================
    
    def print_model_info(self):
        self.log.info('Source word vocabulary size: %s', self.FLAGS.source_vocabulary_size)
        self.log.info('Target word vocabulary size: %s', self.FLAGS.target_vocabulary_size)
        self.log.info('Word hidden units: %s', self.FLAGS.hidden_units)
        self.log.info('Word embedding size: %s', self.FLAGS.embedding_size)
