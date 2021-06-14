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
        
    def get_transition_evaluation_nn(self,  transition, sentence_length):

        transition_matrix = transition
          
        initial_transition = np.zeros((2*sentence_length), dtype=self.float_dtype)
        initial_transition[:sentence_length] = self.float_dtype(1.)
        initial_transition = initial_transition/ np.sum(initial_transition)

        return initial_transition, transition_matrix
        
# =============================================================================
# 
# =============================================================================
    def build_alignment(self):
        
        self.emission = self.get_emission(self.y_hidden, self.y_hidden_null)
        self.transition, _ = self.get_transition(self.y_hidden, self.y_hidden_null)
        
        emission = self.get_emission(self.y_hidden_variable, self.y_hidden_variable_null)
        transition, transition_log = self.get_transition(self.y_hidden_variable, self.y_hidden_variable_null)
        
        emission_ = tf.transpose(tf.tile(tf.expand_dims(emission, axis=1), 
                                        [1,tf.shape(emission)[1],1,1 ]), (3,0,1,2))

        emission_log = tf.ones_like(emission_)
        def forward_function(last_forward, yi):
            last_forward_prob = last_forward[0]
            last_forward_log = last_forward[1]
            
            yi_prob = yi[0]
            yi_log = yi[1]
            
            tmp_prob = tf.multiply(tf.matmul(last_forward_prob, transition), yi_prob)
            tmp_prob = tmp_prob/ tf.reduce_sum(tmp_prob, axis=2, keep_dims=True)
            
            max_j = tf.gather(tf.reduce_max(tf.multiply(tf.matmul(transition,
                                                                  last_forward_prob), yi_prob),axis=-1),[0], axis=-1)
            
            state_previous = tf.reduce_sum(tf.matmul(last_forward_prob, transition), axis=1)
            yi_prob_ = tf.squeeze(tf.gather(yi_prob, [0], axis=1),axis=1)
            state = tf.log(yi_prob_) + tf.log(state_previous) - max_j    
            state_log = tf.log(tf.reduce_sum(tf.exp(state), axis=-1))
            
            tmp_log = tf.squeeze(tf.gather(tf.gather(last_forward_log, [0], axis=1), [0], axis=2)) \
            + state_log + tf.squeeze(max_j,axis=1)
            
            tmp_log = yi_log * tf.reshape(tmp_log, (tf.shape(tmp_log)[0],1,1))
            
            return tmp_prob, tmp_log

        alignment_expectation_prob_log = tf.scan(forward_function, (emission_,emission_log))[1]
        alignment_expectation_prob_log = tf.transpose(tf.squeeze(tf.gather(tf.gather(alignment_expectation_prob_log, 
                                                               [0], axis=2), 
                                                               [0], axis=3)), (1,0))
        
        last_index_source_word= self.source_lengths - 1
        
        input_scan = (alignment_expectation_prob_log, last_index_source_word)
        def get_target_(value):
            target_prob = value[0]
            targets = value[1]
            
            target_prob = tf.gather(target_prob, targets, axis=-1)
            
            return target_prob, 0
            
        alignment_expectation_prob_log, _ = tf.map_fn(get_target_,input_scan, dtype=(self.tf_float_dtype,
                                                                           tf.int32))
        
        self.alignment_expectation = alignment_expectation_prob_log
    
    def get_transition(self, y_hidden_variable, y_hidden_variable_null):
        with tf.variable_scope('transition', reuse=tf.AUTO_REUSE):
            nnLayer_emission = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                        activation=tf.nn.tanh,
                                        use_bias=True,
                                        name='nnLayerTransition')
                                               
            target_state = nnLayer_emission(y_hidden_variable)
            target_state_null = nnLayer_emission(y_hidden_variable_null)
    
            target_state = tf.nn.dropout(x=target_state,
                                           keep_prob=self.keep_prob,
                                           name='nnDropoutNonNull')
                                           
            target_state_null = tf.nn.dropout(x=target_state_null,
                                           keep_prob=self.keep_prob,
                                           name='nnDropoutNull')
            
            nnLayerJumpWidth = tf.layers.Dense(units=self.size_jump_width_set_nn,
                                     name='nnLayerJumpWidth')
            
            jump_width = nnLayerJumpWidth(target_state)
            jump_width_prob = tf.nn.softmax(jump_width)
            
            nnLayerP0 = tf.layers.Dense(units=1,activation=tf.nn.sigmoid,
                                        name='nnLayerP0')
            
            p0 = tf.squeeze(nnLayerP0(target_state), axis=-1)
            
            self.length_target = tf.shape(self.targets)[1]
            def get_jump_width_word(value):
                jump_width = value[0]
                sentence_length = value[1]
                word_idx = value[2]
                jump_width_idx = tf.range(self.max_jump_width-word_idx,
                                          self.max_jump_width+sentence_length-word_idx)
                return tf.gather(jump_width, jump_width_idx),jump_width_idx,0
            
            def get_jump_width_sentence(value):
                jump_width = value[0]
                p0 = value[1]
                sentence_length = tf.ones([self.length_target], 
                                          dtype=tf.int32)*self.length_target
                word_idx = tf.range(0,self.length_target)
                input_scan = (jump_width, sentence_length, word_idx)
                jump_width_word,jump_width_idx,_ = tf.map_fn(get_jump_width_word,
                                            input_scan,
                                            (self.tf_float_dtype, tf.int32,tf.int32))
                
                transition = tf.concat([jump_width_word, tf.diag(p0)], axis=1)
                transition = tf.concat([transition, transition], axis=0)
                
                transition_log = tf.concat([tf.log(jump_width_word), tf.diag(tf.log(p0))], axis=1)
                transition_log = tf.concat([transition_log, transition_log], axis=0)
                return transition, transition_log
    
            transition, transition_log = tf.map_fn(get_jump_width_sentence, (jump_width_prob, p0), (self.tf_float_dtype, self.tf_float_dtype) )
        
        return transition, transition_log
        
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
        
        output_feed = [self.emission,
                       self.transition]
        #----------------------------------------------------------------------
        outputs = self.sess.run(output_feed, input_feed)
        
        emission_final = np.array(outputs[0][0])
        transition_final = np.array(outputs[1][0])
        
        initial_transition, transition = self.get_transition_evaluation_nn(transition_final,
                                                                        target_length)
        
        
        state_seq, likelihood_seq = self.viterbi(np.array(source),
                                        target_length*2,
                                        initial_transition,
                                        transition, 
                                        emission_final)
        
        return [state_seq], [likelihood_seq], [emission_final]