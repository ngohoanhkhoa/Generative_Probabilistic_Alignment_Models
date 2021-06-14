#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, time, math
import numpy as np
import tensorflow as tf
import framework.tools as tools
from tensorflow.python import pywrap_tensorflow

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

                
        def forward_function(last_forward, yi):
            tmp = tf.multiply(tf.matmul(last_forward, transition), yi)
            return tmp

        alignment_expectation_prob = tf.scan(forward_function, emission_)
        alignment_expectation_prob = tf.transpose(tf.squeeze(tf.gather(alignment_expectation_prob,[0],
                                                                       axis=2),axis=2), (1,2,0))
        
        last_index_source_word= self.source_lengths - 1
        
        input_scan = (alignment_expectation_prob, last_index_source_word)
        def get_target_(value):
            target_prob = value[0]
            targets = value[1]
            
            target_prob = tf.gather(target_prob, targets, axis=-1)
            
            return target_prob, 0
            
        alignment_expectation_prob, _ = tf.map_fn(get_target_,input_scan, dtype=(self.tf_float_dtype,
                                                                           tf.int32))
        
        target_mask = tf.sequence_mask(lengths=self.target_lengths, maxlen=tf.shape(self.targets)[1],
                                           dtype=self.tf_float_dtype, name='targetMask')
        
        target_mask = tf.concat([target_mask, target_mask], axis=1)
        
        alignment_expectation = alignment_expectation_prob * target_mask
        
        self.alignment_expectation = tf.log(tf.reduce_sum(alignment_expectation, axis=1)+10e-30)
    
    def get_transition(self, y_hidden_variable, y_hidden_variable_null):
        with tf.variable_scope('transition', reuse=tf.AUTO_REUSE):
            nnLayer_transition = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                        activation=tf.nn.tanh,
                                        use_bias=True,
                                        name='nnLayerTransition')
                                               
            target_state = nnLayer_transition(y_hidden_variable)
            target_state_null = nnLayer_transition(y_hidden_variable_null)
    
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
            
            #p0 = tf.squeeze(nnLayerP0(target_state), axis=-1)
            p0 = self.FLAGS.alpha_p0 * tf.tile(nnLayerP0(target_state_null),[tf.shape(self.targets)[0], tf.shape(self.targets)[1]])
            
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
                
                jump_width_word_sum = tf.reduce_sum(jump_width_word, axis=1, keepdims=True)
                jump_width_word = tf.divide(jump_width_word * (1. - p0), jump_width_word_sum)
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
    
    
    def prepare_batch(self, seqs_x, seqs_y):
        x_lengths = np.array([len(s) for s in seqs_x])
        y_lengths = np.array([len(s) for s in seqs_y])
        x_lengths_max = np.max(x_lengths)
        y_lengths_max = np.max(y_lengths)
        
        self.batch_size = len(seqs_x)
                     
        y_null = [self.FLAGS.target_vocabulary_size - 1]
        
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
    
    def train(self):
        # Training loop
        self.log.info('TRAINING %s', tools.get_time_now())
        
        self.update_with_pretrained_params()
        self.evaluate()
        for epoch_idx in range(self.FLAGS.max_epochs):
            self.log.info('------------------')
            if self.global_epoch_step.eval() >= self.FLAGS.max_epochs:
                self.log.info('Training is complete. Reach the max epoch number: %d ', self.global_epoch_step.eval())
                break
   
            for train_seq in self.train_set:
                start_time = time.time()
                batch = self.prepare_batch(*train_seq)
                update_info = self.train_batch(*batch)
                self.print_log(update_info, start_time)
                
                self.evaluate()
                self.save_model()

            #------------------------------------------------------------------
            #------------------------------------------------------------------
            # Increase the epoch index of the model
            self.log.info('Epoch %d done at %s', 
                          self.global_epoch_step.eval(), 
                          tools.get_time_now())
            self.global_epoch_step_op.eval()
    
    def update_with_pretrained_params(self):
        if self.FLAGS.model_parameter_loaded_from_checkpoint_IBM1 is not None:
            self.pretrained_param = pywrap_tensorflow.NewCheckpointReader(self.FLAGS.model_parameter_loaded_from_checkpoint_IBM1)
            var_to_shape_map = self.pretrained_param.get_variable_to_shape_map()

            assign_ops = []
            trainable_params = tf.trainable_variables()
            
#            for v in trainable_params:
#                self.log.info('tr %s', v.name)
#            
#            for v in var_to_shape_map:
#                self.log.info('pr %s', v)
                
            for v in trainable_params:
                for key in var_to_shape_map:
                    if v.name[:-2] == key:
                        self.log.info(key)
                        assign_ops.append(tf.assign(v,self.pretrained_param.get_tensor(str(v.name)[:-2])))
                        
            for v in trainable_params:
                for key in var_to_shape_map:
                    if v.name[:-2] == 'encoder/lstmTarget/lstm/lstm_cell/kernel' \
                    and key == 'encoder/lstmTarget/lstm/kernel':
                        self.log.info('N %s', key)
                        assign_ops.append(tf.assign(v,self.pretrained_param.get_tensor(key)))
                    if v.name[:-2] == 'encoder/lstmTarget/lstm/lstm_cell/recurrent_kernel' \
                    and key == 'encoder/lstmTarget/lstm/recurrent_kernel':
                        self.log.info('N %s', key)
                        assign_ops.append(tf.assign(v,self.pretrained_param.get_tensor(key)))
                    if v.name[:-2] == 'encoder/lstmTarget/lstm/lstm_cell/bias' \
                    and key == 'encoder/lstmTarget/lstm/bias':
                        self.log.info('N %s', key)
                        assign_ops.append(tf.assign(v,self.pretrained_param.get_tensor(key)))
                        
                    if v.name[:-2] == 'encoder/lstmTarget/lstm_1/lstm_cell/kernel' \
                    and key == 'encoder/lstmTarget/lstm_1/kernel':
                        self.log.info('N %s', key)
                        assign_ops.append(tf.assign(v,self.pretrained_param.get_tensor(key)))
                    if v.name[:-2] == 'encoder/lstmTarget/lstm_1/lstm_cell/recurrent_kernel' \
                    and key == 'encoder/lstmTarget/lstm_1/recurrent_kernel':
                        self.log.info('N %s', key)
                        assign_ops.append(tf.assign(v,self.pretrained_param.get_tensor(key)))
                    if v.name[:-2] == 'encoder/lstmTarget/lstm_1/lstm_cell/bias' \
                    and key == 'encoder/lstmTarget/lstm_1/bias':
                        self.log.info('N %s', key)
                        assign_ops.append(tf.assign(v,self.pretrained_param.get_tensor(key)))
            self.sess.run(tf.group(*assign_ops))
            
            
