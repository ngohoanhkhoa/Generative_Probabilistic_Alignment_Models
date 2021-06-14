#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, time, math
import numpy as np
import tensorflow as tf
import framework.tools as tools
from tensorflow.python import pywrap_tensorflow

from model.model_variational_based.model_variational_bpe_share_params_symmetrical_cost import Model

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
        
    def get_transition_evaluation_nn(self,  transition, sentence_length):

        transition_matrix = transition
          
        initial_transition = np.zeros((2*sentence_length), dtype=self.float_dtype)
        initial_transition[:sentence_length] = self.float_dtype(1.)
        initial_transition = initial_transition/ np.sum(initial_transition)

        return initial_transition, transition_matrix
    
# =============================================================================
# 
# =============================================================================
        
    def build_update(self):
        self.annealing_KL_divergence = tf.Variable(0., trainable=False, name='annealing_KL_divergence' ,
                                                   dtype=self.tf_float_dtype)
        
        # Cost for e -> y -> e, f
        e_y_reconstruction_expectation = tf.reduce_mean(tf.reduce_sum(self.e_y_reconstruction_expectation_log, axis=1))
        f_y_alignment_expectation = tf.reduce_mean(self.f_y_alignment_expectation_log)
        y_KL_divergence = tf.reduce_mean(tf.reduce_sum(self.y_KL_divergence, axis=1))
        
        self.e_y_cost_reconstruction_expectation =  (-e_y_reconstruction_expectation)
        self.f_y_cost_alignment_expectation =  (-f_y_alignment_expectation)
        self.y_cost_KL_divergence = y_KL_divergence * self.annealing_KL_divergence
        self.y_KL_divergence_eval = y_KL_divergence
        
        self.y_cost = (self.FLAGS.alpha_reconstruction_expectation * self.e_y_cost_reconstruction_expectation) \
        + (self.FLAGS.alpha_alignment_expectation * self.f_y_cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.y_cost_KL_divergence)
        
        
        # Cost for f -> x -> f, e
        f_x_reconstruction_expectation = tf.reduce_mean(tf.reduce_sum(self.f_x_reconstruction_expectation_log, axis=1))
        e_x_alignment_expectation = tf.reduce_mean(self.e_x_alignment_expectation_log)
        x_KL_divergence = tf.reduce_mean(tf.reduce_sum(self.x_KL_divergence, axis=1))
        
        self.f_x_cost_reconstruction_expectation =  (-f_x_reconstruction_expectation)
        self.e_x_cost_alignment_expectation =  (-e_x_alignment_expectation)
        self.x_cost_KL_divergence = x_KL_divergence * self.annealing_KL_divergence
        self.x_KL_divergence_eval = x_KL_divergence
        
        self.x_cost = (self.FLAGS.alpha_reconstruction_expectation * self.f_x_cost_reconstruction_expectation) \
        + (self.FLAGS.alpha_alignment_expectation * self.e_x_cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.x_cost_KL_divergence)
        
        
        # Cost for agreement
        target_sequence_mask = tf.transpose(tf.tile(tf.expand_dims(tf.sequence_mask(self.target_lengths,
                                                                         tf.shape(self.f_y_alignment_expectation_prob)[1]), 
                                                                        axis=1),[1, tf.shape(self.e_x_alignment_expectation_prob)[1],1]),perm=[0, 2, 1])
        
        source_sequence_mask = tf.transpose(tf.tile(tf.expand_dims(tf.sequence_mask(self.source_lengths, 
                                                                       tf.shape(self.e_x_alignment_expectation_prob)[1]), 
                                                                        axis=1),
                                                                        [1, tf.shape(self.f_y_alignment_expectation_prob)[1],1]),perm=[0, 2, 1])
        
        f_y_alignment_expectation_prob_non_null = tf.concat([self.f_y_alignment_expectation_prob,self.f_y_alignment_expectation_prob],axis=2)\
                                                            * tf.cast(target_sequence_mask, self.tf_float_dtype)
        e_x_alignment_expectation_prob_non_null = tf.concat([self.e_x_alignment_expectation_prob,self.e_x_alignment_expectation_prob],axis=2)\
                                                            * tf.cast(source_sequence_mask, self.tf_float_dtype)
        
        target_source_mask = tf.cast(target_sequence_mask, self.tf_float_dtype) * tf.transpose(tf.cast(source_sequence_mask, self.tf_float_dtype), perm=[0, 2, 1])
        
        cost_agreement_non_null = target_source_mask * tf.abs(f_y_alignment_expectation_prob_non_null - \
                               tf.transpose(e_x_alignment_expectation_prob_non_null, perm=[0, 2, 1]))
        
        self.cost_agreement_non_null = self.FLAGS.alpha_agreement_non_null * tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(cost_agreement_non_null,axis=2),axis=1))
        
              
        target_sequence_mask_zeros = tf.tile(tf.expand_dims(tf.zeros_like(self.targets, dtype=self.tf_float_dtype), axis=2), [1, 1, tf.shape(self.sources)[1]])
        target_sequence_mask_ones = tf.tile(tf.expand_dims(tf.ones_like(self.targets, dtype=self.tf_float_dtype), axis=2), [1, 1, tf.shape(self.sources)[1]])
        target_sequence_mask_null = tf.concat([target_sequence_mask_zeros, target_sequence_mask_ones], axis=1)
        target_sequence_mask_non_null = tf.concat([target_sequence_mask_ones, target_sequence_mask_zeros], axis=1)
        
        source_sequence_mask_zeros = tf.tile(tf.expand_dims(tf.zeros_like(self.sources, dtype=self.tf_float_dtype), axis=2), [1, 1, tf.shape(self.targets)[1]])
        source_sequence_mask_ones = tf.tile(tf.expand_dims(tf.ones_like(self.sources, dtype=self.tf_float_dtype), axis=2), [1, 1, tf.shape(self.targets)[1]])
        source_sequence_mask_null = tf.concat([source_sequence_mask_zeros, source_sequence_mask_ones], axis=1)
        source_sequence_mask_non_null = tf.concat([source_sequence_mask_ones, source_sequence_mask_zeros], axis=1)
        
        f_y_alignment_expectation_prob_null_ = tf.transpose(self.f_y_alignment_expectation_prob * target_sequence_mask_null, perm=[0,2,1])
        e_x_alignment_expectation_prob_null_ = tf.transpose(self.e_x_alignment_expectation_prob * source_sequence_mask_null, perm=[0,2,1])
        
        f_y_alignment_expectation_prob_null = tf.concat([f_y_alignment_expectation_prob_null_, f_y_alignment_expectation_prob_null_], axis=1)
        e_x_alignment_expectation_prob_non_null_ = tf.tile(tf.expand_dims(tf.reduce_sum(self.e_x_alignment_expectation_prob * source_sequence_mask_non_null,axis=2), axis=2), [1, 1, 2*tf.shape(self.targets)[1]])
        
        e_x_alignment_expectation_prob_null = tf.concat([e_x_alignment_expectation_prob_null_, e_x_alignment_expectation_prob_null_], axis=1)
        f_y_alignment_expectation_prob_non_null_ = tf.tile(tf.expand_dims(tf.reduce_sum(self.f_y_alignment_expectation_prob * target_sequence_mask_non_null, axis=2), axis=2), [1, 1, 2*tf.shape(self.sources)[1]])
        
        
        source_target_mask_null = tf.reverse(tf.transpose(target_source_mask, perm=[0,2,1]), axis=[2])
        target_source_mask_null = tf.reverse(target_source_mask, axis=[2])

        source_null_mask = source_target_mask_null * tf.concat([tf.zeros((tf.shape(self.sources)[0],tf.shape(self.sources)[1]*2,tf.shape(self.targets)[1]*2-1), dtype=self.tf_float_dtype),
                                    tf.ones((tf.shape(self.sources)[0],tf.shape(self.sources)[1]*2,1), dtype=self.tf_float_dtype)],axis=2)
    
        target_null_mask = target_source_mask_null * tf.concat([tf.zeros((tf.shape(self.targets)[0],tf.shape(self.targets)[1]*2,tf.shape(self.sources)[1]*2-1), dtype=self.tf_float_dtype),
                                    tf.ones((tf.shape(self.targets)[0],tf.shape(self.targets)[1]*2, 1), dtype=self.tf_float_dtype)],axis=2)
    
        self.y_null_cost = self.FLAGS.alpha_agreement_y_null * tf.reduce_mean(tf.reduce_sum(source_null_mask * tf.abs((1. - f_y_alignment_expectation_prob_null - e_x_alignment_expectation_prob_non_null_)),axis=[1,2]))
        self.x_null_cost = self.FLAGS.alpha_agreement_x_null * tf.reduce_mean(tf.reduce_sum(target_null_mask * tf.abs((1. - e_x_alignment_expectation_prob_null - f_y_alignment_expectation_prob_non_null_)),axis=[1,2]))
        
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        cost_total = self.y_cost + self.x_cost + self.cost_agreement_non_null + self.y_null_cost + self.x_null_cost
        gradients = tf.gradients(cost_total, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = self.opt.apply_gradients(zip(clip_gradients,trainable_params),
                                                         global_step=self.global_step)
        
    def build_decoder(self):
        with tf.variable_scope('target_y_decoder', reuse=tf.AUTO_REUSE):
            # Reconstruction
            self.e_y_hidden_prob, self.e_y_hidden_prob_null = self.get_softmax(self.y_hidden_variable, 
                                                                         self.y_hidden_variable_null, 
                                                                         self.FLAGS.target_vocabulary_size)
            
            self.e_y_prob, self.e_y_prob_null = self.get_softmax(self.y_hidden, 
                                                                         self.y_hidden_null, 
                                                                         self.FLAGS.target_vocabulary_size)
            
            # Alignment
            self.e_x_hidden_prob, self.e_x_hidden_prob_null = self.get_softmax(self.x_hidden_variable, 
                                                                         self.x_hidden_variable_null, 
                                                                         self.FLAGS.target_vocabulary_size)
            
            self.e_x_prob, self.e_x_prob_null = self.get_softmax(self.x_hidden, 
                                                                         self.x_hidden_null, 
                                                                         self.FLAGS.target_vocabulary_size)
            
            
            self.width_x_hidden_prob_transition, self.width_x_hidden_prob_transition_log \
            = self.get_transition_softmax(self.x_hidden_variable, self.x_hidden_variable_null, self.sources)
            
            self.width_x_prob_transition, self.width_x_prob_transition_log \
            = self.get_transition_softmax(self.x_hidden, self.x_hidden_null, self.sources)
        
        with tf.variable_scope('source_x_decoder', reuse=tf.AUTO_REUSE):
            # Reconstruction
            self.f_x_hidden_prob, self.f_x_hidden_prob_null = self.get_softmax(self.x_hidden_variable, 
                                                                         self.x_hidden_variable_null, 
                                                                         self.FLAGS.source_vocabulary_size)
            
            self.f_x_prob, self.f_x_prob_null = self.get_softmax(self.x_hidden, 
                                                                 self.x_hidden_null, 
                                                                 self.FLAGS.source_vocabulary_size)
            
            # Alignment
            self.f_y_hidden_prob, self.f_y_hidden_prob_null = self.get_softmax(self.y_hidden_variable, 
                                                                         self.y_hidden_variable_null, 
                                                                         self.FLAGS.source_vocabulary_size)
            
            self.f_y_prob, self.f_y_prob_null = self.get_softmax(self.y_hidden, 
                                                                 self.y_hidden_null, 
                                                                 self.FLAGS.source_vocabulary_size)
            
            self.width_y_hidden_prob_transition, self.width_y_hidden_prob_transition_log \
            = self.get_transition_softmax(self.y_hidden_variable, self.y_hidden_variable_null, self.targets)
            
            self.width_y_prob_transition, self.width_y_prob_transition_log \
            = self.get_transition_softmax(self.y_hidden, self.y_hidden_null, self.targets)
            
# =============================================================================
#         
# =============================================================================
            
    def get_transition_softmax(self, hidden_variable, hidden_variable_null, sentences):
        with tf.variable_scope('transition', reuse=tf.AUTO_REUSE):
            nnLayer_transition = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                        activation=tf.nn.tanh,
                                        use_bias=True,
                                        name='nnLayerTransition')
                                               
            output_state = nnLayer_transition(hidden_variable)
            output_state_null = nnLayer_transition(hidden_variable_null)
            
            nnLayerJumpWidth = tf.layers.Dense(units=self.size_jump_width_set_nn,
                                     name='nnLayerJumpWidth')
            
            jump_width = nnLayerJumpWidth(output_state)
            jump_width_prob = tf.nn.softmax(jump_width)
            
            nnLayerP0 = tf.layers.Dense(units=1,activation=tf.nn.sigmoid,
                                        name='nnLayerP0')
            
            #p0 = tf.squeeze(nnLayerP0(output_state), axis=-1)
            p0 = self.FLAGS.alpha_p0 * tf.tile(nnLayerP0(output_state_null),[tf.shape(output_state)[0], tf.shape(output_state)[1]])
            
            self.length_jump = tf.shape(sentences)[1]
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
                sentence_length = tf.ones([self.length_jump], 
                                          dtype=tf.int32)*self.length_jump
                word_idx = tf.range(0,self.length_jump)
                input_scan = (jump_width, sentence_length, word_idx)
                jump_width_word,jump_width_idx,_ = tf.map_fn(get_jump_width_word,
                                            input_scan,
                                            (self.tf_float_dtype, tf.int32,tf.int32))
                
                jump_width_word_sum = tf.reduce_sum(jump_width_word, axis=1, keep_dims=True)
                jump_width_word = tf.divide(jump_width_word * (1. - p0), jump_width_word_sum)
                
                transition = tf.concat([jump_width_word, tf.diag(p0)], axis=1)
                transition = tf.concat([transition, transition], axis=0)
                
                transition_log = tf.concat([tf.log(jump_width_word), tf.diag(tf.log(p0))], axis=1)
                transition_log = tf.concat([transition_log, transition_log], axis=0)
                return transition, transition_log
    
            transition, transition_log = tf.map_fn(get_jump_width_sentence, (jump_width_prob, p0), (self.tf_float_dtype, self.tf_float_dtype) )
        
        return transition, transition_log
    
    
    def get_alignment_expectation(self,
                                  input_prob_softmax, 
                                  input_prob_softmax_null, 
                                  input_sentences, 
                                  output_sentences,
                                  input_lengths,
                                  output_lengths,
                                  transition):
        
        emission = self.get_emission(input_prob_softmax, input_prob_softmax_null, output_sentences)
        
        emission_ = tf.transpose(tf.tile(tf.expand_dims(emission, axis=1), 
                                        [1,tf.shape(emission)[1],1,1 ]), (3,0,1,2))

                
        def forward_function(last_forward, yi):
            tmp = tf.multiply(tf.matmul(last_forward, transition), yi)
            return tmp

        alignment_expectation_prob = tf.scan(forward_function, emission_)
        
        alignment_expectation_prob = tf.transpose(tf.squeeze(tf.gather(alignment_expectation_prob,[0],
                                                                       axis=2),axis=2), (1,2,0))
        
        alignment_expectation_prob_output = alignment_expectation_prob
        
        last_index_source_word= output_lengths - 1
        
        input_scan = (alignment_expectation_prob, last_index_source_word)
        def get_output_alignment(value):
            target_prob = value[0]
            targets = value[1]
            
            target_prob = tf.gather(target_prob, targets, axis=-1)
            
            return target_prob, 0
            
        alignment_expectation_prob, _ = tf.map_fn(get_output_alignment,input_scan, dtype=(self.tf_float_dtype,
                                                                           tf.int32))
        
        output_mask = tf.sequence_mask(lengths=input_lengths, maxlen=tf.shape(alignment_expectation_prob)[1],
                                           dtype=self.tf_float_dtype, name='mask')
        
        alignment_expectation = alignment_expectation_prob * output_mask
        
        alignment_expectation_log = tf.log(tf.reduce_sum(alignment_expectation, axis=1)+10e-30)
        
        return alignment_expectation_log, alignment_expectation_prob_output
    
            
    def build_alignment(self):
        with tf.variable_scope('alignment', reuse=tf.AUTO_REUSE):
        
            self.f_y_emission = self.get_emission(self.f_y_prob, self.f_y_prob_null, self.sources)
            
            self.f_y_alignment_expectation_log, self.f_y_alignment_expectation_prob \
            = self.get_alignment_expectation(input_prob_softmax= self.f_y_hidden_prob, 
                                                                            input_prob_softmax_null= self.f_y_hidden_prob_null,
                                                                            input_sentences= self.targets,
                                                                            output_sentences= self.sources,
                                                                            input_lengths= self.target_lengths,
                                                                            output_lengths= self.source_lengths,
                                                                            transition = self.width_y_hidden_prob_transition)
            
            self.e_x_emission = self.get_emission(self.e_x_prob, self.e_x_prob_null, self.targets)
            
            self.e_x_alignment_expectation_log, self.e_x_alignment_expectation_prob \
            = self.get_alignment_expectation(input_prob_softmax= self.e_x_hidden_prob, 
                                                                            input_prob_softmax_null= self.e_x_hidden_prob_null,
                                                                            input_sentences= self.sources,
                                                                            output_sentences= self.targets,
                                                                            input_lengths= self.source_lengths,
                                                                            output_lengths= self.target_lengths,
                                                                            transition = self.width_x_hidden_prob_transition)
    
    def eval(self,
             sources,
             sources_null,
             source_lengths,
             targets,
             targets_null,
             target_lengths):
        
        target_length = target_lengths[0]
        source_length = source_lengths[0]
        source = sources[0][:source_length]
        target = targets[0][:target_length]
        
        input_feed = {}
        input_feed[self.keep_prob.name] = 1.0
        input_feed[self.sources.name] = sources
        input_feed[self.targets.name] = targets
        input_feed[self.source_lengths.name] = source_lengths
        input_feed[self.target_lengths.name] = target_lengths
        
        input_feed[self.targets_null.name] = targets_null
        input_feed[self.sources_null.name] = sources_null
        
        output_feed = [self.f_y_emission,
                       self.e_x_emission,
                       self.width_y_prob_transition,
                       self.width_x_prob_transition]
        #----------------------------------------------------------------------
        outputs = self.sess.run(output_feed, input_feed)

        
        f_y_emission_final = np.array(outputs[0][0])
        e_x_emission_final = np.array(outputs[1][0])
        w_y_emission_final = np.array(outputs[2][0])
        w_x_emission_final = np.array(outputs[3][0])
        
        f_y_initial_transition, f_y_transition = self.get_transition_evaluation_nn(w_y_emission_final, target_length)
        e_x_initial_transition, e_x_transition = self.get_transition_evaluation_nn(w_x_emission_final, source_length)
        
        
        f_y_state_seq, f_y_likelihood_seq = self.viterbi(np.array(source),
                                        target_length*2,
                                        f_y_initial_transition,
                                        f_y_transition, 
                                        f_y_emission_final)
        
        e_x_state_seq, e_x_likelihood_seq = self.viterbi(np.array(target),
                                        source_length*2,
                                        e_x_initial_transition,
                                        e_x_transition, 
                                        e_x_emission_final)
        
        return [f_y_state_seq], [f_y_likelihood_seq], [f_y_emission_final], \
    [e_x_state_seq], [e_x_likelihood_seq], [e_x_emission_final]          
    
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
            
        # Training loop
        self.log.info('TRAINING %s', tools.get_time_now())
        
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