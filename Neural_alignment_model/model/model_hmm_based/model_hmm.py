#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, time, math
import numpy as np
import tensorflow as tf
import framework.tools as tools

from model.model import BaseModel

class Model(BaseModel):

    def __init__(self, FLAGS, session, log):

        super(Model, self).__init__(FLAGS, session, log)
        
    def build_model(self):
        self.build_optimizer()
        self.build_initializer()
        
        self.build_emission()
        self.build_transition()
        self.build_forward_backward()
        
        self.build_update()

# =============================================================================
# Build model
# =============================================================================
        
    def build_optimizer(self):
        if self.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            self.opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        
    def build_initializer(self):
        self.initializer = tf.contrib.layers.xavier_initializer()
        
        # Initialize encoder_embeddings to have variance=1.
        #sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
        #self.initializer = tf.random_uniform_initializer(-sqrt3,sqrt3,dtype=self.tf_float_dtype)
        
    def build_update_emission(self):
        # =============================================================================
        # Update emission parameters        
        # =============================================================================
        target_sequence_mask_emission = tf.transpose(tf.tile(tf.expand_dims(tf.sequence_mask(self.target_lengths,
                                                                         tf.shape(self.targets)[1]), 
                                                                        axis=1),[1, tf.shape(self.sources)[1],1]),perm=[0, 2, 1])
        
        source_sequence_mask = tf.tile(tf.expand_dims(tf.sequence_mask(self.source_lengths, 
                                                                       tf.shape(self.sources)[1]), 
                                                                        axis=1),
                                                                        [1, tf.shape(self.targets)[1],1])
        
        matrix_mask_bool_emission = tf.logical_and(source_sequence_mask,target_sequence_mask_emission)
        matrix_mask_float_emission = tf.cast(matrix_mask_bool_emission, self.tf_float_dtype)
        matrix_mask_emission = tf.concat([matrix_mask_float_emission,matrix_mask_float_emission], axis=1 )

        trainable_params_emission = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='emission')
        emission_prob_log = -tf.log(self.emission_prob)
        emission_prob_weight = tf.multiply(self.emission_posteriors_expectation, emission_prob_log)
        emission_prob_weight = tf.multiply(emission_prob_weight, matrix_mask_emission)
        cost_emission = tf.reduce_mean(tf.reduce_sum(emission_prob_weight, [1]))

        self.cost_emission = cost_emission
        
        gradients_emission = tf.gradients(cost_emission, trainable_params_emission)
        clip_gradients_emission, _ = tf.clip_by_global_norm(gradients_emission, self.max_gradient_norm)
        self.updates_emission = self.opt.apply_gradients(zip(clip_gradients_emission,trainable_params_emission),
                                                         global_step=self.global_step)
        
    def build_update_transition(self):
        # =============================================================================
        # Update transition parameters
        # =============================================================================
        target_sequence_mask_transition_a = tf.tile(tf.expand_dims(tf.sequence_mask(self.target_lengths,
                                                            tf.shape(self.targets)[1]),axis=1),
                                                                    [1, tf.shape(self.targets)[1],1])
        
        target_sequence_mask_transition_b = tf.transpose(target_sequence_mask_transition_a,perm=[0, 2, 1])

        matrix_mask_bool_transition = tf.logical_and(target_sequence_mask_transition_a, target_sequence_mask_transition_b)
        matrix_mask_float_transition = tf.cast(matrix_mask_bool_transition, self.tf_float_dtype)
        matrix_mask_transition = tf.concat([matrix_mask_float_transition,matrix_mask_float_transition], axis=2 )
        matrix_mask_transition = tf.concat([matrix_mask_transition,matrix_mask_transition], axis=1 )

        trainable_params_transition = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='transition')
        transition_prob_log = -self.transition_prob_log
        transition_prob_weight = tf.multiply(self.transition_posteriors_expectation, transition_prob_log)
        transition_prob_weight = tf.multiply(transition_prob_weight, matrix_mask_transition)
        cost_transition = tf.reduce_mean(tf.reduce_sum(transition_prob_weight, [1]))
        
        self.cost_transition = cost_transition
        
        gradients_transition = tf.gradients(cost_transition, trainable_params_transition)
        clip_gradients_transition, _ = tf.clip_by_global_norm(gradients_transition, self.max_gradient_norm)
        self.updates_transition = self.opt.apply_gradients(zip(clip_gradients_transition,trainable_params_transition),
                                                         global_step=self.global_step)
            
        
    def build_forward_backward_discrete(self):
        # =============================================================================
        # Forward backward from transition discrete
        # =============================================================================
#        input_scan = (self.emission_prob_expectation,
#                      self.initial_transition,
#                      self.transition)
#        
#        self.emission_posteriors_expectation,\
#        self.transition_posteriors_expectation, \
#        _ = tf.map_fn(self.forward_backward_sentence,input_scan, 
#                      dtype=(self.tf_float_dtype,
#                             self.tf_float_dtype,
#                             tf.int32))
        
        self.emission_posteriors_expectation,self.transition_posteriors_expectation =\
        self.forward_backward_batch(self.emission_prob_expectation,self.transition)
        
    def build_forward_backward_nn(self):
        # =============================================================================
        # Forward backward from transition nn       
        # =============================================================================
        
#        input_scan = (self.emission_prob_expectation,
#                      self.initial_transition_prob,
#                      self.transition_prob_expectation)
#        
#        self.emission_posteriors_expectation,\
#        self.transition_posteriors_expectation, \
#        _ = tf.map_fn(self.forward_backward_sentence,input_scan, 
#                      dtype=(self.tf_float_dtype,
#                             self.tf_float_dtype,
#                             tf.int32))
        
        self.emission_posteriors_expectation,self.transition_posteriors_expectation =\
        self.forward_backward_batch(self.emission_prob_expectation,self.transition_prob_expectation)
        
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
        
        # Calculate ALPHA
        # forward_t:
        # (source_size_,batch_size_,1,target_size_)
        # > (source_size_,batch_size_,target_size_,1)
        # > (0:source_size_-1,batch_size_,target_size_,1)
        forward_t = tf.gather(tf.transpose(forward,(0,1,3,2)), tf.range(0, source_size_-1))
        # transition_t from self.transition_prob_expectation:
        # (batch_size_, target_size_, target_size_)
        # > (1, batch_size_, target_size_, target_size_)
        # > (source_size_-1, batch_size_, target_size_, target_size_)
        transition_t = tf.tile(tf.expand_dims(transition_prob_expectation,axis=0),[source_size_-1,1,1,1])
        
        # alpha: (0:source_size_-1,batch_size_,target_size_,target_size_)
        alpha = tf.multiply(forward_t, transition_t)

        # Calculate BETA
        # (source_size_,batch_size_,1,target_size_)
        # > (source_size_,batch_size_,target_size_,1)
        # > (1:source_size_,batch_size_,target_size_,1)
        backward_t = tf.gather(tf.transpose(backward,(0,1,3,2)), tf.range(1, source_size_))
        # obs_seq_b: (B,I,J) > (B,I,J,1) > (J,B,I,1) > (1:J,B,I,1)
        # (batch_size_,target_size_,source_size_) 
        # > (batch_size_,target_size_,source_size_,1) 
        # > (source_size_, batch_size_,target_size_,1)
        # > (1:source_size_, batch_size_,target_size_,1) 
        obs_seq_t = tf.gather(tf.transpose(tf.expand_dims(emission_prob_expectation, axis=-1), (2,0,1,3)), 
                                    tf.range(1, source_size_))
        
        # beta:
        # > (source_size_, batch_size_,target_size_,1)
        # > (source_size_-1, batch_size_,1,target_size_)
        beta = tf.transpose(tf.multiply(backward_t, obs_seq_t), (0,1,3,2))

        # Calculate TRANSITION POSTERIOR
        # transition_posterior: (source_size_, batch_size_,target_size_,target_size_)
        transition_posterior = tf.multiply(alpha, beta)
        transition_posterior_sum = tf.reduce_sum(tf.reduce_sum(transition_posterior,axis=3, keep_dims=True),axis=2,keep_dims=True)
        
        transition_posterior_sum_zero_replaced_by_one = tf.where(tf.is_inf(tf.log(transition_posterior_sum)), 
                                                                 tf.ones_like(transition_posterior_sum), transition_posterior_sum)
                 
        transition_posterior = transition_posterior / transition_posterior_sum_zero_replaced_by_one
                                                              
        # transition_posterior: (J,B,I,I) -> (B,I,I)
        # (source_size_, batch_size_,target_size_,target_size_)
        # > (batch_size_,target_size_,target_size_)
        transition_posterior = tf.reduce_sum(transition_posterior, axis=0)
        
        return emission_posterior, transition_posterior
        
    def forward_backward_sentence(self, value):
        # =============================================================================
        # Code of forward backward algorithm for each sentence     
        # =============================================================================
        obs_prob_seqs = value[0] 
        initial_transition = value[1]  
        transition = value[2] 
        
        def forward_function(last_forward, yi):
            tmp = tf.multiply(tf.matmul(last_forward, self.transition_forward_backward), yi)
            return tmp / tf.reduce_sum(tmp, axis=1, keep_dims=True)
            
        def backward_function(last_backward, yi):
            # combine transition matrix with observations
            combined = tf.multiply(
                tf.expand_dims(self.transition_forward_backward, 0), tf.expand_dims(yi, 1)
            )
            tmp = tf.reduce_sum(
                tf.multiply(combined, tf.expand_dims(last_backward, 1)), axis=2
            )
            return tmp / tf.reduce_sum(tmp, axis=1, keep_dims=True)
            
        def likelihood_forward_function(last_forward, yi):
            tmp = tf.multiply(tf.matmul(last_forward, self.transition_forward_backward), yi)
            return tmp
            
        # obs_prob_seqs: (e,f) -> (f,1,e)
        obs_prob_seqs = tf.transpose(tf.expand_dims(obs_prob_seqs, axis=0), (2,0,1))
       
        #------------------------------------------------------------------
        # Calculate Forward
        # Forward[t = 1] = pi *  obs_prob_seqs[0]            
        obs_prob_seqs_first_obs = tf.gather(obs_prob_seqs, [0])
        initial_transition_f = tf.multiply(initial_transition, obs_prob_seqs_first_obs)
        
        # Concatenate forward[t=1] and obs_prob_seqs[1:T], it is true length of forward
        obs_prob_seqs_removed_first_obs = tf.gather(obs_prob_seqs, tf.range(1, tf.shape(obs_prob_seqs)[0]))
        obs_prob_seqs_f = tf.concat([initial_transition_f, obs_prob_seqs_removed_first_obs], axis=0)
        
        self.transition_forward_backward = transition
        
        # forward: (f,1,e)
        forward = tf.scan(
        forward_function,
        obs_prob_seqs_f
        )
        

        #----------------------------------------------------------------------
        # Calculate Backward
        # Add final transition into obs_prob_seqs, last column = [1...1] -> Not true length anymore
        final_transition = tf.ones((1,1,tf.shape(obs_prob_seqs)[2]), dtype=self.tf_float_dtype)
        obs_prob_seqs_b = tf.concat([obs_prob_seqs,final_transition], axis=0)
        
        #self.transition_forward_backward = tf.transpose(transition)
        
        # backward: (f,1,e)
        backward = tf.scan(
        backward_function,
        tf.reverse(obs_prob_seqs_b, [0])
        )
        
        # Reverse to have the same direction of forward
        backward = tf.reverse(backward, [0])
        
        #----------------------------------------------------------------------
        # Return True Forward and Backward
        forward_true_length = forward
        # backward: Remove the first column  (f,1,e) -> (f[1:],1,e)
        backward_true_length = tf.gather(backward, tf.range(1, tf.shape(backward)[0]))
        
        #----------------------------------------------------------------------
        #likelihood_forward = tf.scan(likelihood_forward_function,obs_prob_seqs_f)
        #likelihood = tf.reduce_sum(tf.gather(likelihood_forward, [tf.shape(forward)[0]-1]) ) 
        #----------------------------------------------------------------------
        #----------------------------------------------------------------------
        # Emission Posterior:

        # emission_posterior: (f,1,e)
        emission_posterior = tf.multiply(forward_true_length, backward_true_length)
        emission_posterior_sum = tf.reduce_sum(emission_posterior, axis=2, keep_dims=True)
        #emission_posterior = emission_posterior / emission_posterior_sum
        
        
        emission_posterior_sum_zeros_mask = tf.log(emission_posterior_sum)
        emission_posterior_sum_zeros_replaced_by_one = \
        tf.where(tf.is_inf(emission_posterior_sum_zeros_mask),
                 tf.ones_like(emission_posterior_sum), emission_posterior_sum)
                 
        emission_posterior = emission_posterior / emission_posterior_sum_zeros_replaced_by_one
        
        
        # emission_posterior: (f_,1,e) -> (1,e,f_) -> (e,f_)
        emission_posterior = tf.squeeze(tf.transpose(emission_posterior, (1, 2, 0)))
        
        #------------------------------------------------------------------
        #------------------------------------------------------------------
        # Transition Posterior:          
        
        # forward_t: (f,1,e) -> (f,e,1) -> (f[:-1], e ,1) Remove the last column forward (to T-1)
        forward_t = tf.gather(tf.transpose(forward_true_length,(0,2,1)), 
                              tf.range(0, tf.shape(forward_true_length)[0]-1))
        # transition_t: (e,e) -> (f_,e,e)
        transition_t = tf.tile(tf.expand_dims(transition,axis=0),[tf.shape(forward_t)[0],1,1])
        
        # alpha
        alpha = tf.multiply(forward_t, transition_t)
        
        #--------------------------------
        
        # backward_t: (f,1,e) -> (f,e,1) -> (f[1:],e,1) Remove the first column backward (from t+1)
        backward_t = tf.gather(tf.transpose(backward_true_length,(0,2,1)), 
                               tf.range(1, tf.shape(backward_true_length)[0]))
        # obs_prob_seqs_t: (f,1,e) -> (f[1:], 1,e) -> (f_,e,1)
        obs_prob_seqs_t = tf.transpose(tf.gather(obs_prob_seqs, tf.range(1, tf.shape(obs_prob_seqs)[0])), (0,2,1))
        
        # beta: (f,1,e)
        beta = tf.multiply(backward_t, obs_prob_seqs_t)
        beta = tf.transpose(beta, (0,2,1))
        #--------------------------------
        # transition_posterior: (f,e,e)
        transition_posterior = tf.multiply(alpha, beta)
        
        
        transition_posterior_sum = tf.reduce_sum(tf.reduce_sum(transition_posterior, 
                                                               axis=2, 
                                                               keep_dims=True), 
                                                               axis=1, 
                                                               keep_dims=True)
        
                                                                                  
        transition_posterior_sum_zeros_mask = tf.log(transition_posterior_sum)
        transition_posterior_sum_zeros_replaced_by_one = \
        tf.where(tf.is_inf(transition_posterior_sum_zeros_mask), 
                 tf.ones_like(transition_posterior_sum), transition_posterior_sum)
                 
        transition_posterior = transition_posterior / transition_posterior_sum_zeros_replaced_by_one
                                                              
        # transition_posterior: (f,e,e) -> (e,e)     
        transition_posterior = tf.reduce_sum(transition_posterior, axis=0)
        
        return emission_posterior, transition_posterior, 0

# =============================================================================
#
# =============================================================================
        
    def build_emission(self):
        # Scope emission:
        with tf.variable_scope('emission', reuse=tf.AUTO_REUSE):
            self.emission_prob \
             = self.get_emission()
            
        # Scope emission_expectation: Keep NN parameters after an epoch
        with tf.variable_scope('expectationEmission'):
            self.emission_prob_expectation \
             = self.get_emission()
        
    def build_transition(self):
        # Scope transition:
        with tf.variable_scope('transition', reuse=tf.AUTO_REUSE):
            self.transition_prob, self.transition_prob_log \
             = self.get_transition()
             
             
        with tf.variable_scope('expectationTransition', reuse=tf.AUTO_REUSE):
            self.transition_prob_expectation, self.transition_prob_log_expectation \
             = self.get_transition()
            
# =============================================================================
# 
# =============================================================================
             
    def assign_variable_training_emission(self):
        if self.emission_update_freq != 0:
            
            emission_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='emission') #TRAINABLE_VARIABLES
            expectation_emission_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='expectationEmission')
            
            assign_ops = []
            for emission_variable, expectation_emission_variable in zip(emission_variables, expectation_emission_variables):
                assign_ops.append(tf.assign(expectation_emission_variable, tf.identity(emission_variable)))
            
            self.sess.run(tf.group(*assign_ops))
            
    def assign_variable_training_transition_(self):
        if self.jump_width_update_freq != 0:
            
            emission_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='transition') #TRAINABLE_VARIABLES
            expectation_emission_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='expectationTransition')
            
            assign_ops = []
            for emission_variable, expectation_emission_variable in zip(emission_variables, expectation_emission_variables):
                assign_ops.append(tf.assign(expectation_emission_variable, tf.identity(emission_variable)))
            
            self.sess.run(tf.group(*assign_ops))
        
# =============================================================================
#         
# =============================================================================
    def train(self):
        # Create a unit matrix from 0 to max_seq_length: speed training
        if 'transition_discrete' in self.FLAGS.model:
            self.update_transition_matrix_unit()
            
        # Training loop
        self.log.info('TRAINING %s', tools.get_time_now())
        
        self.evaluate()
        for epoch_idx in range(self.FLAGS.max_epochs):
            self.log.info('----------------')
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
                
                # Update emission fixed parameters for each batch, emission_update_freq < 0:
                if self.emission_update_freq < 0 and \
                self.global_step.eval() % self.emission_update_freq == 0:
                    self.assign_variable_training_emission()
                    
                # Update transition discrete parameters for each batch, jump_width_update_freq < 0:
                if self.jump_width_update_freq < 0 and \
                self.global_step.eval() % self.jump_width_update_freq == 0:
                    self.assign_variable_training_transition()
            
            
            # Update emission fixed parameters for each epoch, emission_update_freq > 0:
            if self.emission_update_freq > 0 and \
            self.global_step.eval() % self.emission_update_freq == 0:
                self.assign_variable_training_emission()
                
            # Update transition discrete parameters for each epoch, jump_width_update_freq > 0:
            if self.jump_width_update_freq > 0 and \
            self.global_step.eval() % self.jump_width_update_freq == 0:
                self.assign_variable_training_transition()

            #------------------------------------------------------------------
            #------------------------------------------------------------------
            # Increase the epoch index of the model
            self.log.info('Epoch %d done at %s', 
                          self.global_epoch_step.eval(), 
                          tools.get_time_now())
            self.global_epoch_step_op.eval()
        

    def evaluate(self):
        # Execute a validation step
        if self.FLAGS.valid_freq!= 0 and self.global_step.eval() % self.FLAGS.valid_freq == 0:
            start_time = time.time()
            
            #------------------------------------------------------------------
            #------------------------------------------------------------------
            
            alignment_set = []
            likelihood_set = []
            emission_set = []
            
            for valid_seq in self.valid_set:
                batch = self.prepare_batch(*valid_seq)
                eval_info = self.eval(*batch)
                        
                #--------------------------------------------------------------
                
                for idx in range(1):
                    source_length = batch[1][idx]
                    target_length = batch[4][idx]
                    alignment = []
                    for index_source, index_target in enumerate(eval_info[0][idx]):
                        if index_source < source_length and index_target < target_length:
                            # Check alignment reference: Start from 0 or 1 ?
                            alignment.append(str(index_source+self.evaluate_alignment_start_from) \
                            + "-" + str(index_target+self.evaluate_alignment_start_from))
                            #alignment.append(str(index_source) + "-" + str(index_target))
                            
                    alignment_set.append(alignment)

                likelihood_set.append(eval_info[1])
                emission_set.append(eval_info[2])

            AER = self.calculate_AER(alignment_set)
            likelihood_score = np.mean(likelihood_set)
            
            self.log.info('VALIDATION: Epoch %d , Step %d , Likelihood: %.10f , AER: %.5f in %ds at %s',
                          self.global_epoch_step.eval(), 
                          self.global_step.eval() , 
                          likelihood_score, 
                          AER, 
                          time.time() - start_time, 
                          tools.get_time_now())
            
            
            if AER < self.best_AER:
                with io.open(self.get_file_path('result'), 'w') as file_handler:
                    for sent_alignment in alignment_set:
                        sent_alignment = u'{}\n'.format(sent_alignment)
                        sent_alignment = sent_alignment.replace('[','')
                        sent_alignment = sent_alignment.replace(']','')
                        sent_alignment = sent_alignment.replace(',','')
                        sent_alignment = sent_alignment.replace('\'','')
                        file_handler.write(sent_alignment)
            
            self.start_time = time.time()
            
            
# =============================================================================
# Transition
# =============================================================================
            
    def initialize_transition_parameter_nn(self):        
        # =============================================================================
        # Initialize transition parameters nn        
        # =============================================================================
        self.max_jump_width = int(self.FLAGS.max_jump_width)
        self.size_jump_width_set_nn = self.max_jump_width*2 + 1 # Large range of value [-100, ..., 0, ... 100]

    def initialize_transition_parameter_discrete(self):
        # =============================================================================
        # Initialize transition parameters discrete        
        # =============================================================================
        self.max_jump_width = int(self.FLAGS.max_jump_width)
        
        # Transition discrete
        self.size_jump_width_set_discrete = self.max_jump_width*2 + 3
        # Small range of value [<-max_jump_width, -5, ..., 0, ... 5, >max_jump_width]
        
        if self.FLAGS.transition_heuristic_prob is not None:
            self.transition_heuristic_prob = float(self.FLAGS.transition_heuristic_prob)
        
        if self.FLAGS.initialize_jump_set.lower() == 'heuristic' :
            transition_set_prob = np.ones((self.size_jump_width_set_discrete), dtype=self.float_dtype)* \
            ( (1. - self.FLAGS.transition_heuristic_prob)/(self.size_jump_width_set_discrete-1) )
            transition_set_prob[self.max_jump_width + 2] = self.transition_heuristic_prob
        elif self.FLAGS.initialize_jump_set.lower() == 'uniform' :
            transition_set_prob = np.ones((self.size_jump_width_set_discrete), dtype=self.float_dtype)* (1./(self.size_jump_width_set_discrete) )
        else: # Random
            transition_set_prob = np.random.randint(low=1, high=self.size_jump_width_set_discrete, size=[self.size_jump_width_set_discrete])
            transition_set_prob = np.divide(transition_set_prob, np.sum(transition_set_prob))
        
        # Parameters in tensorflow
        self.transition_set_prob = tf.Variable(transition_set_prob,
                                               trainable=False, 
                                               name='transition_set_prob', 
                                               dtype=self.tf_float_dtype)
                                             
        self.p0 = tf.Variable(self.FLAGS.p0,
                              trainable=False, 
                              name='p0', 
                              dtype=self.tf_float_dtype)
        
        # Keep matrix for transition, generate faster transition matrix for each sentence
        self.transition_matrix_unit_prob = np.zeros((self.max_seq_length, 
                                                self.max_seq_length),
                                                dtype=self.float_dtype)
                                                
        self.initial_transition_matrix_unit_prob = np.zeros((self.max_seq_length),
                                                            dtype=self.float_dtype)
            
        # Posterior: Count transition posterior after each batch
        self.posterior_jumps = np.zeros((self.size_jump_width_set_discrete), 
                                        dtype=self.float_dtype)
        
        self.posterior_p0 = self.float_dtype(0)
        
    def prepare_transition_matrix_discrete(self, sentence_lengths, max_seq_length):
        if self.jump_width_update_freq != 0:
            transition_matrix_unit = self.transition_matrix_unit_prob
            transition_matrixs = np.zeros((self.batch_size, max_seq_length*2, max_seq_length*2), dtype=self.float_dtype)
            initial_transitions = np.zeros((self.batch_size, max_seq_length*2), dtype=self.float_dtype)
            for idx, sentence_length in enumerate(sentence_lengths):
                transition_matrix = np.zeros((max_seq_length*2, max_seq_length*2), dtype=self.float_dtype)
                #----------------------------------------------------------------------
                # trans_distance_mask: Keep position of higher or lower than max_jump_width
                trans_distance_mask = np.zeros(self.transition_matrix_unit_prob.shape, dtype=self.float_dtype)
                
                for i in range(sentence_length-self.max_jump_width+1):
                    trans_distance_mask[i, self.max_jump_width + i + 1: sentence_length] = \
                    np.divide(self.float_dtype(1.) , self.float_dtype((sentence_length - self.max_jump_width +1 - i)) )
        
                trans_higher_max_jump_width_matrix = trans_distance_mask * self.transition_set_prob.eval()[-1]
                trans_lower_max_jump_width_matrix = np.transpose(trans_distance_mask) * self.transition_set_prob.eval()[0]
                
                # Add probs into trans_distance_matrix_unit
                transition_matrix_unit = transition_matrix_unit + trans_higher_max_jump_width_matrix
                transition_matrix_unit = transition_matrix_unit + trans_lower_max_jump_width_matrix
                
                # p(EOS|EOS) = 1.
                transition_matrix_unit[:sentence_length, sentence_length:] = self.float_dtype(0.)
                transition_matrix_unit[sentence_length:, :sentence_length] = self.float_dtype(0.)
                transition_matrix_unit[sentence_length:, sentence_length:] = self.float_dtype(1.)
                
                # Add into main matrix: trans_distance_matrix
                transition_matrix[:max_seq_length, : max_seq_length] = transition_matrix_unit[:max_seq_length, : max_seq_length]
        
                #----------------------------------------------------------------------
                # Make null transition
                trans_distance_matrix_unit_null = np.zeros(self.transition_matrix_unit_prob.shape, dtype=self.float_dtype)
                np.fill_diagonal(trans_distance_matrix_unit_null, self.p0.eval())
                
                # Add into main matrix: trans_distance_matrix
                transition_matrix[:max_seq_length, max_seq_length: ] = trans_distance_matrix_unit_null[:max_seq_length, :max_seq_length]
                
                #----------------------------------------------------------------------
                # Sum of all probs = 1.
                prob_sum = (1. - self.p0.eval())
                sum_trans_distance = np.sum(transition_matrix[:sentence_length,:sentence_length], 
                                                                  axis=1, keepdims=True)
                                                                  
                transition_matrix[:sentence_length, :sentence_length] = \
                np.divide(transition_matrix[:sentence_length, :sentence_length] *(prob_sum), sum_trans_distance)
                
                transition_matrix[max_seq_length:] = transition_matrix[:max_seq_length]
                
                #----------------------------------------------------------------------
                # p(EOS|word) = 0
                transition_matrix[:, sentence_length:max_seq_length] = self.float_dtype(0.)
                transition_matrix[:, max_seq_length+sentence_length:max_seq_length*2] = self.float_dtype(0.)
                
                #----------------------------------------------------------------------
                #----------------------------------------------------------------------
                initial_transition = np.zeros((max_seq_length*2), dtype=self.float_dtype)
                initial_transition[:sentence_length] = self.float_dtype(1.)
                initial_transition = initial_transition/ np.sum(initial_transition)
                
                transition_matrixs[idx] = transition_matrix
                initial_transitions[idx] = initial_transition
        else:
            transition_matrixs = np.ones((self.batch_size, max_seq_length*2, max_seq_length*2), dtype=self.float_dtype)
            initial_transitions = np.ones((self.batch_size, max_seq_length*2), dtype=self.float_dtype)
            
            trans_distance_matrix_unit_null = np.zeros((max_seq_length, max_seq_length), dtype=self.float_dtype)
            np.fill_diagonal(trans_distance_matrix_unit_null, self.float_dtype(1.))
            transition_matrixs[:, :max_seq_length, max_seq_length:] = trans_distance_matrix_unit_null
            transition_matrixs[:, max_seq_length:, max_seq_length:] = trans_distance_matrix_unit_null
        
        return initial_transitions, transition_matrixs
    
    def get_transition_evaluation_nn(self,  transition, sentence_length):
        if self.jump_width_update_freq != 0:
            transition_matrix = transition
        else:
            transition_matrix = np.ones((2*sentence_length, 2*sentence_length),dtype=self.float_dtype)
        
            transition_matrix_unit_null = np.zeros((sentence_length, sentence_length), dtype=self.float_dtype)
            np.fill_diagonal(transition_matrix_unit_null, self.float_dtype(1.))
                
            transition_matrix[:sentence_length, sentence_length: ] = transition_matrix_unit_null 
            transition_matrix[sentence_length:] = transition_matrix[:sentence_length]
          
        initial_transition = np.zeros((sentence_length*2), dtype=self.float_dtype)
        initial_transition[:sentence_length] = self.float_dtype(1.)
        initial_transition = initial_transition/ np.sum(initial_transition)

        return initial_transition, transition_matrix
        
    def get_transition_evaluation_discrete(self, sentence_length):
        if self.jump_width_update_freq != 0:
            p0 = self.p0.eval()
            transition_matrix = np.zeros((2*sentence_length, 2*sentence_length),dtype=self.float_dtype)
                    
            for i in range(sentence_length):
                for j in range(sentence_length):
                    indice = j - i
                    if indice < -self.max_jump_width:
                        count_lower_minus_max_jump_width = self.float_dtype((i - self.max_jump_width))
                        transition_matrix[i][j] = np.divide(self.transition_set_prob.eval()[0], count_lower_minus_max_jump_width)
                    elif indice > self.max_jump_width:
                        count_higher_plus_max_jump_width = self.float_dtype((sentence_length - self.max_jump_width + 1 - i))
                        transition_matrix[i][j] = self.transition_set_prob.eval()[-1]/count_higher_plus_max_jump_width
                    else:
                        transition_matrix[i][j] = self.transition_set_prob.eval()[self.max_jump_width + indice +1]
            
            
            prob_sum = (1. - p0)
            sum_transition_matrix = np.sum(transition_matrix[:sentence_length , :sentence_length], 
                                                              axis=1, keepdims=True)
                                                              
            transition_matrix[:sentence_length, :sentence_length] = \
            np.divide(transition_matrix[:sentence_length, :sentence_length] *(prob_sum), sum_transition_matrix)
            
            #----------------------------------------------------------------------
            # Make null transition
            transition_matrix_unit_null = np.zeros((sentence_length, sentence_length), dtype=self.float_dtype)
            np.fill_diagonal(transition_matrix_unit_null, p0)
            
            # Add into main matrix: trans_distance_matrix
            transition_matrix[:sentence_length, sentence_length: ] = transition_matrix_unit_null 
            
            transition_matrix[sentence_length:] = transition_matrix[:sentence_length]
    
            #----------------------------------------------------------------------
        else:
            transition_matrix = np.ones((2*sentence_length, 2*sentence_length),dtype=self.float_dtype)
        
            transition_matrix_unit_null = np.zeros((sentence_length, sentence_length), dtype=self.float_dtype)
            np.fill_diagonal(transition_matrix_unit_null, self.float_dtype(1.))
                
            transition_matrix[:sentence_length, sentence_length: ] = transition_matrix_unit_null 
            transition_matrix[sentence_length:] = transition_matrix[:sentence_length]
            
        #----------------------------------------------------------------------      
        initial_transition = np.zeros((sentence_length*2), dtype=self.float_dtype)
        initial_transition[:sentence_length] = self.float_dtype(1.)
        initial_transition = initial_transition/ np.sum(initial_transition)

        return initial_transition, transition_matrix

    def get_transition_posterior_for_each_batch(self, expected_transition, target_length, max_seq_length):
        if self.jump_width_update_freq != 0:
            #----------------------------------------------------------
            # Get transition posterior: Transition distance
            for j in range(target_length):
                for i in range(target_length):
                    indice = j - i
                    if indice < -self.max_jump_width:
                        self.posterior_jumps[0] += expected_transition[i,j]
                        self.posterior_jumps[0] += expected_transition[i+max_seq_length,j]
                    elif indice > self.max_jump_width:
                        self.posterior_jumps[-1] += expected_transition[i,j]
                        self.posterior_jumps[-1] += expected_transition[i+max_seq_length,j]
                    else:
                        self.posterior_jumps[self.max_jump_width + indice +1] += expected_transition[i,j]
                        self.posterior_jumps[self.max_jump_width + indice +1] += expected_transition[i+max_seq_length,j]
                        
            #----------------------------------------------------------
            # Get transition posterior: p0
            for i in range(target_length):
                self.posterior_p0 += expected_transition[i, i+max_seq_length]
                self.posterior_p0 += expected_transition[i+max_seq_length, i+max_seq_length]

    def update_jump_width_probability_set(self):
        if self.jump_width_update_freq != 0:
            jump_prob_lissage = 1. / self.size_jump_width_set_discrete
            transition_set_prob = np.divide(self.posterior_jumps + jump_prob_lissage,
                                          np.sum(self.posterior_jumps) + (jump_prob_lissage * self.size_jump_width_set_discrete ), dtype=self.float_dtype )
                
            p0 = np.divide(self.posterior_p0, (self.posterior_p0 + np.sum(self.posterior_jumps)), dtype=self.float_dtype )
            
            assign_transition = [tf.assign(self.transition_set_prob, tf.identity(transition_set_prob)),
                                 tf.assign(self.p0, tf.identity(p0)) ]
                                 
            self.sess.run(assign_transition)
            
            # Create a unit matrix from 0 to max_seq_length: speed training
            self.update_transition_matrix_unit()
            
            self.posterior_jumps = np.zeros((self.size_jump_width_set_discrete), dtype=self.float_dtype)
            self.posterior_p0 = self.float_dtype(0)
            
            
    def update_transition_matrix_unit(self):
        for i in range(self.max_seq_length):
            for j in range(self.max_seq_length):
                indice = j - i
                if indice < -self.max_jump_width or indice > self.max_jump_width:
                    self.transition_matrix_unit_prob[i][j] = 0
                else:
                    self.transition_matrix_unit_prob[i][j] = \
                    self.transition_set_prob.eval()[self.max_jump_width + indice + 1]
                    