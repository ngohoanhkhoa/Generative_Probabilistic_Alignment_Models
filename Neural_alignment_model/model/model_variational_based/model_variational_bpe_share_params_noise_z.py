#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, time, math
import numpy as np
import tensorflow as tf
import framework.tools as tools

from tensorflow.python import pywrap_tensorflow

from model.model import BaseModel

class Model(BaseModel):

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
        
    def initialize_transition_parameter_nn(self):
        # =============================================================================
        # Initialize transition parameters nn        
        # =============================================================================
        self.max_jump_width = int(self.FLAGS.max_jump_width)
        self.size_jump_width_set_nn = self.max_jump_width*2 + 1 # Large range of value [-100, ..., 0, ... 100]
    
    def get_reference_BPE_idx(self):
        source_idx_batch = []
        target_idx_batch = []
        
        source_file = open(self.FLAGS.source_idx_data)
        source_lines = source_file.readlines()
        
        target_file = open(self.FLAGS.target_idx_data)
        target_lines = target_file.readlines()
        
        for i in source_lines:
            source_idx_batch.append(i.split())
    
        for i in target_lines:
            target_idx_batch.append(i.split())
            
        self.source_idx_batch = source_idx_batch
        self.target_idx_batch = target_idx_batch
        
    def converse_BPE_to_word(self, alignment_set, source_idx_batch, target_idx_batch):
        word_alignments = []
        for subword_alignment, ref_s, ref_t in zip(alignment_set, source_idx_batch , target_idx_batch):
            word_alignment = []
            for subword in subword_alignment:
                subword_src = subword.split('-')[0]
                subword_tgt = subword.split('-')[1]
                word_src = 0
                word_tgt = 0
                for s in ref_s:
                    word_src_ref = s.split('-')[0]
                    subword_src_ref = s.split('-')[1]
                    if subword_src == subword_src_ref:
                        word_src = word_src_ref
                for t in ref_t:
                    word_tgt_ref = t.split('-')[0]
                    subword_tgt_ref = t.split('-')[1]
                    if subword_tgt == subword_tgt_ref:
                        word_tgt = word_tgt_ref
                if word_src + '-'+ word_tgt not in word_alignment:
                    word_alignment.append(word_src + '-'+word_tgt)
            word_alignments.append(word_alignment)
            
        return word_alignments

# =============================================================================
# Build model
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
        
        self.sources_null = tf.placeholder(dtype=self.tf_int_dtype,
                                          shape=(1),
                                          name='sources_null')
        
        self.targets_noise = tf.placeholder(dtype=self.tf_int_dtype,
                                      shape=(None,None),
                                      name='targets_noise')
        
        self.target_noise_lengths = tf.placeholder(dtype=self.tf_int_dtype,
                                             shape=(None),
                                             name='target_noise_lengths')
        
        self.target_reconstruction_transition_matrix = tf.placeholder(dtype=self.tf_float_dtype,
                                                                      shape=(None, None, None),
                                                                      name='target_reconstruction_transition_matrix')
        
        self.sources_noise = tf.placeholder(dtype=self.tf_int_dtype,
                                      shape=(None,None),
                                      name='sources_noise')
        
        self.source_noise_lengths = tf.placeholder(dtype=self.tf_int_dtype,
                                             shape=(None),
                                             name='source_noise_lengths')
        
        self.source_reconstruction_transition_matrix = tf.placeholder(dtype=self.tf_float_dtype,
                                                                      shape=(None, None, None),
                                                                      name='source_reconstruction_transition_matrix')


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

        
# =============================================================================
# 
# =============================================================================
        
    def build_update(self):
        self.annealing_KL_divergence = tf.Variable(0., trainable=False, name='annealing_KL_divergence' ,
                                                   dtype=self.tf_float_dtype)
        
        # Cost for e -> y -> e, f
        e_y_reconstruction_expectation = tf.reduce_mean(self.e_y_reconstruction_expectation)
        f_y_alignment_expectation = tf.reduce_mean(self.f_y_alignment_expectation)
        y_KL_divergence = tf.reduce_mean(tf.reduce_sum(self.y_KL_divergence, axis=1))
        
        self.e_y_cost_reconstruction_expectation =  (-e_y_reconstruction_expectation)
        self.f_y_cost_alignment_expectation =  (-f_y_alignment_expectation)
        self.y_cost_KL_divergence = y_KL_divergence * self.annealing_KL_divergence
        self.y_cost_KL_divergence_eval = y_KL_divergence
        
        self.y_cost = (self.FLAGS.alpha_reconstruction_expectation * self.e_y_cost_reconstruction_expectation) \
        + (self.FLAGS.alpha_alignment_expectation * self.f_y_cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.y_cost_KL_divergence)
                
        
        # Cost for f -> x -> f, e
        f_x_reconstruction_expectation = tf.reduce_mean(self.f_x_reconstruction_expectation)
        e_x_alignment_expectation = tf.reduce_mean(self.e_x_alignment_expectation)
        x_KL_divergence = tf.reduce_mean(tf.reduce_sum(self.x_KL_divergence, axis=1))
        
        self.f_x_cost_reconstruction_expectation =  (-f_x_reconstruction_expectation)
        self.e_x_cost_alignment_expectation =  (-e_x_alignment_expectation)
        self.x_cost_KL_divergence = x_KL_divergence * self.annealing_KL_divergence
        self.x_cost_KL_divergence_eval = x_KL_divergence
               
        self.x_cost = (self.FLAGS.alpha_reconstruction_expectation * self.f_x_cost_reconstruction_expectation) \
        + (self.FLAGS.alpha_alignment_expectation * self.e_x_cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.x_cost_KL_divergence)
        
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        gradients = tf.gradients(self.y_cost + self.x_cost, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = self.opt.apply_gradients(zip(clip_gradients,trainable_params),
                                                         global_step=self.global_step)
        
# =============================================================================
# 
# =============================================================================
    
    def get_latent_variable(self, input_token, input_null, lengths, vocabulary_size):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            target_embedding= tf.get_variable(name='embedding',
                                                shape=[vocabulary_size,
                                                       self.FLAGS.embedding_size],
                                                initializer=self.initializer,
                                                dtype=self.tf_float_dtype)
            
            target_embedded_word = tf.nn.embedding_lookup(params=target_embedding,
                                                      ids=input_token)
        
            target_embedded_null = tf.nn.embedding_lookup(params=target_embedding,
                                                      ids=input_null)
    
            with tf.variable_scope('lstmTarget'):
                lstm_fw = tf.keras.layers.LSTM(self.FLAGS.hidden_units, return_sequences=True)
                lstm_bw = tf.keras.layers.LSTM(self.FLAGS.hidden_units, return_sequences=True, go_backwards=True)
    
                output_fw = lstm_fw(inputs=target_embedded_word)
                output_bw = lstm_bw(inputs=target_embedded_word)
    
                # Concat and bridge layer
                target_state = tf.concat([output_fw, output_bw], axis=2)
                
                target_state = tf.layers.Dense(units=self.FLAGS.embedding_size,
                                    activation=None,
                                    use_bias=False,
                                    name='nnLayerLSTM')(target_state)
                
                # Sum
                #target_state = output_fw + output_bw
                
            nnLayer_encoder = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                    activation=tf.nn.tanh,
                                    use_bias=True,
                                    name='nnLayerEncoder')
            
            target_state = nnLayer_encoder(target_state)
            target_state_null = nnLayer_encoder(target_embedded_null)
            
            nnLayer_loc = tf.layers.Dense(units=self.FLAGS.embedding_sample_size,
                                        activation=None,
                                        use_bias=True,
                                        name='nnLayerLoc')
            
            nnLayer_scale = tf.layers.Dense(units=self.FLAGS.embedding_sample_size,
                                        activation=tf.nn.softplus,
                                        use_bias=True,
                                        name='nnLayerScale')
            
            location = nnLayer_loc(target_state)
            scale = nnLayer_scale(target_state)
            
            location_null = nnLayer_loc(target_state_null)
            scale_null = nnLayer_scale(target_state_null)
            
            hidden = location
            hidden_null = location_null
    
            eps = tf.random_normal(tf.shape(location),mean=0.0,stddev=1.0,dtype=self.tf_float_dtype)
            eps_null = tf.random_normal(tf.shape(location_null),mean=0.0,stddev=1.0,dtype=self.tf_float_dtype)
            
            hidden_variable = location + eps*tf.exp(scale * .5)
            hidden_variable_null = location_null + eps_null*tf.exp(scale_null * .5)
            
            target_mask = tf.sequence_mask(lengths=lengths, maxlen=tf.shape(location)[1],
                                           dtype=self.tf_float_dtype, name='mask')
            
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(location) + tf.square(scale) - tf.log(tf.square(scale)) - 1, axis=2)
            
            KL_divergence = KL_divergence * target_mask
            
            return hidden_variable, hidden, hidden_variable_null, hidden_null, KL_divergence
        
    def build_variational_encoder(self):
        with tf.variable_scope('target_y_encoder', reuse=tf.AUTO_REUSE):
            self.y_hidden_variable, self.y_hidden, \
            self.y_hidden_variable_null, self.y_hidden_null, \
            self.y_KL_divergence = self.get_latent_variable(self.targets_noise, 
                                                          self.targets_null, 
                                                          self.target_noise_lengths,
                                                          self.FLAGS.target_vocabulary_size)
            
        with tf.variable_scope('source_x_encoder', reuse=tf.AUTO_REUSE):
            self.x_hidden_variable, self.x_hidden, \
            self.x_hidden_variable_null, self.x_hidden_null, \
            self.x_KL_divergence = self.get_latent_variable(self.sources_noise, 
                                                          self.sources_null, 
                                                          self.source_noise_lengths,
                                                          self.FLAGS.source_vocabulary_size)
        
# =============================================================================
#         
# =============================================================================
        
    def build_decoder(self):
        with tf.variable_scope('target_y_decoder', reuse=tf.AUTO_REUSE):
            self.e_y_hidden_prob, self.e_y_hidden_prob_null = self.get_softmax(self.y_hidden_variable, 
                                                                         self.y_hidden_variable_null, 
                                                                         self.FLAGS.target_vocabulary_size)
            
            self.e_y_prob, self.e_y_prob_null = self.get_softmax(self.y_hidden, 
                                                                         self.y_hidden_null, 
                                                                         self.FLAGS.target_vocabulary_size)
            
            self.e_x_hidden_prob, self.e_x_hidden_prob_null = self.get_softmax(self.x_hidden_variable, 
                                                                         self.x_hidden_variable_null, 
                                                                         self.FLAGS.target_vocabulary_size)
            
            self.e_x_prob, self.e_x_prob_null = self.get_softmax(self.x_hidden, 
                                                                         self.x_hidden_null, 
                                                                         self.FLAGS.target_vocabulary_size)
        
        with tf.variable_scope('source_x_decoder', reuse=tf.AUTO_REUSE):
            self.f_x_hidden_prob, self.f_x_hidden_prob_null = self.get_softmax(self.x_hidden_variable, 
                                                                         self.x_hidden_variable_null, 
                                                                         self.FLAGS.source_vocabulary_size)
            
            self.f_x_prob, self.f_x_prob_null = self.get_softmax(self.x_hidden, 
                                                                 self.x_hidden_null, 
                                                                 self.FLAGS.source_vocabulary_size)
            
            self.f_y_hidden_prob, self.f_y_hidden_prob_null = self.get_softmax(self.y_hidden_variable, 
                                                                         self.y_hidden_variable_null, 
                                                                         self.FLAGS.source_vocabulary_size)
            
            self.f_y_prob, self.f_y_prob_null = self.get_softmax(self.y_hidden, 
                                                                 self.y_hidden_null, 
                                                                 self.FLAGS.source_vocabulary_size)
       
                
    def get_softmax(self, hidden_variable, hidden_variable_null, vocabulary_size):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            
            nnLayer_emission = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                        activation=tf.nn.tanh,
                                        use_bias=True,
                                        name='nnLayerEmission')
                                               
            output_state = nnLayer_emission(hidden_variable)
            output_state_null = nnLayer_emission(hidden_variable_null)
            
            nnLayerVocabulary = tf.layers.Dense(units=vocabulary_size,
                                                name='nnLayerVocabulary')
            
            output_value = nnLayerVocabulary(output_state)
            output_value_null = nnLayerVocabulary(output_state_null)
            
            output_prob_softmax = tf.nn.softmax(output_value)
            output_prob_null = tf.nn.softmax(output_value_null)
            
            return output_prob_softmax, output_prob_null
        
# =============================================================================
#         
# =============================================================================
        
    def build_reconstruction(self):
        with tf.variable_scope('reconstruction', reuse=tf.AUTO_REUSE):            
            
            self.e_y_reconstruction = self.get_reconstruction(self.e_y_prob, self.targets)
            
            self.e_y_reconstruction_expectation = self.get_recontruction_expectation(input_prob_softmax= self.e_y_hidden_prob,
                                                                            input_sentences= self.targets_noise,
                                                                            output_sentences= self.targets,
                                                                            input_lengths= self.target_noise_lengths,
                                                                            output_lengths= self.target_lengths)
            
            self.f_x_reconstruction = self.get_reconstruction(self.f_x_prob, self.sources)
            
            self.f_x_reconstruction_expectation = self.get_recontruction_expectation(input_prob_softmax= self.f_x_hidden_prob,
                                                                            input_sentences= self.sources_noise,
                                                                            output_sentences= self.sources,
                                                                            input_lengths= self.source_noise_lengths,
                                                                            output_lengths= self.source_lengths)
        
    def get_recontruction_expectation(self,
                                  input_prob_softmax,
                                  input_sentences, 
                                  output_sentences,
                                  input_lengths,
                                  output_lengths):
        
        emission = self.get_reconstruction(input_prob_softmax, output_sentences)
                
        input_mask = tf.cast(tf.sequence_mask(input_lengths,tf.shape(input_sentences)[1]), self.tf_float_dtype)
        transition = tf.div(input_mask, tf.cast(tf.expand_dims(input_lengths, -1), self.tf_float_dtype))
        transition = tf.expand_dims(transition, 1)
        transition = tf.tile(transition, [1, tf.shape(output_sentences)[1], 1])
        
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
        
    def get_reconstruction(self, input_prob_softmax, sentences):
        input_scan = (input_prob_softmax, sentences)
                
        def get_output_emission(value):
            prob = value[0]
            sent = value[1]
            
            emission_prob = tf.gather(prob, sent, axis=-1)
            
            return emission_prob, 0
    
        emission_prob, _ = tf.map_fn(get_output_emission,input_scan, dtype=(self.tf_float_dtype,tf.int32))
        
        return emission_prob

# =============================================================================
# 
# =============================================================================
        
        
    def build_alignment(self):
        with tf.variable_scope('alignment', reuse=tf.AUTO_REUSE):
        
            self.f_y_emission = self.get_emission(self.f_y_prob, self.f_y_prob_null, self.sources)
            
            self.f_y_alignment_expectation = self.get_alignment_expectation(input_prob_softmax= self.f_y_hidden_prob, 
                                                                            input_prob_softmax_null= self.f_y_hidden_prob_null,
                                                                            input_sentences= self.targets_noise,
                                                                            output_sentences= self.sources,
                                                                            input_lengths= self.target_noise_lengths,
                                                                            output_lengths= self.source_lengths)
            

            
            self.e_x_emission = self.get_emission(self.e_x_prob, self.e_x_prob_null, self.targets)
            
            self.e_x_alignment_expectation = self.get_alignment_expectation(input_prob_softmax= self.e_x_hidden_prob, 
                                                                            input_prob_softmax_null= self.e_x_hidden_prob_null,
                                                                            input_sentences= self.sources_noise,
                                                                            output_sentences= self.targets,
                                                                            input_lengths= self.source_noise_lengths,
                                                                            output_lengths= self.target_lengths)
    
    def get_alignment_expectation(self,
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
        
    def get_emission(self, input_prob_softmax, input_prob_softmax_null, sentences):
        input_scan = (input_prob_softmax, sentences)
        
        self.null_value = input_prob_softmax_null
                
        def get_output_emission(value):
            prob = value[0]
            sent = value[1]
            
            prob = tf.gather(prob, sent, axis=-1)
            prob_null = tf.gather(self.null_value, sent, axis=-1)
            
            prob_null = tf.tile(prob_null, [tf.shape(prob)[0], 1])
            
            emission_prob = tf.concat([prob, prob_null], axis=0)
            
            return emission_prob, 0
    
        emission_prob, _ = tf.map_fn(get_output_emission,input_scan, dtype=(self.tf_float_dtype,tf.int32))
        
        return emission_prob

# =============================================================================
# 
# =============================================================================
    def print_log(self, update_info, start_time):
        y_cost = update_info[0]
        e_y_cost_reconstruction = update_info[1]
        f_y_cost_alignment = update_info[2]
        y_cost_KL = update_info[3]
        
        x_cost = update_info[4]
        f_x_cost_reconstruction = update_info[5]
        e_x_cost_alignment = update_info[6]
        x_cost_KL = update_info[7]
        
        if self.global_step.eval() % self.FLAGS.display_freq == 0:
            self.log.info('Epoch %d , Step %d , Cost e -> y: %.5f (R: %.5f, A: %.5f, KL: %.5f) in %ds at %s', 
                          self.global_epoch_step.eval(),
                          self.global_step.eval(),
                          y_cost,e_y_cost_reconstruction,f_y_cost_alignment, y_cost_KL,
                          time.time() - self.start_time,
                          tools.get_time_now())
            
            self.log.info('Epoch %d , Step %d , Cost f -> x: %.5f (R: %.5f, A: %.5f, KL: %.5f) in %ds at %s', 
                          self.global_epoch_step.eval(),
                          self.global_step.eval(),
                          x_cost,f_x_cost_reconstruction,e_x_cost_alignment, x_cost_KL,
                          time.time() - self.start_time,
                          tools.get_time_now())
            
            self.start_time = time.time()
        
    def train_batch(self,
             sources,
             sources_null,
             source_lengths,
             targets,
             targets_null,
             target_lengths,
             sources_noise,
             source_noise_lengths,
             source_mask_noise,
             targets_noise,
             target_noise_lengths,
             target_mask_noise):
        
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
        
        input_feed[self.sources_noise.name] = sources_noise
        input_feed[self.targets_noise.name] = targets_noise
        
        input_feed[self.source_noise_lengths.name] = source_noise_lengths
        input_feed[self.target_noise_lengths.name] = target_noise_lengths
        
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
    
    def decode_reconstruction(self,
             sources,
             sources_null,
             source_lengths,
             targets,
             targets_null,
             target_lengths,
             sources_noise,
             source_noise_lengths,
             source_mask_noise,
             targets_noise,
             target_noise_lengths,
             target_mask_noise):
        
        source_length = source_lengths[0]
        target_length = target_lengths[0]
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
        
        input_feed[self.sources_noise.name] = sources
        input_feed[self.targets_noise.name] = targets
        
        input_feed[self.source_noise_lengths.name] = source_lengths
        input_feed[self.target_noise_lengths.name] = target_lengths
        
        output_feed = [self.e_y_reconstruction,
                       self.f_x_reconstruction]
        #----------------------------------------------------------------------
        outputs = self.sess.run(output_feed, input_feed)
        
        e_y_reconstruction_final = np.array(outputs[0][0])
        f_x_reconstruction_final = np.array(outputs[1][0])
        
        e_y_initial_transition = np.ones((target_length), dtype=self.float_dtype)
        e_y_transition = np.ones((target_length, target_length),dtype=self.float_dtype)
        f_x_initial_transition = np.ones((source_length), dtype=self.float_dtype)
        f_x_transition = np.ones((source_length, source_length),dtype=self.float_dtype)

        
        e_y_state_seq, e_y_likelihood_seq = self.viterbi(np.array(target),
                                        target_length,
                                        e_y_initial_transition,
                                        e_y_transition, 
                                        e_y_reconstruction_final)
        
        f_x_state_seq, f_x_likelihood_seq = self.viterbi(np.array(source),
                                        source_length,
                                        f_x_initial_transition,
                                        f_x_transition, 
                                        f_x_reconstruction_final)
        
        
        f_x_state_seq_final = [source[i] for i in f_x_state_seq] 
        e_y_state_seq_final = [target[i] for i in e_y_state_seq]
        
        return [e_y_state_seq_final], [e_y_likelihood_seq], [e_y_reconstruction_final], \
    [f_x_state_seq_final], [f_x_likelihood_seq], [f_x_reconstruction_final]
    
    
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
        
    def eval(self,
             sources,
             sources_null,
             source_lengths,
             targets,
             targets_null,
             target_lengths,
             sources_noise,
             source_noise_lengths,
             source_mask_noise,
             targets_noise,
             target_noise_lengths,
             target_mask_noise):
        
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
        
        input_feed[self.sources_noise.name] = sources
        input_feed[self.targets_noise.name] = targets
        
        input_feed[self.source_noise_lengths.name] = source_lengths
        input_feed[self.target_noise_lengths.name] = target_lengths
        
        output_feed = [self.f_y_emission,
                       self.e_x_emission]
        #----------------------------------------------------------------------
        outputs = self.sess.run(output_feed, input_feed)
        
        f_y_emission_final = np.array(outputs[0][0])
        e_x_emission_final = np.array(outputs[1][0])
        
        f_y_initial_transition, f_y_transition = self.get_transition_evaluation_nn(target_length)
        e_x_initial_transition, e_x_transition = self.get_transition_evaluation_nn(source_length)
        
        
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
    
    
    def get_kl(self,
             sources,
             sources_null,
             source_lengths,
             targets,
             targets_null,
             target_lengths,
             sources_noise,
             source_noise_lengths,
             source_mask_noise,
             targets_noise,
             target_noise_lengths,
             target_mask_noise):
        
        input_feed = {}
        input_feed[self.keep_prob.name] = 1.0
        input_feed[self.sources.name] = sources
        input_feed[self.targets.name] = targets
        input_feed[self.source_lengths.name] = source_lengths
        input_feed[self.target_lengths.name] = target_lengths
        
        input_feed[self.targets_null.name] = targets_null
        input_feed[self.sources_null.name] = sources_null
        
        input_feed[self.sources_noise.name] = sources
        input_feed[self.targets_noise.name] = targets
        
        input_feed[self.source_noise_lengths.name] = source_lengths
        input_feed[self.target_noise_lengths.name] = target_lengths
        
        output_feed = [self.f_y_cost_alignment_expectation,
                       self.e_y_cost_reconstruction_expectation,
                       self.y_cost_KL_divergence_eval,
                       self.e_x_cost_alignment_expectation,
                       self.f_x_cost_reconstruction_expectation,
                       self.x_cost_KL_divergence_eval]
        
        #----------------------------------------------------------------------
        outputs = self.sess.run(output_feed, input_feed)
        
        return outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5]
    
# =============================================================================
#         
# =============================================================================
        
    def prepare_batch(self, seqs_x, seqs_y):
        x_lengths = np.array([len(s) for s in seqs_x])
        y_lengths = np.array([len(s) for s in seqs_y])
        x_lengths_max = np.max(x_lengths)
        y_lengths_max = np.max(y_lengths)
        
        self.batch_size = len(seqs_x)
                     
        y_null = [self.FLAGS.target_vocabulary_size - 1]
        x_null = [self.FLAGS.source_vocabulary_size - 1]
        
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
            
        word_y_noise, word_y_noise_lengths, mask_ys = self.add_noise(word_y, seqs_y, self.batch_size)
        word_x_noise, word_x_noise_lengths, mask_xs = self.add_noise(word_x, seqs_x, self.batch_size)
            
        return word_x, x_null, word_x_lengths, word_y, y_null, word_y_lengths, \
                word_x_noise, word_x_noise_lengths, mask_xs,\
                word_y_noise, word_y_noise_lengths, mask_ys,
                

    def add_noise(self, word_y, seqs_y, batch_size):
        #-----------------
        # Add noise
        
        prob_drop_word_noise = self.FLAGS.prob_drop_word_noise
        max_jump_width_noise = self.FLAGS.max_jump_width_noise
        seqs_y_noise = []
        for idx, s_y in enumerate(seqs_y):
            s_y_drop_word = []
            s_y_mask = np.random.choice([0, 1], size=np.shape(s_y), p=[prob_drop_word_noise, 1. - prob_drop_word_noise])
            if np.sum(s_y_mask) == 0:
                s_y_mask[0] = 1
            for mask, w_y in zip(s_y_mask, s_y):
                if mask != 0:
                    s_y_drop_word.append(w_y)
            
            s_y_shuffle = []
            
            random_vector = np.random.uniform(0, max_jump_width_noise + 1, len(s_y_drop_word))
            noise_idx = np.array(range(len(s_y_drop_word))) + random_vector
            new_idx = np.argsort(noise_idx)
            for idx_ in new_idx:
                s_y_shuffle.append(s_y_drop_word[idx_])
                
            seqs_y_noise.append(s_y_shuffle)
            
        y_noise_lengths = np.array([len(s) for s in seqs_y_noise])
        y_noise_lengths_max = np.max(y_noise_lengths)
        
        word_y_noise = np.ones((batch_size,
                     y_noise_lengths_max),
                    dtype=self.int_dtype) * tools.end_token
        
        word_y_noise_lengths = np.ones((batch_size),
                                    dtype=self.int_dtype)
        
        for idx, s_y in enumerate(seqs_y_noise):
            word_y_noise[idx, :y_noise_lengths[idx]] = s_y
            word_y_noise_lengths[idx] = y_noise_lengths[idx]
            
        
        mask_ys = np.zeros((np.shape(word_y)[0],np.shape(word_y)[1],np.shape(word_y_noise)[1]), dtype=self.float_dtype)
        for s_idx, [s_y, s_y_noise] in enumerate(zip(word_y, word_y_noise)):
            for w_idx_y, w_y in enumerate(s_y):
                for w_idx_y_noise, w_y_noise in enumerate(s_y_noise):
                    if w_y == w_y_noise:
                        mask_ys[s_idx, w_idx_y, w_idx_y_noise] = 1.
                        
                        
        return word_y_noise, word_y_noise_lengths, mask_ys
    
# =============================================================================
# 
# =============================================================================

    def train(self):
            
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
        

    def evaluate(self):
        if self.FLAGS.valid_freq!= 0 and self.global_step.eval() % self.FLAGS.valid_freq == 0:
            self.log.info('------------------')
            self.evaluate_alignment()
            self.evaluate_reconstruction()
            self.evaluate_kl()
            self.log.info('------------------')
        
    def evaluate_reconstruction(self):
        # Execute a validation step
            start_time = time.time()
            
            y_reconstruction_set = []
            y_accuracy_set = []
            
            x_reconstruction_set = []
            x_accuracy_set = []
            
            for valid_seq in self.valid_set:
                batch = self.prepare_batch(*valid_seq)
                eval_info = self.decode_reconstruction(*batch)
                
                
                for idx in range(1):
                    target_length = batch[5][idx]
                    target = batch[3][idx]
                    source_length = batch[2][idx]
                    source = batch[0][idx]
                    
                    y_num_word_correct = 0
                    for index_word, (predited, correct) in enumerate(zip(eval_info[0][idx], target)):
                        if index_word < target_length:
                            if predited == correct:
                                y_num_word_correct+=1
                                
                    x_num_word_correct = 0
                    for index_word, (predited, correct) in enumerate(zip(eval_info[3][idx], source)):
                        if index_word < source_length:
                            if predited == correct:
                                x_num_word_correct+=1
                                
                    y_accuracy_set.append(y_num_word_correct/target_length)
                    x_accuracy_set.append(x_num_word_correct/source_length)
                    
                y_reconstruction_set.append(np.prod(eval_info[1]))
                x_reconstruction_set.append(np.prod(eval_info[4]))
                    
            y_ACC = np.mean(y_accuracy_set)
            y_likelihood_score = np.mean(y_reconstruction_set)
            
            x_ACC = np.mean(x_accuracy_set)
            x_likelihood_score = np.mean(x_reconstruction_set)
            
            time_output = time.time() - start_time
            
            self.log.info('VALIDATION f-e (e->y): Epoch %d , Step %d , Likelihood: %.10f , ACC: %.5f in %ds at %s',
                          self.global_epoch_step.eval(), 
                          self.global_step.eval() , 
                          y_likelihood_score, 
                          y_ACC, 
                          time_output, 
                          tools.get_time_now())
            
            self.log.info('VALIDATION e-f (f->x): Epoch %d , Step %d , Likelihood: %.10f , ACC: %.5f in %ds at %s',
                          self.global_epoch_step.eval(), 
                          self.global_step.eval() , 
                          x_likelihood_score, 
                          x_ACC, 
                          time_output, 
                          tools.get_time_now())

            self.start_time = time.time()
            
    def evaluate_alignment(self):
        # Execute a validation step
            start_time = time.time()
            
            f_y_alignment_set = []
            f_y_likelihood_set = []
            f_y_emission_set = []
            
            e_x_alignment_set = []
            e_x_likelihood_set = []
            e_x_emission_set = []
            
            for valid_seq in self.valid_set:
                batch = self.prepare_batch(*valid_seq)
                eval_info = self.eval(*batch)
                        
                #--------------------------------------------------------------
                                
                for idx in range(1):
                    source_length = batch[2][idx]
                    target_length = batch[5][idx]
                    
                    f_y_alignment = []
                    for index_source, index_target in enumerate(eval_info[0][idx]):
                        if index_source < source_length and index_target < target_length:
                            # Check alignment reference: Start from 0 or 1 ?
                            f_y_alignment.append(str(index_source+self.evaluate_alignment_start_from) \
                            + "-" + str(index_target+self.evaluate_alignment_start_from))
                            
                    f_y_alignment_set.append(f_y_alignment)
                    
                    e_x_alignment = []
                    for index_target, index_source in enumerate(eval_info[3][idx]):
                        if index_source < source_length and index_target < target_length:
                            # Check alignment reference: Start from 0 or 1 ?
                            e_x_alignment.append(str(index_target+self.evaluate_alignment_start_from) \
                            + "-" + str(index_source+self.evaluate_alignment_start_from))
                            
                    e_x_alignment_set.append(e_x_alignment)

                f_y_likelihood_set.append(eval_info[1])
                f_y_emission_set.append(eval_info[2])
                
                e_x_likelihood_set.append(eval_info[4])
                e_x_emission_set.append(eval_info[5])
                
            f_y_alignment_set = self.converse_BPE_to_word(f_y_alignment_set, self.source_idx_batch, self.target_idx_batch)
            e_x_alignment_set = self.converse_BPE_to_word(e_x_alignment_set, self.target_idx_batch, self.source_idx_batch)

            f_y_AER = self.calculate_AER(f_y_alignment_set, self.f_e_sure_batch, self.f_e_possible_batch)
            e_x_AER = self.calculate_AER(e_x_alignment_set, self.e_f_sure_batch, self.e_f_possible_batch)
            
            f_y_likelihood_score = np.mean(f_y_likelihood_set)
            e_x_likelihood_score = np.mean(e_x_likelihood_set)
            
            time_output = time.time() - start_time
            
            self.log.info('VALIDATION f-e (e->y): Epoch %d , Step %d , Likelihood: %.10f , AER: %.5f in %ds at %s',
                          self.global_epoch_step.eval(), 
                          self.global_step.eval() , 
                          f_y_likelihood_score, 
                          f_y_AER, 
                          time_output, 
                          tools.get_time_now())
            
            self.log.info('VALIDATION e-f (f->x): Epoch %d , Step %d , Likelihood: %.10f , AER: %.5f in %ds at %s',
                          self.global_epoch_step.eval(), 
                          self.global_step.eval() , 
                          e_x_likelihood_score, 
                          e_x_AER, 
                          time_output, 
                          tools.get_time_now())
            
            if f_y_AER < self.best_AER_f_e or e_x_AER < self.best_AER_e_f:
                self.best_AER_f_e = f_y_AER
                with io.open(self.get_file_path('result_f_e'), 'w') as file_handler:
                    for sent_alignment in f_y_alignment_set:
                        sent_alignment = u'{}\n'.format(sent_alignment)
                        sent_alignment = sent_alignment.replace('[','')
                        sent_alignment = sent_alignment.replace(']','')
                        sent_alignment = sent_alignment.replace(',','')
                        sent_alignment = sent_alignment.replace('\'','')
                        file_handler.write(sent_alignment)                    
                        
            
                self.best_AER_e_f = e_x_AER
                with io.open(self.get_file_path('result_e_f'), 'w') as file_handler:
                    for sent_alignment in e_x_alignment_set:
                        sent_alignment = u'{}\n'.format(sent_alignment)
                        sent_alignment = sent_alignment.replace('[','')
                        sent_alignment = sent_alignment.replace(']','')
                        sent_alignment = sent_alignment.replace(',','')
                        sent_alignment = sent_alignment.replace('\'','')
                        file_handler.write(sent_alignment)
            
            self.start_time = time.time()
            
    def evaluate_kl(self):
        # Execute a validation step
            
            y_alignment_cost_batch = []
            y_reconstruction_cost_batch = []
            y_kl_cost_batch = []
            
            x_alignment_cost_batch = []
            x_reconstruction_cost_batch = []
            x_kl_cost_batch = []
            
            for valid_seq in self.valid_set:
                batch = self.prepare_batch(*valid_seq)
                eval_info = self.get_kl(*batch)
                
                y_alignment_cost_batch.append(eval_info[0])
                y_reconstruction_cost_batch.append(eval_info[1])
                y_kl_cost_batch.append(eval_info[2])
                
                x_alignment_cost_batch.append(eval_info[3])
                x_reconstruction_cost_batch.append(eval_info[4])
                x_kl_cost_batch.append(eval_info[5])
                    
            y_alignment_cost = np.mean(y_alignment_cost_batch)
            y_reconstruction_cost = np.mean(y_reconstruction_cost_batch)
            y_kl_cost = np.mean(y_kl_cost_batch)
            
            x_alignment_cost = np.mean(x_alignment_cost_batch)
            x_reconstruction_cost = np.mean(x_reconstruction_cost_batch)
            x_kl_cost = np.mean(x_kl_cost_batch)
            
            self.log.info('VALIDATION f-e (e->y): Epoch %d , Step %d , Alignment: %.5f, Reconstruction: %.5f, KL: %5f',
                          self.global_epoch_step.eval(),
                          self.global_step.eval() ,
                          y_alignment_cost, 
                          y_reconstruction_cost,y_kl_cost)
            
            self.log.info('VALIDATION e-f (f->x): Epoch %d , Step %d , Alignment: %.5f, Reconstruction: %.5f, KL: %5f',
                          self.global_epoch_step.eval(),
                          self.global_step.eval() , 
                          x_alignment_cost, 
                          x_reconstruction_cost,x_kl_cost)

# =============================================================================
# 
# =============================================================================
            
    def get_reference(self):
        target_file = open(self.FLAGS.reference_valid_data_f_e)
        target_lines = target_file.readlines()
        target_lines = [str(line[:-1]) for line in target_lines]
        target_lines = np.reshape(target_lines, (np.int(len(target_lines)/2), 2))
        
        sure_batch = []
        possible_batch = []
        
        sure = target_lines[:,0]
        possible = target_lines[:,1]
        
        for i in sure:
            sure_batch.append(i.split())
            
        for i in possible:
            possible_batch.append(i.split())
            
        self.f_e_sure_batch = sure_batch
        self.f_e_possible_batch = possible_batch
        
        
        target_file = open(self.FLAGS.reference_valid_data_e_f)
        target_lines = target_file.readlines()
        target_lines = [str(line[:-1]) for line in target_lines]
        target_lines = np.reshape(target_lines, (np.int(len(target_lines)/2), 2))
        
        sure_batch = []
        possible_batch = []
        
        sure = target_lines[:,0]
        possible = target_lines[:,1]
        
        for i in sure:
            sure_batch.append(i.split())
            
        for i in possible:
            possible_batch.append(i.split())
            
        self.e_f_sure_batch = sure_batch
        self.e_f_possible_batch = possible_batch
        
    def calculate_AER(self, alignment_set, sure_batch, possible_batch):
        
        sure_alignment = 0.
        possible_alignment = 0.
        count_alignment = 0.
        count_sure = 0.
        for sure, possible, alignment in zip(sure_batch, possible_batch, alignment_set ):
            for w in alignment:
                if w in sure:
                    sure_alignment+=1.
                if w in possible:
                    possible_alignment+=1.
                    
            count_alignment += float(len(alignment))
            count_sure += float(len(sure))
            
        return 1. - (sure_alignment*2 + possible_alignment)/ (count_alignment + count_sure)
    
# =============================================================================
# 
# =============================================================================
        
    def print_model_info(self):
        self.log.info('Source word vocabulary size: %s', self.FLAGS.source_vocabulary_size)
        self.log.info('Target word vocabulary size: %s', self.FLAGS.target_vocabulary_size)
        self.log.info('Word hidden units: %s', self.FLAGS.hidden_units)
        self.log.info('Word embedding size: %s', self.FLAGS.embedding_size)
        
        self.log.info('Alpha reconstruction expectation: %s', self.FLAGS.alpha_reconstruction_expectation)
        self.log.info('Alpha alignment expectation: %s', self.FLAGS.alpha_alignment_expectation)
        self.log.info('Alpha KL divergence: %s', self.FLAGS.alpha_KL_divergence)
        self.log.info('Alpha KL divergence freq: %s', self.FLAGS.alpha_KL_divergence_freq)
        
        self.log.info('Drop noise rob: %s', self.FLAGS.prob_drop_word_noise)
        self.log.info('Shuffle noise width: %s', self.FLAGS.max_jump_width_noise)
