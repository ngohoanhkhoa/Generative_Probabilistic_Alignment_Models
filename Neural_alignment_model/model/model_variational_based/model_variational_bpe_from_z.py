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
        
        self.get_reference_BPE_idx()
        
        self.build_optimizer()
        self.build_initializer()
        self.build_word_model_placeholders()
        self.initialize_transition_parameter_nn()
        
        self.build_variational_encoder()
        #self.build_alignment()
        
        #self.build_update()
        
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
        
    def converse_BPE_to_word(self, alignment_set):
        word_alignments = []
        for subword_alignment, ref_s, ref_t in zip(alignment_set, self.source_idx_batch, self.target_idx_batch):
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
        
        self.z_length_max = tf.Variable(80, trainable=False,
                                        name='z_length_max',
                                        dtype=self.tf_int_dtype)
        
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
        
        reconstruction_expectation = tf.reduce_mean(tf.reduce_sum(self.reconstruction_expectation_log, axis=1))
        
        alignment_expectation = tf.reduce_mean(tf.reduce_sum(self.alignment_expectation, axis=1))
        
        KL_divergence = tf.reduce_mean(tf.reduce_sum(self.KL_divergence, axis=1))
        
        self.cost_reconstruction_expectation =  (-reconstruction_expectation)
        self.cost_alignment_expectation =  (-alignment_expectation)
        self.cost_KL_divergence = KL_divergence * self.annealing_KL_divergence
        
        self.cost = (self.FLAGS.alpha_reconstruction_expectation * self.cost_reconstruction_expectation) \
        + (self.FLAGS.alpha_alignment_expectation * self.cost_alignment_expectation) \
        + (self.FLAGS.alpha_KL_divergence * self.cost_KL_divergence)
        
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        gradients = tf.gradients(self.cost, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = self.opt.apply_gradients(zip(clip_gradients,trainable_params),
                                                         global_step=self.global_step)
        
        
    def get_encoder(self, token, null , vocabulary_size):
        embedding= tf.get_variable(name='embedding',
                                   shape=[vocabulary_size,
                                          self.FLAGS.embedding_size],
                                          initializer=self.initializer,
                                          dtype=self.tf_float_dtype)
            
        embedded_word = tf.nn.embedding_lookup(params=embedding,
                                                  ids=token)
    
        embedded_null = tf.nn.embedding_lookup(params=embedding,
                                                  ids=null)

        with tf.variable_scope('lstmEncoder', reuse=tf.AUTO_REUSE):
            lstm_fw = tf.keras.layers.LSTM(self.FLAGS.hidden_units, return_sequences=True)
            lstm_bw = tf.keras.layers.LSTM(self.FLAGS.hidden_units, return_sequences=True, go_backwards=True)

            output_fw = lstm_fw(inputs=embedded_word)
            output_bw = lstm_bw(inputs=embedded_word)
            
            output_fw_null = lstm_fw(inputs=tf.expand_dims(embedded_null, axis=1))
            output_bw_null = lstm_bw(inputs=tf.expand_dims(embedded_null, axis=1))

            # Concat and bridge layer
            state = tf.concat([output_fw, output_bw], axis=2)
            state_null = tf.concat([output_fw_null, output_bw_null], axis=2)
        
        return state, state_null
        
        
    def build_variational_encoder(self):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('sourceEncoder', reuse=tf.AUTO_REUSE):
                source_state, source_state_null = self.get_encoder(self.sources,
                                                                   self.sources_null,
                                                                   self.FLAGS.source_vocabulary_size)
            with tf.variable_scope('targetEncoder', reuse=tf.AUTO_REUSE):
                target_state, target_state_null = self.get_encoder(self.targets,
                                                                   self.targets_null,
                                                                   self.FLAGS.target_vocabulary_size)
                
            source_state = tf.div(tf.reduce_sum(source_state, axis=1),
                                  tf.tile(tf.expand_dims(tf.cast(self.target_lengths, 
                                                                 self.tf_float_dtype),axis=-1),
                                        [1,self.FLAGS.hidden_units*2]))
                
            target_state = tf.div(tf.reduce_sum(target_state, axis=1),
                                  tf.tile(tf.expand_dims(tf.cast(self.source_lengths, 
                                                                 self.tf_float_dtype),axis=-1),
                                        [1,self.FLAGS.hidden_units*2]))
                
            z_state = tf.concat([source_state, target_state], axis=1)
            z_state_null = tf.concat([source_state_null, target_state_null], axis=1)
            
            
            nnLayer_loc = tf.layers.Dense(units=self.FLAGS.embedding_sample_size,
                                        activation=tf.nn.tanh,
                                        use_bias=True,
                                        name='nnLayerLoc')
            
            nnLayer_scale = tf.layers.Dense(units=self.FLAGS.embedding_sample_size,
                                        activation=tf.nn.softplus,
                                        use_bias=True,
                                        name='nnLayerScale')
            
            location = nnLayer_loc(z_state)
            scale = nnLayer_scale(z_state)
            
            location = tf.tile(tf.expand_dims(location, axis=1), [1,self.z_length_max,1])
            scale = tf.tile(tf.expand_dims(scale, axis=1), [1,self.z_length_max,1])
            
            
            location_null = nnLayer_loc(z_state_null)
            scale_null = nnLayer_scale(z_state_null)
            
            location_null = tf.tile(tf.expand_dims(location_null, axis=1), [1,self.z_length_max,1])
            scale_null = tf.tile(tf.expand_dims(scale_null, axis=1), [1,self.z_length_max,1])
            
            self.z_hidden = location
            self.z_hidden_null = location_null
            
            eps = tf.random_normal((tf.shape(location)[0],
                                    self.z_length_max,
                                    self.FLAGS.embedding_sample_size),
                                   mean=0.0,stddev=1.0,dtype=self.tf_float_dtype)
            
            self.z_hidden_variable = location + eps*tf.exp(scale * .5)
            self.z_hidden_variable_null = location_null + eps*tf.exp(scale_null * .5)
          
            
            self.KL_divergence = 0.5 * tf.reduce_sum(tf.square(location) + tf.square(scale) - tf.log(tf.square(scale)) - 1, axis=2)
            
# =============================================================================
#         
# =============================================================================
        
    def build_alignment(self):
        
        self.emission = self.get_emission(self.z_hidden, self.z_hidden_null)
        
        emission = self.get_emission(self.z_hidden_variable, self.z_hidden_variable_null)
                
        target_mask = tf.ones_likes(self.z_hidden_variable, self.tf_float_dtype)
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
        
        self.alignment_expectation = alignment_expectation
        
    def get_emission(self, y_hidden_variable, y_hidden_variable_null):
        with tf.variable_scope('emission', reuse=tf.AUTO_REUSE):
            
            nnLayer_emission = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                        activation=tf.nn.tanh,
                                        use_bias=True,
                                        name='nnLayerEmission')
                                               
            target_state = nnLayer_emission(y_hidden_variable)
            target_state_null = nnLayer_emission(y_hidden_variable_null)
            
            target_state = tf.nn.dropout(x=target_state,
                                           keep_prob=self.keep_prob,
                                           name='nnDropoutNonNull')
                                           
            target_state_null = tf.nn.dropout(x=target_state_null,
                                           keep_prob=self.keep_prob,
                                           name='nnDropoutNull')
            
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

# =============================================================================
# 
# =============================================================================
    def print_log(self, update_info, start_time):
        cost = update_info[0]
        cost_reconstruction = update_info[1]
        cost_alignment = update_info[2]
        cost_KL = update_info[3]
        
        if self.global_step.eval() % self.FLAGS.display_freq == 0:
            self.log.info('Epoch %d , Step %d , Cost: %.5f (T: %.5f, A: %.5f, KL: %.5f) in %ds at %s', 
                          self.global_epoch_step.eval(),
                          self.global_step.eval(),
                          cost,cost_reconstruction,cost_alignment, cost_KL,
                          time.time() - self.start_time,
                          tools.get_time_now())
            self.start_time = time.time()
        
    def train_batch(self,
                    sources,source_lengths,
                    targets,null,target_lengths,
                    z_length_max):
        
#        if self.global_step.eval() % self.FLAGS.alpha_KL_divergence_freq == 0:
#            if self.annealing_KL_divergence.eval() <= 1.:
#                self.sess.run(tf.assign(self.annealing_KL_divergence,
#                                        self.annealing_KL_divergence +
#                                        tf.cast(10e-3, dtype=self.tf_float_dtype)))
#            else:
#                self.sess.run(tf.assign(self.annealing_KL_divergence,
#                                        tf.cast(1., dtype=self.tf_float_dtype)))
        
        input_feed = {}
        input_feed[self.keep_prob.name] = self.FLAGS.keep_prob
        input_feed[self.sources.name] = sources
        input_feed[self.targets.name] = targets
        input_feed[self.source_lengths.name] = source_lengths
        input_feed[self.target_lengths.name] = target_lengths
        
        input_feed[self.targets_null.name] = null
        input_feed[self.sources_null.name] = null
        
        input_feed[self.z_length_max.name] = z_length_max
        
        output_feed = [#self.updates,
                       #self.cost,
                       #self.cost_reconstruction_expectation,
                       #self.cost_alignment_expectation,
                       #self.cost_KL_divergence
                       self.test
                       ]
                       
        outputs = self.sess.run(output_feed, input_feed)
        
        print(np.shape(outputs[0]))
        
        return outputs[1], outputs[2], outputs[3], outputs[4]
        return 0,0,0,0
        
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
    
# =============================================================================
#         
# =============================================================================
        
    def prepare_batch(self, seqs_x, seqs_y):
        x_lengths = np.array([len(s) for s in seqs_x])
        y_lengths = np.array([len(s) for s in seqs_y])
        x_lengths_max = np.max(x_lengths)
        y_lengths_max = np.max(y_lengths)

        z_length_max = x_lengths_max if x_lengths_max > y_lengths_max else y_lengths_max
        
        self.batch_size = len(seqs_x)
        self.update_freq = len(seqs_x)
                     
        null = [self.FLAGS.target_vocabulary_size - 1]
        
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

        return word_x, word_x_lengths, word_y, null, word_y_lengths, z_length_max
    
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
            #self.evaluate_alignment()
            #self.evaluate_reconstruction()
            self.log.info('------------------')
        
    def evaluate_reconstruction(self):
        # Execute a validation step
        if self.FLAGS.valid_freq!= 0 and self.global_step.eval() % self.FLAGS.valid_freq == 0:
            start_time = time.time()
            
            reconstruction_set = []
            accuracy_set = []
            
            for valid_seq in self.valid_set:
                batch = self.prepare_batch(*valid_seq)
                eval_info = self.decode_reconstruction(*batch)
                
                for idx in range(1):
                    target_length = batch[4][idx]
                    target = batch[2][idx]
                    
                    num_word_correct = 0
                    for index_word, (predited, correct) in enumerate(zip(eval_info[0][idx], target)):
                        if index_word < target_length:
                            if predited == correct:
                                num_word_correct+=1
                                
                    accuracy_set.append(num_word_correct/target_length)
                    
                reconstruction_set.append(np.prod(eval_info[1]))
                    
            ACC = np.mean(accuracy_set)
            likelihood_score = np.mean(reconstruction_set)
            
            self.log.info('VALIDATION: Epoch %d , Step %d , Likelihood: %.10f , ACC: %.5f in %ds at %s',
                          self.global_epoch_step.eval(), 
                          self.global_step.eval() , 
                          likelihood_score, 
                          ACC, 
                          time.time() - start_time, 
                          tools.get_time_now())

            self.start_time = time.time()
            
    def evaluate_alignment(self):
        # Execute a validation step
        if self.FLAGS.valid_freq!= 0 and self.global_step.eval() % self.FLAGS.valid_freq == 0:
            start_time = time.time()
            
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
                
            alignment_set = self.converse_BPE_to_word(alignment_set)

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
# 
# =============================================================================
        
    def print_model_info(self):
        self.log.info('Source word vocabulary size: %s', self.FLAGS.source_vocabulary_size)
        self.log.info('Target word vocabulary size: %s', self.FLAGS.target_vocabulary_size)
        self.log.info('Word hidden units: %s', self.FLAGS.hidden_units)
        self.log.info('Word embedding size: %s', self.FLAGS.embedding_size)
        
        self.log.info('Alpha reconstruction expectation: %s', self.FLAGS.alpha_reconstruction_expectation)
        self.log.info('Alpha alignment expectation: %s', self.FLAGS.alpha_alignment_expectation)
        self.log.info('Alpha KL divergence: %s', self.FLAGS.alpha_KL_divergence_freq)
        self.log.info('Transition uniform: %s', self.FLAGS.uniform_transition)
