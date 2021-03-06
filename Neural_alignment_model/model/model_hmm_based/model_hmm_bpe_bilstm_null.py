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
# Transition: Bi-LSTM
# =============================================================================

#==============================================================================
# Extra vocabulary BPE symbols 
unk_token_bpe = 0
start_token_bpe = 1
end_token_bpe = 2
#==============================================================================
from model.model_hmm_based.model_hmm import Model

class Model(Model):

    def __init__(self, FLAGS, session, log):

        super(Model, self).__init__(FLAGS, session, log)
        
    def build_model(self):
        self.get_reference_BPE_idx()
        
        self.initialize_transition_parameter_nn()
        
        self.build_optimizer()
        self.build_initializer()
        
        self.build_word_model_placeholders()
        
        self.build_emission()
        self.build_transition()
        self.build_forward_backward_nn()
        
        self.build_update_emission()
        self.build_update_transition()
        
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

    def assign_variable_training_transition(self):
        self.assign_variable_training_transition_()

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
        #target_state = nnLayer(target_state)

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
    
    def get_transition(self):
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
                
                nnLayer_lstm = tf.layers.Dense(units=self.FLAGS.embedding_size,
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

    def train_batch(self,
                  sources,
                  source_lengths,
                  targets,
                  targets_null,
                  target_lengths):
        
        input_feed = {}
        input_feed[self.keep_prob.name] = self.FLAGS.keep_prob
        input_feed[self.sources.name] = sources
        input_feed[self.targets.name] = targets
        input_feed[self.source_lengths.name] = source_lengths
        input_feed[self.target_lengths.name] = target_lengths
        
        input_feed[self.targets_null.name] = targets_null
        
        output_feed = [self.updates_emission,
                       self.cost_emission,
                       self.updates_transition,
                       self.cost_transition
                       ]
                       
        outputs = self.sess.run(output_feed, input_feed)

        cost = outputs[1] + outputs[3]
        
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
        
        output_feed = [self.emission_prob, 
                       self.transition_prob]
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
        
        self.batch_size = len(seqs_x)
        self.update_freq = len(seqs_x)
                     
        y_null = [self.FLAGS.target_vocabulary_size - 1]
        
        word_x = np.ones((self.batch_size,
                     x_lengths_max),
                    dtype=self.int_dtype) * end_token_bpe
                
        word_x_lengths = np.ones((self.batch_size),
                                    dtype=self.int_dtype)
        
        word_y = np.ones((self.batch_size,
                     y_lengths_max),
                    dtype=self.int_dtype) * end_token_bpe
        
        word_y_lengths = np.ones((self.batch_size),
                                    dtype=self.int_dtype)
                
        for idx, s_x in enumerate(seqs_x):
            word_x[idx, :x_lengths[idx]] = s_x
            word_x_lengths[idx] = x_lengths[idx]
            
        for idx, s_y in enumerate(seqs_y):
            word_y[idx, :y_lengths[idx]] = s_y
            word_y_lengths[idx] = y_lengths[idx]

        return word_x, word_x_lengths, word_y, y_null, word_y_lengths
    
    
# =============================================================================
# 
# =============================================================================

    def train(self):
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
# 
# =============================================================================
    
    def print_model_info(self):
        self.log.info('Source word vocabulary size: %s', self.FLAGS.source_vocabulary_size)
        self.log.info('Target word vocabulary size: %s', self.FLAGS.target_vocabulary_size)
        self.log.info('Word hidden units: %s', self.FLAGS.hidden_units)
        self.log.info('Word embedding size: %s', self.FLAGS.embedding_size)
