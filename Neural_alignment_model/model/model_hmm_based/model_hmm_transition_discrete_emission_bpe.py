#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import io, time
import framework.tools as tools

# =============================================================================
# Target BPE
# Embedding: Feed forward
# Emission: Feed forward
# Transition: Discrete
# =============================================================================

#==============================================================================
# Extra vocabulary BPE symbols 
unk_token_bpe = 0
start_token_bpe = 1
end_token_bpe = 2
#==============================================================================


from model.model_hmm_based.model_hmm_transition_discrete_emission_word import Model

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
        target_state = nnLayer(target_embedded_word)
        
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
        
#==============================================================================
# 
#==============================================================================
    
    def print_model_info(self):
        self.log.info('Source word vocabulary size: %s', self.FLAGS.source_vocabulary_size)
        self.log.info('Target word vocabulary size: %s', self.FLAGS.target_vocabulary_size)
        self.log.info('Word hidden units: %s', self.FLAGS.hidden_units)
        self.log.info('Word embedding size: %s', self.FLAGS.embedding_size)
