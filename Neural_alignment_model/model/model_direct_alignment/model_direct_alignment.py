#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import time, io
import random
import framework.tools as tools

from model.model import BaseModel


class Model(BaseModel):

    def __init__(self, FLAGS, session, log):

        super(Model, self).__init__(FLAGS, session, log)
        
    def build_model(self):
        self.build_word_model_placeholders()
        
        self.build_optimizer()
        self.build_initializer()
        
        self.build_alignment()
        
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
# =============================================================================
#             
# =============================================================================
        
    def build_word_model_placeholders(self):
        
        self.sources = tf.placeholder(dtype=self.tf_int_dtype,
                                      shape=(None,None),
                                      name='sources')
        
        self.targets_positive = tf.placeholder(dtype=self.tf_int_dtype,
                                      shape=(None,None),
                                      name='targets_positive')
        
        
        self.targets_negative = tf.placeholder(dtype=self.tf_int_dtype,
                                      shape=(None,None),
                                      name='targets_negative')
                                      
        self.source_lengths = tf.placeholder(dtype=self.tf_int_dtype, 
                                             shape=(None),
                                             name='source_lengths')
                                             
        self.target_positive_lengths = tf.placeholder(dtype=self.tf_int_dtype,
                                             shape=(None),
                                             name='target_positive_lengths')
        
        self.target_negative_lengths = tf.placeholder(dtype=self.tf_int_dtype,
                                             shape=(None),
                                             name='target_negative_lengths')
        
        
    
# =============================================================================
# 
# =============================================================================
        
    def build_update(self):
        positive_alignment = self.positive_alignment
        negative_alignment = self.negative_alignment
        
        positive_alignment = tf.reduce_max(positive_alignment,axis=1)
        negative_alignment = tf.reduce_max(negative_alignment,axis=1)
        
        positive_alignment = tf.log(1+ tf.exp(-positive_alignment) )
        negative_alignment = tf.log(1+ tf.exp(negative_alignment) )

        positive_alignment = tf.reduce_sum(positive_alignment,axis=1)
        negative_alignment = tf.reduce_sum(negative_alignment,axis=1)
        
        self.cost_positive_alignment = tf.reduce_mean(positive_alignment)
        self.cost_negative_alignment = tf.reduce_mean(negative_alignment)
        
        trainable_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        
        self.cost = self.cost_positive_alignment + self.cost_negative_alignment
        gradients = tf.gradients(self.cost, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.updates = self.opt.apply_gradients(zip(clip_gradients,trainable_params),
                                                         global_step=self.global_step)
        
    def build_alignment(self):
        with tf.variable_scope('source', reuse=tf.AUTO_REUSE):
            hidden_state_source = self.get_hidden_state(self.sources, self.source_lengths, self.FLAGS.source_vocabulary_size)
        with tf.variable_scope('target', reuse=tf.AUTO_REUSE):
            hidden_state_target_positive = self.get_hidden_state(self.targets_positive, self.target_positive_lengths, self.FLAGS.target_vocabulary_size)
            hidden_state_target_negative = self.get_hidden_state(self.targets_negative, self.target_negative_lengths, self.FLAGS.target_vocabulary_size)

        self.positive_alignment = tf.matmul(a=hidden_state_source,
                                       b=hidden_state_target_positive,
                                       transpose_b=True)
        
        self.negative_alignment = tf.matmul(a=hidden_state_source,
                                       b=hidden_state_target_negative,
                                       transpose_b=True)
        
        
                                         
    def get_hidden_state(self, input_batch, sentence_lengths, vocabulary_size):
        # NN parameters Null
        embedding= tf.get_variable(name='embedding',
                                   shape=[vocabulary_size,
                                   self.FLAGS.embedding_size],
                                   initializer=self.initializer,
                                   dtype=self.tf_float_dtype)
        
        embedded_word = tf.nn.embedding_lookup(params=embedding,
                                                      ids=input_batch)
        
        
        lstm_cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.FLAGS.hidden_units)
        lstm_cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.FLAGS.hidden_units)
                                                                       
        with tf.variable_scope('lstmTarget'):

                (output_fw, output_bw),\
                (output_state_fw, output_state_bw) = \
                tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_cell_fw,
                                                cell_bw=lstm_cell_bw, 
                                                inputs=embedded_word,
                                                sequence_length=sentence_lengths,
                                                dtype=self.tf_float_dtype)
                                                
                hidden_state = tf.concat([output_fw, output_bw], axis=2)
                
                nnLayer_lstm = tf.layers.Dense(units=self.FLAGS.hidden_units,
                                    activation=None,
                                    use_bias=False,
                                    name='nnLayerLSTM')

                hidden_state = nnLayer_lstm(hidden_state)
        
        return hidden_state
            
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
        
        word_x = np.ones((self.batch_size,
                     x_lengths_max),
                    dtype=self.int_dtype) * tools.end_token
                
        word_x_lengths = np.ones((self.batch_size),
                                    dtype=self.int_dtype)
        
        word_y_positive = np.ones((self.batch_size,
                                   y_lengths_max),
                                dtype=self.int_dtype) * tools.end_token
        
        word_y_negative = np.ones((self.batch_size,
                                   y_lengths_max),
                                dtype=self.int_dtype) * tools.end_token
        
        word_y_positive_lengths = np.ones((self.batch_size),
                                          dtype=self.int_dtype)
        
        word_y_negative_lengths = np.ones((self.batch_size),
                                          dtype=self.int_dtype)

            
        for idx in range(len(seqs_x)):
            word_x[idx, :x_lengths[idx]] = seqs_x[idx]
            word_x_lengths[idx] = x_lengths[idx]
            
            word_y_positive[idx, :y_lengths[idx]] = seqs_y[idx]
            word_y_positive_lengths[idx] = y_lengths[idx]
            
            if idx != len(seqs_x)-1:
                s_y_ = np.array(seqs_y[idx+1])
                random.shuffle(s_y_)
                word_y_negative[idx, :y_lengths[idx+1]] = s_y_
                word_y_negative_lengths[idx] = y_lengths[idx+1]
            else:
                s_y_ = np.array(seqs_y[0])
                random.shuffle(s_y_)     
                word_y_negative[idx, :y_lengths[0]] = s_y_
                word_y_negative_lengths[idx] = y_lengths[0]

        return word_x, word_x_lengths, \
    word_y_positive, word_y_positive_lengths, \
    word_y_negative, word_y_negative_lengths
    
    def train_batch(self,
                  sources,
                  source_lengths,
                  targets_positive,
                  target_positive_lengths,
                  targets_negative,
                  target_negative_lengths):
        
        input_feed = {}
        input_feed[self.keep_prob.name] = self.FLAGS.keep_prob
        input_feed[self.sources.name] = sources
        input_feed[self.targets_positive.name] = targets_positive
        input_feed[self.targets_negative.name] = targets_negative
        input_feed[self.source_lengths.name] = source_lengths
        input_feed[self.target_positive_lengths.name] = target_positive_lengths
        input_feed[self.target_negative_lengths.name] = target_negative_lengths
        
        output_feed = [self.updates,
                       self.cost
                       ]
                       
        outputs = self.sess.run(output_feed, input_feed)
        
        return outputs[1]
    
    
    def eval(self,
                  sources,
                  source_lengths,
                  targets_positive,
                  target_positive_lengths,
                  targets_negative,
                  target_negative_lengths):
        
        target_length = target_positive_lengths[0]
        source_length = source_lengths[0]
        source = sources[0][:source_length]
        
        input_feed = {}
        input_feed[self.keep_prob.name] = 1.0
        input_feed[self.sources.name] = sources
        input_feed[self.targets_positive.name] = targets_positive
        input_feed[self.targets_negative.name] = targets_negative
        input_feed[self.source_lengths.name] = source_lengths
        input_feed[self.target_positive_lengths.name] = target_positive_lengths
        input_feed[self.target_negative_lengths.name] = target_negative_lengths

        
        output_feed = [self.positive_alignment]
        #----------------------------------------------------------------------
        outputs = self.sess.run(output_feed, input_feed)
        
        positive_alignment = outputs[0][0]
        
        state_seqs = np.argmax(positive_alignment,axis=1)
                    
        return [state_seqs]
    
    
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
            
            for valid_seq in self.valid_set:
                batch = self.prepare_batch(*valid_seq)
                eval_info = self.eval(*batch)
                        
                #--------------------------------------------------------------
                
                for idx in range(1):
                    source_length = batch[1][idx]
                    target_length = batch[3][idx]
                    alignment = []
                    for index_source, index_target in enumerate(eval_info[idx]):
                        if index_source < source_length and index_target < target_length:
                            # Check alignment reference: Start from 0 or 1 ?
                            alignment.append(str(index_source+self.evaluate_alignment_start_from) \
                            + "-" + str(index_target+self.evaluate_alignment_start_from))
                            #alignment.append(str(index_source) + "-" + str(index_target))
                            
                    alignment_set.append(alignment)

            AER = self.calculate_AER(alignment_set)
            
            self.log.info('VALIDATION: Epoch %d , Step %d , AER: %.5f in %ds at %s',
                          self.global_epoch_step.eval(), 
                          self.global_step.eval() ,
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
                    
#==============================================================================
# 
#==============================================================================
    
    def print_model_info(self):
        self.log.info('Source word vocabulary size: %s', self.FLAGS.source_vocabulary_size)
        self.log.info('Target word vocabulary size: %s', self.FLAGS.target_vocabulary_size)
        self.log.info('Word hidden units: %s', self.FLAGS.hidden_units)
        self.log.info('Word embedding size: %s', self.FLAGS.embedding_size)
