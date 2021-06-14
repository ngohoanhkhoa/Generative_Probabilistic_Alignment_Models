#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io, time, math
import numpy as np
import tensorflow as tf
import framework.tools as tools

from model.model_variational_based.model_variational_bpe_ibm1 import Model

class Model(Model):

    def __init__(self, FLAGS, session, log):
        
        super(Model, self).__init__(FLAGS, session, log)
        
    def build_variational_encoder(self):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            target_embedding= tf.get_variable(name='targetEmbedding',
                                                shape=[self.FLAGS.target_vocabulary_size,
                                                       self.FLAGS.embedding_size],
                                                initializer=self.initializer,
                                                dtype=self.tf_float_dtype)
            
            target_embedded_word = tf.nn.embedding_lookup(params=target_embedding,
                                                      ids=self.targets)
    
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
            
            self.y_hidden = location

    
            eps = tf.random_normal(tf.shape(location),mean=0.0,stddev=1.0,dtype=self.tf_float_dtype)
            eps_null = tf.random_normal(tf.shape(tf.reduce_mean(location, axis=[0,1])),mean=0.0,stddev=1.0,dtype=self.tf_float_dtype)
            
            self.y_hidden_variable = location + eps*tf.exp(scale * .5)
            
            self.y_hidden_variable_null = tf.reduce_mean(location, axis=[0,1]) + eps_null
            self.y_hidden_null = tf.reduce_mean(location, axis=[0,1]) + eps_null
            
            target_mask = tf.sequence_mask(lengths=self.target_lengths, maxlen=tf.shape(location)[1],
                                           dtype=self.tf_float_dtype, name='targetMask')
            
            KL_divergence = 0.5 * tf.reduce_sum(tf.square(location) + tf.square(scale) - tf.log(tf.square(scale)) - 1, axis=2)
            
            self.KL_divergence = KL_divergence * target_mask
            
            
    def train_batch(self,
                  sources,
                  source_lengths,
                  targets,
                  targets_null,
                  target_lengths):
        
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
        
        output_feed = [self.updates,
                       self.cost,
                       self.cost_reconstruction_expectation,
                       self.cost_alignment_expectation,
                       self.cost_KL_divergence
                       ]
                       
        outputs = self.sess.run(output_feed, input_feed)
        
        return outputs[1], outputs[2], outputs[3], outputs[4]
    
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